from __future__ import annotations
import logging
import multiprocessing
import os
import subprocess
import pandas as pd
import numpy as np
import argparse
import json
import time
from typing import List, Dict, Tuple, Any

from implementation.seqcache import read_cache

logger = logging.getLogger('base-logger')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


class PropetsClassifier(object):

    def __init__(self, ip_entropies: pd.Series, primary_set: Dict[str, List[str]],
                 secondary_set: Dict[str, List[str]]):
        self.ip_entropies: pd.Series = ip_entropies
        self.primary_set: Dict[str, List[str]] = primary_set
        self.secondary_set: Dict[str, List[str]] = secondary_set
        self.index_misses: int = 0
        self.total_queries: int = 0

    def fit(self):
        return self

    def _predict(self, file_name: str, url: str) -> str:
        df = pd.read_csv(file_name, sep=';')
        domain_name = get_domain_name(url)
        try:
            primary_ip = df.set_index('domain').loc[domain_name, 'ip']
            self.total_queries += 1
        except Exception as e:
            logger.exception(f"Did not find domain {domain_name} in ip-resolution {file_name}.")
            self.index_misses += 1
            return 'unknown'
        if type(primary_ip) is not str:
            primary_ip = primary_ip[0]
        candidate_labels = []
        for lbl, ips in self.primary_set.items():
            if primary_ip in ips:
                candidate_labels.append(lbl)
        if len(candidate_labels) == 0:
            return 'unknown'
        secondary_ips = df.loc[np.logical_not(df.loc[:, 'ip'].isin([primary_ip])), 'ip']
        entropies = []
        for clbl in candidate_labels:
            entropy = 0
            sips = self.secondary_set[clbl]
            for secondary_ip in secondary_ips:
                if secondary_ip in sips:
                    entropy += self.ip_entropies[secondary_ip]
            entropies.append(entropy)
        return candidate_labels[np.argmax(entropies)]

    def predict(self, file_names: List[str], urls: List[str]) -> List[str]:
        predictions = []
        for file_name, url in zip(file_names, urls):
            predictions.append(self._predict(file_name, url))
        return predictions


def make_fname(name: str) -> str:
    return f'/opt/project/data/popets-data/{name}-ip-resolutions.csv'


def get_domain_name(url: str) -> str:
    """
    Extract the domain name from an URL. URL has to start with https://.

    Args:
        url: URL that has been queried.

    Returns:
        Name of the domain from the URL.
    """
    assert url.startswith('https://'), "URL expected to start with https://"
    hostname = url[len('https://'):]
    idx = hostname.find('/')
    # assert idx > 0, f"Could not find slash in url {hostname}."
    if idx < 0:
        return hostname
    else:
        return hostname[:idx]


def get_closed_world_labels() -> List[str]:
    with open("/opt/project/closed-world-labels.json", "r") as fh:
        closed_world_labels = json.load(fh)
    return closed_world_labels


def get_open_world_labels() -> List[str]:
    with open("/opt/project/open-world-labels.json", "r") as fh:
        open_world_labels = json.load(fh)
    return open_world_labels


def get_resolution_files(meta_data: List[Tuple[str, List[Dict[str, Any]]]],
                         indicator: Dict[str, str], labels: Dict[str, str],
                         closed_world_labels: List[str], is_open_world: bool) -> List[Dict[str, str | int]]:
    cached_result = f'/opt/project/data/popets-fingerprints/training-files-{"open" if is_open_world else "closed"}.json'
    if os.path.exists(cached_result):
        logger.info(f"Restore from {cached_result}")
        with open(cached_result, 'r') as fh:
            training_files = json.load(fh)
    else:
        training_files = []
        for day, url_metas in meta_data:
            # Needed to detect if a file for a label has already been included.
            # Reset for each day.
            unknown_set = []
            for url_meta in url_metas:
                url_id = str(url_meta['url_id'])
                if url_id not in labels: continue
                if url_id not in indicator: continue
                lbl = labels[url_id]
                # Get all files in the training and validation set.
                if indicator[url_id] in ['train', 'val']:
                    fname = make_fname(url_meta['filename'])
                    if os.path.exists(fname):
                        if lbl in closed_world_labels:
                            training_files.append({'fname': fname, 'url_id': int(url_meta['url_id'])})
                        elif is_open_world and lbl not in unknown_set:
                            training_files.append({'fname': fname, 'url_id': int(url_meta['url_id'])})
                            unknown_set.append(lbl)
                        else:
                            pass
        assert len(training_files) > 0
        with open(cached_result, "w") as fh:
            json.dump(obj=training_files, fp=fh)
    return training_files


def _merge_ip_resolutions_mp(resolution_files: List[Dict[str, str | int]]) -> Tuple[np.array, pd.DataFrame]:
    num_processes = 30
    bins = [[] for _ in range(num_processes)]
    for i, rf in enumerate(resolution_files):
        bins[i % num_processes].append(rf)
    logger.info(f"Create {num_processes} bins with {np.mean([len(b) for b in bins])} elements each on average.")
    logger.info(f"Start Scattering...")
    pool = multiprocessing.Pool(processes=num_processes)
    ret = pool.map(_merge_ip_resolutions, bins)
    pool.close()

    logger.info("Gather results.")
    all_domains = None
    all_resolutions = None
    for domains, resolutions in ret:
        all_domains = domains if all_domains is None else np.concatenate([all_domains, domains])
        all_resolutions = resolutions if all_resolutions is None else pd.concat([all_resolutions, resolutions], axis=0)
    return all_domains, all_resolutions


def _merge_ip_resolutions(resolution_files: List[Dict[str, str | int]]) -> Tuple[np.array, pd.DataFrame]:
    """
    Reads the resolution files and merges them into a single data-frame.
    For each call, note the unique IP <-> domain mappings, since one domain
    is contacted multiple times with the same IP in some files, i.e., the
    file contains duplicate rows.

    Args:
        resolution_files:

    Returns:

    """
    t_start = time.time()
    logger.info(f"Start merging of {len(resolution_files)} files.")
    resolutions = None
    domains = np.array([])
    for i, data in enumerate(resolution_files):
        f = data['fname']
        url_id = data['url_id']
        if not f.endswith("ip-resolutions.csv"):
            continue
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f, sep=";").drop_duplicates()
            df['url_id'] = url_id
            if resolutions is None:
                resolutions = df
            else:
                resolutions = pd.concat([resolutions, df], axis=0)
            # Get all unique domains in this webpage call, and concatenate
            # these with the already pulled ones. The array will contain
            # for each webpage call the domain names that were contacted.
            domains = np.concatenate([domains, df.loc[:, 'domain'].unique()])
        except Exception as e:
            logger.error(f"Unexpected error processing file {f}")
            logger.exception(e)
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i} of {len(resolution_files)} in {time.time() - t_start}s, "
                        f"{(len(resolution_files) - i) / (time.time() - t_start)}s "
                        f"per file on average.")
    return domains, resolutions


def merge_ip_resolutions(resolution_files: List[Dict[str, str | int]],
                         use_mp=True) -> Tuple[pd.Series, pd.DataFrame]:
    h5_file_res = "/opt/project/data/popets-fingerprints/all-resolutions2.h5"
    h5_file_dom = "/opt/project/data/popets-fingerprints/domain-entropy2.h5"
    if os.path.exists(h5_file_res):
        logger.info(f"Restore from {h5_file_dom} and {h5_file_res}")
        resolutions = pd.read_hdf(path_or_buf=h5_file_res, key='all-resolution')
        domain_entropies = pd.read_hdf(path_or_buf=h5_file_dom, key='domain-entropies')
    else:
        logger.info(f"Use Multiprocessing to merge: {'yes' if use_mp else 'no'}")
        domains, resolutions = _merge_ip_resolutions_mp(resolution_files) if use_mp else \
            _merge_ip_resolutions(resolution_files)
        domain_entropies: pd.Series = -1. * np.log(pd.Series(domains).value_counts(normalize=True, dropna=True))
        domain_entropies.to_hdf(key="domain-entropies", path_or_buf=h5_file_dom)
        resolutions.to_hdf(key='all-resolution', path_or_buf=h5_file_res)
    return domain_entropies, resolutions


def make_ip_entropies(resolutions: pd.DataFrame, domain_entropies: pd.Series) -> pd.Series:
    h5_file_res = "/opt/project/data/popets-fingerprints/ip-entropies2.h5"
    if os.path.exists(h5_file_res):
        logger.info(f"Restore from {h5_file_res}")
        ip_entropies = pd.read_hdf(path_or_buf=h5_file_res, key='ip-entropies')
    else:
        unique_ips = resolutions.loc[:, 'ip'].unique()
        resolutions = resolutions.set_index('ip')
        ip_entropies = pd.Series(dtype=np.float32)
        for ip in unique_ips:
            domains = resolutions.loc[ip, 'domain']
            if type(domains) == str:
                domains = [domains]
            else:
                domains = domains.unique()
            entropy = domain_entropies.loc[domains].mean()
            ip_entropies[ip] = entropy
        ip_entropies.to_hdf(path_or_buf=h5_file_res, key='ip-entropies')
    return ip_entropies


def _make_primary_set(meta_data: List[Tuple[str, List[Dict[str, Any]]]],
                     indicator: Dict[str, str], labels: Dict[str, str],
                     closed_world_labels: List[str], all_resolutions: pd.DataFrame,
                     is_open_world: bool) -> Dict[str, List[str]]:
    all_resolutions = all_resolutions.set_index('domain')
    # Maps a label to a list of primary IPs that were first contacted.
    primary_set: Dict[str, List[str]] = {}
    for day, url_metas in meta_data:
        logger.info(f"Process day {day}")
        # Needed to detect if a file for a label has already been included.
        # Reset for each day.
        unknown_set = []
        for i, url_meta in enumerate(url_metas):
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} of {len(url_metas)}")
            url_id = str(url_meta['url_id'])
            if url_id not in labels: continue
            if url_id not in indicator: continue
            lbl = labels[url_id]
            # Get all files in the training and validation set.
            if indicator[url_id] in ['train', 'val']:
                to_add = lbl in closed_world_labels
                if is_open_world and lbl not in unknown_set:
                    unknown_set.append(lbl)
                    to_add = True
                if to_add:
                    # Get the domain name as used in the DNS query from the URL.
                    domain_name = get_domain_name(url_meta['url'])
                    # Get all IPs that are associated with this domain name.
                    ips = all_resolutions.loc[domain_name, 'ip']
                    if type(ips) == str:
                        ips = np.array([ips])
                    else:
                        ips = ips.unique()
                    if lbl not in primary_set:
                        primary_set[lbl] = []
                    # Add each primary IP that is not yet part of the set to the set.
                    primary_set[lbl].extend([str(ip) for ip in ips if ip not in primary_set[lbl]])
    return primary_set


def _merge_ip_set_dicts(destination: Dict[str, List[str]], source: Dict[str, List[str]]):
    for lbl, ips in source.items():
        if lbl not in destination:
            destination[lbl] = []
        for ip in ips:
            if ip not in destination[lbl]:
                destination[lbl].append(ip)


def _make_primary_set_mp(days: List[str]) -> Dict[str, List[str]]:
    h5_file_res = "/opt/project/data/popets-fingerprints/all-resolutions2.h5"
    indicator: Dict[str, str] = read_cache('indicator.json')
    labels: Dict[str, str] = read_cache('labels.json')
    meta_data: List[Tuple[str, List[Dict[str, Any]]]] = read_cache("meta_data.json")
    closed_world_labels = get_closed_world_labels()
    all_resolutions: pd.DataFrame = pd.read_hdf(path_or_buf=h5_file_res, key='all-resolution')

    primary_set: Dict[str, List[str]] = {}
    for day, url_metas in meta_data:
        if day not in days:
            continue
        tmp = _make_primary_set(
            meta_data=[(day, url_metas)],
            indicator=indicator,
            labels=labels,
            closed_world_labels=closed_world_labels,
            all_resolutions=all_resolutions,
            is_open_world=False
        )
        _merge_ip_set_dicts(primary_set, tmp)
    return primary_set


def make_primary_set(meta_data: List[Tuple[str, List[Dict[str, Any]]]]):
    is_open_world = False
    cached_result = f'/opt/project/data/popets-fingerprints/primary-set-{"open" if is_open_world else "closed"}2.json'
    if os.path.exists(cached_result):
        logger.info(f"Restore from {cached_result}")
        with open(cached_result, 'r') as fh:
            primary_set = json.load(fh)
    else:
        num_p = 22
        bins = [[] for _ in range(num_p)]
        for i, (day, _) in enumerate(meta_data):
            bins[i % len(bins)].append(day)
        logger.info(f"Create {num_p} bins with {np.mean([len(b) for b in bins])} elements each on average.")
        logger.info(f"Start Scattering...")
        pool = multiprocessing.Pool(processes=num_p)
        ret = pool.map(_make_primary_set_mp, bins)
        pool.close()

        logger.info(f"Start Gathering...")
        primary_set: Dict[str, List[str]] = {}
        for tmp in ret:
            _merge_ip_set_dicts(primary_set, tmp)
        with open(cached_result, 'w') as fh:
            json.dump(primary_set, fh, indent=1)
    return primary_set


def _make_secondary_set(meta_data: List[Tuple[str, List[Dict[str, Any]]]],
                       indicator: Dict[str, str], labels: Dict[str, str],
                       closed_world_labels: List[str], all_resolutions: pd.DataFrame,
                       is_open_world: bool) -> Dict[str, List[str]]:
    all_resolutions = all_resolutions.set_index('url_id')
    # Maps a label to a list of primary IPs that were first contacted.
    secondary_set: Dict[str, List[str]] = {}
    for day, url_metas in meta_data:
        logger.info(f"Process day {day}")
        # Needed to detect if a file for a label has already been included.
        # Reset for each day.
        unknown_set = []
        for i, url_meta in enumerate(url_metas):
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} of {len(url_metas)}")
            url_id = str(url_meta['url_id'])
            if url_id not in labels: continue
            if url_id not in indicator: continue
            lbl = labels[url_id]
            # Get all files in the training and validation set.
            if indicator[url_id] in ['train', 'val']:
                to_add = lbl in closed_world_labels
                if is_open_world and lbl not in unknown_set:
                    unknown_set.append(lbl)
                    to_add = True
                if to_add:
                    # Get the domain name as used in the DNS query from the URL.
                    domain_name = get_domain_name(url_meta['url'])
                    tmp = all_resolutions.loc[int(url_id), :]
                    ips = tmp.loc[np.logical_not(tmp.domain.isin([domain_name])), 'ip']
                    if type(ips) == str:
                        ips = np.array([ips])
                    else:
                        ips = np.unique(ips)
                    if lbl not in secondary_set:
                        secondary_set[lbl] = []
                        secondary_set[lbl].extend([str(ip) for ip in ips])
    return secondary_set


def _make_secondary_set_mp(days: List[str]) -> Dict[str, List[str]]:
    h5_file_res = "/opt/project/data/popets-fingerprints/all-resolutions2.h5"
    indicator: Dict[str, str] = read_cache('indicator.json')
    labels: Dict[str, str] = read_cache('labels.json')
    meta_data: List[Tuple[str, List[Dict[str, Any]]]] = read_cache("meta_data.json")
    closed_world_labels = get_closed_world_labels()
    all_resolutions: pd.DataFrame = pd.read_hdf(path_or_buf=h5_file_res, key='all-resolution')

    secondary_set: Dict[str, List[str]] = {}
    for day, url_metas in meta_data:
        if day not in days:
            continue
        tmp = _make_secondary_set(
            meta_data=[(day, url_metas)],
            indicator=indicator,
            labels=labels,
            closed_world_labels=closed_world_labels,
            all_resolutions=all_resolutions,
            is_open_world=False
        )
        _merge_ip_set_dicts(secondary_set, tmp)
    return secondary_set


def make_secondary_set(meta_data: List[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, List[str]]:
    is_open_world = False
    cached_result = f'/opt/project/data/popets-fingerprints/secondary-set-{"open" if is_open_world else "closed"}2.json'
    if os.path.exists(cached_result):
        logger.info(f"Restore from {cached_result}")
        with open(cached_result, 'r') as fh:
            secondary_set = json.load(fh)
    else:
        num_p = 22
        bins = [[] for _ in range(num_p)]
        for i, (day, _) in enumerate(meta_data):
            bins[i % len(bins)].append(day)
        logger.info(f"Create {num_p} bins with {np.mean([len(b) for b in bins])} elements each on average.")
        logger.info(f"Start Scattering...")
        pool = multiprocessing.Pool(processes=num_p)
        ret = pool.map(_make_secondary_set_mp, bins)
        pool.close()

        logger.info(f"Start Gathering...")
        secondary_set: Dict[str, List[str]] = {}
        for tmp in ret:
            _merge_ip_set_dicts(secondary_set, tmp)
        with open(cached_result, 'w') as fh:
            json.dump(secondary_set, fh, indent=1)
    return secondary_set


def main(scenario: str):
    logger.info(f"Train and evaluate POPETS on the testset for the {scenario}-world scenario.")
    logger.info("Read indicator...")
    indicator: Dict[str, str] = read_cache('indicator.json')
    logger.info("Read labels...")
    labels: Dict[str, str] = read_cache('labels.json')
    logger.info("Read metadata")
    meta_data: List[Tuple[str, List[Dict[str, Any]]]] = read_cache("meta_data.json")
    closed_world_labels = get_closed_world_labels()

    logger.info("Get Training Files.")
    training_files = get_resolution_files(
        meta_data=meta_data,
        indicator=indicator,
        labels=labels,
        closed_world_labels=closed_world_labels,
        is_open_world=False # need only closed world. Open would should miss the fingerprint then.
    )
    logger.info(f"Retrieved {len(training_files)} files for training.")
    logger.info("Compute the domain entropies.")
    domain_entropies, all_resolutions = merge_ip_resolutions(training_files)
    logger.info("Compute the IP entropies.")
    ip_entropies = make_ip_entropies(all_resolutions, domain_entropies)
    logger.info("Retrieve the primary set of IP addresses for each class.")
    # primary_set = make_primary_set(
    #     meta_data=meta_data,
    #     indicator=indicator,
    #     labels=labels,
    #     closed_world_labels=closed_world_labels,
    #     is_open_world=scenario == 'open',
    #     all_resolutions=all_resolutions
    # )
    primary_set = make_primary_set(meta_data)
    logger.info("Retrieve the secondary set of IP addresses for each class.")
    # secondary_set = make_secondary_set(
    #     meta_data=meta_data,
    #     indicator=indicator,
    #     labels=labels,
    #     closed_world_labels=closed_world_labels,
    #     is_open_world=scenario == 'open',
    #     all_resolutions=all_resolutions
    # )
    secondary_set = make_secondary_set(meta_data)
    logger.info("Build the classifier.")
    classifier = PropetsClassifier(ip_entropies, primary_set, secondary_set)
    confusion_matrices = []
    logger.info("Evaluate performance.")
    for day, url_metas in meta_data:
        logger.info(f"Evaluate day {day}.")
        confusion_matrix = {}
        for url_meta in url_metas:
            url_id = str(url_meta['url_id'])
            if url_id not in labels: continue
            if url_id not in indicator: continue
            if indicator[url_id] != 'test':
                continue
            lbl = labels[url_id]
            if scenario == 'closed' and lbl not in closed_world_labels: continue
            url = url_meta['url']
            file_name = make_fname(url_meta['filename'])
            if not os.path.exists(file_name): continue
            prediction = classifier._predict(file_name, url)
            if lbl not in confusion_matrix:
                confusion_matrix[lbl] = {}
            if prediction not in confusion_matrix[lbl]:
                confusion_matrix[lbl][prediction] = 0
            confusion_matrix[lbl][prediction] += 1
        confusion_matrices.append(confusion_matrix)
    with open(f'/opt/project/data/popets-fingerprints/classification-result-{scenario}.json', 'w') as fh:
        json.dump(confusion_matrices, fh)


if __name__ == "__main__":
    main("open")
