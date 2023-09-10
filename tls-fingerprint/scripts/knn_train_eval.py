from __future__ import annotations

import os
from datetime import datetime
# import multiprocessing as mp
import argparse
import numpy as np
from typing import Tuple, List, Dict, Any, Union
import json
import gc
import logging
import sys
import time
import itertools as itt

import uuid

import implementation.rediswq as rediswq
import implementation.data_conversion.dataprep as dprep
import implementation.classification.knn as knnmod
import implementation.classification.mc as mcmod
import implementation.classification.phmm as phmmmod
import implementation.classification.binary as bnmod
from implementation.classification.seq_classifier import SeqClassifier
import implementation.data_conversion.tls_flow_extraction as tlsex
import implementation.logging_factory as logging_factory
import implementation.data_conversion.constants as constants
from implementation.seqcache import is_cached, add_to_cache, read_cache
use_caching = True

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filemode='w',
    filename=logging_factory.get_log_file(
        name=f'{os.environ.get("POD_NAME")}',
        log_dir='/opt/project/data/grid-search-results/logs'
    ),
    level=logging.INFO
)

class Config(object):

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        return cls(**d)

    def __init__(self, classifier: str, binning_method: str, num_bins: int,
                 seq_length: int, seq_element_type: str, hmm_length: int,
                 day_train_start: int, day_train_end: int, knn_num_neighbors: int,
                 hmm_init_prior: str, hmm_num_iter: int, ano_density_estimator: str,
                 seed: int, uuid_: str=None, trial_dir: str=None, accuracy: float=None,
                 bin_edges: List[float]=None, max_bin_size: int=None,
                 label: str=None, tp:int = None, fp:int = None, tn:int = None,
                 fn:int = None, knn_fraction=None, skip_tls_handshake=False, **kwargs):
        self.uuid = uuid.uuid4().hex if uuid_ is None else uuid_
        self.classifier = classifier
        self.binning_method = binning_method
        self.num_bins = num_bins
        self.day_train_start = day_train_start
        self.day_train_end = day_train_end
        self.days_trained = day_train_end - day_train_start
        self.seq_length = seq_length
        self.seq_element_type = seq_element_type
        self.hmm_length = hmm_length
        self.knn_num_neighbors = knn_num_neighbors
        self.knn_fraction = knn_fraction
        self.seed = seed
        self.hmm_init_prior = hmm_init_prior
        self.hmm_num_iter = hmm_num_iter
        self.ano_density_estimator = ano_density_estimator
        self.trial_dir = trial_dir
        self.accuracy = accuracy
        self._bin_edges = None
        self.bin_edges = bin_edges
        self.max_bin_size = max_bin_size
        self.label = label
        self.tn = tn
        self.tp = tp
        self.fn = fn
        self.fp = fp
        self.skip_tls_handshake = skip_tls_handshake

    @property
    def bin_edges(self) -> List[float]:
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, edges: Union[np.array, List[float]]):
        if type(edges) == np.array:
            self._bin_edges = [float(x) for x in edges]
        elif type(edges) == list:
            self._bin_edges = edges
        elif edges is None:
            self._bin_edges = None
        else:
            raise ValueError(f"Unknown data type {type(edges)}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid_": self.uuid,
            "accuracy": self.accuracy,
            "trial_dir": self.trial_dir,
            "classifier": self.classifier,
            "binning_method": self.binning_method,
            "num_bins": self.num_bins,
            "day_train_start": self.day_train_start,
            "day_train_end": self.day_train_end,
            "seq_length": self.seq_length,
            "seq_element_type": self.seq_element_type,
            "hmm_length": self.hmm_length,
            "hmm_duration": self.hmm_length,
            "knn_num_neighbors": self.knn_num_neighbors,
            "seed": self.seed,
            "hmm_init_prior": self.hmm_init_prior,
            "hmm_num_iter": self.hmm_num_iter,
            "ano_density_estimator": self.ano_density_estimator,
            "bin_edges": self.bin_edges,
            "max_bin_size": self.max_bin_size,
            "label": self.label,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "tn": self.tn,
            "knn_fraction": self.knn_fraction,
            "skip_tls_handshake": self.skip_tls_handshake
        }


def make_trial_dir_name(config: Config, exp_dir: str) -> str:
    count = 0
    file_name = f'{config.uuid}_{count}'
    while os.path.exists(os.path.join(exp_dir, file_name)):
        count += 1
        file_name = f'{config.uuid}_{count}'
    return file_name


def save_config(config: Config) -> None:
    with open(os.path.join(config.trial_dir, 'config.json'), "w") as fh:
        json.dump(config.to_dict(), fh)


def save_classification_results(config: Config, results: List[Dict[str, Dict[str, float]]]) -> None:
    with open(os.path.join(config.trial_dir, 'conf-mats.json'), 'w') as fh:
        json.dump(results, fh)


def make_knn(config: Config) -> knnmod.KnnClassifier:
    knn = knnmod.KnnClassifier(config.knn_num_neighbors, config.knn_fraction)
    return knn


def make_mc(config: Config) -> mcmod.MarkovChainClassifier:
    mc = mcmod.MarkovChainClassifier()
    return mc


def make_phmm(config: Config) -> phmmmod.PhmmClassifier:
    phmm = phmmmod.PhmmClassifier(
        duration=config.hmm_length,
        init_prior=config.hmm_init_prior,
        seed=config.seed,
        num_iter=config.hmm_num_iter
    )
    return phmm


def make_ano(config: Union[Config, Dict[str, Config]]) -> bnmod.MultiBinaryClassifier:
    if type(config) == Config:
        anoc = bnmod.MultiBinaryClassifier(config.to_dict())
    else:
        anoc = bnmod.MultiBinaryClassifier({l: c.to_dict() for l, c in config.items()})
    return anoc


def make_classifier(config: Union[Config, Dict[str, Config]]) -> SeqClassifier:
    if type(config) == dict:
        return make_ano(config)
    else:
        return {
            'knn': make_knn,
            'mc': make_mc,
            'phmm': make_phmm,
            'ano': make_ano
        }[config.classifier](config)


def load_sequences(dset: str, meta_data: List[List[Dict[str, Any]]],
                   indicator: Dict[str, str], labels: Dict[str, str],
                   seq_length: int, seq_element_type: object,
                   bin_edges: Union[None, np.array]) -> Tuple[List[List[str]], List[str]]:
    X = []
    y = []
    skipped = 0
    total = 0
    main_flow_to_symbol = tlsex.MainFlowToSymbol(
        seq_length=seq_length,
        to_symbolize=seq_element_type,
        bin_edges=bin_edges
    )
    for meta_data_day in meta_data:
        for data in meta_data_day:
            total += 1
            if data['url_id'] not in indicator or data['url_id'] not in labels:
                skipped += 1
                continue
            if indicator[data['url_id']] != dset:
                skipped += 1
                continue
            main_flow = dprep.load_flow_dict(f"{data['filename']}.json")
            if main_flow is None:
                skipped += 1
                continue
            main_flow = dprep.make_main_flows([main_flow])[data['filename']]
            seq = main_flow_to_symbol(main_flow)
            X.append(seq)
            y.append(labels[data['url_id']])
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>> Skipped {skipped} of {total} entries.")
    return X, y


def load_flow_dicts(dset: str, meta_data_day: List[Dict[str, Any]],
                    day_index: int, indicator: Dict[str, str],
                    labels: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str], int]:
    X = []
    y = []
    skipped = 0
    total = 0
    local_logger = logging.getLogger(uuid.uuid4().hex)
    local_logger.addHandler(logging.StreamHandler(sys.stdout))
    local_logger.setLevel(logging.DEBUG)
    local_logger.info(f"Extract {len(meta_data_day)} items.")
    for i, data in enumerate(meta_data_day):
        if int(total / len(meta_data_day) * 100) % 5 == 0:
            perc = total / len(meta_data_day) * 100
            local_logger.info(f"Processed {total} items ({perc}%%), {len(X)} with success, skipped {skipped}")
        total += 1
        if data['url_id'] not in indicator or data['url_id'] not in labels:
            local_logger.debug("URL ID does not exist in indicator or labels")
            skipped += 1
            continue
        if indicator[data['url_id']] != dset:
            local_logger.debug('URL Id not in correct data set')
            skipped += 1
            continue
        main_flow = dprep.load_flow_dict(f"{data['filename']}.json")
        if main_flow is None:
            local_logger.debug("Main FLow retrieval failed")
            skipped += 1
        else:
            reduced_main_flow = {
                'frames': main_flow['frames'][:35]
            }
            X.append(reduced_main_flow)
            y.append(labels[data['url_id']])
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>> Skipped {skipped} of {total} entries.")
    return X, y, day_index


def load_flow_dicts_mp(args):
    return load_flow_dicts(args[0], args[1], args[2], args[3], args[4])


def train_predict(config_ds: List[Dict[str, Any]], X_train_raw: List[Dict[str, Any]],
                  y_train: List[str], X_val_raw: List[List[Dict[str, Any]]],
                  y_val: List[List[str]], defense: None | Dict[str, Any],
                  direction_to_filter: int, logger: logging.Logger):
    local_logger = logger
    config = Config.from_dict(config_ds[0])
    first_config = config
    local_logger.info(f'Create sequences from {len(X_train_raw)} main-flows for training set.')

    unique_labels = []
    for y in y_train:
        if y not in unique_labels:
            unique_labels.append(y)

    main_flow_to_symbol = tlsex.MainFlowToSymbol(
        seq_length=config.seq_length,
        to_symbolize=config.seq_element_type,
        bin_edges=config.bin_edges,
        direction_to_filter=direction_to_filter
    )
    if defense is None:
        defend = lambda x: x
    else:
        random = np.random.RandomState(seed=defense['seed'])
        defend = tlsex.RandomRecordSizeDefense(
            max_seq_length=config.seq_length,
            get_random_length=lambda x: int(random.randint(defense['min_record_size'], defense['max_record_size']))
        )
    X_train = []
    for i, main_flow_dict in enumerate(X_train_raw):
        X_train.append(main_flow_to_symbol(defend(main_flow_dict)))

    local_logger.info('Create sequences from main flows for validation set.')
    X_val = []
    for day, X_day in enumerate(X_val_raw):
        X_val_day = []
        for i, main_flow_dict in enumerate(X_day):
            X_val_day.append(main_flow_to_symbol(defend(main_flow_dict)))
        X_val.append(X_val_day)

    for config_idx, config_d in enumerate(config_ds):
        exp_dir = config_d.pop("exp_dir", '/opt/project/data/grid-search-results')
        config = Config.from_dict(config_d)
        if config.trial_dir is None:
            config.uuid = uuid.uuid4().hex
            config.trial_dir = os.path.join(exp_dir, make_trial_dir_name(config, exp_dir))
        if not (first_config.binning_method == config.binning_method and
                first_config.seq_element_type == config.seq_element_type and
                first_config.seq_length == config.seq_length):
            logger.error("Configs do not have the same data prep stuff.")
            continue
        if not os.path.exists(config.trial_dir):
            os.mkdir(config.trial_dir)
        save_config(config)

        local_logger.info('Train Classifier.')
        classifier = make_classifier(config)
        classifier.trial_dir = config.trial_dir
        t1 = time.perf_counter()
        classifier.fit(X_train, y_train)
        training_times = [time.perf_counter() - t1]
        with open(os.path.join(config.trial_dir, 'training-timings.json'), 'w') as fh:
            json.dump(training_times, fh)

        conf_mats = []
        nominator = 0.
        denominator = 0.
        local_logger.info('Evaluate Classifier.')
        predicted_labels = classifier.predict_mp(X_val)
        with open(os.path.join(config.trial_dir, 'inference-timings.json'), 'w') as fh:
            json.dump(classifier.inference_times, fh)
        # for i, (X, y) in enumerate(zip(X_val, y_val)):
        for i, (y_hat, y) in enumerate(zip(predicted_labels, y_val)):
            local_logger.info(f"Evaluate day {i} - has {len(y_hat)} predictions.")
            conf_mat = {}
            # y_hat, _ = classifier.predict((X, i))
            for true_label, predicted_label in zip(y, y_hat):
                if true_label not in conf_mat:
                    conf_mat[true_label] = {}
                if predicted_label not in conf_mat[true_label]:
                    conf_mat[true_label][predicted_label] = 0
                conf_mat[true_label][predicted_label] += 1

                denominator += 1
                nominator += int(true_label == predicted_label)
            conf_mats.append(conf_mat)
            save_classification_results(config, conf_mats)
            with open(os.path.join(config.trial_dir, 'agreement.json'), 'w') as fh:
                json.dump(classifier.votes, fh)
        config.accuracy = nominator / denominator
        save_config(config)
        del conf_mats
        gc.collect()


def train_predict_mp(args):
    return train_predict(args[0], args[1], args[2], args[3], args[4])


def add_bin_edges(config: Config):
    if config.binning_method == constants.BINNING_SINGLE:
        config.bin_edges = [0.]
    elif config.binning_method == constants.BINNING_NONE:
        config.bin_edges = None
    else:
        raise ValueError(f"Unknown binning method {config.binning_method}.")
    return config


def make_knn_configs(binning_method: str, seq_element_type: str) -> List[Config]:
    knn_num_neighbors = 10
    configs = [
        add_bin_edges(Config(
            classifier='knn',
            binning_method=binning_method,
            num_bins=0 if binning_method == constants.BINNING_NONE else 1,
            seq_length=30,
            seq_element_type=seq_element_type,
            hmm_length=None,
            day_train_start=0,
            day_train_end=30,
            knn_num_neighbors=knn_num_neighbors,
            hmm_init_prior=None,
            hmm_num_iter=None,
            ano_density_estimator=None,
            seed=1
        ))
    ]
    for i, config in enumerate(configs):
        config.trial_dir = f'/opt/project/data/grid-search-results/knn-{i}'
    return configs


def make_mc_configs(binning_method: str, seq_element_type: str) -> List[Config]:
    return [
        add_bin_edges(Config(
            classifier='mc',
            binning_method=binning_method,
            num_bins=0 if binning_method == 'no-bins' else 1,
            seq_length=30,
            seq_element_type=seq_element_type,
            hmm_length=None,
            day_train_start=0,
            day_train_end=30,
            knn_num_neighbors=None,
            hmm_init_prior=None,
            hmm_num_iter=None,
            ano_density_estimator=None,
            seed=1
        ))
    ]


def make_phmm_configs(binning_method: str, seq_element_type: str) -> List[Config]:
    return [
        add_bin_edges(Config(
            classifier='phmm',
            binning_method=binning_method,
            num_bins=0 if binning_method == 'no-bins' else 1,
            seq_length=30,
            seq_element_type=seq_element_type,
            hmm_length=l,
            day_train_start=0,
            day_train_end=30,
            knn_num_neighbors=None,
            hmm_init_prior='uniform',
            hmm_num_iter=30,
            ano_density_estimator=None,
            seed=1
        ))
    for l in [30, 25, 20, 15, 10]]


def make_ano_phmm_config(binning_method: str, seq_element_type: str) -> List[Config]:
    configs = make_phmm_configs(binning_method, seq_element_type)
    for config in configs:
        config.classifier = 'ano'
        config.ano_density_estimator = 'phmm'
    return configs


def make_ano_mc_config(binning_method: str, seq_element_type: str) -> List[Config]:
    configs = make_mc_configs(binning_method, seq_element_type)
    for config in configs:
        config.classifier = 'ano'
        config.ano_density_estimator = 'mc'
    return configs


def make_configs() -> List[Dict[str, Any]]:
    logger.info("Create configs...")
    configs = []
    for binning_method in ['one-bin', 'no-bins']:
        for seq_element_type in ['frame', 'record']:
            tmp = []
            # tmp.extend([c.to_dict() for c in make_knn_configs(binning_method, seq_element_type)])
            # tmp.extend([c.to_dict() for c in make_mc_configs(binning_method, seq_element_type)])
            tmp.extend([c.to_dict() for c in make_phmm_configs(binning_method, seq_element_type)])
            # tmp.extend([c.to_dict() for c in make_ano_phmm_config(binning_method, seq_element_type)])
            # tmp.extend([c.to_dict() for c in make_ano_mc_config(binning_method, seq_element_type)])
            configs.extend(tmp)
    return configs


def run2(logger):
    logger.info("Retrieve datasets")
    if is_cached('indicator.json'):
        indicator = read_cache('indicator.json')
        indicator = {int(k): v for k, v in indicator.items()}
        labels = read_cache('labels.json')
        labels = {int(k): v for k, v in labels.items()}
    else:
        _, indicator, labels = dprep.create_data_sets()
        add_to_cache('indicator.json', indicator)
        add_to_cache('labels.json', labels)

    logger.info("Retrieve metadata")
    if is_cached('meta_data.json'):
        logger.info("Retrieve from cache.")
        meta_data = read_cache('meta_data.json')
        meta_data = [(datetime.strptime(a, '%Y-%m-%d %H:%M:%S'), b) for a, b in meta_data]
    else:
        logger.info("Retrieve days")
        days = dprep.get_days()
        # pool = mp.Pool(10)
        # meta_data = pool.map(dprep.get_days_metadata, days)
        # pool.close()
        meta_data = [dprep.get_days_metadata(d) for d in days]
        meta_data.sort(key=lambda x: x[0])
        add_to_cache('meta_data.json', [(str(a), b) for a, b in meta_data])
    logger.info("Serialize meta data")
    meta_data = [m for _, m in meta_data]

    configs = make_configs()

    logger.info("Make training data.")
    if is_cached('X_train.json'):
        logger.info("Read training data from cache")
        X_train = read_cache('X_train_small.json')
        y_train = read_cache('y_train_small.json')
    else:
        s = configs[0]['day_train_start']
        e = configs[0]['day_train_end']
        logger.info(f"Create training data from raw data, have {len(meta_data)}, use {s} to {e} for training.")
        # pool = mp.Pool(30)
        # rets = pool.map(load_flow_dicts_mp, [('train', m, i, indicator, labels) for i, m in enumerate(meta_data[s:e])])
        # pool.close()
        rets = [load_flow_dicts_mp(('train', m, i, indicator, labels)) for i, m in enumerate(meta_data[s:e])]
        rets.sort(key=lambda x: x[2])
        X_train = []
        y_train = []
        for x, y, _ in rets:
            X_train.extend(x)
            y_train.extend(y)

        add_to_cache('X_train.json', X_train)
        add_to_cache('y_train.json', y_train)

    logger.info("Make validation data.")
    if is_cached('X_val.json'):
        logger.info("Read validation data from cache")
        X_val = read_cache('X_val_small.json')
        y_val = read_cache('y_val_small.json')
    else:
        logger.info("Create validation data from raw data")
        # pool = mp.Pool(30)
        # rets = pool.map(load_flow_dicts_mp, [('val', m, i, indicator, labels) for i, m in enumerate(meta_data)])
        # pool.close()
        rets = [load_flow_dicts_mp(('val', m, i, indicator, labels)) for i, m in enumerate(meta_data[s:e])]
        rets.sort(key=lambda x: x[2])
        logger.info(f"Retrieved validation data for {len(rets)} days.")
        X_val = [r[0] for r in rets]
        y_val = [r[1] for r in rets]
        add_to_cache('X_val.json', X_val)
        add_to_cache('y_val.json', y_val)

    logger.info(f"Train and evaluate classifier. Train {len(configs)} configurations.")
    for config in configs:
        try:
            train_predict(
                config_ds=[config],
                X_train_raw=X_train,
                y_train=y_train,
                X_val_raw=X_val,
                y_val=y_val,
                logger=logger
            )
        except Exception as e:
            logger.exception(e)


def run_redis(logger):
    with open('/opt/project/closed-world-labels.json', 'r') as fh:
        closed_world_labels = json.load(fh)
    # Create a flat list from the cached files. Training set consists of a
    # list of main flows.
    X_train = []
    y_train = []
    logger.info("Retrieve training...")
    for label in closed_world_labels:
        fname = f'{label}_train.json'
        if is_cached(fname):
            tmp = read_cache(fname)
            for day in tmp.values():
                X_train.extend(day)
                y_train.extend([label for _ in range(len(day))])
    # Create a list of lists from cached files. Validation set consists of a list.
    # This list contains for each list another list. This list contains the
    # main flows from all web-sites.
    X_val = []
    y_val = []
    logger.info("Retrieve validation...")
    for label in closed_world_labels:
        fname = f'{label}_val.json'
        if is_cached(fname):
            tmp = read_cache(fname)
            for day, flows in tmp.items():
                day = int(day)
                while len(X_val) <= day:
                    X_val.append([])
                    y_val.append([])
                X_val[day].extend(flows)
                y_val[day].extend([label for _ in range(len(flows))])

    wait = 10
    redis_q = rediswq.RedisWQ(name='gridsearch', host='tueilkn-swc06.forschung.lkn.ei.tum.de')
    reported_queue_empty = False
    count = 0
    while count <= 10:
        try:
            if redis_q.empty():
                if not reported_queue_empty:
                    logger.info(f"Queue empty, wait for new jobs...")
                    reported_queue_empty = True
                time.sleep(wait)
            else:
                reported_queue_empty = False
                item = redis_q.lease(lease_secs=30, block=True, timeout=2)
                if item is not None:
                    redis_q.complete(item)
                    config = json.loads(item.decode("utf-8"))
                    count += 1
                    msg = f"{config['classifier']}-{config['seq_element_type']}-" + \
                          f"{config['binning_method']}-{config['num_bins']}-" + \
                          f"{config['seq_length']}-{config['hmm_length']}"
                    sep = '\n\t====================================================='
                    logger.info(f"{sep}\n\tTrain new model: {msg}{sep}")
                    t1 = time.perf_counter()
                    train_predict(
                        config_ds=[config],
                        X_train_raw=X_train,
                        y_train=y_train,
                        X_val_raw=X_val,
                        y_val=y_val,
                        logger=logger
                    )
                    logger.info(f"Trained model number {count} {msg} in {time.perf_counter() - t1}s")
                    gc.collect()
        except Exception as e:
            logger.exception(e)


def run_single_config(config: Dict[str, Any], labels_train: List[str], labels_eval: List[str],
                      dset_eval: str, defense: None | Dict[str, Any],
                      direction_to_filter: int) -> None:
    # Create a flat list from the cached files. Training set consists of a
    # list of main flows.
    X_train = []
    y_train = []
    logger = logging.getLogger('run-single-config')
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.info("Retrieve training...")
    for label in labels_train:
        fname = f'{label}_train.json'
        if is_cached(fname):
            tmp = read_cache(fname)
            for day, data in tmp.items():
                if int(day) > config['day_train_end']:
                    continue
                X_train.extend(data)
                y_train.extend([label for _ in range(len(data))])
    # In case of test set add also the validation set to the training set.
    if dset_eval == 'test':
        for label in labels_train:
            fname = f'{label}_val.json'
            if is_cached(fname):
                tmp = read_cache(fname)
                for day, data in tmp.items():
                    if int(day) > config['day_train_end']:
                        continue
                    X_train.extend(data)
                    y_train.extend([label for _ in range(len(data))])
    # Create a list of lists from cached files. Validation set consists of a list.
    # This list contains for each list another list. This list contains the
    # main flows from all web-sites.
    X_val = []
    y_val = []
    logger.info("Retrieve validation...")
    for label in labels_eval:
        fname = f'{label}_{dset_eval}.json'
        if is_cached(fname):
            tmp = read_cache(fname)
            for day, flows in tmp.items():
                day = int(day)
                while len(X_val) <= day:
                    X_val.append([])
                    y_val.append([])
                X_val[day].extend(flows)
                y_val[day].extend([label for _ in range(len(flows))])
    for day, tmp in enumerate(X_val):
        logger.info(f"Day {day} has {len(tmp)} elements.")
    train_predict(
        config_ds=[config],
        X_train_raw=X_train,
        y_train=y_train,
        X_val_raw=X_val,
        y_val=y_val,
        logger=logger,
        defense=defense,
        direction_to_filter=direction_to_filter
    )


def get_open_world_labels() -> List[str]:
    with open('/opt/project/open-world-labels.json', 'r') as fh:
        ret = json.load(fh)
    return ret


def get_closed_world_labels() -> List[str]:
    with open('/opt/project/closed-world-labels.json', 'r') as fh:
        ret = json.load(fh)
    return ret


def get_all_labels() -> List[str]:
    all = get_closed_world_labels()
    all.extend(get_open_world_labels())
    return all


def get_best_config() -> Dict[str, Any]:
    # "trial_dir": "/opt/project/data/grid-search-results/047f435fd54642f4b2674ac21e08d27f_agreement",
    config = {"uuid_": "047f435fd54642f4b2674ac21e08d27f", "accuracy": 0.9039107646821664,
              "trial_dir": "/opt/project/data/grid-search-results/047f435fd54642f4b2674ac21e08d27f",
              "classifier": "knn", "binning_method": "equal_width", "num_bins": 40, "day_train_start": 0,
              "day_train_end": 72, "seq_length": 30, "seq_element_type": "record", "hmm_length": None,
              "hmm_duration": None, "knn_num_neighbors": 9, "seed": 1, "hmm_init_prior": None, "hmm_num_iter": None,
              "ano_density_estimator": None, "bin_edges": None, "max_bin_size": 16384, "label": None, "fp": None,
              "fn": None, "tp": None, "tn": None}
    config['exp_dir'] = '/opt/project/data/grid-search-results'
    config['num_samples'] = 1
    return config


if __name__ == '__main__':
    pod_name = os.environ.get("POD_NAME")
    if pod_name is None:
        pod_name = "test"
    logger = logging_factory.produce_logger(
        name=f'{pod_name}',
        log_dir='/opt/project/data/grid-search-results/logs'
    )
    mode = sys.argv[1]
    if mode == 'redis':
        run_redis(logger)
    elif mode == 'singleconfig':
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--scenario",
            help="Scenario that should be evaluated, must be in {open, closed}.",
            default='None'
        )
        parser.add_argument(
            "--train-for-days",
            help="Number of days that should be used for training.",
            default=70,  # Use all days.
            type=int
        )
        parser.add_argument(
            "--filter-client-to-server",
            help="Remove all packets/frames travelling from the client to the server",
            action="store_true"
        )
        parser.add_argument(
            "--filter-server-to-client",
            help="Remove all packets/frames travelling from the server to the client",
            action="store_true"
        )
        parser.add_argument(
            "--use-defense",
            help="Randomly change the size of record lengths. If set, specify min-record-size and max-record-size",
            action="store_true"
        )
        parser.add_argument(
            "--min-record-size",
            help="Minimum Record size for defense.",
            type=int,
            default=-1
        )
        parser.add_argument(
            '--max-record-size',
            help="Maximum record size for defense, must be smaller 2^14.",
            type=int,
            default=-1
        )
        parsed_args, _ = parser.parse_known_args(sys.argv[2:])
        if parsed_args.use_defense:
            defense = {
                'name': 'RandomRecordSizeDefense',
                'min_record_size': int(parsed_args.min_record_size),
                'max_record_size': int(parsed_args.max_record_size),
                'seed': 1
            }
            dfstr = f'-{defense["name"]}-{defense["min_record_size"]}-{defense["max_record_size"]}'
        else:
            defense = None
            dfstr = ''
        scenario = parsed_args.scenario
        fstr = ''
        filter_direction = 0
        if parsed_args.filter_client_to_server:
            fstr = '-filter-client2server'
            filter_direction = -1
        if parsed_args.filter_server_to_client:
            fstr = '-filter-server2client'
            filter_direction = 1
        config = get_best_config()
        config['trial_dir'] = f'/opt/project/data/grid-search-results/eval-results/{scenario}-world-knn-days-{parsed_args.train_for_days}{fstr}{dfstr}'
        if not os.path.exists(config['trial_dir']):
            os.mkdir(config['trial_dir'])
        if scenario == 'open':
            config['knn_fraction'] = 0.55
            eval_labels = get_all_labels()
        else:
            config['knn_fraction'] = None
            eval_labels = get_closed_world_labels()
        logger.info(f"Store results in {config['trial_dir']}")
        run_single_config(
            config=config,
            labels_train=get_closed_world_labels(),
            labels_eval=eval_labels,
            dset_eval='test',
            defense=defense,
            direction_to_filter=filter_direction
        )
    elif mode == 'overdays':
        # for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 45, 50, 60, 65]:
        for i in [5, 15, 25, 35, 45, 55, 65]:
            eval_labels = get_all_labels()
            config = get_best_config()
            config['knn_fraction'] = 0.55
            config['exp_dir'] = '/opt/project/data/grid-search-results'
            config['num_samples'] = 1
            config['trial_dir'] += f'-open-world-eval-over-days-{i}'
            config['day_train_end'] = i
            run_single_config(
                config=config,
                labels_train=get_closed_world_labels(),
                labels_eval=eval_labels,
                dset_eval='test',
                defense=None,
                direction_to_filter=0
            )
    else:
        raise KeyError

