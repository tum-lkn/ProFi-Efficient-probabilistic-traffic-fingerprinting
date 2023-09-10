import os
from datetime import datetime

import argparse
import joblib
import numpy as np
from typing import Tuple, List, Dict, Any, Union
import json
import gc
import logging
import sys
import time
import itertools as itt
import uuid
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
import sklearn.preprocessing as skprep
import pandas as pd
import multiprocessing as mp
from joblib import dump, load

from implementation.seqcache import is_cached, add_to_cache, read_cache


logger = logging.getLogger('cumul-train-eval')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def get_closed_world_labels() -> List[str]:
    with open('/opt/project/closed-world-labels.json', 'r') as fh:
        closed_world_labels = json.load(fh)
    return closed_world_labels


def get_open_world_labels() -> List[str]:
    with open('/opt/project/open-world-labels.json', 'r') as fh:
        closed_world_labels = json.load(fh)
    return closed_world_labels


def get_training_data(closed_world_labels: List[str]) -> Tuple:
    X_train = []
    y_train = []
    for label in closed_world_labels:
        fname = f'{label}_train.json'
        tmp = read_cache(fname)
        for day in tmp.values():
            X_train.extend(day)
            y_train.extend([label for _ in range(len(day))])
    return X_train, y_train


def multilabel_confusion_matrix(ground_truth: List[Any], predictions: List[Any]) -> Dict[str, Dict[str, float]]:
    matrix = {}
    for y, z in zip(predictions, ground_truth):
        y = str(y)
        z = str(z)
        if z not in matrix:
            matrix[z] = {}
        if y not in matrix[z]:
            matrix[z][y] = 0.
        matrix[z][y] += 1.
    return matrix


def train_svm(args):
    def open_worldify_gt(some_y: List[List[str]]) -> List[List[str]]:
        other_y = []
        for d in some_y:
            other_y.append([l if l in cwls else 'unknown' for l in d])
        return other_y

    X_train = args['X_train']  # 2d numpy array.
    X_val = args["X_val"]  # List of 2d numpy arrays.
    y_train = args['y_train']  # List of strings.
    y_val = args['y_val']  # List of List of strings
    X_test = args['X_test']  # List of 2d numpy arrays.
    y_test = args['y_test']  # List of List of strings
    c = args['c']

    minmax_scaler = skprep.MinMaxScaler((-1, 1)).fit(X_train)
    X_train = minmax_scaler.transform(X_train)
    X_val = [minmax_scaler.transform(x) for x in X_val]
    X_test = [minmax_scaler.transform(x) for x in X_test]

    gamma = args['gamma']
    num_days = args.get('num_days', 65)
    save_model = args.get("save_model", False)
    if save_model:
        trial_dir = args['trial_dir']
        if not os.path.exists(trial_dir):
            os.mkdir(trial_dir)
    cwls = get_closed_world_labels()

    t1 = time.time()  # use time and not perf counter because fit delegates to other process.
    if os.path.exists(os.path.join(trial_dir, 'model.pkl')):
        svm = SVC(C=c, kernel='rbf', gamma=gamma).fit(X_train, y_train)
        # svm = load(os.path.join(trial_dir, 'model.pkl'))
    else:
        svm = SVC(C=c, kernel='rbf', gamma=gamma).fit(X_train, y_train)
        if save_model:
            dump(svm, os.path.join(trial_dir, 'model.pkl'))
    fitting_time = time.time() - t1
    z_train = svm.predict(X_train)

    evaluation_times = []
    z_val = []
    for X_val_ in X_val:
        z_vals_ = []
        for i in range(X_val_.shape[0]):
            t1 = time.time()
            z_vals_.append(svm.predict(np.expand_dims(X_val_[i, :], axis=0)))
            evaluation_times.append(time.time() - t1)
        z_val.append(np.concatenate(z_vals_))
    evaluation_time = np.sum(evaluation_times)
    y_val_ = open_worldify_gt(y_val)

    z_test = [svm.predict(X_test_) for X_test_ in X_test]
    y_test_ = open_worldify_gt(y_test)

    try:
        ret = {
            "c": float(c),
            "gamma": gamma if type(gamma) == str else float(gamma),
            "num_days": num_days,
            "validation": {
                "confusion_matrix": [multilabel_confusion_matrix(yv, zv) for yv, zv in zip(y_val, z_val)],
                'accuracy': [float(accuracy_score(yv, zv)) for yv, zv in zip(y_val_, z_val)],
                'precision': [{str(svm.classes_[i]): float(x) for i, x in enumerate(precision_score(yv, zv, average=None))} for yv, zv in zip(y_val_, z_val)],
                'recall': [{str(svm.classes_[i]): float(x) for i, x in enumerate(recall_score(yv, zv, average=None))} for yv, zv in zip(y_val_, z_val)]
            },
            "test": {
                "confusion_matrix": [multilabel_confusion_matrix(yt, zt) for yt, zt in zip(y_test, z_test)],
                'accuracy': [float(accuracy_score(yt_, zt)) for yt_, zt in zip(y_test_, z_test)],
                'precision': [{str(svm.classes_[i]): float(x) for i, x in enumerate(precision_score(yt_, zt, average=None))} for yt_, zt in zip(y_test_, z_test)],
                'recall': [{str(svm.classes_[i]): float(x) for i, x in enumerate(recall_score(yt_, zt, average=None))} for yt_, zt in zip(y_test_, z_test)]
            },
            "train": {
                "confusion_matrix": multilabel_confusion_matrix(y_train, z_train),
                'accuracy': float(accuracy_score(y_train, z_train)),
                'precision': {str(svm.classes_[i]): float(x) for i, x in enumerate(precision_score(y_train, z_train, average=None))},
                'recall': {str(svm.classes_[i]): float(x) for i, x in enumerate(recall_score(y_train, z_train, average=None))}
            },
            "fitting_time": fitting_time,
            "evaluation_time": evaluation_time,
            "evaluation_time_per_sample": evaluation_times
        }
        logger.info(f"Fitting took {fitting_time}s = {fitting_time / 3600}h, " +
                    f"Evaluation took {evaluation_time}s, i.e., {evaluation_time / len(evaluation_times)}s "
                    f"per sample.")
        with open(os.path.join(trial_dir, 'results.json'), 'w') as fh:
            json.dump(ret, fh)
    except Exception as e:
        logger.exception(e)
    return ret


def read_data() -> Tuple[np.array, List[str], np.array, List[str], np.array, List[str]]:
    f = h5py.File('/opt/project/data/cumul-interpolations/n100-X-train-val.h5', 'r')
    X_train = f['X_train'][()]
    X_val = f['X_val'][()]
    X_test = f['X_test'][()]
    f.close()
    with open("/opt/project/data/cumul-interpolations/n100-y-train-val.json", "r") as fh:
        d = json.load(fh)
    return X_train, d['y_train'], X_val, d['y_val'], X_test, d['y_test']


def make_dataset_from_bin(meta_data: List[Tuple[str, List[Dict[str, Any]]]],
                          indicator: Dict[str, str], labels: Dict[str, str],
                          closed_world_labels: List[str], dsets: List[str],
                          is_open_world: bool, defense: bool, keep_days=False,
                          num_days: int = 100) -> Tuple[np.array, List[str]]:
    def load_array(url_meta: Dict[str, Any], lbl: str, bin_data: List[np.array], targets: List[str]) -> bool:
        pcap_p = os.path.join('/opt/project/data/k8s-json', f'{url_meta["filename"]}.json')
        if not os.path.exists(pcap_p):
            # logger.debug(f"JSON {pcap_p} does not exist.")
            # Make sure its the same data as for the flow based methods.
            return False
        p = os.path.join(bin_dir, f'{url_meta["filename"]}.bin')
        if os.path.exists(p):
            bin_data.append(np.fromfile(p, dtype=np.float32))
            targets.append(lbl)
            return True
        else:
            logger.debug(f'file {p} does not exist.')
            return False
    if defense:
        bin_dir = '/opt/project/data/cumul-interpolations-defense'
    else:
        bin_dir = '/opt/project/data/cumul-interpolations-cpp'
    logger.info(f"Retrieve data set(s) {json.dumps(dsets)} from {bin_dir}")
    bin_data = []
    targets = []
    for day, url_metas in meta_data[:num_days]:
        unknown_set = []
        if keep_days:
            bin_data.append([])
            targets.append([])
        for url_meta in url_metas:
            url_id = str(url_meta['url_id'])
            if url_id not in labels:
                # logger.debug(f"URL ID {url_id} not in labels")
                continue
            if url_id not in indicator:
                # logger.debug(f"URL ID {url_id} not in indicator")
                continue
            lbl = labels[url_id]
            if indicator[url_id] in dsets:
                if lbl in closed_world_labels:
                    load_array(url_meta, lbl, bin_data[-1] if keep_days else bin_data, targets[-1] if keep_days else targets)
                elif is_open_world and not lbl in unknown_set:
                    if load_array(url_meta, lbl, bin_data[-1] if keep_days else bin_data, targets[-1] if keep_days else targets):
                        unknown_set.append(lbl)
                else:
                    pass
    if keep_days:
        tmp = []
        tmp_y = []
        for d, x in enumerate(bin_data):
            if len(x) == 0:
                print(f"No datay for day {d}")
            else:
                tmp.append(np.row_stack(x))
                tmp_y.append(targets[d])
        targets = tmp_y
        X = tmp
        # X = [np.row_stack(x) for x in bin_data]
    else:
        X = np.row_stack(bin_data)
    logger.info(f'\tRetrieved {len(targets)} samples.')
    return X, targets


def get_background_samples(indicator: Dict[str, str], labels: Dict[str, str],
                           closed_world_labels: List[str], interpolations: pd.DataFrame,
                           dset: str, all_urls: bool) -> Tuple[np.array, List[str]]:
    def flatten(a):
        if a.shape[0] == 2:
            return np.concatenate([a[0, :], a[1, :]])
        else:
            return np.concatenate([a[:, 0], a[:, 1]])
    X = []
    y = []
    has_one_url = []
    for url_id, dset_ in indicator.items():
        if labels[url_id] in closed_world_labels:
            continue
        elif dset != dset:
            continue
        elif labels[url_id] in has_one_url and not all_urls:
            continue
        slice = interpolations.loc[int(url_id), 'interpolation']
        X.extend([flatten(slice.iloc[i]).astype(np.float32) for i in range(slice.shape[0]) if slice.iloc[i].size >= 200])
        if dset == 'train':
            y.extend(['unknown' for i in range(slice.shape[0]) if slice.iloc[i].size >= 200])
        else:
            y.extend([labels[url_id] for i in range(slice.shape[0]) if slice.iloc[i].size >= 200])
        if labels[url_id] not in has_one_url:
            # Prevent appending the same label multiple times since urls of one
            # label can be added multiple times because of the all_urls arg.
            has_one_url.append(labels[url_id])
    X = np.row_stack(X)
    return X, y


def flatten(a):
    if a.shape[0] == 2:
        return np.concatenate([a[0, :], a[1, :]])
    else:
        return np.concatenate([a[:, 0], a[:, 1]])


def write_data(indicator: Dict[str, str], labels: Dict[str, str], class_labels: List[str],
               interpolations: pd.DataFrame):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    for url_id, dset in indicator.items():
        if labels[url_id] not in class_labels:
            continue
        elif dset == 'train':
            X = X_train
            y = y_train
        elif dset == 'val':
            X = X_val
            y = y_val
        else:
            X = X_test
            y = y_test
        slice = interpolations.loc[int(url_id), 'interpolation']
        X.extend([flatten(slice.iloc[i]).astype(np.float32) for i in range(slice.shape[0]) if slice.iloc[i].size >= 200])
        y.extend([labels[url_id] for i in range(slice.shape[0]) if slice.iloc[i].size >= 200])
    X_train = np.row_stack(X_train)
    X_val = np.row_stack(X_val)
    X_test = np.row_stack(X_test)
    minmax_scaler = skprep.MinMaxScaler((-1, 1)).fit(X)
    X_train = minmax_scaler.transform(X_train)
    X_val = minmax_scaler.transform(X_val)
    X_test = minmax_scaler.transform(X_test)
    f = h5py.File('/opt/project/data/cumul-interpolations/n100-X-train-val.h5', 'w')
    f.create_dataset(name='X_train', data=X_train)
    f.create_dataset(name='X_val', data=X_val)
    f.create_dataset(name='X_test', data=X_test)
    f.close()
    with open("/opt/project/data/cumul-interpolations/n100-y-train-val.json", "w") as fh:
        json.dump({'y_val': y_val, 'y_train': y_train, 'y_test': y_test}, fh)
    return X_train, y_train, X_val, y_val, X_test, y_test


def grid_search(scenario: str):
    logger.info(f"Train and evaluate CUMUL on the testset for the {scenario}-world scenario.")
    logger.info("Read indicator...")
    indicator: Dict[str, str] = read_cache('indicator.json')
    logger.info("Read labels...")
    labels: Dict[str, str] = read_cache('labels.json')
    logger.info("Read metadata")
    meta_data: List[Tuple[str, List[Dict[str, Any]]]] = read_cache("meta_data.json")
    closed_world_labels = get_closed_world_labels()

    X_test, y_test = make_dataset_from_bin(
        meta_data=meta_data,
        indicator=indicator,
        labels=labels,
        closed_world_labels=closed_world_labels,
        is_open_world=scenario == 'open',
        dsets=['test'],
        defense=None,
        keep_days=True
    )
    X_val, y_val = make_dataset_from_bin(
        meta_data=meta_data,
        indicator=indicator,
        labels=labels,
        closed_world_labels=closed_world_labels,
        is_open_world=scenario == 'open',
        dsets=['val'],
        defense=None
    )
    X_train, y_train = make_dataset_from_bin(
        meta_data=meta_data,
        indicator=indicator,
        labels=labels,
        closed_world_labels=closed_world_labels,
        is_open_world=scenario == 'open',
        dsets=['train'],
        defense=None
    )

    # cs = [2**i for i in range(-5, 18)]
    # gammas = [2**i for i in range(-15, 4)]
    cs = [2**i for i in range(11, 18)]
    gammas = [2**i for i in range(-3, 4)]
    gammas.extend(['auto', 'scale'])
    args_list = [{
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'c': c,
        'gamma': gamma
    } for c, gamma in itt.product(cs, gammas)]
    pool = mp.Pool(processes=mp.cpu_count() - 1)
    res = pool.map(train_svm, args_list)
    pool.close()
    with open('/opt/project/data/svm-grid-search.json', 'w') as fh:
        json.dump(res, fh)
    res.sort(key=lambda x: x['validation']['accuracy'])
    logger.info(f"Best model: {json.dumps(res[-1], indent=1)}")


def evaluate_cumul(scenario: str, defense: bool) -> None:
    logger.info(f"Train and evaluate CUMUL on the testset for the {scenario}-world scenario, defense: {defense}.")
    logger.info("Read indicator...")
    indicator: Dict[str, str] = read_cache('indicator.json')
    logger.info("Read labels...")
    labels: Dict[str, str] = read_cache('labels.json')
    logger.info("Read metadata")
    meta_data: List[Tuple[str, List[Dict[str, Any]]]] = read_cache("meta_data.json")
    closed_world_labels = get_closed_world_labels()

    logger.info("Get best config:")
    with open("/opt/project/data/svm-grid-search.json", 'r') as fh:
        gs_results = json.load(fh)
    gs_results.sort(key=lambda x: np.mean(list(x['validation']['precision'].values())))
    best_config = gs_results[-1]
    logger.info(f"Best config is: {json.dumps(best_config, indent=1)}")
    X_test, y_test = make_dataset_from_bin(
        meta_data=meta_data,
        indicator=indicator,
        labels=labels,
        closed_world_labels=closed_world_labels,
        is_open_world=scenario == 'open',
        dsets=['test'],
        defense=defense,
        keep_days=True
    )
    X_train, y_train = make_dataset_from_bin(
        meta_data=meta_data,
        indicator=indicator,
        labels=labels,
        closed_world_labels=closed_world_labels,
        is_open_world=scenario == 'open',
        dsets=['train', 'val'],
        defense=defense
    )
    dfstr = '-RandomRecordSizeDefense-50-100' if defense else ''
    config = {
        # 'X_train': X_train[indices],
        'X_train': X_train,
        'X_val': X_test,  # Hyperparameter selection is done, thus its fine to use X_test here.
        'X_test': X_test,
        # 'y_train': [y_train[i] for i in indices],
        'y_train': y_train,
        'y_val': y_test,
        'y_test': y_test,
        'c': best_config['c'],
        'gamma': best_config['gamma'],
        'save_model': True,
        'scenario': scenario,
        'num_days': 70,
        'defense': {
            'name': 'RandomRecordSizeDefense',
            'min_record_size': 50,
            'max_record_size': 100,
        } if defense else None,
        'trial_dir': f'/opt/project/data/grid-search-results/eval-results/cumul-{scenario}-world{dfstr}/'
    }
    if not os.path.exists(config['trial_dir']):
        os.mkdir(config['trial_dir'])
    logger.info("Start training")
    res = train_svm(config)
    with open(os.path.join(config['trial_dir'], 'result.json'), 'w') as fh:
        json.dump(res, fh)


def retrain_from_grid_search_open(scenario: str, defend: bool):
    logger.info("Read indicator...")
    indicator = read_cache('indicator.json')
    logger.info("Read labels...")
    labels = read_cache('labels.json')
    closed_world_labels = get_closed_world_labels()
    logger.info("Read interpolations...")
    interpolations = pd.read_hdf('/opt/project/data/cumul-interpolations/n100.h5', key='interpolation')
    interpolations.set_index('url_id', inplace=True)
    with open("/opt/project/data/svm-grid-search.json", 'r') as fh:
        gs_results = json.load(fh)
    gs_results.sort(key=lambda x: np.mean(list(x['validation']['precision'].values())))
    best_config = gs_results[-1]
    X_train, y_train, X_val, y_val, X_test, y_test = read_data()
    logger.info("Get background samples for training")
    X, y = get_background_samples(indicator, labels, closed_world_labels, interpolations, 'train', False)
    X_train = np.concatenate((X_train, X_val, X), axis=0)
    y_train.extend(y_val)
    y_train.extend(y)
    logger.info("Get background samples for testing")
    X, y = get_background_samples(indicator, labels, closed_world_labels, interpolations, 'test', True)
    X_test = np.concatenate((X_test, X), axis=0)
    y_test.extend(y)
    indices = np.random.randint(0, X_train.shape[0], size=1000)
    indices = np.concatenate((indices, np.arange(X.shape[0] - 10, X.shape[0])))
    config = {
        # 'X_train': X_train[indices],
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        # 'y_train': [y_train[i] for i in indices],
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'c': best_config['c'],
        'gamma': best_config['gamma'],
        'save_model': True,
        'scenario': 'open',
        'trial_dir': '/opt/project/data/cumul-interpolations/cumul-open-world/'
    }
    if not os.path.exists(config['trial_dir']):
        os.mkdir(config['trial_dir'])
    logger.info("Start training")
    res = train_svm(config)
    with open(os.path.join(config['trial_dir'], 'result.json'), 'w') as fh:
        json.dump(res, fh)


def get_best_grid_search_config() -> Dict[str, Any]:
    with open("/opt/project/data/svm-grid-search.json", 'r') as fh:
        gs_results = json.load(fh)
    gs_results.sort(key=lambda x: np.mean(list(x['validation']['precision'].values())))
    best_config = gs_results[-1]
    return best_config


def make_day_data(indicator: Dict[str, str], labels: Dict[str, str],
                  closed_world_labels: List[str], meta_data: List[Dict[str, Any]],
                  interpolations: pd.DataFrame, num_days: int, dset: List[str],
                  open_world: bool, keep_days: bool) -> Tuple[List[np.array], List[List[str]]]:
    X = []
    y = []
    for _, day_datas in meta_data[:num_days]:
        unknown_set = []
        if keep_days:
            X.append([])
            y.append([])
        for url_meta in day_datas:
            url_id = str(url_meta['url_id'])

        background_indis = {}
        X_day = []
        y_day = []
        for day in day_datas:
            url_id = int(day['url_id'])
            if str(url_id) not in indicator: continue
            if indicator[str(url_id)] != dset: continue
            if str(url_id) not in labels: continue
            label = labels[str(url_id)]
            if not open_world and label not in closed_world_labels: continue
            if open_world and dset == 'train' and label not in closed_world_labels \
                    and label in background_indis: continue
            background_indis[label] = True
            meta_data_id = day['meta_data_id']
            if int(meta_data_id) not in interpolations.index: continue
            slice = interpolations.loc[int(meta_data_id), :]
            if type(slice) == pd.DataFrame:
                slice = slice.iloc[0]
            X_day.append(np.expand_dims(flatten(slice.interpolation), 0))
            y_day.append('unknown' if dset == 'train' and label not in closed_world_labels else label)
        if len(X_day) == 0: continue
        X.append(np.concatenate(X_day, axis=0))
        y.append(y_day)
    return X, y


def train_over_days(scenario: str):
    def make_dict(num_days) -> Dict[str, Any]:
        X_tr = np.concatenate(X_train[:num_days], axis=0)
        y_tr = []
        for lbls in y_train[:num_days]:
            y_tr.extend(lbls)
        logger.info(f"X train for up to {num_days} has {X_tr.shape} dims, y_train has {len(y_tr)}")
        logger.debug(f"X_val lengths [{', '.join([str(x.shape[0]) for x in X_test])}], y_val lengths: [{', '.join([str(len(y)) for y in y_test])}")
        return {
            'X_train': X_tr,
            'X_val': X_test,
            'X_test': X_test,
            'y_train': y_tr,
            'y_val': y_test,
            'y_test': y_test,
            'c': best_config['c'],
            'gamma': best_config['gamma'],
            'save_model': True,
            'num_days': num_days,
            'save_model': True,
            'trial_dir': f'/opt/project/data/grid-search-results/eval-results/cumul-{scenario}-over-days-{num_days}/'
    }

    closed_world_labels = get_closed_world_labels()
    best_config = get_best_grid_search_config()
    logger.info("Read indicator...")
    indicator = read_cache('indicator.json')
    logger.info("Read labels...")
    labels = read_cache('labels.json')
    logger.info("Read meta-data...")
    with open('/opt/project/data/cache/meta_data.json', 'r') as fh:
        meta_data = json.load(fh)
    logger.info("Make training set")
    X_train, y_train = make_dataset_from_bin(
        meta_data=meta_data,
        indicator=indicator,
        labels=labels,
        closed_world_labels=closed_world_labels,
        is_open_world=scenario == 'open',
        dsets=['train', 'val'],
        defense=False,
        keep_days=True
    )
    logger.info("Make test set")
    X_test, y_test = make_dataset_from_bin(
        meta_data=meta_data,
        indicator=indicator,
        labels=labels,
        closed_world_labels=closed_world_labels,
        is_open_world=scenario == 'open',
        dsets=['test'],
        keep_days=True,
        defense=False
    )
    logger.info(f"X_val has {len(X_test)} dims, y_train has {len(y_test)}")
    logger.info("Create configs...")
    # configs = [make_dict(i) for i in range(1, 66)]
    configs = [make_dict(n) for n in range(1, 72, 2)]

    pool = mp.Pool(processes=18)
    pool.map(train_svm, configs)
    pool.close()
    # train_svm(configs[0])


def retrain_from_grid_search():
    best_config = get_best_grid_search_config()
    X_train, y_train, X_val, y_val, X_test, y_test = read_data()
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train.extend(y_val)
    config = {
        'X_train': X_train,
        'X_val': [X_val],
        'X_test': [X_test],
        # 'y_train': [y_train[i] for i in indices],
        'y_train': y_train,
        'y_val': [y_val],
        'y_test': [y_test],
        'c': best_config['c'],
        'gamma': best_config['gamma'],
        'save_model': True,
        'trial_dir': '/opt/project/data/cumul-interpolations/cumul-closed-world/'
    }
    if not os.path.exists(config['trial_dir']):
        os.mkdir(config['trial_dir'])
    res = train_svm(config)
    with open(os.path.join(config['trial_dir'], 'result.json'), 'w') as fh:
        json.dump(res, fh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        help="Scenario that should be evaluated, must be in {open, closed}.",
        default='None'
    )
    parser.add_argument(
        "--evaluation",
        help="Evaluation that should be performed. Either evaluate over days, or evaluate a single config {overdays, singleconf}.",
        default='None'
    )
    parser.add_argument(
        "--defend",
        help="Train of the features extracted from the simulated defense.",
        action="store_true"
    )
    parsed_args, _ = parser.parse_known_args()
    assert parsed_args.scenario in ['open', 'closed']
    assert parsed_args.evaluation in ['singleconf', 'overdays']
    if parsed_args.evaluation == 'overdays':
        train_over_days(parsed_args.scenario)
    elif parsed_args.evaluation == 'singleconf':
        # retrain_from_grid_search_open(parsed_args.scenario, parsed_args.defend)
        evaluate_cumul(parsed_args.scenario, parsed_args.defend)
