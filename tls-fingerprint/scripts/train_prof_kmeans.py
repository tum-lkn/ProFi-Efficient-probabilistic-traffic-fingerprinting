import logging
import uuid

import numpy as np
import json
import sys
import os
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans

import implementation.data_conversion.tls_flow_extraction as tlsex
from scripts.knn_train_eval import Config
import implementation.classification.phmm as wrappers
from implementation.phmm_np.grid_search import save_config


def get_url_ids(label: str, dset: str) -> List[int]:
    assert dset in ['train', 'val', 'test']
    ids_to_use = [int(k) for k, v in labels.items() if v == label and indicator[k] == dset]
    return ids_to_use


def load_filenames(ids_to_use: List[int]) -> List[str]:
    filenames = []
    for date, day in meta_data:
        filenames.extend([data['filename'] for data in day if data['url_id'] in ids_to_use])
    return filenames


def load_main_flows(filenames: List[str]) -> List[Dict[str, Any]]:
    main_flows = []
    for filename in filenames:
        with open(f'/opt/project/data/k8s-json/{filename}.json', 'r') as fh:
            main_flows.append(json.load(fh))
    return main_flows


def get_frame_sizes(filenames: List[str]) -> List[float]:
    frame_sizes = []
    for i, filename in enumerate(filenames):
        if os.path.exists(f'/opt/project/data/k8s-json/{filename}.json'):
            try:
                with open(f'/opt/project/data/k8s-json/{filename}.json', 'r') as fh:
                    main_flow = json.load(fh)
                if len(main_flow['frames']) < 10:
                    continue
                for frame in main_flow['frames'][:30]:
                    frame_sizes.append(frame['tcp_length'] * frame['direction'])
            except Exception as e:
                print(f"Error for filename {filename}")
                print(e)
    return frame_sizes


def train_kmeans(frame_sizes: List[float], num_centers: int) -> KMeans:
    return KMeans(num_centers).fit(np.array(frame_sizes)[:, None])


def make_sequences(filenames: List[str], edges: np.array, seq_length: int) -> List[List[str]]:
    sequences = []
    for i, filename in enumerate(filenames):
        if os.path.exists(f'/opt/project/data/k8s-json/{filename}.json'):
            try:
                with open(f'/opt/project/data/k8s-json/{filename}.json', 'r') as fh:
                    main_flow = json.load(fh)
                if len(main_flow['frames']) <= 10:
                    continue
                sequences.append(tlsex.main_flow_to_symbol(main_flow, seq_length, tlsex.Frame, edges, True))
            except Exception as e:
                print(f"Error for filename {filename}")
                print(e)
    return sequences


def clean_filenames(filenames: List[str], lbl: str) -> List[str]:
    if lbl == 'www.inquirer.net':
        return [f for f in filenames if not f.startswith('chaturbate')]
    else:
        return filenames


def classify_and_predict(lbl: str):
    logger = logging.getLogger(f"logger-{lbl}")
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    uid = uuid.uuid4().hex
    config = Config(
        classifier='phmm',
        binning_method='kmeans',
        num_bins=10,
        seq_length=15,
        seq_element_type='frame',
        hmm_length=15,
        day_train_start=0,
        day_train_end=70,
        knn_num_neighbors=None,
        hmm_init_prior='uniform',
        hmm_num_iter=20,
        ano_density_estimator='',
        seed=1,
        uuid_=uid,
        trial_dir='/opt/project/data/grid-search-results/kmeans-clustering',
        label=lbl
    )
    logger.info("Retrieve Data")
    url_ids_train = get_url_ids(lbl, 'train')
    url_ids_val = get_url_ids(lbl, 'val')
    files_train = clean_filenames(load_filenames(url_ids_train), lbl)
    files_val = clean_filenames(load_filenames(url_ids_val), lbl)

    frame_sizes = get_frame_sizes(files_train)
    frame_sizes.extend(get_frame_sizes(files_val))
    logger.info("Get frame sizes and cluster")
    kmeans = KMeans(10).fit(np.array(frame_sizes, dtype=np.float32)[:, None])
    edges = np.sort(kmeans.cluster_centers_.flatten())

    sequences_train = make_sequences(files_train, edges, config.seq_length)
    sequences_val = make_sequences(files_val, edges, config.seq_length)

    logger.info("Train models")
    best_log_prob = -1e12
    for seed in range(1, 11):
        wrapper = wrappers.CPhmm(
            duration=config.hmm_length,
            init_prior=config.hmm_init_prior,
            seed=seed,
            label=config.label,
            num_iter=config.hmm_num_iter
        ).fit(sequences_train, sequences_val)
        log_prob = np.median(wrapper.log_prob_train_all)
        if log_prob > best_log_prob:
            logger.info(f"New best model {best_log_prob} < {log_prob}")
            worst_log_prob = np.min(wrapper.log_prob_train_all)
            config.accuracy = float(np.mean(wrapper.log_prob_val_all >= worst_log_prob))
            d = {
                'sum_log_prob_train': float(np.sum(wrapper.log_prob_train_all)),
                'sum_log_prob_val': float(np.sum(wrapper.log_prob_val_all)),
                'avg_log_prob_train': float(np.mean(wrapper.log_prob_train_all)),
                'avg_log_prob_val': float(np.mean(wrapper.log_prob_val_all)),
                'min_nll_train': float(np.min(np.abs(wrapper.log_prob_train_all))),
                'min_nll_val': float(np.min(np.abs(wrapper.log_prob_val_all))),
                'max_nll_train': float(np.max(np.abs(wrapper.log_prob_train_all))),
                'max_nll_val': float(np.max(np.abs(wrapper.log_prob_val_all)))
            }
            save_config(config, addendum=d)
            best_model = wrapper
            best_model.max_nll = d['max_nll_train']
            best_model.edges = edges
            best_log_prob = np.abs(d['avg_log_prob_val'])
            logger.debug(f"\tSave model to {os.path.join(config.trial_dir, 'model.json')}")
            model_d = wrapper.to_json_dict()
            name = 'model.json'
            with open(os.path.join(config.trial_dir, name), "w") as fh:
                json.dump(model_d, fh)

    all_labels = [l for l in closed_world_labels]
    all_labels.extend([l for l in open_world_labels])
    d['binary_classification'] = {}
    logger.info("Evaluate other pages in binary classification.")
    conv_cm = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for i, ylbl in enumerate(all_labels):
        logger.info(f"Evaluate {i} of {len(all_labels)}: {ylbl} with {lbl} model.")
        if ylbl == config.label:
            sequences_ = sequences_val
        else:
            url_ids_ = get_url_ids(ylbl, 'val')
            files_ = clean_filenames(load_filenames(url_ids_), ylbl)
            sequences_ = make_sequences(files_, edges, config.seq_length)
        lls = best_model.score_c(sequences_)
        scores = np.abs(lls) / best_model.max_nll
        pred_true = np.sum(scores <= 1)
        if ylbl == config.label:
            d['binary_classification'][ylbl] = {
                'tp': int(pred_true),
                'tn': 0,
                'fp': 0,
                'fn': int(scores.size - pred_true)
            }
            conv_cm['tp'] += int(pred_true)
            conv_cm['fn'] += int(scores.size - pred_true)
        else:
            d['binary_classification'][ylbl] = {
                'tp': 0,
                'tn': int(scores.size - pred_true),
                'fp': int(pred_true),
                'fn': 0
            }
            conv_cm['tn'] += int(scores.size - pred_true)
            conv_cm['fp'] += int(pred_true)
    logger.info(f"Evaluation finished: {json.dumps(conv_cm)}")
    save_config(config, addendum=d)
    return best_model


if __name__ == '__main__':
    with open('/opt/project/data/cache/meta_data.json', 'r') as fh:
        meta_data = json.load(fh)
    with open('/opt/project/data/cache/indicator.json', 'r') as fh:
        indicator = json.load(fh)
    with open('/opt/project/data/cache/labels.json', 'r') as fh:
        labels = json.load(fh)
    with open('/opt/project/closed-world-labels.json', 'r') as fh:
        closed_world_labels = json.load(fh)
    with open('/opt/project/open-world-labels.json', 'r') as fh:
        open_world_labels = json.load(fh)
    classify_and_predict('www.inquirer.net')
