import logging
import time
import json
import os
from typing import Dict, List, Tuple, Any

from scripts.evaluate_multi_binary import phmm_models, mc_models, train_model
from implementation.seqcache import read_cache, is_cached
import implementation.classification.binary as bcmod
from scripts.knn_train_eval import run_single_config

logger = logging.getLogger('trace-timings')
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def make_configs() -> List[Dict[str, Any]]:
    configs = [
        {
            'day_train_end': 65,
            'scenario': 'open',
            'trial_dirs': list(mc_models().values()),
            'trial_dir': '/opt/project/data/grid-search-results/no-handshake-mc',
            'classifier': 'mc'
        },
        {
            'day_train_end': 65,
            'scenario': 'open',
            'trial_dirs': list(phmm_models().values()),
            'trial_dir': '/opt/project/data/grid-search-results/no-handshake-phmm',
            'classifier': 'phmm'
        }
        # {
        #     'scenario': 'open',
        #     'trial_dirs': mc_models(),
        #     'trial_dir': '/opt/project/data/grid-search-results/timings-knn',
        #     'classifier': 'knn',
        #     'binning_method': None,
        #     'num_bins': 0,
        #     'seq_length': 30,
        #     'seq_element_type': 'frame',
        #     'hmm_length': None,
        #     'day_train_start': 0,
        #     'day_train_end': 65,
        #     'knn_num_neighbors': 7,
        #     'knn_fraction': 4,
        #     'hmm_init_prior': None,
        #     'hmm_num_iter': None,
        #     'ano_density_estimator': None,
        #     'seed': 1,
        #     'exp_dir': '/opt/project/data/grid-search-results'
        # }
    ]
    return configs


def make_multi_binary_classifier(trial_dirs: List[str], dset: str, dst_trial_dir: str,
                                 updates: Dict[str, Any]) -> bcmod.MultiBinaryClassifier:
    training_times = []
    binary_classifiers = []
    for td in trial_dirs:
        logger.info(f"Train {td}")
        binary_classifiers.append(train_model(td, dst_trial_dir, updates))
    mbc = bcmod.MultiBinaryClassifier(None)
    for nll, edges, config, bc in binary_classifiers:
        training_times.extend(bc.training_times)
        wrapper = bcmod.BinaryClassifier(
            ano_density_estimator=None,
            seq_length=config.seq_length,
            seq_element_type=config.seq_element_type,
            bin_edges=edges
        )
        wrapper.density_estimator = bc
        wrapper.threshold = nll
        mbc.bcs[bc.label] = wrapper
    mbc.training_times = training_times
    return mbc


def evaluate_pgm(config, closed_world_labels, open_world_labels) -> None:
    scenario = config['scenario']
    trial_dirs = config['trial_dirs']
    dset = 'test'
    updates = {
        'skip_tls_handshake': True,
        'seq_length': 10,
        'hmm_length': 10
    }
    result_dir = config['trial_dir']

    logger.info("Make binary classifiers")
    mbc = make_multi_binary_classifier(trial_dirs, dset, result_dir, updates)
    mbc.skip_tls_handshake = updates['skip_tls_handshake']
    with open(os.path.join(result_dir, 'training-timings.json'), 'w') as fh:
        json.dump(mbc.training_times, fh)

    mbc.scenario = scenario
    conf_mats = {}
    web_sites = [l for l in closed_world_labels]
    web_sites.extend(open_world_labels)
    for i, web_site in enumerate(web_sites):
        true_label = web_site
        logger.info(f"Evaluate traces of {web_site} - {i} of {len(web_sites)}.")
        if not is_cached(f"{web_site}_{dset}.json"):
            logger.info(f"test data for {web_site} does not exist.")
            continue
        x_test = read_cache(f"{web_site}_{dset}.json")
        for day, x_day in x_test.items():
            logger.info(f"Evaluate day {day}.")
            day = int(day)
            if day not in conf_mats:
                conf_mats[day] = {}
            conf_mats[day][true_label] = {}
            predicted_labels = mbc.predict(x_day)
            for predicted_label in predicted_labels:
                if predicted_label not in conf_mats[day][true_label]:
                    conf_mats[day][true_label][predicted_label] = 0
                conf_mats[day][true_label][predicted_label] += 1
    with open(os.path.join(result_dir, 'inference-timings.json'), 'w') as fh:
        json.dump(mbc.inference_times, fh)
    with open(os.path.join(result_dir, 'classification-timings.json'), 'w') as fh:
        json.dump(mbc.classification_times, fh)


def evaluate_knn(config, closed_world_labels, open_world_labels) -> None:
    all_labels = [s for s in closed_world_labels]
    all_labels.extend([s for s in open_world_labels])
    run_single_config(config, open_world_labels, all_labels, 'test')


def main() -> None:
    with open("/opt/project/closed-world-labels.json", "r") as fh:
        closed_world_labels = json.load(fh)
    with open("/opt/project/open-world-labels.json", "r") as fh:
        open_world_labels = json.load(fh)
    configs = make_configs()
    for config in configs:
        logger.info(f"Evaluate {config['classifier']}")
        if config['classifier'] == 'knn':
            evaluate_knn(config, closed_world_labels, open_world_labels)
        else:
            evaluate_pgm(config, closed_world_labels, open_world_labels)


if __name__ == '__main__':
    main()
