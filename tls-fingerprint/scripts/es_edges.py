from __future__ import annotations
import numpy as np
import logging
import time
import json
from typing import List, Tuple, Dict, Any

import implementation.phmm_np.grid_search as gs
from implementation.seqcache import is_cached, add_to_cache, read_cache
import implementation.data_conversion.tls_flow_extraction as tlsex
from implementation.classification.mc import MarkovChain


class MakeNoise(object):
    def __init__(self, population_size: int, num_features: int, seed: int):
        self.population_size = population_size
        self.num_features = num_features
        self.seed = seed
        self.last_noise = None
        self.random = np.random.RandomState(seed=seed)
        self.loc = 0
        self.scale = 1

    def __call__(self):
        self.last_noise = self.random.normal(self.loc, self.scale, size=[self.population_size, self.num_features])
        return self.last_noise


def train_mc_model(params: Dict[str, Any], closed_world_labels: List[str],
                   edges: np.array, logger: logging.Logger=None,
                   defense: None | Dict[str, Any] = None) -> MarkovChain:
    """
    Trains a phmm model with a given parameter set
    Args:
        params (dict):  dict with parameters
    Returns:
        /
    """
    eval_dset = 'val'
    if logger is None:
        logger = logging.getLogger("train_model")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
    config = gs.Config.from_dict(params)
    logger.debug(f"\tRetrieve training data from file {params['label']}_train.json")
    _, X_train = gs.unpack({k: v for k, v in read_cache(f"{params['label']}_train.json").items() if int(k) < config.day_train_end})
    logger.debug(f"\tRetrieve validation data from file {params['label']}_val.json")
    lengths_val, X_val = gs.unpack(read_cache(f"{params['label']}_{eval_dset}.json"))

    logger.debug(f"\tSkip TLS Handshake {config.skip_tls_handshake}.")
    if defense is None:
        the_defense = lambda x: x
    else:
        logger.debug(f"Apply RandomRecordSizeDefense with record lengths between {defense['min_record_size']} and {defense['max_record_size']}")
        random = np.random.RandomState(seed=defense['seed'])
        the_defense = tlsex.RandomRecordSizeDefense(
            max_seq_length=config.seq_length,
            get_random_length=lambda x: int(random.randint(defense['min_record_size'], defense['max_record_size']))
        )
    logger.debug(f"\tConvert sequences in training set, {len(X_train)} in total, to symbols.")
    logger.debug(f"\tSeq length {config.seq_length}, HMM length {config.hmm_length}, Skip Handshake: {config.skip_tls_handshake}")
    main_flow_to_symbol = tlsex.MainFlowToSymbol(
        seq_length=config.seq_length,
        to_symbolize=config.seq_element_type,
        bin_edges=edges,
        skip_handshake=config.skip_tls_handshake
    )
    X_train = [main_flow_to_symbol(the_defense(m)) for m in X_train]
    logger.debug(f"\tConvert sequences in validation set, {len(X_val)} in total, to symbols.")
    X_val = [main_flow_to_symbol(the_defense(m)) for m in X_val]

    training_times = []
    config.seed = 1
    t1 = time.perf_counter()
    wrapper = MarkovChain(config.label).fit(X_train, X_val)
    training_times.append(time.perf_counter() - t1)

    # Log prob is negative, loss is negative log likelihood --> take the
    # minimum and then check for larger equal. Classification would fail if
    # the log-likelihood in the validation set would be smaller than the
    # minimum of the training set.
    worst_log_prob = np.min(wrapper.log_prob_train_all)
    config.accuracy = float(np.mean(wrapper.log_prob_val_all >= worst_log_prob))
    d = {}
    best_model = wrapper
    best_model.edges = edges

    max_ll = np.max(np.abs(wrapper.log_prob_train_all))
    metrics = gs._evaluate_model(
        logger=logger,
        config=config,
        edges=edges,
        max_ll=max_ll,
        closed_world_labels=closed_world_labels,
        model=wrapper,
        dset=eval_dset,
        defense=defense
    )
    d.update(metrics)
    wrapper.stats = d
    best_model.training_times = training_times
    return best_model


def make_utilities(population_size: int) -> np.array:
    """
    Make utility values to weight samples with.

    Args:
        population_size:

    Returns:
        utility_values: Shape (S,).
    """
    utility_values = np.log(population_size / 2. + 1) - \
                     np.log(np.arange(1, population_size + 1))
    utility_values[utility_values < 0] = 0.
    utility_values /= np.sum(utility_values)
    utility_values -= 1. / population_size
    utility_values = utility_values.astype(np.float32)
    return utility_values


def calculate_gradient(indices: np.array, noise: np.array,
                       utilities: np.array) -> np.array:
    scaled_noise = np.expand_dims(utilities, axis=1) * noise[indices]
    grad = np.sum(scaled_noise, 0)
    return grad


def main_mc(population_size: int, num_edges: int):
    with open("/opt/project/closed-world-labels.json", "r") as fh:
        closed_world_labels = json.load(fh)
    logger = logging.getLogger('es-edges')
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
    random = np.random.RandomState(seed=2)
    make_noise = MakeNoise(population_size, num_edges, 1)
    make_noise.scale = 100
    edges = random.randint(0, 1500, size=(1, num_edges)).astype(np.float32)
    utilities = make_utilities(population_size)
    best_fitness = 0
    best_edges = edges
    estr = ''
    for epoch in range(1000):
        noise = make_noise()
        edges_ = edges + noise
        fitness = np.zeros(population_size)
        for j in range(population_size):
            m = train_mc_model(
                params={
                    'label': 'www.inquirer.net',
                    'classifier': 'mc',
                    'binning_method': 'es',
                    'num_bins': num_edges,
                    'seq_length': 20,
                    'seq_element_type': 'Frame',
                    'hmm_length': None,
                    'day_train_start': 0,
                    'day_train_end': 30,
                    'knn_num_neighbors': None,
                    'hmm_init_prior': None,
                    'hmm_num_iter': None,
                    'ano_density_estimator': None,
                    'seed': 1
                },
                closed_world_labels=closed_world_labels,
                edges=edges_[j, :],
                logger=logger,
                defense=None
            )
            fitness[j] = -1. * m.stats['tp'] / (m.stats['tp'] + m.stats['fp'])
        indices = np.argsort(fitness)
        gradient = calculate_gradient(indices, noise, utilities)
        edges += np.expand_dims(gradient, 0)
        make_noise.scale *= 0.99

        avg_fitness = np.mean(fitness)
        tmp = np.min(fitness)
        if tmp < best_fitness:
            best_fitness = tmp
            best_edges = edges_[np.argmin(fitness)]
            estr = ' ,'.join([str(x) for x in best_edges.astype(np.int)])
        logger.info(f'{epoch:3d}\t{avg_fitness:.4f}\t{best_fitness:.4f}\t{estr}')
    return best_edges


def main(population_size: int):
    make_noise = MakeNoise(population_size, 20, 1)
    thres = np.arange(20)
    edges = np.random.rand(1, 20)
    utilities = make_utilities(population_size)
    for epoch in range(1000):
        noise = make_noise()
        edges_ = edges + noise
        fitness = np.sum(np.abs(thres - edges_), axis=1)
        indices = np.argsort(fitness)
        gradient = calculate_gradient(indices, noise, utilities)
        edges += np.expand_dims(gradient, 0)
        # print(np.mean(fitness), edges)
        print(np.mean(fitness))
        make_noise.scale *= 0.99


if __name__ == '__main__':
    main_mc(100, 100)
