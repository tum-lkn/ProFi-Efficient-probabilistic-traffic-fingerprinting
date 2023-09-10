from __future__ import annotations

import os
import sys
import time

import numpy as np
from typing import Any, Dict, List, Tuple, Union
import json as js
import itertools as itt
import multiprocessing as mp
import ctypes
from ctypes import cdll
import json
from datetime import datetime
import logging
import uuid
import gc
import h5py

#   Files from this Project
import implementation.data_conversion.dataprep as dprep
import baum_welch.phmm as phmm
import implementation.data_conversion.data_aggregation as data_aggregation
import implementation.data_conversion.constants as constants
import implementation.data_conversion.tls_flow_extraction as tlsex
import implementation.classification.phmm as wrappers
from scripts.knn_train_eval import Config
from implementation.classification.mc import MarkovChain
from implementation.seqcache import is_cached, add_to_cache, read_cache
applications = constants.applications
use_caching = True


def load_clib() -> Tuple[callable, callable, callable]:
    lib = cdll.LoadLibrary('/opt/project/implementation/ctypes/phmm.so')
    # Argument types for baum_welch function that performs <num_iterations> steps
    # of the Baum-Welch algorithm. Trains on dedicated training sequences and
    # evaluates training progress on a separate validation set.
    # Returns parameters of best iteration.
    lib.baum_welch.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # Transition Matrix.
        ctypes.POINTER(ctypes.c_double),  # Observation Matrix.
        ctypes.POINTER(ctypes.c_int),     # Sequences for training.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in training.
        ctypes.c_int,                     # Number of sequences in training.
        ctypes.POINTER(ctypes.c_int),     # Sequences for validation.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in validation.
        ctypes.c_int,                     # Number of sequences in validation.
        ctypes.c_int,                     # Number of unique symbols, i.e., observation space.
        ctypes.c_int,                     # HMM duration.
        ctypes.c_int                      # Number of iterations in baum welch.
        ]
    # Defines the return type. Returns the transition and the observation matrix.
    lib.baum_welch.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
    # Perform a single step of the Baum-Welch algorithm corresponding to a
    # forward pass, a backward pass and an update of the parameters.
    # Returns the updated transistion and emission model.
    lib.baum_welch_step.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # Transition Matrix.
        ctypes.POINTER(ctypes.c_double),  # Observation Matrix.
        ctypes.POINTER(ctypes.c_int),     # Sequences for training.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in training.
        ctypes.c_int,                     # Number of sequences in training.
        ctypes.POINTER(ctypes.c_int),     # Sequences for validation.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in validation.
        ctypes.c_int,                     # Number of sequences in validation.
        ctypes.c_int,                     # Number of unique symbols, i.e., observation space.
        ctypes.c_int,                     # HMM duration.
        ]
    lib.baum_welch_step.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
    # Free the memory allocated during whatever.
    lib.free_mem.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
    lib.free_mem.restype = None
    return lib.baum_welch, lib.baum_welch_step, lib.free_mem


__baum_welch, __baum_welch_step, __free = load_clib()


def read_dataset(company: str, browser: str, dataset: str) -> List[str]:
    data_file = os.path.join(constants.data_dir, 'benedikts-dataset.json')
    with open(data_file, 'r') as fh:
        data = js.load(fh)
    return data[company][browser][dataset]


def load_flow_dicts(dset: str, meta_data_day: List[Dict[str, Any]],
                    day_index: int, indicator: Dict[str, str],
                    labels: Dict[str, str], target_label: str) -> Tuple[List[Dict[str, Any]], List[str], int]:
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
        if labels[data['url_id']] != target_label:
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
    return X, y, day_index


def load_flow_dicts_mp(args) -> Tuple[List[Dict[str, Any]], List[str], int]:
    return load_flow_dicts(
        dset=args[0],
        meta_data_day=args[1],
        day_index=args[2],
        indicator=args[3],
        labels=args[3],
        target_label=args[5]
    )


def _get_edges(main_flows: List[Dict[str, Any]], seq_element: str, binning_method: str,
               num_bins: int, max_val_geo_bin: int, seq_length: int) -> Union[None, np.array]:
    """
    Calculate the edges of the bins for various binning methods.

    Retrieves a dataset based on company and browser and calculates the bins
    for the traces in this dataset.

    Args:
        company:
        browser:
        binning_method:
        num_bins:
        max_val_geo_bin:

    Returns:
        edges: The edges of the bins or None if binning method has no edges.
    """
    def get_sizes() -> np.array:
        sizes = []
        for main_flow in main_flows:
            rid = None
            # Take at most seq_element packets.
            for frame in main_flow['frames'][:seq_length]:
                if seq_element == 'frame':
                    sizes.append(frame['tcp_length'])
                else:
                    for record in frame['tls_records']:
                        # since records can span multiple packets, check here
                        # if record changed.
                        if record['record_number'] == rid:
                            continue
                        else:
                            rid = record['record_number']
                            sizes.append(record['length'])
        return np.array(sizes)


    if binning_method == constants.BINNING_FREQ:
        sizes = get_sizes()
        edges = data_aggregation.equal_frequency_binning(
            x=sizes,
            num_bins=num_bins
        )
    elif binning_method == constants.BINNING_GEOM:
        edges = data_aggregation.log_binning(
            max_val=max_val_geo_bin,
            num_bins=num_bins
        )
    elif binning_method == constants.BINNING_EQ_WIDTH:
        edges = data_aggregation.equal_width_binning(
            max_val=max_val_geo_bin,
            num_bins=num_bins
        )
    elif binning_method == constants.BINNING_SINGLE:
        edges = np.array([0.])
    else:
        edges = None
    return edges


def convert_log_prob(log_prob: ctypes.POINTER(ctypes.c_double), num_seq: int) -> Tuple[np.array, float]:

    """
    Converts the log_probs into a np.array

    Args:
        log_prob (ctypes.POINTER):  pointer to the log_prob array
        num_seq (int):  number of sequences in the array
    Returns:
        log_prob_a (np.array):  array of log_probs
        log_prob_s (float): sum of log_prob_a
    """

    log_prob_a = np.zeros(num_seq)
    for i in range(num_seq):
        log_prob_a[i] = log_prob[i]
    return log_prob_a, np.sum(log_prob_a)


def convert_c_lib_output(res: ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), hmm: phmm.PhmmC, num_traces_train: int, num_traces_val: int):

    """
    Convert the c lib output into python objects
    Args:
        res (ctypes.POINTER): pointer to the output arrays
        hmm (phmm.PhmmC): hmm object
        num_traces_train (int): number of traces train, needed for conversion of log_probs
        num_traces_val (int): number of traces val, needed for conversion of log_probs
    Returns:
        hmm_r (phmm.HMM): new hmm which is updated with the transistion and emission matrices
        log_prob_train (np.array): array of log_probs of the training sequences
        log_prob_train_all (float): sum of log_prob_train
        log_prob_val (np.array): array of log_probs of the validation sequences
        log_prob_val_all (float): sum of log_prob_train
    """

    hmm.update_p_ij(res[0])
    hmm.update_p_o_in_i(res[1])

    log_prob_train, log_prob_train_all = convert_log_prob(res[2], num_traces_train)
    log_prob_val, log_prob_val_all = convert_log_prob(res[3], num_traces_val)

    __free(res)

    hmm_r = phmm.Hmm(hmm.duration)
    hmm_r.p_ij = dict(zip(hmm._p_ij_keys, hmm._p_ij))
    hmm_r.p_o_in_i = dict(zip(hmm._p_o_in_i_keys, hmm._p_o_in_i))    

    return (hmm_r, log_prob_train, log_prob_train_all, log_prob_val, log_prob_val_all)


def save_params(params: Dict[str, Any], path: str = None) -> None:
    """
    Saves the training configuration.

    Args:
        params: Dictionary with the parameters for which a model has been trained.
    Returns:
        None.
    """
    if path is None:
        model_path = os.path.join('/mounted-data/data/grid_results', params['company'] + '_' + str(params['seed']))
        os.makedirs(model_path, exist_ok=True)
        path = os.path.join(model_path, 'hmm_' + params['company'] + '_' + params['browser'] + '_params.json')
    with open(path, 'w', encoding='utf-8') as f:
        js.dump(params, f, ensure_ascii=False, indent=4)


def save_config(config: Config, addendum: Dict[str, Any]=None, prefix=None):
    if prefix is None:
        name = 'config.json'
    else:
        name = f'config-{prefix}.json'
    with open(os.path.join(config.trial_dir, name), 'w') as fh:
        d = config.to_dict()
        if addendum is not None:
            d.update(addendum)
        json.dump(d, fh)


def save_log_probs(log_probs: List[Tuple[np.array, float, np.array, float]], trial_dir: str, num_samples: int) -> None:

    """
    Saves the log_prob lists
    Args:
        log_probs (list):   list of log_probs of training and validation sequences
        trial_dir (str):    path to the file
        num_samples (int):  number of samples of multiprocessing
    Returns:
        /
    """

    log_probs = [[log_probs[i][1].tolist(), log_probs[i][2], log_probs[i][3].tolist(), log_probs[i][4]] for i in range(num_samples)]
    with open(os.path.join(trial_dir, 'all_log_probs.json'), 'w', encoding='utf-8') as fh:
        json.dump(log_probs, fh)


def save_model(hmm: phmm.Hmm, params: Dict[str, Any], log_prob_train: List[float], log_prob_train_all: float, log_prob_val: List[float], log_prob_val_all: float, path: str) -> None:
    """
    Saves a python PHMM object. Stores the parameters, i.e., transition and
    emission model together with the lob-prob the model achieved.

    Args:
        hmm:
        params:
        log_prob:

    Returns:
        /
    """

    model = {}
    model['trans'] = {str(k): v for k, v in hmm.p_ij.items()}
    model['obs'] = {str(k): v for k, v in hmm.p_o_in_i.items()}
    model['log_prob_train'] = log_prob_train.tolist()
    model['log_prob_train_all'] = log_prob_train_all
    model['log_prob_val'] =  log_prob_val.tolist()
    model['log_prob_val_all'] = log_prob_val_all

    with open(path, 'w', encoding='utf-8') as f:
        js.dump(model, f, ensure_ascii=False)


def baum_welch_step(hmm_duration: int, obs_space: List[str], seed: int,
               traces_train: List[int], traces_val: List[int],
               traces_train_lens: List[int], traces_val_lens: List[int],
               num_traces_train: int, num_traces_val: int,
               num_obs: int, iterations: int) -> Tuple[phmm.Hmm, np.array, float, np.array, float]:
    """
    Trains a PHMM with the Baum Welch algorithm.

    Args:
        hmm_duration: Number of time steps in the PHMM.
        obs_space: Unique symbols in the observation space.
        seed: Random number initializer.
        traces_train: The traces used to train the model, i.e., perform parameter
            updates.
        traces_val: The traces used to validate the model, not used for
            parameter updates.
        trans_offsets: The offsets in the transition matrix that mark the
            beginnings of the match, insert and delete states.
        obs_offsets: The offsets in the emission model that mark the beginning
            of the missions of the match and insert states.
        traces_train_lens: A list with the lenghts of the individual sequences
            in the training set.
        traces_val_lens: L list with the lengths of the individual sequences in
            the validation set.
        num_traces_train: The total number of traces in the training set.
        num_traces_val: The total number of traces in the validation set.
        num_obs: The total number of observations in the observation space.
        iterations: Number of iterations the Baum Welch algorithm should perform.

    Returns:

    """
    traces_train = (ctypes.c_int * len(traces_train))(*traces_train)
    traces_train_lens = (ctypes.c_int * len(traces_train_lens))(*traces_train_lens)
    traces_val = (ctypes.c_int * len(traces_val))(*traces_val)
    traces_val_lens = (ctypes.c_int * len(traces_val_lens))(*traces_val_lens)

    hmm = phmm.basic_phmm_c(hmm_duration, obs_space, seed=seed)

    for iter in range(iterations):
        res = __baum_welch_step(
            hmm._p_ij_p,
            hmm._p_o_in_i_p,
            traces_train,
            traces_train_lens,
            num_traces_train,
            traces_val,
            traces_val_lens,
            num_traces_val,
            num_obs,
            hmm.duration
        )
        hmm_r, log_prob_train, log_prob_train_all, log_prob_val, log_prob_val_all = convert_c_lib_output(
            res=res,
            hmm=hmm,
            num_traces_train=num_traces_train,
            num_traces_val=num_traces_val
        )
        hmm = hmm_r
        # @TODO: store progress, store models, whatever.

    return (hmm_r, log_prob_train, log_prob_train_all, log_prob_val, log_prob_val_all)


def get_traces() -> Dict[Tuple[str, str, str], List[str]]:

    """
    Read the trace files of the local dataset
    Args:
        /
    Returns:
        traces (dict):  dict with company, browser, dataset as keys and List as values 
    """

    companys = ['google']
    browsers = [constants.BROWSER_MOZILLA, constants.BROWSER_CHROME]
    traces = {}
    for company, browser in itt.product(companys, browsers):
        traces[(company, browser, 'train')] = read_dataset(company, browser, 'train')
        traces[(company, browser, 'val')] = read_dataset(company, browser, 'val')
    return traces


def make_trial_dir_name(config: Config, exp_dir: str) -> str:
    count = 0
    file_name = f'{config.uuid}_{count}'
    while os.path.exists(os.path.join(exp_dir, file_name)):
        count += 1
        file_name = f'{config.uuid}_{count}'
    return file_name


def unpack(some_x: Dict[int, List[Dict[str, Any]]]) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Unpack the dataset into a flat one, i.e., remove the first dimension.
    Args:
        some_x: List of tuples, first tuple element is the trace day, the second
            element is a list of main flows of that day

    Returns:
        lengths: List of individual day lengths. Can be used to reconstruct the
            days if needed.
        unpacked: Flattened some_x, i.e., removed time dimension. Contains all
            main flows of all days concatenated after each other.
    """
    unpacked = []
    lengths = []
    for v in some_x.values():
        lengths.append(len(v))
        unpacked.extend(v)
    return lengths, unpacked


def _evaluate_model(logger: logging.Logger, config: Config, edges: Union[None, np.array],
                    max_ll: float, closed_world_labels: List[str],
                    model: Union[wrappers.CPhmm, MarkovChain],
                    dset: str, defense: Dict[str, Any | None],
                    direction_to_filter=0) -> Dict[str, float]:
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    lls = None
    # g_v = f.create_group("val")
    # all_labels = [x for x in closed_world_labels]
    # all_labels.extend(open_world_labels)
    main_flow_to_symbol = tlsex.MainFlowToSymbol(
        seq_length=config.seq_length,
        to_symbolize=config.seq_element_type,
        bin_edges=edges,
        skip_handshake=config.skip_tls_handshake,
        direction_to_filter=direction_to_filter
    )
    if defense is None:
        the_defense = lambda x: x
    else:
        random = np.random.RandomState(seed=defense['seed'])
        the_defense = tlsex.RandomRecordSizeDefense(
            max_seq_length=config.seq_length,
            get_random_length=lambda x: int(random.randint(defense['min_record_size'], defense['max_record_size']))
        )
    for cwl in closed_world_labels:
        try:
            # g_cwl = g_v.create_group(cwl)
            if not is_cached(f'{cwl}_{dset}.json'):
                logger.warning(f"Validation set for URL {cwl} does not exist.")
                continue
            lengths, X = unpack(read_cache(f"{cwl}_{dset}.json"))
            X = [main_flow_to_symbol(the_defense(m)) for m in X]
            lls = model.score_c(X)

            if cwl in closed_world_labels:
                scores = np.abs(lls) / max_ll
                pred_true = np.sum(scores <= 1)
                if cwl == model.label:
                    tp += pred_true
                    fn += scores.size - pred_true
                else:
                    fp += pred_true
                    tn += scores.size - pred_true

            # offset = 0
            # for i, length in enumerate(lengths):
            #     g_cwl.create_dataset(name=f"day-{i}", data=lls[offset:offset + length])
            #     offset += length
        except Exception as e:
            logger.error(f"Error during processing of validation data of {cwl}")
            logger.exception(e)
    # f.close()
    return {
        'tn': int(tn),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'max_nll_val': float(np.max(np.abs(lls))),
        'min_nll_val': float(np.min(np.abs(lls)))
    }


def evaluate_model(config: Dict[str, Any], model_params: Dict[str, Any],
                   closed_world_labels: List[str], dset: str, defense: callable,
                   logger: logging.Logger=None, direction_to_filter=0) -> Dict[str, float]:
    if logger is None:
        logger = logging.getLogger("train_model")
    if not is_cached(f"{model_params['label']}_train.json"):
        logger.warning(f"No training data exists for label {model_params['label']}.")
        return
    logger.debug(f"\tRetrieve training data from file {model_params['label']}_train.json")
    lengths, X = unpack(read_cache(f"{model_params['label']}_train.json"))
    config = Config.from_dict(config)

    edges = _get_edges(
        main_flows=X,
        seq_element=config.seq_element_type,
        binning_method=config.binning_method,
        num_bins=config.num_bins,
        max_val_geo_bin=config.max_bin_size,
        seq_length=config.seq_length
    )
    main_flow_to_symbol = tlsex.MainFlowToSymbol(
        seq_length=config.seq_length,
        to_symbolize=config.seq_element_type,
        bin_edges=edges,
        direction_to_filter=direction_to_filter
    )
    X = [main_flow_to_symbol(defense(m)) for m in X]
    phmm = wrappers.CPhmm.from_dict(model_params)
    lls = phmm.score_c(X)
    max_ll = np.max(np.abs(lls))
    # f = h5py.File(os.path.join(config.trial_dir, 'perf.h5'), "w")
    # try:
    #     g = f.create_group('train')
    #     offset = 0
    #     for i, length in enumerate(lengths):
    #         g.create_dataset(f"day-{i}", data=lls[offset:offset + length])
    #         offset += length
    # except Exception as e:
    #     logger.exception(e)
    #     f.close()
    #     os.rmdir(os.path.join(config.trial_dir, 'perf.h5'))
    #     raise e

    ret = _evaluate_model(
        logger=logger,
        config=config,
        edges=edges,
        max_ll=max_ll,
        closed_world_labels=closed_world_labels,
        model=phmm,
        dset=dset,
        defense=defense
    )
    ret['min_nll_train'] = float(np.min(np.abs(lls)))
    ret['max_nll_train'] = float(max_ll)
    return ret


def train_phmm_model(params: Dict[str, Any], closed_world_labels: List[str],
                     logger: logging.Logger=None, eval_dset='val',
                     defense: None | Dict[str, Any] = None, edges: np.array = None,
                     direction_to_filter=0) -> Union[MarkovChain, wrappers.CPhmm]:
    """
    Trains a phmm model with a given parameter set
    Args:
        params (dict):  dict with parameters
    Returns:
        /
    """
    if logger is None:
        logger = logging.getLogger("train_model")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
    num_samples = np.min([3, params.pop("num_samples")])
    exp_dir = params.pop("exp_dir")
    config = Config.from_dict(params)
    logger.debug(f"\tRetrieve training data from file {params['label']}_train.json")
    _, X_train = unpack({k: v for k, v in read_cache(f"{params['label']}_train.json").items() if int(k) < config.day_train_end})
    logger.debug(f"\tRetrieve validation data from file {params['label']}_val.json")
    lengths_val, X_val = unpack(read_cache(f"{params['label']}_{eval_dset}.json"))

    logger.debug(f"\tGet bin edges for {config.binning_method}.")
    logger.debug(f"\tSkip TLS Handshake {config.skip_tls_handshake}.")
    if edges is None:
        edges = _get_edges(
            main_flows=X_train,
            seq_element=config.seq_element_type,
            binning_method=config.binning_method,
            num_bins=config.num_bins,
            max_val_geo_bin=config.max_bin_size,
            seq_length=config.seq_length
        )
    if eval_dset == 'test':
        _, X_val2 = unpack({k: v for k, v in read_cache(f"{params['label']}_val.json").items() if int(k) < config.day_train_end})
        X_train.extend(X_val2)
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
        skip_handshake=config.skip_tls_handshake,
        direction_to_filter=direction_to_filter
    )
    X_train = [main_flow_to_symbol(the_defense(m)) for m in X_train]
    logger.debug(f"\tConvert sequences in validation set, {len(X_val)} in total, to symbols.")
    X_val = [main_flow_to_symbol(the_defense(m)) for m in X_val]
    gc.collect()

    best_log_prob = 1e12
    best_model = None
    training_times = []
    for seed in range(1, num_samples + 1):
        logger.debug(f"\tFit {seed} of {num_samples} models.")
        if params['trial_dir'] is None:
            # Use params, config is overwritten every time.
            config.uuid = uuid.uuid4().hex
            config.trial_dir = os.path.join(exp_dir, make_trial_dir_name(config, exp_dir))
            logger.info(f"\tSet trial dir to {config.trial_dir}")
        os.makedirs(config.trial_dir, exist_ok=True)
        config.seed = seed
        save_config(config)
        t1 = time.perf_counter()
        if config.classifier == 'phmm':
            wrapper = wrappers.CPhmm(
                duration=config.hmm_length,
                init_prior=config.hmm_init_prior,
                seed=seed,
                label=config.label,
                num_iter=config.hmm_num_iter
            ).fit(X_train, X_val)
        elif config.classifier == 'mc':
            wrapper = MarkovChain(config.label).fit(X_train, X_val)
        else:
            raise KeyError(f"Unknown model {config.classifier}.")
        training_times.append(time.perf_counter() - t1)

        # Log prob is negative, loss is negative log likelihood --> take the
        # minimum and then check for larger equal. Classification would fail if
        # the log-likelihood in the validation set would be smaller than the
        # minimum of the training set.
        logger.debug(f"\tSave config to {os.path.join(config.trial_dir, 'config.json')}")
        worst_log_prob = np.min(wrapper.log_prob_train_all)
        config.accuracy = float(np.mean(wrapper.log_prob_val_all >= worst_log_prob))
        d = {}
        d['sum_log_prob_train'] = float(np.sum(wrapper.log_prob_train_all))
        d['sum_log_prob_val'] = float(np.sum(wrapper.log_prob_val_all))
        d['avg_log_prob_train'] = float(np.mean(wrapper.log_prob_train_all))
        d['avg_log_prob_val'] = float(np.mean(wrapper.log_prob_val_all))
        d['min_nll_train'] = float(np.min(np.abs(wrapper.log_prob_train_all)))
        d['min_nll_val'] = float(np.min(np.abs(wrapper.log_prob_val_all)))
        d['max_nll_train'] = float(np.max(np.abs(wrapper.log_prob_train_all)))
        d['max_nll_val'] = float(np.max(np.abs(wrapper.log_prob_val_all)))
        if eval_dset == 'test':
            d['nlls'] = []
            start = 0
            for length in lengths_val:
                d['nlls'].append([float(x) for x in np.abs(wrapper.log_prob_val_all)[start:start + length]])
                start = start + length
        save_config(config, addendum=d)

        if best_log_prob > np.abs(d['avg_log_prob_val']):
            best_model = wrapper
            best_model.nlls = d['nlls']

        if eval_dset != 'test' or best_log_prob > np.abs(d['avg_log_prob_val']):
            best_log_prob = np.abs(d['avg_log_prob_val'])
            best_model.best_nll = best_log_prob
            best_model.max_nll = d['max_nll_train']
            best_model.edges = edges

            logger.debug(f"\tSave model to {os.path.join(config.trial_dir, 'model.json')}")
            logger.info(f"New best log prob with {d['avg_log_prob_val']}.")
            model_d = wrapper.to_json_dict()
            if eval_dset == 'test':
                name = f'model-{config.label}.json'
            else:
                name = 'model.json'
            with open(os.path.join(config.trial_dir, name), "w") as fh:
                json.dump(model_d, fh)

            max_ll = np.max(np.abs(wrapper.log_prob_train_all))
            metrics = _evaluate_model(
                logger=logger,
                config=config,
                edges=edges,
                max_ll=max_ll,
                closed_world_labels=closed_world_labels,
                model=wrapper,
                dset=eval_dset,
                defense=defense,
                direction_to_filter=direction_to_filter
            )
            d.update(metrics)
            wrapper.stats = d
            save_config(config, addendum=d, prefix=config.label if eval_dset == 'test' else None)
    best_model.training_times = training_times
    return best_model


def get_data(logger: logging.Logger, company: str) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[str]]:
    def load_data(dset, meta_data, start_day, end_day):
        if is_cached(f'X_{dset}_for_{company}.json'):
            X_train = read_cache(f'X_{dset}_for_{company}.json')
            y_train = read_cache(f'y_{dset}_for_{company}.json')
        else:
            logger.info("Serialize meta data")
            meta_data = [m for _, m in meta_data]
            pool = mp.Pool(30)
            rets = pool.map(
                load_flow_dicts_mp,
                [('train', m, i, indicator, labels, company)
                 for i, m in enumerate(meta_data[start_day:end_day])])
            pool.close()
            rets.sort(key=lambda x: x[2])
            X_train = []
            y_train = []
            for x, y, _ in rets:
                X_train.extend(x)
                y_train.extend(y)
            add_to_cache(f'X_{dset}_for_{company}.json', X_train)
            add_to_cache(f'y_{dset}_for_{company}.json', y_train)
        return X_train, y_train

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
        pool = mp.Pool(10)
        meta_data = pool.map(dprep.get_days_metadata, days)[:50]
        pool.close()
        meta_data.sort(key=lambda x: x[0])
        add_to_cache('meta_data.json', [(str(a), b) for a, b in meta_data])

    # Start and end-days.
    X_train, y_train = load_data('train', meta_data, 0, 30)
    X_val, y_val = load_data('val', meta_data, 0, 30)
    return X_train, y_train, X_val, y_val


def run_grid_search(company: str, model: str):
    logger = logging.getLogger("grid-search-logger")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
    exp_dir = '/opt/project/data/grid-search-results/'

    binning_method = [constants.BINNING_EQ_WIDTH]
    number_bins = list(range(30, 41, 10))
    hmm_duration = list(range(4, 7, 2))
    trace_lengths = list(range(8, 11, 2))
    iterations = 10
    n_samples = 1

    for binning, num_bins, dur, length in itt.product(binning_method, number_bins, hmm_duration, trace_lengths):
        with open("/opt/project/closed-world-labels.json", "r") as fh:
            closed_world_labels = json.load(fh)
        if binning == constants.BINNING_NONE:
            num_bins = 0
        logger.debug("Create Config")
        config = Config(
            classifier=model,
            binning_method=binning,
            num_bins=num_bins,
            seq_length=length,
            seq_element_type='frame',
            hmm_length=dur,
            day_train_start=0,
            day_train_end=30,
            knn_num_neighbors=None,
            hmm_num_iter=iterations,
            hmm_init_prior='uniform',
            ano_density_estimator=None,
            seed=None,
            max_bin_size=1500,
            label=company
        )
        params = config.to_dict()
        params['num_samples'] = n_samples
        params['exp_dir'] = exp_dir
        logger.debug("Train model")
        train_phmm_model(params, closed_world_labels, logger)
        break


if __name__ == '__main__':
    run_grid_search('bitly.com', 'mc')