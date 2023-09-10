# Imports:
#   External Libraries

import os
import sys
import numpy as np
import multiprocessing as mp
from itertools import repeat
from typing import Any, Dict, List, Tuple

#   Files from this Project

sys.path.append('../')
sys.path.append('/mounted-data/code/')

from data_conversion import data_aggregation
from data_conversion import constants
from data_conversion import data_aggregation_hmm_np
from baum_welch import phmm, learning


def init_HMM(observation_space: List[str], seed: int, hmm_duration: int, init_prior: str='uniform') -> phmm.Hmm:
    """
    Inits a Phmm

    Args:
        observation_space (list):  list of observable states in the traces
        seed (int):  seed to init the Hmm probabilities
        hmm_duration (int): length of the init Phmm
        init_prior (str):   distribution of first transition

    Returns:
        hmm (phmm.hmm): Phmm
    """
    hmm = phmm.basic_phmm(
        duration=hmm_duration,
        observation_space=observation_space,
        init_prior=init_prior,
        seed = seed
    )
    return hmm

def acquire_training_data(company: str, browser: str) -> Tuple[Dict[str, Any], np.array, List[str], List[str]]:
    """
    Loads the parameters for Phmm and trace conversion

    Args:
        company (str):  specified service to load
        browser (str): specified browser to load

    Returns:
        params (dict): dictonary of parameters
        centers (np.array): array of packet sizes to quantize to
        traces_train (list):    list of converted training sequences
        traces_val (list):  list of converted validation sequences
    """

    params = data_aggregation_hmm_np.load_phmm_params(company=company, browser=browser)
    pcaps = None
    if params['binning_method'] == constants.BINNING_FREQ:
        pcaps = data_aggregation.acquire_pcap_files(company=params['company'], browser=params['browser'], data_set='train')
    centers = data_aggregation.get_binning_centers(num_bins=params['num_bins'], binning=params['binning_method'], pcaps=pcaps)

    traces_train = data_aggregation_hmm_np.convert_pcap_to_states(
        company=company,
        browser=browser,
        flow=params['flow'],
        trace_length=params['trace_length'],
        centers=centers,
        data_set='train'
    )
    traces_val = data_aggregation_hmm_np.convert_pcap_to_states(
        company=company,
        browser=browser,
        flow=params['flow'],
        trace_length=params['trace_length'],
        centers=centers,
        data_set='val'
    )

    return params, centers, traces_train, traces_val


def train_hmm_step(hmm_duration: int, observation_space: List[str], seed: int, traces_train: List[List[str]], traces_val: List[List[str]], init_prior: str='uniform') -> Tuple[phmm.Hmm, float]:
    """
    Trains one Hmm for number of epochs

    Args:
        hmm_duration (int): length of the init Phmm
        observation_space (list):  list of observable states in the traces
        seed (int):  seed to init the Hmm probabilities
        traces_train (list):    list of converted training sequences
        traces_val (list):  list of converted validation sequences
        init_prior (str):   distribution of first transition

    Returns:
        hmm_best (phmm.hmm): best trained hmm
        log_prob_best (float): best log prob of the hmm
    """
    hmm = init_HMM(
                    observation_space=observation_space,
                    seed=seed,
                    hmm_duration=hmm_duration,
                    init_prior=init_prior
                    )
    # hmm = phmm.hmm_from_sequences(traces_train, seed)
    epochs = 15
    log_prob_best = np.NINF
    hmm_best = None
    for _ in range(epochs):
        hmm, _, _ = learning.baum_welch(hmm, traces_train)
        log_prob_l = learning.calc_log_prob(hmm, traces_val)
        log_prob = np.sum(log_prob_l)
        if log_prob > log_prob_best and log_prob < 0:
            log_prob_best = log_prob
            hmm_best = hmm
    return (hmm_best, log_prob_best)


def calc_best_hmm(samples: int, return_list: List[Tuple[phmm.Hmm, float]]) -> phmm.Hmm:

    """
    Extracts the best Hmm of the trained Hmms based on the best log_prob

    Args:
        samples (int):  number of trained Hmms
        return_list (list): List of all Hmms and log_probs
    
    Returns:
        hmm_best (phmm.hmm): best trained hmm
    """

    log_prob_best = np.NINF
    hmm_best = None

    for i in range(samples):
        if return_list[i][1] > log_prob_best:
            log_prob_best = return_list[i][1]
            hmm_best = return_list[i][0]

    return hmm_best


def calc_threshold(hmm_best: phmm.Hmm, traces: List[str]) -> float:

    """
    Calculates the threshold out of the given traces

    Args:
        hmm_best (phmm.Hmm):    best trained hmm
        traces (list): list of converted sequences

    Returns:
        threshold (float):  threshold for binary decision
    """

    log_prob_l = []
    for trace in traces:
        log_prob = learning.calc_log_prob(hmm_best, [trace])
        log_prob_l.append(log_prob)

    threshold = np.percentile(log_prob_l, 100 - 100)

    return threshold


def save_hmm_params(hmm: phmm.Hmm, threshold: float, params: Dict[str, Any], centers: np.array) -> None:

    """
    Saves the Hmm and parameters of the training

    Args:
        hmm (phmm.Hmm): hmm to save
        threshold (float):  threshold for binary classification
        params (dict):  dict of parameters to save
        centers (np.array): array of packet size quantization centers

    Returns:
        /
    """

    data_aggregation_hmm_np.save_phmm(hmm, params['company'], params['browser'], params['trace_length'], centers, threshold)
    data_aggregation_hmm_np.save_params_phmm(
            path=None,
            company=params['company'],
            browser=params['browser'],
            flow=params['flow'],
            binning_method=params['binning_method'],
            num_bins=params['num_bins'],
            trace_length=params['trace_length'],
            hmm_duration=params['hmm_duration'],
            init_prior=params['init_prior']
            )


def train_hmm(company: str, browser: str) -> None:
    """
    Trains n number of Hmms of a specified company and saves the best Hmm

    Args:
        company (str):  name of the company
        browser (str):  browser to train on

    Returns:
        /
    """

    params, centers, traces_train, traces_val = acquire_training_data(company, browser)

    if not traces_train or not traces_val:
        return 0

    traces = traces_train + traces_val
    observation_space = data_aggregation.get_observation_space(traces)

    samples = 12
    with mp.Pool(samples) as p:
        return_list = p.starmap(
                                train_hmm_step, 
                                zip(
                                    repeat(params['hmm_duration'], samples),
                                    repeat(observation_space, samples), 
                                    range(samples),
                                    repeat(traces_train, samples), 
                                    repeat(traces_val, samples),
                                    repeat(params['init_prior'], samples)
                                    ))
    hmm_best = calc_best_hmm(samples, return_list)
    threshold = calc_threshold(hmm_best, traces_val)

    save_hmm_params(hmm_best, threshold, params, centers)


def run_fingerprint_hmm() -> None:
    """
    Runs the fingerprints of the Hmms

    Args:
        /

    Returns:
        /
    """

    companys = data_aggregation.get_services()
    data_aggregation.create_folders(companys)

    browser = constants.BROWSER_MOZILLA

    for company in companys:
        train_hmm(company=company, browser=browser)

    
if __name__ == '__main__' :
    run_fingerprint_hmm()