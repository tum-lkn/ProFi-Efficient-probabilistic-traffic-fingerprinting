import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import repeat
from typing import Any, Dict, List, Tuple

sys.path.append('../')

from data_conversion import data_aggregation
from data_conversion import constants
from data_conversion import data_aggregation_hmm_np
import fhmm, learning

def set_params(company: str, browser: str) -> None:

    trace_length = 8
    hmm_duration = 8
    binning_method = 'None'
    num_bins = 0

    params = data_aggregation_hmm_np.load_fhmm_params(company=company, browser=browser)

    params['trace_length'] = trace_length
    params['hmm_duration'] = hmm_duration
    params['binning_method'] = binning_method
    params['num_bins'] = num_bins

    data_aggregation_hmm_np.save_params_fhmm(
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


def init_HMM(observation_space: List[str], seed: int, hmm_duration: int, init_prior: str='uniform') -> fhmm.Hmm:
    """
    Inits a Hmm

    Args:
        observation_space (list):  list of observable states in the traces
        seed (int):  seed to init the Hmm probabilities

    Returns:
        hmm (phmm.hmm): trained hmm
    """
    hmm = fhmm.basic_hmm(
        duration=hmm_duration,
        observation_space=observation_space,
        init_prior=init_prior,
        seed = seed
    )
    return hmm

def acquire_training_data(company: str, browser: str) -> Tuple[Dict[str, Any], np.array, List[str], List[str]]:

    params = data_aggregation_hmm_np.load_fhmm_params(company=company, browser=browser)
    pcaps = None
    if params['binning_method'] == constants.BINNING_FREQ:
        pcaps = data_aggregation.acquire_pcap_files(company=params['company'], browser=params['browser'], data_set='train')
    centers = data_aggregation.get_binning_centers(browser=browser, num_bins=params['num_bins'], binning=params['binning_method'], pcaps=pcaps)

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

def train_hmm_step(hmm_duration: int, observation_space: List[str], seed: int, traces_train: List[List[str]], traces_val: List[List[str]], init_prior: str='uniform') -> Tuple[fhmm.Hmm, float]:
    """
    Trains one Hmm 

    Args:
        observation_space (list):  list of observable states in the traces
        seed (int):  seed to init the Hmm probabilities
        traces (list):  list of traces in state representation

    Returns:
        hmm (phmm.hmm): trained hmm
        log_prob (float): log prob of the hmm
    """
    hmm = init_HMM(
                    observation_space=observation_space,
                    seed=seed,
                    hmm_duration=hmm_duration,
                    init_prior=init_prior
                    )
    epochs = 15
    log_prob_best = np.NINF
    hmm_best = None
    for _ in range(epochs):
        hmm = learning.baum_welch(hmm, traces_train)
        log_prob_l = learning.calc_log_prob(hmm, traces_val)
        log_prob = np.sum(log_prob_l)
        if log_prob > log_prob_best and log_prob < 0:
            log_prob_best = log_prob
            hmm_best = hmm
    return (hmm_best, log_prob_best)

def calc_best_hmm(samples: int, return_list: List[Tuple[fhmm.Hmm, float]]) -> fhmm.Hmm:

    log_prob_best = np.NINF
    hmm_best = None

    for i in range(samples):
        if return_list[i][1] > log_prob_best:
            log_prob_best = return_list[i][1]
            hmm_best = return_list[i][0]

    return hmm_best


def calc_threshold(hmm_best: fhmm.Hmm, traces: List[str]) -> float:

    log_prob_l = []
    for trace in traces:
        log_prob = learning.calc_log_prob(hmm_best, [trace])
        log_prob_l.append(log_prob)

    threshold = np.percentile(log_prob_l, 100 - 100)

    return threshold

def save_hmm_params(hmm: fhmm.Hmm, threshold: float, params: Dict[str, Any], centers: np.array) -> None:

    data_aggregation_hmm_np.save_fhmm(hmm, params['company'], params['browser'], params['trace_length'], centers, threshold)
    data_aggregation_hmm_np.save_params_fhmm(
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
    data_aggregation.create_folders(companys, 'fhmm')

    browser = constants.BROWSER_CHROME

    for company in companys:
        set_params(company=company, browser=browser)
        train_hmm(company=company, browser=browser)

    
if __name__ == '__main__' :
    run_fingerprint_hmm()