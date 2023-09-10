import os
import sys
import numpy as np
from typing import Any, Dict, List, Tuple

sys.path.append('../')

import fhmm, learning
from data_conversion import data_aggregation_hmm_np
from data_conversion import data_aggregation
from data_conversion import constants
from fhmm_np import check_classification_fhmm


def calc_binary_class(pcap_file: str, hmm: fhmm.Hmm, threshold: float, trace_length: int, center: np.array, flow: str='main_flow'):

    trace = data_aggregation_hmm_np.convert_trace_to_states(pcap_file, trace_length=trace_length, centers=center, flow=flow)
    log_prob = learning.calc_log_prob(hmm, [trace])

    class_aff = 0
    if log_prob >= threshold:
        class_aff = 1

    return class_aff, log_prob


def classify_trace(pcap_file: str, hmms: List[Dict[str, fhmm.Hmm]], thresholds: Dict[str, float], trace_lengths: Dict[str, int], centers: Dict[str, np.array], flow: str='main_flow') -> None:
    """
    Classifies a trace with all hmms and saves result

    Args:
        pcap_file (str):    string to pacp path
        hmms (dict):    dict of hmms with companys as keys and phmm.hmm as values

    Returns:
        /
    """

    pcap_file_test = os.path.join(constants.trace_path, pcap_file + '.pcapng')
    if not os.path.exists(pcap_file_test):
        return 0

    log_probs = {}
    class_affs = {}
    for company, hmm in hmms.items():
        class_aff, log_prob = calc_binary_class(pcap_file_test, hmm, thresholds[company], trace_lengths[company], centers[company])
        class_affs[company] = class_aff
        log_probs[company] = log_prob

    data_aggregation.save_predict_binary(pcap_file, class_affs, log_probs, thresholds)


def run_classification_hmm() -> None:
    """
    Classifies all test traces

    Args:
        /

    Returns:
        /
    """
    browser = constants.BROWSER_NOT_WGET
    hmms, thresholds, trace_lengths, centers = data_aggregation_hmm_np.load_fingerprints_fhmm(browser)
    data_aggregation.create_folders(hmms.keys(), 'fhmm')
    pcap_files = data_aggregation.get_traces_test(browser)
    for pcap_file in pcap_files:
        classify_trace(pcap_file, hmms, thresholds, trace_lengths, centers)

    assignment = check_classification_fhmm.check_assignment(browser)

    # return assignment


if __name__ == '__main__' :
    
    run_classification_hmm()