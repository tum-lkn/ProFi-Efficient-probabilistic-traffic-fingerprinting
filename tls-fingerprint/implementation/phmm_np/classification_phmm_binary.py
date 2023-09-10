import os
import sys
import numpy as np
from typing import Any, Dict, List, Tuple

sys.path.append('../')
sys.path.append('../../..')

from baum_welch import phmm, learning
from data_conversion import data_aggregation
from data_conversion import data_aggregation_hmm_np
from data_conversion import constants
from phmm_np import check_classification_phmm


def calc_binary_class(pcap_file: str, hmm: phmm.Hmm, threshold: float, trace_length: int, center: np.array, flow: str='main_flow') -> Tuple[int, float]:

    """
    Converts the trace into state representation and calculates the binary class of a trace

    Args:
        pcap_file (str):    path to pcap file
        hmm (phmm.Hmm): hmm to calc log_prob of
        threshold (float):  threshold to compare log_prob to
        trace_length (int): length of converted trace
        center (np.array):  centers to quantize packet size to
        flow (str): flow to extract of trace

    Returns:
        class_aff (int):    binary class identifier
        log_prob (float):   log_prob of trace
    """

    trace = data_aggregation_hmm_np.convert_trace_to_states(pcap_file, trace_length=trace_length, centers=center, flow=flow)
    log_prob = learning.calc_log_prob(hmm, [trace])

    gamma = 1.1

    norm_log = log_prob / (gamma * threshold)

    class_aff = 0
    if norm_log <= 1.0:
        class_aff = 1

    return class_aff, log_prob


def classify_trace(pcap_file: str, hmms: List[Dict[str, phmm.Hmm]], thresholds: Dict[str, float], trace_lengths: Dict[str, int], centers: Dict[str, np.array], flow: str='main_flow') -> None:
    """
    Classifies a trace with all hmms by calculating the binary classes and saves result

    Args:
        pcap_file (str):    string to pacp path
        hmms (dict):    dict of hmms with companys as keys and phmm.hmm as values
        thresholds (dict):  dict of thresholds with companys as keys and thresholds as values
        trace_lengths (dict):   dict of trace_lengths with companys as keys and trace_lengths as values
        centers (dict): dict of centers with companys as keys and centers as values
        flow (str): flow to extract of trace

    Returns:
        /
    """

    pcap_file_test = os.path.join(constants.trace_path, pcap_file + '.pcapng')
    if not os.path.exists(pcap_file_test):
        return 0

    log_probs = {}
    class_affs = {}
    for company, hmm in hmms.items():
        class_aff, log_prob = calc_binary_class(pcap_file_test, hmm, thresholds[company], trace_lengths[company], centers[company], flow)
        class_affs[company] = class_aff
        log_probs[company] = log_prob

    data_aggregation.save_predict_binary(pcap_file, class_affs, log_probs, thresholds, flow)


def run_classification_hmm() -> None:
    """
    Classifies all test traces

    Args:
        /

    Returns:
        /
    """
    browser = constants.BROWSER_NOT_WGET
    flow = constants.FLOW_MAIN
    hmms, thresholds, trace_lengths, centers = data_aggregation_hmm_np.load_fingerprints_phmm(browser)
    data_aggregation.create_folders(hmms.keys(), 'phmm')
    pcap_files = data_aggregation.get_traces_test(browser)
    for pcap_file in pcap_files:
        classify_trace(pcap_file, hmms, thresholds, trace_lengths, centers, flow)

    assignment = check_classification_phmm.check_assignment(browser)

    # return assignment


if __name__ == '__main__' :
    
    run_classification_hmm()