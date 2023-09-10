import os
import sys
import numpy as np
from typing import Any, Dict, List, Tuple

sys.path.append('../')
sys.path.append('../../..')

from baum_welch import phmm, learning
from data_conversion import data_aggregation
from data_conversion import constants
from phmm_np import check_classification_hmm


def classify_trace(pcap_file: str, hmms: List[Dict[str, phmm.Hmm]], flow: str='main_flow', trace_length: int, centers: np.array) -> None:
    """
    Classifies a trace with all hmms and saves result

    Args:
        pcap_file (str):    string to pcap path
        hmms (dict):    dict of hmms with companys as keys and phmm.hmm as values
        flow (str): flow of packets to extract
        trace_length (int): length of trace
        centers (np.array): centers of packet size quantization

    Returns:
        /
    """
    pcap_file_test = os.path.join(constants.trace_path, pcap_file + '.pcapng')
    if not os.path.exists(pcap_file_test):
        return 0

    trace = data_aggregation.convert_trace_to_states(pcap_file_test, trace_length=trace_length, centers=centers, flow=flow)
    log_prob = {}
    for company, hmm in hmms.items():
        log_prob[company] = learning.calc_log_prob(hmm, [trace])

    data_aggregation.save_predict(pcap_file, log_prob, flow)


def run_classification_hmm() -> None:
    """
    Classifies all test traces

    Args:
        /

    Returns:
        /
    """
    flow = constants.FLOW_MAIN
    browser = constants.BROWSER_MOZILLA
    trace_length = 8

    num_bins = 30
    binning_method = constants.BINNING_SINGLE
    centers = data_aggregation.get_binning_centers(browser=browser, num_bins=num_bins, binning_method=binning_method)

    hmms, _ = data_aggregation.load_fingerprints_phmm_np()
    data_aggregation.create_folders(hmms.keys())
    pcap_files = data_aggregation.get_traces_test(browser)
    for pcap_file in pcap_files:
        classify_trace(pcap_file, hmms, flow)

    assignment = check_classification_hmm.check_assignment(browser)

    return assignment


if __name__ == '__main__' :
    
    run_classification_hmm()