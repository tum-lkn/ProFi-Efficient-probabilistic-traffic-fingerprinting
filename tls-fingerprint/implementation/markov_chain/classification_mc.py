"""
Classifies a trace with all the current fingerprints
"""

# Imports:
#   External Libraries

import os
import sys
import sqlalchemy
import numpy as np
import pandas as pd

#   Files from this Project

sys.path.append('../')

from markov_chain import check_classification_mc
from data_conversion import data_aggregation
from data_conversion import constants

    
# def predict_cluster(bigram_matrix, cluster_centers, bigram):

#     """
#     Classifies the given traffic with regards to the loaded application fingerprints
    
#     Args:
#         bigram_matrix (np.array): 2D array with shape (num_applications, num_clusters)
#         cluster_center (np.array): 2D array with shape (num_applications, 2)
#         bigram (np.array): attribute bigram of trace with shape (1, 2)
        
#     Returns:
#         log_vec (np.array): 1D array with shape (num_applications) is the predicted column of the bigram matrix
#     """

#     index = np.argmin(np.linalg.norm(cluster_centers - bigram, ord=2, axis=1))

#     log_vec = np.log(bigram_matrix[:, index])

#     return log_vec


def calculate_loglikelihood(markov_chain, trace):
    
    """
    Calculates the loglikelihoods of a trace with respect to a single application
    
    Args:
        company (string): name of the application
        trace (list): trace as list of states
        
    Returns:
        likelihood (double): likelihood of the classified application
    """

    pd_ENPD = markov_chain[0]
    pd_EXPD = markov_chain[1]
    pd_Trans = markov_chain[2]
    likelihood = 0
    ENPD = 1e-64
    EXPD = 1e-64

    try:
        ENPD_tmp = pd_ENPD[(pd_ENPD['prev'] == trace[0]) & (pd_ENPD['curr'] == trace[1])]
        EXPD_tmp = pd_EXPD[(pd_EXPD['prev'] == trace[len(trace) - 2]) & (pd_EXPD['curr'] == trace[len(trace) - 1])]
    except:
        print(trace)

    if not ENPD_tmp.empty:
        ENPD = ENPD_tmp['prob'].iloc[0]
    if not EXPD_tmp.empty:
        EXPD = EXPD_tmp['prob'].iloc[0]
    likelihood += np.log(ENPD) + np.log(EXPD)
    
    for i in range(len(trace) - 2):
        prob = 1e-32
        Trans_tmp = pd_Trans[(pd_Trans['prev'] == trace[i]) & (pd_Trans['curr'] == trace[i + 1]) & (pd_Trans['next'] == trace[i + 2])]
        if not Trans_tmp.empty:
            prob = Trans_tmp['prob'].iloc[0]
        likelihood += np.log(prob)

    return likelihood


def classify_trace(pcap_file, mcs, included_packets):

    """
    Converts the pcap file into a state representation and computes the likelihoods for all applications
    
    Args:
        pcap_file (string): path to the pcap file
        companys (list): list of applications
        bigram_matrix (np.array): 2D array with shape (num_applications, num_clusters) contains the bigram probabilities
        cluster_centers (np.array): 2D array with shape (num_clusters, 2)
        
    Returns:
        likelihoods (list): contains the likelihoods for all applications
    """

    pcap_file_test = os.path.join(constants.trace_path, pcap_file + '.pcapng')
    if not os.path.exists(pcap_file_test):
        return 0
    trace = data_aggregation.convert_trace_to_states_mc(pcap_file_test, num_packets=constants.number_packets, centers=constants.centers, included_packets=included_packets)
    likelihoods = []

    for markov_chain in mcs.values():

        likelihoods.append(calculate_loglikelihood(markov_chain, trace))

    log_prob = dict(zip(mcs.keys(), likelihoods))
    data_aggregation.save_predict(pcap_file, log_prob)


def run_classify_traffic():

    """
    Classifys the traces and outputs the results

    Args:
        /

    Returns:
        /
    """
    flow = constants.FLOW_MAIN
    if constants.co_flows:
        flow = constants.FLOW_CO
    browser = None
    data_aggregation.set_constants()
    mcs = data_aggregation.load_fingerprints_mc()
    data_aggregation.create_folders(mcs.keys())
    pcap_files = data_aggregation.get_traces_test(browser)

    for pcap_file in pcap_files:
        classify_trace(pcap_file, mcs, flow)

    assignment = check_classification_mc.check_assignment(browser)
    return assignment


if __name__ == '__main__' :
    run_classify_traffic()

