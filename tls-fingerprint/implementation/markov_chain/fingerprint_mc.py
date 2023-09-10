"""
Converts the pcap traces of an application into a state representation, calculates the fingerprints of the specified applications
"""

# Imports:
#   External Libraries

import sys
import numpy as np
import pandas as pd

#   Files from this Project

sys.path.append('../')

from data_conversion import data_aggregation
from data_conversion import constants

def markov_chain(traces):
    
    """
    Calculates the 2nd order markov-chain fingerprint for a given trace.
    
    Args:
        traces (pd.DataFrame): traffic of a single application (like youtube or amazon)
        
    Returns:
        pd_ENPD (pd.DataFrame): contains the enter states and probabilities of the markov chain
        pd_EXPD (pd.DataFrame): contains the exit states and probabilities of the markov chain
        pd_Trans (pd.DataFrame): contains the transition states and probabilities of the markov chain
        
    """
    
    ENPD_states = list()
    EXPD_states = list()
    Trans_states = list()
    
    for trace in traces:
        if len(trace) == 1:
            continue
        ENPD_states.append((trace[0], trace[1]))
        EXPD_states.append((trace[len(trace) - 2], trace[len(trace) - 1]))
        
        for j in range(0, len(trace) - 2): 
            Trans_states.append((trace[j], trace[j + 1], trace[j + 2]))
        
    pd_ENPD = pd.DataFrame(ENPD_states, columns = ['prev', 'curr'])
    pd_ENPD = pd_ENPD.groupby(['prev', 'curr']).size().reset_index(name='count')
    pd_EXPD = pd.DataFrame(EXPD_states, columns = ['prev', 'curr'])
    pd_EXPD = pd_EXPD.groupby(['prev', 'curr']).size().reset_index(name='count')
    pd_Trans = pd.DataFrame(Trans_states, columns = ['prev', 'curr', 'next'])
    pd_Trans = pd_Trans.groupby(['prev', 'curr', 'next']).size().reset_index(name='count')
    ENPD = np.divide(pd_ENPD['count'], np.sum(pd_ENPD['count']))
    EXPD = np.divide(pd_EXPD['count'], np.sum(pd_EXPD['count']))
    Trans = np.zeros((len(set(Trans_states))))

    for i in range(len(set(Trans_states))):
        prev = pd_Trans.loc[i]['prev']
        curr = pd_Trans.loc[i]['curr']
        tmp = pd_Trans[(pd_Trans['prev'] == prev) & (pd_Trans['curr'] == curr)]        
        Trans[i] = np.divide(pd_Trans.loc[i]['count'], np.sum(tmp['count']))

    pd_ENPD['prob'] = ENPD
    pd_EXPD['prob'] = EXPD
    pd_Trans['prob'] = Trans
    return pd_ENPD, pd_Trans, pd_EXPD


def run_fingerprint_mc():

    """
    Loads the specified traces, calculates the fingerprints and saves the fingerprints as hdf5 files

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
    companys = data_aggregation.get_services()
    data_aggregation.create_folders(companys)

    for company in companys:
        traces = data_aggregation.convert_pcap_to_states_mc(company, num_packets=constants.number_packets, browser=browser, centers=constants.centers, included_packets=flow)
        if not traces:
            continue
        pd_ENPD, pd_Trans, pd_EXPD = markov_chain(traces)
        data_aggregation.save_mc(pd_ENPD, pd_Trans, pd_EXPD, company)


if __name__ == '__main__' :
    run_fingerprint_mc()

