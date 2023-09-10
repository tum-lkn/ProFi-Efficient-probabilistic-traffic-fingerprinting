import os
import sys
import json as js
import numpy as np
from typing import List, Dict, Tuple, Any

sys.path.append('/mounted-data/code/baum-welch')
sys.path.append('/mounted-data/code/tls-fingerprint/implementation/data_conversion')
sys.path.append('/mounted-data/code/tls-fingerprint/implementation/fhmm_np')

sys.path.append('/opt/project/baum_welch')
sys.path.append('/opt/project/implementation/data_conversion')
sys.path.append('/opt/project/implementation/fhmm_np')

import data_aggregation
import constants
import fhmm
import phmm


#############################################################################################################################

def shape_trace(trace: List[str], length: int) -> List[str]:
    """
    Cuts the trace to a specified length

    Args:
        trace (list): list with TLS traffic only
        length (int):   length to cut the trace to

    Returns:
        trace (list): cutted trace
    """
    if len(trace) <= 0:
        return trace

    if len(trace) == 1:
        trace.extend(['0'])

    return trace[:length]

def convert_trace_to_states(pcap_file: str, trace_length: int, centers: np.array=None, flow: str='main_flow') -> List[str]:
    """
    Converts the pcap representation of a trace into the state representation of markov chains

    Args:
        pcap_file (string): path to the pcap file
        centers (np.array): The centers of the bins in which the packets should
            be mapped.
        included_packets (str): The filter that should be applied to the trace. Supported
            are {main_flow, and co_flows}.

    Note:
        Use centers as follows:
            centers <-- np.array([0]) to disregard the packet size info completly.
                This is the default behavior in case None is passed.
            centers <-- None uses the raw packet size.
            centers <-- np.array([..]) use custom centers.

    Returns:
        trace (list): list containing the state sequences
    """

    trace = []
    pd_data, convertible = data_aggregation.convert_pcap_to_dataframe(pcap_file)

    if not convertible:
        return trace
    if flow == constants.FLOW_MAIN:
        pd_data = data_aggregation.extract_main_flow(pd_data)
    elif flow == constants.FLOW_CO:
        pd_data = data_aggregation.extract_co_flow(pd_data)
    else:
        raise ValueError("Value {} not known for argument included_packets".format(flow))

    pd_data = pd_data[['ip.src', 'frame.len', '_ws.col.Info']]

    client_ip = pd_data[pd_data['_ws.col.Info'].str.contains('Client Hello')].iloc[0].values[0]

    for info, length, source in zip(pd_data['_ws.col.Info'], pd_data['frame.len'], pd_data['ip.src']):
        info_splitted = info.split(',')
        # Check if server or client packet
        direction = ';S'
        if source == client_ip:
            direction = ';C'

        # If multiple application or whatever in the packet assume each
        # application data has the same length.
        length_orig = length / len(info_splitted)
        # Quantizes the packet size to some range. constants.centers contains the
        # edges of the bins. Project the packet size into the bin with the
        # smallest distance. Use then the bin number.
        # packet_size = ';' + str(int(length_orig))

        # packet_size = ';' + str(int(constants.centers[(np.abs(constants.centers - length_orig)).argmin()]))
        if centers is None:
            packet_size = ';{:d}'.format(int(length_orig))
        elif centers.size == 1:
            packet_size = ';'
        else:
            packet_size = ';{:d}'.format(int(np.argmin(np.abs(centers - length_orig))))

        # Due to whitespaces at begin and end try to find the substring in the
        # info_state.
        # Then, for each info state, insert a packet into the trace.
        for info_state in info_splitted:
            if 'Application Data' in info_state:
                info_state = 'Application Data'
            if 'Certificate' in info_state:
                info_state = 'Certificate'
            if 'Continuation Data' in info_state:
                info_state = 'Continuation Data'
            original_state = constants.tls_states_mc.get(info_state.strip(), '0')
            trace.append(original_state + packet_size + direction)
    trace = shape_trace(trace, trace_length)
    return trace


def _convert_pcap_to_states(trace_length: int, pcap_files: List[str], centers: np.array=None, flow: str='main_flow') -> List[List[str]]:
    """
    Converts the pcaps of a company into the state representation

    Args:
        num_packets (int): Maximum length of sequence, i.e., maximum numner of
            packets used for sequence creation.
        centers (np.array): Array giving the bin edges for binning the packet size.
        flow (str): The filter that should be applied to the trace. Supported
            are {main_flow, and co_flows}.

    Returns:
        traces (list):  list of state sequences
    """
    traces = []
    for pcap_file in pcap_files:
        pcap = os.path.join(constants.trace_path, pcap_file)
        trace = convert_trace_to_states(pcap, trace_length, centers, flow)
        if trace:
            if len(trace) < trace_length:
                print(f"Trace {pcap_file} shorter than anticipated, {len(trace)} vs {trace_length}: {trace}")
            traces.append(trace)
    return traces


def convert_pcap_to_states(company: str, trace_length: int, browser: str=None, centers: np.array=None, flow: str='main_flow', data_set: str='train') -> List[List[str]]:
    """
    Converts the pcaps of a company into the state representation

    Args:
        company (str): string containing the service
        num_packets (int): Maximum length of sequence, i.e., maximum numner of
            packets used for sequence creation.
        browser: Browser name with which packets have been gathered. Must be in
            {Mozilla, Chromium, Wget, not_wget}.
        centers (np.array): Array giving the bin edges for binning the packet size.
        included_packets (str): The filter that should be applied to the trace. Supported
            are {main_flow, and co_flows}.
        data_set (name) Name of data set that should be retrieved. Must be in
            {train, val, test}.

    Returns:
        traces (list):  list of state sequences
    """
    pcap_files = data_aggregation.acquire_pcap_files(company, browser=browser, data_set=data_set)
    return _convert_pcap_to_states(
        trace_length=trace_length,
        pcap_files=pcap_files,
        centers=centers,
        flow=flow
    )

#############################################################################################################################


def save_phmm(hmm: phmm.Hmm, company: str, browser: str, trace_length: int, centers: np.array, log_prob: float=0., path: str=None) -> None:
    """
    Saves the phmm as json file

    Args:
        hmm (phmm): phmm with the inital, transitions and emissions probabilities
        company (str):  name of company
        browser (str):  name of browser
        trace_length (int): length of the traces
        centers (np.array): centers of the packet size quantization
        log_prob (float):   threshold of the phmm
        path (str):  path to save phmm to

    Returns:
        / 
    """

    if path is None:
        path = os.path.join(constants.model_path, 'phmm', company)
    json_file = os.path.join(path, 'hmm_' + company + '_' + browser + '.json')

    model = {}
    model['duration'] = hmm.duration
    model['trans'] = {str(k): v for k, v in hmm.p_ij.items()}
    model['obs'] = {str(k): v for k, v in hmm.p_o_in_i.items()}
    model['log_prob'] = log_prob
    model['trace_length'] = trace_length
    if centers is not None:
        model['centers'] = centers.tolist()
    else:
        model['centers'] = centers

    with open(json_file, 'w', encoding='utf-8') as f:
        js.dump(model, f, ensure_ascii=False, indent=4)


def convert_to_phmm(duration: int, trans: Dict[Tuple[str, str], float], obs: Dict[Tuple[Any, str], float]) -> phmm.Hmm:

        """
        Converts the loaded distributions into a phmm class

        Args:
            duration (int): length of the phmm
            trans (dict):   dict of transition probabilities
            obs (dict): dict of observation probabilities

        Returns:
            hmm (phmm.Hmm): phmm with specified distributions
        """

        hmm = phmm.basic_phmm(duration, [])
        hmm.p_ij = trans
        hmm.p_o_in_i = obs

        return hmm


def load_phmm(company: str, browser: str) -> Tuple[phmm.Hmm, float, int, np.array]:
    """
    Loads a phmm

    Args:
        company (str):  name of company
        browser (str):  name of browser

    Returns:
        hmm (phmm.Hmm): phmm with loaded distributions
        log_prob (float):   log_prob of the phmm
        trace_length (int): length of the traces
        centers (np.array): centers of the packet size quantization
    """

    json_file = os.path.join(constants.model_path, 'phmm', company, 'hmm_' + company + '_' + browser + '.json')

    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='utf-8') as f:
        fingerprint = js.load(f)

    duration = fingerprint['duration']
    log_prob = fingerprint['log_prob']
    trans = fingerprint['trans']
    obs = fingerprint['obs']
    trace_length = fingerprint['trace_length']
    if fingerprint['centers'] is not None:
        centers = np.array(fingerprint['centers'])
    else:
        centers = fingerprint['centers']

    trans = {eval(k): v for k, v in trans.items()}
    obs = {eval(k): v for k, v in obs.items()}

    hmm = convert_to_phmm(duration, trans, obs)

    return hmm, log_prob, trace_length, centers


def load_fingerprints_phmm(browser: str) -> Tuple[Dict[str, phmm.Hmm], Dict[str, float], Dict[str, int], Dict[str, np.array]]:
    """
    Loads all phmms

    Args:
        browser (str):  name of browser

    Returns:
        hmms (dict):    dictionary with services as keys and phmms as values 
        threshold (dict): dict with companys as keys and thresholds as values
        trace_length (dict):    dict with companys as keys and trace_lengths as values
        centers (dict): dict with companys as keys and centers as values
    """
    companys = data_aggregation.get_services()
    hmms = {}
    thresholds = {}
    trace_lengths = {}
    centers = {}
    for company in companys:
        hmm, log_prob, trace_length, center = load_phmm(company, browser)
        if hmm is not None:
            hmms[company] = hmm
            thresholds[company] = log_prob
            trace_lengths[company] = trace_length
            centers[company] = center

    return hmms, thresholds, trace_lengths, centers


def save_params_phmm(path: str, company: str, browser: str, binning_method: str, num_bins: int, trace_length: int, hmm_duration: int, init_prior: str, flow: str='main_flow') -> None:

    """
    Saves the parameters of a phmm

    Args:
        path (str): path to save file to
        company (str):   name of company
        browser (str):  name of browser
        binning_method (str):   name of binning_method
        num_bins (int): number of bins
        trace_length (int): length of traces
        hmm_duration (int): length of hmm
        init_prior (str):   prior of the initialize
        flow (str): flow of the traces

    Returns:
        /
    """

    if path is None:
        path = os.path.join(constants.model_path, 'phmm', company)
    json_file = os.path.join(path, 'hmm_' + company + '_' + browser + '_params.json')

    params = {}

    params['company'] = company
    params['browser'] = browser
    params['flow'] = flow
    params['binning_method'] = binning_method
    params['num_bins'] = num_bins
    params['trace_length'] = trace_length
    params['hmm_duration'] = hmm_duration
    params['init_prior'] = init_prior

    with open(json_file, 'w', encoding='utf-8') as f:
        js.dump(params, f, ensure_ascii=False, indent=4)


def load_phmm_params(company: str, browser: str) -> Dict[str, Any]:

    """
    Loads the parameters of a company browser combination

    Args:
        company (str):   name of company
        browser (str):  name of browser

    Returns:
        params (dict):   dict containing parameters for training
    """

    json_file = os.path.join(constants.model_path, 'phmm', company, 'hmm_' + company + '_' + browser + '_params.json')

    with open(json_file, 'r', encoding='utf-8') as f:
        params = js.load(f)

    return params

#############################################################################################################################

def _convert_seq_to_int(hmm: phmm.PhmmC, sequence: List[Any]) -> List[int]:
    """
    Converts a single sequence of Anys into int representation

    Args:
        hmm (phmm.PhmmC): 
        sequence (List): List of symbols

    Returns:
        seq_n (List): List of int symbols
    """
    seq_n = []
    for sym in sequence:
        seq_n.append(hmm._sym_to_int[sym] if sym in hmm._sym_to_int else -1)
    return seq_n

def convert_seq_to_int(hmm: phmm.PhmmC, sequence: List[List[Any]]) -> List[List[int]]:
    """
    Args:
        hmm (phmm.PhmmC): 
        sequence (List): List of List of symbols

    Returns:
        seq_n (List): List of List of int symbols
    """
    seq_n = []
    seq_len = []
    for seq in sequence:
        seq_n.extend(_convert_seq_to_int(hmm, seq))
        seq_len.append(len(seq))
    return seq_n, seq_len

def convert_seq_to_int_lstm(hmm: phmm.PhmmC, sequence: List[List[Any]]) -> List[List[int]]:

    """
    Args:
        hmm (phmm.PhmmC): 
        sequence (List): List of List of symbols

    Returns:
        seq_n (List): List of List of int symbols
    """
    seq_n = []
    seq_len = []
    for seq in sequence:
        seq_n.append(_convert_seq_to_int(hmm, seq))
        seq_len.append(len(seq))
    return seq_n, seq_len

#############################################################################################################################


def save_fhmm(hmm: fhmm.Hmm, company: str, browser: str, trace_length: int, centers: np.array, log_prob: float=0., path: str=None) -> None:
    """
    Saves the phmm as json

    Args:
        hmm (phmm): phmm with the inital, transitions and emissions probabilities
        company (str):  string containing the service

    Returns:
        / 
    """

    if path is None:
        path = os.path.join(constants.model_path, 'fhmm', company)
    json_file = os.path.join(path, 'hmm_' + company + '_' + browser + '.json')

    model = {}

    model['duration'] = hmm.duration
    model['init'] = {str(k): v for k, v in hmm.init.items()}
    model['trans'] = {str(k): v for k, v in hmm.p_ij.items()}
    model['obs'] = {str(k): v for k, v in hmm.p_o_in_i.items()}
    model['log_prob'] = log_prob
    model['trace_length'] = trace_length
    if centers is not None:
        model['centers'] = centers.tolist()
    else:
        model['centers'] = centers

    with open(json_file, 'w', encoding='utf-8') as f:
        js.dump(model, f, ensure_ascii=False, indent=4)


def convert_to_fhmm(duration: int, init: Dict[int, float], trans: Dict[Tuple[str, str], float], obs: Dict[Tuple[Any, str], float]) -> fhmm.Hmm:

        hmm = fhmm.basic_hmm(duration, [])
        hmm.init = init
        hmm.p_ij = trans
        hmm.p_o_in_i = obs

        return hmm


def load_fhmm(company: str, browser: str) -> Tuple[fhmm.Hmm, float, int, np.array]:
    """
    Loads a fingerprint as the phmm module

    Args:
        company (str):  string as the service

    Returns:
        hmm (phmm): phmm with the inital, transitions and emissions probabilities
    """

    json_file = os.path.join(constants.model_path, 'fhmm', company, 'hmm_' + company + '_' + browser + '.json')

    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='utf-8') as f:
        fingerprint = js.load(f)

    duration = fingerprint['duration']
    log_prob = fingerprint['log_prob']
    init = fingerprint['init']
    trans = fingerprint['trans']
    obs = fingerprint['obs']
    trace_length = fingerprint['trace_length']
    if fingerprint['centers'] is not None:
        centers = np.array(fingerprint['centers'])
    else:
        centers = fingerprint['centers']

    init = {eval(k): v for k, v in init.items()}
    trans = {eval(k): v for k, v in trans.items()}
    obs = {eval(k): v for k, v in obs.items()}

    hmm = convert_to_fhmm(duration, init, trans, obs)

    return hmm, log_prob, trace_length, centers


def load_fingerprints_fhmm(browser: str) -> Tuple[Dict[str, phmm.Hmm], Dict[str, float], Dict[str, int], Dict[str, np.array]]:
    """
    Loads all fingerprints of all services

    Args:
        /

    Returns:
        hmms (dict):    dictionary with services as keys and phmms as values 
    """
    companys = data_aggregation.get_services()
    hmms = {}
    thresholds = {}
    trace_lengths = {}
    centers = {}
    for company in companys:
        hmm, log_prob, trace_length, center = load_fhmm(company, browser)
        if hmm is not None:
            hmms[company] = hmm
            thresholds[company] = log_prob
            trace_lengths[company] = trace_length
            centers[company] = center

    return hmms, thresholds, trace_lengths, centers


def save_params_fhmm(path: str, company: str, browser: str, binning_method: str, num_bins: int, trace_length: int, hmm_duration: int, init_prior: str, flow: str='main_flow') -> None:

    if path is None:
        path = os.path.join(constants.model_path, 'fhmm', company)
    json_file = os.path.join(path, 'hmm_' + company + '_' + browser + '_params.json')

    params = {}

    params['company'] = company
    params['browser'] = browser
    params['flow'] = flow
    params['binning_method'] = binning_method
    params['num_bins'] = num_bins
    params['trace_length'] = trace_length
    params['hmm_duration'] = hmm_duration
    params['init_prior'] = init_prior

    with open(json_file, 'w', encoding='utf-8') as f:
        js.dump(params, f, ensure_ascii=False, indent=4)


def load_fhmm_params(company: str, browser: str) -> Dict[str, Any]:

    json_file = os.path.join(constants.model_path, 'fhmm', company, 'hmm_' + company + '_' + browser + '_params.json')

    with open(json_file, 'r', encoding='utf-8') as f:
        params = js.load(f)

    return params

