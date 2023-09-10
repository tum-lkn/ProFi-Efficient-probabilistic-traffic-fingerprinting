import h5py
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import os
from implementation.data_conversion import constants


def shape_trace_mc(trace: List[str], length: int) -> List[str]:
    """
    Cuts the trace to a specified length

    Args:
        trace (list): list with Server Messages and TLS traffic only

    Returns:
        trace (list): original list with randomly dropped packets
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
            centers <-- None to disregard the packet size info completly.
                This is the default behavior in case None is passed.
            centers <-- np.array([0]) uses the raw packet size.
            centers <-- np.array([..]) use custom centers.

    Returns:
        trace (list): list containing the state sequences
    """

    trace = []
    pd_data, convertible = convert_pcap_to_dataframe(pcap_file)

    if not convertible:
        return trace
    if flow == constants.FLOW_MAIN:
        pd_data = extract_main_flow(pd_data)
    elif flow == constants.FLOW_CO:
        pd_data = extract_co_flow(pd_data)
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
            packet_size = ';{:d}'.format(
                int(np.argmin(np.abs(centers - length_orig)))
            )

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
        pcap = os.path.join(constants.trace_path, pcap_file + '.pcapng')
        trace = convert_trace_to_states_mc(pcap, trace_length, centers, flow)
        if trace:
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


def save_mc(pd_ENPD: pd.DataFrame, pd_Trans: pd.DataFrame, pd_EXPD: pd.DataFrame, company: str) -> None:
    """
    Saves the markov as a hdf5 file, creates path to the file if it does not exist

    Args:
        pd_ENPD: pandas Dataframe with enter probability
        pd_Trans: pandas Dataframe with transition probability
        pd_EXPD: pandas Dataframe with exit probability
        company (string): name of company

    Returns:
        /
    """

    markov_chain_dict = {}

    markov_chain_dict['enpd'] = pd_ENPD
    markov_chain_dict['trans'] = pd_Trans
    markov_chain_dict['expd'] = pd_EXPD

    hdf5_file = os.path.join(constants.model_path, company, 'mc_' + company + '.h5')

    for key in markov_chain_dict:

        if key == 'enpd':

            markov_chain_dict[key].to_hdf(hdf5_file, key=key, mode='w')

        else:

            markov_chain_dict[key].to_hdf(hdf5_file, key=key, mode='a')


def load_mc(company: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads a markov chain

    Args:
        company (str):  string containing the service

    Returns:
        markov_chain (list):    list containing the enter, transition and exit probabilities
    """

    hdf5_file = os.path.join(constants.model_path, company, 'mc_' + company + '.h5')

    if not os.path.exists(hdf5_file):

        return None

    pd_ENPD = pd.read_hdf(hdf5_file, key='enpd')
    pd_EXPD = pd.read_hdf(hdf5_file, key='expd')
    pd_Trans = pd.read_hdf(hdf5_file, key='trans')

    markov_chain = [pd_ENPD, pd_EXPD, pd_Trans]

    return markov_chain


def load_fingerprints_mc() -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Loads all fingerprints of the markov chains

    Args:
        /

    Retruns:
        mcs (dict): dict with services as keys and markov chain as values
    """

    companys = get_services()
    mcs = {}
    for company in companys:
        mc = load_mc(company)
        if mc is not None:
            mcs[company] = mc

    return mcs


def save_bigram_matrix(bigram_matrix: np.array, cluster_center: np.array) -> None:
    """
    Saves the bigram matrix and cluster centers of the model as a hdf5 file, creates path to the file if it does not exist

    Args:
        bigram_matrix (np.array): 2D array with shape (num_applications, num_clusters)
        cluster_center (np.array): 2D array with shape (num_applications, 2)
    Returns:
        /
    """

    hdf5_file = os.path.join(constants.model_path, 'mc', 'bigram.h5')
    with h5py.File(hdf5_file, 'w') as hf:
        hf.create_dataset('bigram_matrix', data = bigram_matrix)
    with h5py.File(hdf5_file, 'a') as hf:
        hf.create_dataset('centers', data = cluster_center)


def load_bigram_matrix() -> Tuple[np.array, np.array]:
    """
    Loads the bigram matrix and the cluster centers

    Args:
        hdf5_file (string): path to the hdf5 file
    Returns:
        bigram_matrix (np.array): 2D array with shape (num_applications, num_clusters)
        cluster_center (np.array): 2D array with shape (num_applications, 2)
    """

    hdf5_file = os.path.join(constants.model_path, 'mc', 'bigram.h5')
    with h5py.File(hdf5_file, 'r') as hf:
        bigram_matrix = hf['bigram_matrix'][:]
        cluster_centers = hf['centers'][:]

    return bigram_matrix, cluster_centers
