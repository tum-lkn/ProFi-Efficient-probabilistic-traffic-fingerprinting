"""
Holds the functions to convert a pcap TLS trace into the corresponding state representation

i.e. if the Server sends a packet with 'application data', this is then saved as state '23'
"""

# Imports:
#   External Libraries

import os
import sys
import subprocess
import sqlalchemy
import pandas as pd
import numpy as np
import numpy.random as rd
import json as js
import logging
import itertools as itt
from typing import Any, Dict, List, Tuple

#   Files from this Project

sys.path.append('/mounted-data/code/tls-fingerprint/implementation/data_conversion')

import implementation.data_conversion.constants as constants

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("DataAgg")

#############################################################################################################################


def create_folders(companys: List[str], model: str) -> None:
    """
    Creates all the necessary folders for training and classification

    Args:
        company (list): list of strings containing all services

    Returns:
        /
    """
    os.makedirs(constants.trace_path, exist_ok=True)
    os.makedirs(os.path.join(constants.model_path, model), exist_ok=True)
    os.makedirs(constants.classification_path, exist_ok=True)
    os.makedirs(constants.result_path, exist_ok=True)
    os.makedirs(constants.tf_log_dir, exist_ok=True)

    for company in companys:
        os.makedirs(os.path.join(constants.model_path, model, company), exist_ok=True)


#############################################################################################################################


def equal_frequency_binning(x: np.array, num_bins: int) -> np.array:
    """
    Create bins for an equal frequency histogram. Each bin in the histogram
    has the same number of observations

    Args:
        x: The data values that should be binned
        num_bins: The number of bins that should be used.

    Returns:
        edges: The edges of the bins.
    """
    edges = np.interp(
        np.linspace(0, x.size, num_bins + 1),
        np.arange(x.size),
        np.sort(x)
    )
    return edges


def log_binning(max_val: int, num_bins: int) -> np.array:
    """
    Perform logarithmic binning using an upper value.

    Args:
        max_val:
        num_bins:

    Returns:
        edges: The edges of the bins.
    """
    # edges = np.unique(np.rint(np.geomspace(1, max_val, num=num_bins)))
    edges = equal_width_binning(np.log(max_val), num_bins)
    edges = np.exp(edges)
    return edges


def equal_width_binning(max_val: int, num_bins: int) -> np.array:
    """
    Creates bins of equal width between 0 and max_val

    Args:
        max_val:
        num_bins:

    Returns:

    """
    return np.linspace(0, max_val, num_bins)


def _get_packet_sizes(pcap_files: List[str], flow: str='main_flow') -> np.array:
    sizes = []
    for pcap_file in pcap_files:
        pcap_file = os.path.join(constants.trace_path, pcap_file + '.pcapng')
        pd_data, convertible = convert_pcap_to_dataframe(pcap_file)

        if not convertible:
            continue
        if flow == constants.FLOW_MAIN:
            pd_data = extract_main_flow(pd_data)
        sizes = np.concatenate([sizes, pd_data['frame.len'].values])
    return np.array(sizes, dtype=np.float32)


def get_packet_sizes(company: str, browser: str=None, data_set='train', flow: str='main_flow') -> np.array:
    """
    Get the packet sizes of traces from a company or specific browser.
    Args:
        compay:
        browser:

    Returns:

    """
    pcap_files = acquire_pcap_files(company, browser=browser, data_set=data_set)
    return _get_packet_sizes(pcap_files, flow)


def get_binning_centers(num_bins: int=30, binning: str='frequency', pcaps: List[str]=None) -> np.array:
    """
    Calculates the binning centers

    Args:
        num_bins (int): number of bins to use
        binning (str):  method of binning
        pcaps (list): list of converted sequences 

    Returns:
        edges (np.array):   edges of the bins
    """
    if binning == constants.BINNING_NONE:
        return None
    elif binning == constants.BINNING_SINGLE:
        return np.array([0.])
    elif binning == constants.BINNING_FREQ:
        assert pcaps is not None
        sizes = _get_packet_sizes(pcaps)
        edges = equal_frequency_binning(sizes, num_bins)
    elif binning == constants.BINNING_GEOM:
        edges = log_binning(1500, num_bins)
    elif binning == constants.BINNING_EQ_WIDTH:
        edges = equal_width_binning(1500, num_bins)
    else:
        raise KeyError
    return edges

#############################################################################################################################


def get_services() -> List[str]:
    """
    Returns all names of services

    Args:
        /

    Retruns:
        companys (list):    list with all names of services
    """
    engine = sqlalchemy.create_engine(constants.SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        sql_query = """SELECT DISTINCT tag FROM tags"""
        companys = connection.execute(sql_query).fetchall()
        companys = [company[0] for company in companys]
        companys.sort()
    companys = check_valid_company(companys)
    return companys


def check_valid_company(companys: List[str]) -> List[str]:

    """
    Checks if models can be trained for the services

    Args:
        companys (list): list of companys

    Returns:
         company_new (list):    companys with valid training data  
    """

    company_new = []
    for company in companys:
        pcap_files = acquire_pcap_files(company)
        if pcap_files:
            company_new.append(company)
    return company_new


def _acquire_pcap_files(url_ids: [List[List[int]]], browser: str=None) -> List[str]:
    pcap_files_l = []
    engine = sqlalchemy.create_engine(constants.SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        for url_id in url_ids:
            if browser is None:
                q = 'SELECT filename FROM traces_metadata WHERE url = {:d}'.format(url_id[0])
            elif browser == constants.BROWSER_NOT_WGET:
                q = 'SELECT filename FROM traces_metadata WHERE url = {:d} AND browser NOT LIKE "%%{:s}%%"'.format(
                    url_id[0],
                    "Wget"
                )
            else:
                q = 'SELECT filename FROM traces_metadata WHERE url = {:d} AND browser LIKE "%%{:s}%%"'.format(
                    url_id[0],
                    browser
                )
            pcap_files = connection.execute(q).fetchall()
            pcap_files = [pcap_file[0] for pcap_file in pcap_files]
            pcap_files_l.extend(pcap_files)
    return pcap_files_l


def acquire_pcap_files(company: str, browser: str=None, data_set: str='train') -> List[str]:
    """
    Collects the names of traces for a combination of company and browser.

    Args:
        company (string): name of application
        browser (string): Name of a browser for which traces should be limited.
            See file constants.py for details.
        data_set (string): Which type of dataset to use. Must be in {train, test, val}.

    Returns:
        traces (list): list of traces in state representation
    """
    engine = sqlalchemy.create_engine(constants.SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        sql_query_url_ids = """SELECT id FROM urls WHERE tags = %s"""
        url_ids = connection.execute(sql_query_url_ids, (company)).fetchall()
        # np.random.shuffle(url_ids)
        if data_set == 'train':
            url_ids = url_ids[:20]
        elif data_set == 'val':
            url_ids = url_ids[20:25]
        elif data_set == 'test':
            url_ids = url_ids[25:]
        else:
            raise ValueError("Unknown argument {} for data_set".format(data_set))
    return _acquire_pcap_files(url_ids, browser)


def get_traces_test(browser: str=None) -> List[str]:
    """
    Returns all the filenames of test traces

    Args:
        browser (str):  browser to get test traces of

    Returns:
        pcap_files_all (list):  list with all test filenames
    """
    pcap_files_all = []
    engine = sqlalchemy.create_engine(constants.SQLALCHEMY_DATABASE_URI)
    companys = get_services()

    with engine.connect() as connection:
        for company in companys:
           pcap_files = acquire_pcap_files(company=company, browser=browser, data_set='test')
           pcap_files_all.extend(pcap_files)
    return pcap_files_all


def convert_pcap_to_dataframe(pcap_file: str) -> Tuple[pd.DataFrame, int]:
    """
    Converts the pcap trace into a csv trace

    Args:
        pcap_file (string): path to the pcap file

    Returns:
        pd_data (pd.DataFrame): pandas data frame with TLS traffic only
    """
    convertible = 1
    if not os.path.exists(pcap_file):
        logger.warning("PCAP file >{}< does not exist.".format(pcap_file))
        convertible = 0
        pd_data = pd.DataFrame()
        return pd_data, convertible

    csv_file = pcap_file.replace('pcapng', 'csv')
    if os.path.exists(csv_file):
        pd_data = pd.read_csv(csv_file)
        return pd_data, convertible

    command = ('tshark -r {} -T fields '
               '-e frame.number '
               '-e frame.time_epoch '
               '-e ip.src '
               '-e ip.dst '
               '-e _ws.col.Protocol '
               '-e frame.len '
               '-e tcp.srcport '
               '-e tcp.dstport '
               '-e _ws.col.Info '
               '-E header=y -E separator=, -E quote=d > {}').format(
        pcap_file,
        csv_file
    )
    subprocess.check_call(command, shell=True)

    pd_data = pd.read_csv(csv_file, on_bad_lines='skip')
    pd_data = pd_data[(pd_data['_ws.col.Protocol'] == 'TLSv1.2') | (pd_data['_ws.col.Protocol'] == 'TLSv1.3') | (pd_data['_ws.col.Protocol'] == 'TLSv1')]
    pd_data.to_csv(csv_file, index = False)

    return pd_data, convertible


def _extract_flow(pd_data: pd.DataFrame, src_ip: str, dst_ip: str,
                  src_port: int, dst_port: int) -> pd.DataFrame:

    """
    Extracts the flow based on the 4 tuple (src_ip, dst_ip, src_port, dst_port)

    Args:
        pd_data (pd.DataFrame): pandas dataframe containing all TLS traffic
        src_ip (str): source ip address
        dst_ip (str):   destination ip address
        src_port (int): source port
        dst_port (int): destination port

    Returns:
        pd_data (pd.DataFrame) pandas dataframe containing only the flow
    """
    
    pd_data = pd_data.set_index(['ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport'])
    pd_data = pd_data.sort_index()
    client_server = pd_data.loc[pd.IndexSlice[src_ip, dst_ip, src_port, dst_port], :]
    server_client = pd_data.loc[pd.IndexSlice[dst_ip, src_ip, dst_port, src_port], :]
    pd_data = pd.concat([client_server, server_client])
    pd_data = pd_data.reset_index()
    pd_data = pd_data.sort_values(by='frame.number')
    pd_data = pd_data.reset_index(drop=True)

    return pd_data


def extract_main_flow(pd_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the main-flow of a trace

    Args:
        pd_data (pd.DataFrame): pandas data frame TLS traffic only

    Returns:
        pd_data (pd.DataFrame): pandas data frame TLS traffic only
    """
    client_index = pd_data[pd_data['_ws.col.Info'].str.contains('Client Hello')].index.values
    if len(client_index) < 2:
        return pd_data
    main_flow = pd.DataFrame([[]])
    i = 0
    try:
        main_flow = _extract_flow(
            pd_data=pd_data,
            src_ip=pd_data.loc[client_index[i], 'ip.src'],
            dst_ip=pd_data.loc[client_index[i], 'ip.dst'],
            src_port=int(pd_data.loc[client_index[i], 'tcp.srcport']),
            dst_port=int(pd_data.loc[client_index[i], 'tcp.dstport'])
        )
    except Exception as e:
        print("index ", i, " indices ", client_index)
        print(pd_data)
        raise e

    return main_flow


def extract_co_flow(pd_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the co-flow of a trace

    Args:
        pd_data (pd.DataFrame): pandas data frame TLS traffic only

    Returns:
        pd_data (pd.DataFrame): pandas data frame TLS traffic only
    """
    client_index = pd_data[pd_data['_ws.col.Info'].dropna().str.contains('Client Hello')].index.values
    assert client_index.size > 1, "Trace contains only one flow, cannot extract a co-flow."

    co_flow = None
    client_index = np.delete(client_index, 0)
    rd.shuffle(client_index)

    for index in client_index:
        try:
            co_flow = _extract_flow(
                pd_data=pd_data,
                src_ip=pd_data.loc[index, 'ip.src'],
                dst_ip=pd_data.loc[index, 'ip.dst'],
                src_port=int(pd_data.loc[index, 'tcp.srcport']),
                dst_port=int(pd_data.loc[index, 'tcp.dstport'])
            )
            return co_flow
        except Exception as e:
            print(e)
    assert co_flow is not None, "Could not retrieve a co-flow"
    return co_flow


def get_observation_space(traces: List[List[Any]]) -> List[Any]:
    """
    Extracts all different symbols from a list of traces.

    Args:
        traces (list): list of state sequences

    Returns:
        observation_space (list): list with all different symbols in the traces
    """
    observation_space = []
    for trace in traces:
        observation_space.extend(trace)
        observation_space = list(set(observation_space))
    return observation_space


def simulate_packet_loss(trace: List[Any]) -> List[Any]:
    """
    Simulates packet loss by randomly dropping various packets of list

    Args:
        trace (list): list with Server Messages and TLS traffic only

    Returns:
        trace (list): original list with randomly dropped packets
    """
    package_count = len(trace)
    packet_loss = rd.randint(0, 2)

    if packet_loss:
        num_lost_packets = rd.randint(0, package_count/2)
        packets = np.unique(rd.randint(0, package_count, num_lost_packets))
        trace = [element for index, element in enumerate(trace) if index not in packets]

    return trace


#############################################################################################################################


def save_predict(pcap_file: str, log_prob: Dict[str, float], flow: str='main_flow') -> None:

    """
    Computes the output of the prediction

    Args:
        pcap_file (string): path to the pcap file
        log_prob (dict): dict of log_probs
        flow (str): flow of the trace

    Returns:
        /
    """
    application_pre = max(log_prob, key = log_prob.get)
    result = log_prob[application_pre]
    application_act = 'unknown'

    if flow == 'main_flow':
        engine = sqlalchemy.create_engine(constants.SQLALCHEMY_DATABASE_URI)
        with engine.connect() as connection:
            sql_query_url = """SELECT url FROM traces_metadata WHERE filename = %s"""
            sql_query_app = """SELECT tags FROM urls WHERE id = %s"""
            url_id = connection.execute(sql_query_url, (os.path.basename(pcap_file))).fetchall()
            application_act = connection.execute(sql_query_app, (url_id[0])).fetchall()[0][0]

    output = dict()
    output['filename'] = pcap_file
    output['predicted application'] = application_pre
    output['actual application'] = application_act
    output['result'] = result
    output['likelihoods'] = log_prob

    output_file = os.path.join(constants.classification_path, pcap_file + '_clas.json')
    with open(output_file, 'w', encoding = 'utf-8') as f:
        js.dump(output, f, ensure_ascii = False, indent = 4)


def save_predict_binary(pcap_file: str, class_affs: Dict[str, int], log_probs: Dict[str, float], thresholds: Dict[str, float], flow: str='main_flow') -> None:

    """
    Saves the binary prediction of a trace

    Args:
        pcap_file (string): path to the pcap file
        class_affs (dict):  dict of class identifiers
        log_probs (dict): dict of log_probs
        thresholds (dict):  dict of thresholds
        flow (str): flow of the trace

    Returns:
        /
    """

    if np.sum(list(class_affs.values())) == 0:
        application_pre = 'unknown'
    elif np.sum(list(class_affs.values())) == 1:
        application_pre = max(class_affs, key = class_affs.get)
    else:
        norm_log_probs = np.divide(np.array(list(log_probs.values())), np.array(list(thresholds.values())))
        norm_log_probs = dict(zip(log_probs.keys(), norm_log_probs))
        application_pre = min(norm_log_probs, key = norm_log_probs.get)
    application_act = 'unknown'
    if flow == 'main_flow':
        engine = sqlalchemy.create_engine(constants.SQLALCHEMY_DATABASE_URI)
        with engine.connect() as connection:

            sql_query_url = """SELECT url FROM traces_metadata WHERE filename = %s"""
            sql_query_app = """SELECT tags FROM urls WHERE id = %s"""
            url_id = connection.execute(sql_query_url, (os.path.basename(pcap_file))).fetchall()
            application_act = connection.execute(sql_query_app, (url_id[0])).fetchall()[0][0]

    if application_pre == 'unknown':
        result = 0
    else:
        result = log_probs[application_pre]

    output = dict()
    output['filename'] = pcap_file
    output['predicted application'] = application_pre
    output['actual application'] = application_act
    output['result'] = result
    output['likelihoods'] = log_probs
    output['thresholds'] = thresholds
    output['classes'] = class_affs


    output_file = os.path.join(constants.classification_path, pcap_file + '_clas.json')
    with open(output_file, 'w', encoding = 'utf-8') as f:
        js.dump(output, f, ensure_ascii = False, indent = 4)


def load_prediction(json_file: str) -> Dict[Any, Any]:
    """
    Loads a prediction

    Args:
        json_file (string): path to json file

    Returns:
        js_data (dict): json data in dict format
    """
    json_file = os.path.join(constants.classification_path, json_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        js_data = js.load(f)

    return js_data


def save_quality_measurements(output: Dict[str, Any], json_file: str) -> None:
    """
    Writes the quality measurements of the classification to file

    Args:
        output (dict): output in dict format
        json_file (string): path to json file

    Returns:
        /
    """
    with open(json_file, 'w', encoding='utf-8') as f:
        js.dump(output, f, ensure_ascii=False, indent=4)

#############################################################################################################################


def export_dataset(export_path: str) -> None:
    """
    Reads pcap file names from the database and stores them in a json file
    identified by `export_path`.

    Args:
        export_path:

    Returns:

    """
    companies = ['google', 'amazon', 'facebook', 'wikipedia', 'youtube', 'google_drive', 'google_maps']
    browsers = ['Mozilla', 'Chromium', 'not_wget']
    sets = ['train', 'val', 'test']
    data = {}
    for company, browser, dataset in itt.product(companies, browsers, sets):
        if company not in data:
            data[company] = {}
        if browser not in data[company]:
            data[company][browser] = {}
        data[company][browser][dataset] = acquire_pcap_files(company, browser, dataset)
    with open(export_path, 'w') as fh:
        js.dump(data, fh, indent=1)