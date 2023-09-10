from leven import levenshtein
import sqlalchemy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm as mcm
import matplotlib as mpl
from sklearn import cluster
from itertools import repeat
import multiprocessing as mp

from data_conversion import data_aggregation
from data_conversion import constants


def optimize_bins(nums, dataset):

    k_means = cluster.KMeans(n_clusters = nums)
    k_means.fit(dataset)

    centers = np.sort(np.rint(k_means.cluster_centers_.squeeze()))

    return centers

def convert_trace_to_packet_length_hmm(pcap_file):

    trace = []

    pd_data, convertible = data_aggregation.convert_pcap_to_dataframe(pcap_file)

    pd_data = pd_data['frame.len']

    pd_data = pd_data.iloc[:constants.num_packets_hidden]

    return pd_data.tolist()


def acquire_packet_length_hmm(company):

    engine = sqlalchemy.create_engine(constants.SQLALCHEMY_DATABASE_URI)
    sql_query_url_ids = """SELECT id FROM urls WHERE tags = %s"""
    url_ids = engine.execute(sql_query_url_ids, (company)).fetchall()

    url_ids = url_ids[0:25]

    traces = []

    for url_id in url_ids:

        sql_query_pcaps = """SELECT filename FROM traces_metadata WHERE url = %s"""
        pcap_files = engine.execute(sql_query_pcaps, (url_id[0])).fetchall()

        
        for pcap_file in pcap_files:

            pcap = os.path.join(constants.pcap_path, pcap_file[0] + '.pcapng')
            trace = convert_trace_to_packet_length_hmm(pcap)

            if trace:

                traces.append(trace)


    result = []
    [result.extend(el) for el in traces]

    return result


def acquire_all_traces_hmm():

    traces_all = []

    for company in constants.applications:

        data = data_aggregation.acquire_traces_hmm(company)

        traces_all.append(data)

    return traces_all

def acquire_all_pcaps():

    companys = data_aggregation.get_services()

    pcap_files_l = []

    for company in companys:

        pcap_files = data_aggregation.acquire_pcap_files(company)

        pcap_files_l.append(pcap_files)

    return pcap_files_l



def acquire_all_traces_mc():

    companys = data_aggregation.get_services()
    traces_all = []

    for company in companys:

        traces = data_aggregation.convert_pcap_to_states_mc(company)

        traces_all.append(traces)

    return traces_all


def convert_traces_to_strings_hmm(traces):

    traces_str = []

    for trace in traces:

        traces_str.append(u''.join([str(i) for i in trace]))

    return traces_str


def convert_all_traces_to_strings_hmm(traces_all):

    traces_str_all = []

    for traces in traces_all:

        traces_str_all.append(convert_traces_to_strings_hmm(traces))

    return traces_str_all


def concat_strings_mc(traces):

    traces_str = []

    for trace in traces:

        empty_str = ''

        for string in trace:

            empty_str += string

        traces_str.append(empty_str)

    return traces_str


def concat_all_strings_mc(traces_all):

    traces_str_all = []

    for traces in traces_all:

        traces_str_all.append(concat_strings_mc(traces))

    return traces_str_all

def calculate_levenshtein(traces_str_1, traces_str_2):

    avg_leven = 0

    for i in range(len(traces_str_1)):

        for j in range(len(traces_str_2)):

            avg_leven += 2 * levenshtein(traces_str_1[i], traces_str_2[j])

    avg_leven /= (len(traces_str_1) * len(traces_str_2))

    return avg_leven


def calculate_all_levenshtein(traces_str_all):

    leven_mat = np.zeros((len(traces_str_all), len(traces_str_all)))

    for i in range(len(traces_str_all)):

        j = 0

        while j <= i:

            leven_mat[i, j] = calculate_levenshtein(traces_str_all[i], traces_str_all[j])
            leven_mat[j, i] = leven_mat[i, j]

            j += 1

    return leven_mat


def get_fig(ncols, aspect_ratio=0.618):

    """
    Create figure and axes objects of correct size.

    Args:
        ncols (float): Percentage of one column in paper.
        aspect_ratio (float): Ratio of width to height. Default is golden ratio.

    Returns:
        fig (plt.Figure)
        ax (plt.Axes)
    """
    COLW = 3.45
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figwidth(ncols * COLW)
    fig.set_figheight(ncols * COLW * aspect_ratio)


    return fig, ax


def draw_levenshtein_matrix(leven_mat):

    """
    Plots the confusion matrix

    Args:
        confusion_matrix (np.array): 2D array with shape (num_applications, num_applications)
        applications (list): list of applications

    Returns:
        /
    """

    leven_mat_file = os.path.join(constants.result_path, 'levenshtein_distance_mat.pdf')

    # leven_mat = np.divide(leven_mat, np.sum(leven_mat, axis = 1).reshape(-1, 1))

    plt.set_cmap('Set2')
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'font.family': 'serif'})

    fig, ax = get_fig(1)
    cax = ax.imshow(leven_mat, cmap=mcm.get_cmap("Greens"))
    ax.set_yticks(np.arange(-0.5, leven_mat.shape[0] - 1, 1))
    ax.set_xticks(np.arange(-0.5, leven_mat.shape[1] - 1, 1))

    ax.set_xticklabels(np.arange(leven_mat.shape[1])) 
    ax.set_yticklabels(np.arange(leven_mat.shape[0]))

    dx = 10/72. 
    dy = 0/72.  

    offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans) 

    for label in ax.xaxis.get_majorticklabels(): 

        label.set_transform(label.get_transform() + offset) 

    dx = 0 
    dy = -10/72. 

    offset_y = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans) 

    for label in ax.yaxis.get_majorticklabels():

        label.set_transform(label.get_transform() + offset_y)

    ax.set_xticklabels(constants.applications_short)
    ax.set_yticklabels(constants.applications_short)
    # ax.set_xlabel('prediction')
    # ax.set_ylabel('ground truth')
    plt.grid(color='black')
    m = mcm.ScalarMappable(cmap=mcm.get_cmap("Greens"))
    m.set_array(leven_mat)
    m.set_clim(leven_mat.min(), leven_mat.max())
    plt.colorbar(m)

    # fig.tight_layout()

    plt.savefig(leven_mat_file, bbox_inches = 'tight')


def calculate_avg_distance(leven_mat):

    distance = np.zeros(leven_mat.shape[0])

    for i in range(len(distance)):

        first = np.amin(leven_mat[i, :])

        second = np.sort(leven_mat[i, :])[1]

        distance[i] = np.sqrt(np.square(np.add(first, -second)))

    avg_distance = (np.sum(np.logical_and(distance, np.ones(len(distance)))) - 1) * np.sum(1 / (1 + np.exp(-distance)))

    return avg_distance


def run_levenshtein_mc(traces_str_all):

    leven_mat = np.rint(calculate_all_levenshtein(traces_str_all))

    # avg_distance = calculate_avg_distance(leven_mat)

    draw_levenshtein_matrix(leven_mat)

    # return avg_distance


def run_levenshtein_hmm():

    traces_all = acquire_all_traces_hmm()
    traces_str_all = convert_all_traces_to_strings_hmm(traces_all)

    # leven_mat = np.rint(calculate_all_levenshtein(traces_str_all))

    # print(leven_mat)
    # print('----------------------------------------------------')

    # avg_distance = calculate_avg_distance(leven_mat)

    # print(avg_distance)

    # draw_levenshtein_matrix(leven_mat)

    return avg_distance

def set_constants(nums):

    constants.nums = nums
    constants.centers = np.unique(np.rint(np.geomspace(1, 2000, num = nums)))
    constants.bins = len(constants.centers) 

    # constants.tls_states_hmm_c = dict()
    # constants.tls_states_hmm_c['Client Hello'] = 0 * constants.bins
    # constants.tls_states_hmm_c['Client Key Exchange'] = 1 * constants.bins
    # constants.tls_states_hmm_c['Change Cipher Spec'] = 2 * constants.bins
    # constants.tls_states_hmm_c['Encrypted Handshake Message'] = 3 * constants.bins
    # constants.tls_states_hmm_c['Application Data'] = 4 * constants.bins
    # constants.tls_states_hmm_c['Continuation Data'] = 4 * constants.bins

    # constants.tls_states_hmm_s = dict()
    # constants.tls_states_hmm_s['Server Hello'] = 5 * constants.bins
    # constants.tls_states_hmm_s['Server Key Exchange'] = 6 * constants.bins
    # constants.tls_states_hmm_s['Change Cipher Spec'] = 7 * constants.bins
    # constants.tls_states_hmm_s['Encrypted Handshake Message'] = 8 * constants.bins
    # constants.tls_states_hmm_s['Application Data'] = 9 * constants.bins
    # constants.tls_states_hmm_s['Continuation Data'] = 9 * constants.bins
    # constants.tls_states_hmm_s['Certificate'] = 10 * constants.bins
    # constants.tls_states_hmm_s['Certificate Status'] = 11 * constants.bins
    # constants.tls_states_hmm_s['New Session Ticket'] = 12 * constants.bins
    # constants.tls_states_hmm_s['Server Hello Done'] = 13 * constants.bins

    # constants.default_state = (constants.client_states + constants.server_states) * constants.bins

def run_search_packets(packets, pcap_files_all):

    constants.number_packets = packets

    runs = range(70, 101)

    distances = np.zeros(len(runs))

    for i in runs:

        set_constants(i)

        traces_all = acquire_all_traces_mc(pcap_files_all)
        traces_str_all = concat_all_strings_mc(traces_all)

        distances[i - 70] = run_levenshtein_mc(traces_str_all)

    index = np.argmax(distances)

    return index + 70

def run_search():

    pcap_files_all = acquire_all_pcaps()

    num_bins_l = []
    pd_file = os.path.join(constants.result_path, 'opt_bins.csv')

    packets_l = range(2, 31)

    for packets in packets_l:

        index = run_search_packets(packets, pcap_files_all)

        num_bins_l.append(index)

    pd_data = pd.DataFrame(columns = ['packets', 'coflows', 'bins'])
    pd_data['packets'] = packets_l
    pd_data['coflows'] = [constants.co_flows] * len(packets_l)
    pd_data['bins'] = num_bins_l

    pd_data.to_csv(pd_file, index = False, mode = 'a', header = False)


# run_search()

data_aggregation.set_constants()

traces_all = acquire_all_traces_mc()

traces_str_all = concat_all_strings_mc(traces_all)

run_levenshtein_mc(traces_str_all)
