import os
import sys
import numpy as np
import numpy.random as rd
import itertools as itt
import time
import ctypes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json as js
from ctypes import cdll
from typing import Any, Dict, List, Tuple, Union

sys.path.append('/mounted-data/code/')
sys.path.append('/opt/project/implementation')

from baum_welch import phmm, learning
from data_conversion import data_aggregation, data_aggregation_hmm_np


def load_clib() -> Tuple[callable, callable, callable]:
    if os.path.exists('./ctypes/phmm.so'):
        lib = cdll.LoadLibrary('./ctypes/phmm.so')
    elif os.path.exists('/opt/project/implementation/ctypes/phmm.so'):
        lib = cdll.LoadLibrary('/opt/project/implementation/ctypes/phmm.so')
    else:
        raise FileNotFoundError("phmm.so Not found!")
    # Argument types for baum_welch function that performs <num_iterations> steps
    # of the Baum-Welch algorithm. Trains on dedicated training sequences and
    # evaluates training progress on a separate validation set.
    # Returns parameters of best iteration.
    lib.baum_welch.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # Transition Matrix.
        ctypes.POINTER(ctypes.c_double),  # Observation Matrix.
        ctypes.POINTER(ctypes.c_int),     # Sequences for training.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in training.
        ctypes.c_int,                     # Number of sequences in training.
        ctypes.POINTER(ctypes.c_int),     # Sequences for validation.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in validation.
        ctypes.c_int,                     # Number of sequences in validation.
        ctypes.c_int,                     # Number of unique symbols, i.e., observation space.
        ctypes.c_int,                     # HMM duration.
        ctypes.c_int                      # Number of iterations in baum welch.
        ]
    # Defines the return type. Returns the transition and the observation matrix.
    lib.baum_welch.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
    # Perform a single step of the Baum-Welch algorithm corresponding to a
    # forward pass, a backward pass and an update of the parameters.
    # Returns the updated transistion and emission model.
    lib.baum_welch_step.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # Transition Matrix.
        ctypes.POINTER(ctypes.c_double),  # Observation Matrix.
        ctypes.POINTER(ctypes.c_int),     # Sequences for training.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in training.
        ctypes.c_int,                     # Number of sequences in training.
        ctypes.POINTER(ctypes.c_int),     # Sequences for validation.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in validation.
        ctypes.c_int,                     # Number of sequences in validation.
        ctypes.c_int,                     # Number of unique symbols, i.e., observation space.
        ctypes.c_int,                     # HMM duration.
        ]
    lib.baum_welch_step.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
    # Free the memory allocated during whatever.
    lib.free_mem.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
    lib.free_mem.restype = None
    return lib.baum_welch, lib.baum_welch_step, lib.free_mem


def load_logprob2() -> Tuple[callable, callable]:
    if os.path.exists('./ctypes/phmm.so'):
        lib = cdll.LoadLibrary('./ctypes/phmm.so')
    elif os.path.exists('/opt/project/implementation/ctypes/phmm.so'):
        lib = cdll.LoadLibrary('/opt/project/implementation/ctypes/phmm.so')
    else:
        raise FileNotFoundError("phmm.so Not found!")

    lib.calc_log_prob_seq.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # Transition Matrix.
        ctypes.POINTER(ctypes.c_double),  # Observation Matrix.
        ctypes.POINTER(ctypes.c_int),     # Sequences for evaluation.
        ctypes.POINTER(ctypes.c_int),     # Lengths of sequences in third argument.
        ctypes.c_int,                     # Number of sequences in training.
        ctypes.c_int                      # HMM duration.
    ]
    # Defines the return type. Returns the log likelihoods.
    lib.calc_log_prob_seq.restype = ctypes.POINTER(ctypes.c_double)
    # Free the memory allocated during whatever.
    lib.free_mem_single.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.free_mem_single.restype = None
    return lib.calc_log_prob_seq, lib.free_mem_single


__baum_welch, __baum_welch_step, __free = load_clib()
# __log_prob = load_logprob2()


def convert_log_prob(log_prob: ctypes.POINTER(ctypes.c_double), num_seq: int) -> Tuple[np.array, float]:
    log_prob_a = np.zeros(num_seq)
    for i in range(num_seq):
        log_prob_a[i] = log_prob[i]
    return log_prob_a, np.sum(log_prob_a) / log_prob_a.size


def time_phmm(num_train, num_val):

    hmm_durations = list(range(10, 31, 10))
    trace_lengths = list(range(10, 31, 10))
    iterations = 20

    time_phmm_l = []
    for trace_length, hmm_duration, in itt.product(trace_lengths, hmm_durations):

        traces_train = rd.randint(0, 20, (num_train, trace_length)).tolist()
        # traces_val = rd.randint(0, 20, (num_val, trace_length))
        obs_space = data_aggregation.get_observation_space(traces_train)
        hmm = phmm.basic_phmm(hmm_duration, obs_space)

        tmp = []
        for _ in range(30):
            start_time = time.time()
            for _ in range(iterations):
                hmm, _, _ = learning.baum_welch(hmm, traces_train)
            tmp.append((time.time() - start_time))

        time_phmm_l.append(tmp)

    return time_phmm_l


def time_cphmm(num_train, num_val):

    hmm_durations = list(range(10, 31, 10))
    trace_lengths = list(range(10, 31, 10))
    iterations = 20
    time_cphmm_l = []

    for trace_length, hmm_duration, in itt.product(trace_lengths, hmm_durations):

        traces_train_r = rd.randint(0, 20, (num_train, trace_length)).tolist()
        traces_val_r = rd.randint(0, 20, (num_val, trace_length)).tolist()
        obs_space = data_aggregation.get_observation_space(traces_train_r)

        num_traces_train = len(traces_train_r)
        num_traces_val = len(traces_val_r)
        num_obs = len(obs_space)

        tmp = []
        for _ in range(30):

            start_time = time.time()

            hmm_c = phmm.basic_phmm_c(hmm_duration, obs_space)

            traces_train, traces_train_lens = data_aggregation_hmm_np.convert_seq_to_int(hmm_c, traces_train_r)
            traces_val, traces_val_lens = data_aggregation_hmm_np.convert_seq_to_int(hmm_c, traces_val_r)

            traces_train_p = (ctypes.c_int * len(traces_train))(*traces_train)
            traces_train_lens_p = (ctypes.c_int * len(traces_train_lens))(*traces_train_lens)
            traces_val_p = (ctypes.c_int * len(traces_val))(*traces_val)
            traces_val_lens_p = (ctypes.c_int * len(traces_val_lens))(*traces_val_lens)

            res = __baum_welch(
                hmm_c._p_ij_p,
                hmm_c._p_o_in_i_p,
                traces_train_p,
                traces_train_lens_p,
                num_traces_train,
                traces_val_p,
                traces_val_lens_p,
                num_traces_val,
                num_obs,
                hmm_duration,
                iterations
            )
    
            hmm_c.update_p_ij(res[0])
            hmm_c.update_p_o_in_i(res[1])

            log_prob_train, log_prob_train_all = convert_log_prob(res[2], num_traces_train)
            log_prob_val, log_prob_val_all = convert_log_prob(res[3], num_traces_val)

            __free(res)

            hmm_r = phmm.Hmm(hmm_duration)
            hmm_r.p_ij = dict(zip(hmm_c._p_ij_keys, hmm_c._p_ij))
            hmm_r.p_o_in_i = dict(zip(hmm_c._p_o_in_i_keys, hmm_c._p_o_in_i))

            tmp.append((time.time() - start_time))

        time_cphmm_l.append(tmp)

    return time_cphmm_l

def time_iter_forward(num_train):

    hmm_durations = list(range(10, 31, 10))
    trace_lengths = list(range(10, 31, 10))
    iterations = 30
    time_cphmm_l = []

    for trace_length, hmm_duration, in itt.product(trace_lengths, hmm_durations):

        tmp = []
        for i in range(num_train):

            traces_train_r = rd.randint(0, 20, (1, trace_length)).tolist()
            obs_space = data_aggregation.get_observation_space(traces_train_r)
            hmm_c = phmm.basic_phmm_c(hmm_duration, obs_space)

            traces_train, _ = data_aggregation_hmm_np.convert_seq_to_int(hmm_c, traces_train_r)

            for _ in range(iterations):

                traces_train_p = (ctypes.c_int * len(traces_train))(*traces_train)

                start_time = time.time()

                log_prob = __forward(
                    hmm_c._p_ij_p,
                    hmm_c._p_o_in_i_p,
                    traces_train_p,
                    trace_length,
                    hmm_duration
                )
                tmp.append((time.time() - start_time))
        time_cphmm_l.append(tmp)
    return time_cphmm_l

def save_timings(time_phmm_l, time_cphmm_l):

    output = {'phmm': time_phmm_l, 'cphmm': time_cphmm_l}
    with open('/mounted-data/data/results/timings/timings_1.json', 'w', encoding='utf-8') as f:
        js.dump(output, f, ensure_ascii=False, indent=4)


def load_timings():

    with open('/mounted-data/data/results/timings/timings_1.json', 'r', encoding='utf-8') as f:
        timings = js.load(f)
    return timings['phmm'], timings['cphmm']


def set_size(width, fraction=1, subplot=[1, 1]):
    
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt 
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in * (subplot[0] / subplot[1]), fig_height_in)

    return fig_dim


def plot_times(time_phmm_l, time_cphmm_l):

    width = 426.79135

    positions_l = np.arange(1., 19.)

    positions_c = positions_l[::2]
    positions_r = positions_l[1::2]

    labels_c = [str(item) for item in itt.product(list(range(10, 31, 10)), list(range(10, 31, 10)))]
    labels = []
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    fig, ax = plt.subplots(1, figsize=set_size(width))
    plt.xticks(fontsize = 8)
    add_label(ax.violinplot(dataset = time_phmm_l, positions = positions_c), 'Phmm')
    add_label(ax.violinplot(dataset = time_cphmm_l, positions = positions_r), 'CPhmm')
    plt.xticks(np.arange(1.5, 19.5, 2), labels_c)
    plt.legend(*zip(*labels), loc=2)
    ax.grid()
    ax.set_ylabel('time in s')
    ax.set_xlabel('configuration (trace_length, hmm_duration)')

    plt.savefig('c_lib_timing.pdf')
    plt.close()


def time_phmm_cphmm():

    num_train = 20
    num_val = 10

    # time_cphmm_l = time_iter_forward(num_train)

    # time_phmm_l = time_phmm(num_train, num_val)
    # print('Doing CPhmm')
    # time_cphmm_l = time_cphmm(num_train, num_val)

    # save_timings([], time_cphmm_l)
    time_phmm_l, time_cphmm_l = load_timings()

    plot_times(time_cphmm_l, time_cphmm_l)

    

if __name__ == '__main__':
    time_phmm_cphmm()