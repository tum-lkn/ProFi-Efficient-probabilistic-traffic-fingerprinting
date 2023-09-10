import sys
import numpy as np
import numpy.random as rd
import time
from typing import List, Dict, Tuple, Any
import ctypes
from ctypes import cdll

sys.path.append('/mounted-data/code/')

from baum_welch import phmm, learning
from data_conversion import data_aggregation, data_aggregation_hmm_np

# load library
lib = cdll.LoadLibrary('./ctypes/phmm.so')

def convert_transitions_a_c(transitions_a_c, hmm_duration):

    """
    Converts the result of the c implementation into np.array
    Args:
        transitions_a_c (Pointer): pointer to c array
        hmm_duration (int): hmm length, used for space alloc
    Returns:
        transitions_n (np.array): transitions calculated by c lib
    """

    transitions_n = np.zeros(9 * hmm_duration + 3, dtype=np.float64)
    for i in range(len(transitions_n)):
        transitions_n[i] = transitions_a_c[i]
    return transitions_n

def convert_emissions_a_c(emissions_a_c, num_obs, hmm_duration):

    """
    Converts the result of the c implementation into np.array
    Args:
        emissions_a_c (Pointer): pointer to c array
        hmm_duration (int): hmm length, used for space alloc
    Returns:
        emissions_n (np.array): emissions calculated by c lib
    """

    emissions_n = np.zeros((2*hmm_duration + 1)*num_obs, dtype=np.float64)
    for i in range(len(emissions_n)):
        emissions_n[i] = emissions_a_c[i]
    return emissions_n

###############################################################################################

# define the lengths of training and validation sequences and the observation in them
traces_train = [[1, 5, 3, 6], [1, 1, 2], [2, 3, 3, 4], [1, 6, 4, 4]]
traces_val = [[0, 4, 2, 1]]
obs_space = data_aggregation.get_observation_space(traces_train)

# define the length of a hmm and create it
hmm_duration = 3
hmm = phmm.basic_phmm(hmm_duration, obs_space)
hmm_c = phmm.basic_phmm_c(hmm_duration, obs_space)

new_hmm, _, _ = learning.baum_welch(hmm, traces_train)

###############################################################################################

#  set number of sequences for training and validation set and number of iterations
num_traces_train = len(traces_train)
num_traces_val = len(traces_val)
num_obs = len(obs_space)
trans_offsets = [0, 3 * hmm_duration - 1, 6 * hmm_duration + 1]
obs_offsets = [0, 0, hmm_duration + 1]

# create the em_lookup and convert the sequences according to the lookup
traces_train, traces_train_lens = data_aggregation_hmm_np.convert_seq_to_int(hmm_c, traces_train)
traces_val, traces_val_lens = data_aggregation_hmm_np.convert_seq_to_int(hmm_c, traces_val)

# Convert all input parameters for the c lib into c types
traces_train = (ctypes.c_int * len(traces_train))(*traces_train)
traces_train_lens = (ctypes.c_int * len(traces_train_lens))(*traces_train_lens)

traces_val = (ctypes.c_int * len(traces_val))(*traces_val)
traces_val_lens = (ctypes.c_int * len(traces_val_lens))(*traces_val_lens)

trans_offsets = (ctypes.c_int * len(trans_offsets))(*trans_offsets)
obs_offsets = (ctypes.c_int * len(obs_offsets))(*obs_offsets)

# set the argtypes and restypes of the c functions
lib.baum_welch_step.argtypes = [ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_int),
                            ctypes.POINTER(ctypes.c_int),
                            ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int),
                            ctypes.POINTER(ctypes.c_int),
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int]
lib.baum_welch_step.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
lib.free_mem.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.free_mem.restype = None

# exec the c function get the transition and emission pointer of the return value
res = lib.baum_welch_step(hmm_c._p_ij_p, hmm_c._p_o_in_i_p, traces_train, traces_train_lens, num_traces_train, traces_val, traces_val_lens, num_traces_val, num_obs, hmm_duration)
hmm_c.update_p_ij(res[0])
hmm_c.update_p_o_in_i(res[1])

# Release the allocated memory of the c lib
