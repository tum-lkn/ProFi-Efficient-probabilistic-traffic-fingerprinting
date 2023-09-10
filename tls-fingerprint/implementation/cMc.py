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


class markov_chain_t(ctypes.Structure):
    _fields_ = [
        ('num_symbols', ctypes.c_int),
        ('offsets', ctypes.POINTER(ctypes.c_int)),
        ('tails', ctypes.POINTER(ctypes.c_uint64)),
        ('heads', ctypes.POINTER(ctypes.c_uint64)),
        ('log_probs', ctypes.POINTER(ctypes.c_double))
    ]


CLIB = ctypes.CDLL('/opt/project/implementation/ctypes/mc.so')
CLIB.transition_probability.argtypes = [
    ctypes.POINTER(markov_chain_t),
    ctypes.c_uint64,
    ctypes.c_uint64
]
CLIB.transition_probability.restype = ctypes.c_double

tails   = np.array([1, 100, 101, 102, 103, 104, 105, 1000], dtype=np.int64)
offsets = np.array([0,   3,   5,   7,   9,  11,  13,   15], dtype=np.int32)
heads = np.array([
    100, 101, 102,  # 1
    101, 102,       # 100
    102, 103,       # 101
    103, 104,       # 102
    104, 105,       # 103
    105, 1000,      # 104
    1000            # 105
])
log_probs = np.log(1. / np.arange(2, heads.size + 2), dtype=np.double)

mc = markov_chain_t(
    7,
    offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    tails.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    heads.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    log_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)

print(CLIB.transition_probability(ctypes.byref(mc), 1, 100), log_probs[0])
print(CLIB.transition_probability(ctypes.byref(mc), 1, 101), log_probs[1])
print(CLIB.transition_probability(ctypes.byref(mc), 1, 102), log_probs[2])

print(CLIB.transition_probability(ctypes.byref(mc), 100, 101), log_probs[3])
print(CLIB.transition_probability(ctypes.byref(mc), 100, 102), log_probs[4])

print(CLIB.transition_probability(ctypes.byref(mc), 101, 102), log_probs[5])
print(CLIB.transition_probability(ctypes.byref(mc), 101, 103), log_probs[6])

print(CLIB.transition_probability(ctypes.byref(mc), 102, 103), log_probs[7])
print(CLIB.transition_probability(ctypes.byref(mc), 102, 104), log_probs[8])

print(CLIB.transition_probability(ctypes.byref(mc), 103, 104), log_probs[9])
print(CLIB.transition_probability(ctypes.byref(mc), 103, 105), log_probs[10])

print(CLIB.transition_probability(ctypes.byref(mc), 104, 105), log_probs[11])
print(CLIB.transition_probability(ctypes.byref(mc), 104, 1000), log_probs[12])

print(CLIB.transition_probability(ctypes.byref(mc), 105, 1000), log_probs[13])


import implementation.classification.mc as pymc
sequences = [
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;65;S'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;65;S'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;65;S'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;65;S'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C'],
    ['22:1;67;C', '22:2|22:11;79;S', '22:11|22:22;79;S', '22:22|22:12|22:14;75;S', '22:16|20|22:0;49;C']
]
m = pymc.MarkovChain('test')
m.fit(sequences)
export = m.c_export()
print(export)
time.sleep(1)

mc = markov_chain_t(
    export['num_nodes'],
    np.array(export['offsets'], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    np.array(export['tails'], dtype=np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    np.array(export['heads'], dtype=np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    np.array(export['log_probs'], dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)
score_py = m.score([sequences[0]])
int_seq = [pymc.omc.compute_hash(s) for s in sequences[0]]
int_seq.insert(0, 1)
lp = 0
for t, h in zip(int_seq[:-1], int_seq[1:]):
    tmp = CLIB.transition_probability(ctypes.byref(mc), t, h)
    lp += tmp
print(score_py, lp)
print(export)
import json
print(json.dumps(export, indent=1))
