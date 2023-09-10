import ctypes
import numpy as np
from typing import List, Tuple, Any, Dict


CLIB = ctypes.CDLL('/opt/project/cparse/build/lib.linux-x86_64-3.8/cparse.cpython-38-x86_64-linux-gnu.so')
CLIB.cumul_with_defense.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char)
]
CLIB.cumul_with_defense.restype = ctypes.c_int
CLIB.cumul_features.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char)
]
CLIB.cumul_features.restype = ctypes.c_int

filename = "/opt/project/data/devel-traces/chaturbate_firefox_gatherer-01-d2ghn_78644108.pcapng"
buffer = np.zeros(200, dtype=np.float32)
CLIB.cumul_with_defense(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(100), ctypes.c_char_p(filename.encode('utf-8')))
print(buffer)
