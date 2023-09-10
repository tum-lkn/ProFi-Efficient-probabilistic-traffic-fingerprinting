import os
import subprocess
import numpy as np
from typing import List


part_file_pattern = 'measure-part-trace-{:02d}.pcapng'
pcap_folder = '/home/sim/TRACES'


def merge_files(to_merge: List[str], out_file: str) -> None:
    names = ' '.join(to_merge)
    subprocess.call(f"mergecap {names} -w {out_file}", shell=True)
    
    
def merge_stage_one(seed: int, out_dir: str) -> List[str]:
    pcap_files = os.listdir(pcap_folder)
    random = np.random.RandomState(seed)
    # Make sure to draw each PCAP only once. Random sampling could duplicate
    # Traces due to the birthday paradoxon.
    indices = np.arange(pcap_files).astype(np.int32)
    random.shuffle(indices)
    make_pcap = lambda idx: os.path.join(pcap_folder, pcap_files[idx])
    part_files = []
    for i in range(40):
        print(f"Create {i:2d} of 40 files", end=' ')
        to_merge = []
        total_size = 0
        while total_size < 1e9:
            p = make_pcap()
            total_size += os.path.getsize(p)
            to_merge.append(p)
        print(f"Merge {len(to_merge)} Files")
        part_files.append(os.path.join(out_dir, part_file_pattern.format(i)))
        merge_files(to_merge, part_files[-1])
    return part_files
    
    
if __name__ == '__main__':
    for seed in range(1, 2):
        merge_files(
            to_merge=merge_stage_one(seed, '/home/sim'),
            out_file=f'/mnt/nfs/full-measure-trace-{seed}.pcapng'
        )
    
    
