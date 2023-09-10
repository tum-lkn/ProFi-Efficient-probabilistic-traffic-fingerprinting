import subprocess
import os
import multiprocessing as mp
import sys
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('/home/sim/tls-fingerprint/cumul-feature.log'))


prefix = '/home/sim/tls-fingerprint'
# prefix = '/opt/project'
# prefix = '/home/patrick/Documents/GitHub/lkn/tls-fingerprint'


def get_files(data_dir: str) -> List[str]:
    pcaps = []
    for f in os.listdir(data_dir):
        if f.endswith('pcapng') and not f.startswith('mainflow'):
            pcaps.append(os.path.join(data_dir, f))
    return pcaps


def get_files_ct() -> List[str]:
    thresh = 1651075376.4362242
    pcaps = []
    p = '/mnt/nfs/cumul-interpolations-defense'
    for f in os.listdir(p):
        pf = os.path.join(p, f)
        if os.path.getctime(pf) > thresh:
            pcaps.append(os.path.join('/mnt/nfs/k8s-traces', f'{f[:-4]}.pcapng'))
    return pcaps


def get_output_files(ft_dir: str, pcaps: List[str]) -> List[str]:
    ft_dirs = []
    for pcap in pcaps:
        _, f = os.path.split(pcap)
        ft_dirs.append(os.path.join(ft_dir, f'{f[:-7]}.bin'))
    return ft_dirs


def make_trace(args: Dict[str, str]) -> None:
    pcap_file = args['pcap_file']
    ft_file = args['ft_file']
    defend = args['defend']
    cmd = f'{prefix}/cparse/cparse {1 if defend else 2} {pcap_file} {ft_file}'
    try:
        subprocess.check_output(cmd, shell=True)
    except Exception as e:
        logger.error(f"Error converting {pcap_file} with command {cmd}")
        logger.exception(e)


if __name__ == '__main__':
    # pcap_files = get_files(f'{prefix}/data/devel-traces')
    if sys.argv[1] == 'defend':
        ft_root = '/mnt/nfs/cumul-interpolations-defense'
    else:
        ft_root = '/mnt/nfs/cumul-interpolations-cpp'
    logger.info("get pcap files...")
    pcap_files = get_files(f'/mnt/nfs/k8s-traces')
    logger.info(f'Got {len(pcap_files)} files to convert.')
    assert os.path.exists(f'{prefix}/cparse/cparse')
    for f in pcap_files:
        assert os.path.exists(f)
    assert os.path.exists(ft_root)
    logger.info("make bin file paths...")
    ft_files = get_output_files(ft_root, pcap_files)
    args = [{'pcap_file': p, 'ft_file': f, 'defend': sys.argv[1] == 'defend'} for (p, f) in zip(pcap_files, ft_files)]
    logger.info("start conversion")
    pool = mp.Pool(50)
    pool.map(make_trace, args)
    pool.close()
