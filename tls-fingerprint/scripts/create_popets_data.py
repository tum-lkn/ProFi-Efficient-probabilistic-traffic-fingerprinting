import logging
import multiprocessing as mp
import subprocess
import os
import json
from typing import List, Dict, Tuple, Any


logger = logging.getLogger('base-logger')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())
prefix = '/opt/project'


def get_files() -> List[str]:
    pcaps = []
    for f in os.listdir("/opt/project/data/k8s-json"):
        if f.endswith('.json') and not f.startswith('mainflow'):
            pcaps.append(os.path.join('/opt/project/data/k8s-traces', f"{f[:-5]}.pcapng"))
    return pcaps


def make_args(pcaps: List[str]) -> List[Dict[str, str]]:
    args = []
    for pcap in pcaps:
        _, pcap_file_name = os.path.split(pcap)
        pcap_file_name = pcap_file_name[:-7]
        args.append({
            "pcap_file_name": pcap,
            "data_dir": "/opt/project/data/popets-data/",
            "file_name_prefix": pcap_file_name,
            "exe_dir": prefix
        })
    return args


def make_resolutions(args: Dict[str, str]) -> None:
    pcap_file_name = args['pcap_file_name']
    data_dir = args['data_dir']
    file_name_prefix = args['file_name_prefix']
    exe_dir = args["exe_dir"]
    cmd = f'{exe_dir}/cparse/popets {pcap_file_name} {data_dir} {file_name_prefix}'
    try:
        subprocess.check_output(cmd, shell=True)
    except Exception as e:
        logger.error(f"Error converting {pcap_file_name} with command {cmd}")
        logger.exception(e)


if __name__ == "__main__":
    assert os.path.exists(f'{prefix}/cparse/popets'), f"Executable {prefix}/cparse/popets does not exist, build it!"
    assert os.path.exists("/opt/project/data/popets-data/")
    logger.info("get pcap files...")
    pcap_files = get_files()
    logger.info(f'Got {len(pcap_files)} files to convert.')
    for f in pcap_files:
        assert os.path.exists(f)
    logger.info("make arguments.")
    args = make_args(pcap_files)
    logger.info("start conversion")
    pool = mp.Pool(30)
    pool.map(make_resolutions, args)
    pool.close()
