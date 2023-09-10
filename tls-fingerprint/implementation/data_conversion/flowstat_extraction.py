from __future__ import annotations
import os
import pandas as pd
import subprocess
import numpy as np
import json
import re
import logging
import multiprocessing as mp
from typing import Dict, List, Any, Tuple


logger = logging.getLogger('flowstat-extraction')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


PATTERN = re.compile("(?P<src>[0-9.:]+)( +<-> )(?P<dst>[0-9.:]+) +(?P<frames_out>\d+) +"
          "(?P<bytes_out>\d+) +(?P<frames_in>\d+) +(?P<bytes_in>\d+) +"
          "(?P<total_frames>\d+) +(?P<total_bytes>\d+) +"
          "(?P<start>\d+.\d+) +(?P<duration>\d+.\d+)")


def read_flow_stats(pcap_path: str) -> str:
    cmd = f'tshark -r {pcap_path} -qz conv,tcp'
    output = subprocess.check_output(cmd, shell=True)
    return output.decode('ascii')


def parse_tshark_stats(stats: str) -> List[Dict[str, float | str]] | None:
    lines = stats.split('\n')
    flow_infos = []
    if len(lines) < 8:
        return None
    for line in lines[5:-2]:
        m = re.match(PATTERN, line)
        if m is None:
            continue
        else:
            flow_infos.append({
                'src': m.group('src'),
                'dst': m.group('dst'),
                'frames_out': float(m.group('frames_out')),
                'bytes_out': float(m.group('bytes_out')),
                'frames_in': float(m.group('frames_in')),
                'bytes_in': float(m.group('bytes_in')),
                'total_frames': float(m.group('total_frames')),
                'total_bytes': float(m.group('total_bytes')),
                'start': float(m.group('start')),
                'duration': float(m.group('duration'))
            })
    return flow_infos


def aggregate_flow_infos(flow_infos: List[Dict[str, any]]) -> Dict[str, float]:
    aggregate = {
        'frames_out': 0,
        'bytes_out': 0,
        'frames_in': 0,
        'bytes_in': 0,
        'total_frames': 0,
        'total_bytes': 0,
        'duration': 0
    }
    for flow_info in flow_infos:
        for k in aggregate.keys():
            aggregate[k] += flow_info[k]
    aggregate['duration'] /= len(flow_infos)
    return aggregate


def mp_driver(all_args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rets = []
    for args in all_args:
        filename = args['filename']
        pcap_path = f'/opt/project/data/k8s-traces/{filename}.pcapng'
        if not os.path.exists(pcap_path):
            continue
        try:
            flow_infos = parse_tshark_stats(read_flow_stats(pcap_path))
            if flow_infos is None:
                continue
            if args['aggregate']:
                flow_infos = [aggregate_flow_infos(flow_infos)]
            for fi in flow_infos:
                fi.update(args)
            rets.extend(flow_infos)
        except Exception as e:
            logger.exception(e)
    return rets


if __name__ == '__main__':
    with open('/opt/project/data/cache/meta_data.json', 'r') as fh:
        meta_data = json.load(fh)
    with open('/opt/project/data/cache/labels.json', 'r') as fh:
        labels = json.load(fh)

    n_procs = mp.cpu_count() - 2
    all_args = [[] for _ in range(n_procs)]
    for i, (_, day) in enumerate(meta_data):
        flat_infos = []
        with open('log.txt', 'a') as fh:
            fh.write(f"Process Day {i}\n")
        for i, md in enumerate(day):
            if str(md['url_id']) in labels:
                all_args[i % n_procs].append({
                    'filename': md['filename'],
                    'url_id': md['url_id'],
                    'label': labels[str(md['url_id'])],
                    'url': md['url'],
                    'meta_data_id': md['meta_data_id'],
                    'aggregate': False
                })
        pool = mp.Pool(n_procs)
        infos = pool.map(mp_driver, all_args)
        pool.close()
        for i in infos:
            flat_infos.extend(i)
        df = pd.DataFrame.from_dict(flat_infos)
        df.to_hdf(f'/opt/project/data/cumul-interpolations/flow-infos-{i}.h5', key='flow-infos')

