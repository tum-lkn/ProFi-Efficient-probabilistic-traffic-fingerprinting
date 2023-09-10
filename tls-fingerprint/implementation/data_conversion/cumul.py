import pandas as pd
import logging
import numpy as np
import subprocess
import os
import sqlalchemy
import h5py
from io import StringIO
import time
import multiprocessing as mp
from typing import List, Dict, Tuple, Any
import json


logger = logging.getLogger('base-logger')
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())

with open("CONFIG.json", 'r') as fh:
    CONFIG = json.laod(fh)


def get_metadata(limit=None):
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://root:{CONFIG["db_password"]}@{CONFIG["db_host"]}:3306/gatherer_upscaled'
    logger.info("Retrieve data from Database.")
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    t1 = time.time()
    with engine.connect() as connection:
        query = """
        SELECT
            urls.id as url_id,
            urls.url,
            traces_metadata.id as metadata_id,
            traces_metadata.filename
        FROM 
            gatherer_upscaled.traces_metadata INNER JOIN urls ON urls.id = traces_metadata.url
        """
        if limit is not None:
            query += f' LIMIT {limit}'
        data = connection.execute(query).fetchall()
    t2 = time.time()
    logger.info(f"Retrieved {len(data)} URLs - took {t2-t1}s.")
    return [{
        'url_id': url_id,
        'url': url,
        'metadata_id': metadata_id,
        'filename': filename,
        'n_points': 100
    } for url_id, url, metadata_id, filename in data]


def read_trace(trace_path: str) -> pd.DataFrame:
    assert os.path.exists(trace_path), f"Trace {trace_path} does not exist."
    cmd = f'tshark -r {trace_path} -Y "tcp && tcp.len > 0" ' \
          f'-T fields -e ip.src -e tcp.len -E header=y -E separator=";"'
    out = subprocess.check_output(cmd, shell=True)
    page_load = pd.read_table(StringIO(out.decode('utf8')), sep=';')
    return page_load


def get_client_ip(page_load: pd.DataFrame) -> str:
    client_ip = None
    idx = 0
    while client_ip is None and idx < page_load.shape[0]:
        ip = page_load.iloc[idx, 0]
        if ip.startswith('172.17'):
            client_ip = ip
        idx += 1
    return client_ip


def calculate_features(page_load: pd.DataFrame, client_ip: str) -> np.array:
    weights = np.ones(page_load.shape[0])
    weights += (page_load.iloc[:, 0] == client_ip).values.astype(np.float32) * -2
    cum_sizes = np.cumsum(page_load.iloc[:, 1].values)
    changes = np.cumsum(page_load.iloc[:, 1].values * weights)
    return np.column_stack((cum_sizes, changes))


def interpolate(features: np.array, n_points: int) -> np.array:
    points = np.linspace(0, features.shape[0] - 1, n_points)
    lower = np.floor(points).astype(np.int32)
    upper = np.ceil(points).astype(np.int32)
    step = np.expand_dims(points - lower, axis=1)
    interpolation = features[lower, :] + step * (features[upper, :] - features[lower, :])
    return interpolation


def strip_ending(pcap_path: str) -> str:
    assert pcap_path.endswith(".pcapng"), f'Expected file to end with pcapng: {pcap_path}'
    return pcap_path[:-1 * len(".pcapng")]


def make_features(trace_path: str) -> np.array:
    trace = read_trace(trace_path)
    client_ip = get_client_ip(trace)
    features = calculate_features(trace, client_ip)
    return features


def mp_driver(args_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    times = []
    for i, args in enumerate(args_list):
        trace_path = os.path.join('/opt/project/data/k8s-traces', f'{args["filename"]}.pcapng')
        h5_name = os.path.join('/opt/project/data/cumul-features', f'{args["filename"]}.h5')
        if os.path.exists(h5_name):
            continue
        try:
            t1 = time.time()
            features = make_features(trace_path)
            # interpolation = interpolate(features, args['n_points'])
            # args['interpolation'] = interpolation

            f = h5py.File(h5_name, 'w')
            f.create_dataset(name='features', data=features)
            f.close()
            times.append(time.time() - t1)
            if len(times) > 100:
                logger.info(f"Average for 100 files {float(np.mean(times))}")
                times = []
        except Exception as e:
            logger.exception(e)
            # args['interpolation'] = []
    return args_list


def mp_make_dataset(args_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    ret_list = []
    num_fails = 0
    for args in args_list:
        h5_name = os.path.join('/opt/project/data/cumul-features', f'{args["filename"]}.h5')
        if not os.path.exists(h5_name):
            continue
        try:
            f = h5py.File(h5_name, 'r')
            # Has shape [num_packets, 2]
            if 'features' in f.keys():
                features = f['features'][()]
                args['interpolation'] = interpolate(features, args['n_points'])
                ret_list.append(args)
            else:
                num_fails += 1
        except Exception as e:
            logger.exception(e)
            num_fails += 1
        finally:
            f.close()
    return ret_list, num_fails


def timing():
    p = '/opt/project/data/devel-traces'
    pcaps = [os.path.join(p, f) for f in os.listdir(p) if f.endswith('pcapng')]
    times = []
    for f in pcaps:
        print(f)
        t = time.time()
        features = make_features(f)
        inter = interpolate(features, 100)
        fp = h5py.File(os.path.join(p, 'tmp', f'{strip_ending(os.path.split(f)[1])}.h5'), "w")
        fp.create_dataset(name='features', data=features)
        fp.close()
        times.append(time.time() - t)
    print(pd.Series(times).describe())


if __name__ == '__main__':
    logger.info("Get metadata")
    meta_data = get_metadata()
    args_list = [[] for _ in range(mp.cpu_count() - 1)]
    logger.info("Map to jobs")
    for i, args in enumerate(meta_data):
        args_list[i % len(args_list)].append(args)
    logger.info("Extract Features")
    pool = mp.Pool()
    data = pool.map(mp_driver, args_list)
    pool.close()
    # linear_data = []
    # for x in data:
    #     linear_data.extend(x)
    # logger.info("Save Features")
    # df = pd.DataFrame(data)
    # df.to_hdf('/opt/project/data/cumul-interpolations/n100.h5', key='interpolation')
