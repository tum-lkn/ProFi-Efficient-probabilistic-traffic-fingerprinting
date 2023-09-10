import pandas as pd
import os
import numpy as np
import subprocess
from typing import List, Dict, Tuple, Any

BASE_DIR = '/opt/project/data/bigdata/protype-measures-ccs'
BASE_DIR_TTL = '/opt/project/data/bigdata/protype-measures-ccs'


def get_exp_start(results: pd.DataFrame) -> int:
    df_tmp = results.set_index('INSTANCEID').loc[1, :]
    df_tmp = df_tmp.loc[df_tmp.RX.values == 0, :]
    return df_tmp.iloc[-1]['TS']


def get_exp_end(results: pd.DataFrame) -> int:
    df_tmp = results.set_index('INSTANCEID').loc[1, :]
    df_tmp = df_tmp.loc[df_tmp.RX.values == df_tmp.iloc[-1, :]['RX'], :]
    return df_tmp.iloc[0]['TS']


def slice_experiment_period(results: pd.DataFrame) -> pd.DataFrame:
    start_ts = get_exp_start(results)
    end_ts = get_exp_end(results)
    tmp = results.loc[results.TS.values > start_ts, :]
    tmp = tmp.loc[tmp.TS.values < end_ts, :]
    return tmp


def _read_result(p) -> pd.DataFrame:
    df = pd.read_csv(p, sep=';')
    df['TS'] = pd.to_datetime(df['TS'], unit='ms')
    return df


def read_result(num_hmms: int, length: int, base_dir=BASE_DIR) -> pd.DataFrame:
    p = os.path.join(
        base_dir,
        f'num-zoom-{num_hmms:d}-hmm-len-{length:d}-trace-len-{length:d}',
        'VNF_stats.txt'
    )
    return _read_result(p)


def calc_cost_pp(df: pd.DataFrame, service_id: int) -> pd.Series:
    """
    Returns the cost in cycles per packet.

    Args:
        df:
        service_id:

    Returns:

    """
    df = df.loc[:, ['RX', 'COST', 'SERVICEID']].set_index('SERVICEID').loc[service_id].diff().dropna()
    df = df.loc[df.RX > 0]
    cost_pp = df.COST.values / df.RX.values
    return cost_pp


def extract_archives(base_dir: str) -> None:
    for f in os.listdir(base_dir):
        if f.endswith('.tar.gz'):
            name = f[:f.find('.tar.gz')]
            if not os.path.exists(os.path.join(base_dir, name)):
                os.mkdir(os.path.join(base_dir, name))
            subprocess.call(f"mv {os.path.join(base_dir, f)} {os.path.join(base_dir, name)}", shell=True)
            subprocess.call(f'tar -xzf {os.path.join(base_dir, name, f)} -C {os.path.join(base_dir, name)}', shell=True)
        if os.path.isdir(os.path.join(base_dir, f)):
            cmd = f'tar -xzf {os.path.join(base_dir, f, f"{f}.tar.gz")} -C {os.path.join(base_dir, f)}'
            subprocess.call(cmd, shell=True)


def retrieve_num_pgms():
    def is_big_model(p: str):
        return p.find('big-mc') >= 0 or p.find('hmm-len-30') >= 0

    samples = []
    for f in os.listdir(BASE_DIR):
        td = os.path.join(BASE_DIR, f)
        if not os.path.isdir(td):
            continue
        parts = f.split('-')
        print(f, parts)
        data = slice_experiment_period(_read_result(os.path.join(td, 'VNF_stats.txt')))
        try:
            sample = {
                'model': parts[0],
                'workload': parts[1] if parts[1] == 'average' else f'{parts[1]}-{parts[2]}',
                'model_size': 'big' if is_big_model(f) else 'small',
                'num_pgms': int(parts[5]) if parts[1] == 'average' else int(parts[6]),
                'dropped': hmm_or_data_agg_drops(data)
            }
            samples.append(sample)
        except Exception as e:
            print(f"Error for file {td}")
            print(e)
    return samples


def hmm_or_data_agg_drops(result: pd.DataFrame) -> bool:
    sids = np.sort(result.SERVICEID.unique())[3:]
    tmp = result.set_index("SERVICEID").loc[sids, :]
    if tmp.TX_DROP.max() > 1000 or tmp.RX_DROP.max() > 1000:
        return True
    else:
        return False


def ttl_config_from_name(name: str) -> Dict[str, Any]:
    def is_big_model(p: str):
        return p.find('big-mc') >= 0 or p.find('hmm-len-30') >= 0

    parts = name.split('-')
    config = {
        'model': parts[0],
        'workload': parts[1] if parts[1] == 'average' else f'{parts[1]}-{parts[2]}',
        'model_size': 'big' if is_big_model(name) else 'small',
        'num_pgms': int(parts[5]) if parts[1] == 'average' else int(parts[6]),
        'onvm_mode': parts[2] if parts[1] == 'average' else parts[3]
    }
    return config


def calculate_ttls(enter: pd.DataFrame, exit: pd.DataFrame) -> np.array:
    ttls = []
    exit = exit.set_index(['src_ip', 'src_port', 'dst_ip', 'dst_port']).sort_index()
    tuple_known = {}
    for _, row in enter.iterrows():
        k = (row.src_ip, row.src_port, row.dst_ip, row.dst_port) 
        k_str = f'{row.src_ip};{row.src_port};{row.dst_ip};{row.dst_port}'
        if k_str in tuple_known:
            continue
        else:
            tuple_known[k_str] = True
        if k in exit.index:
            times = exit.loc[[k], :]
            ts_exit = times.tv_sec.values.astype(np.float64) + times.tv_nsec.values.astype(np.float64) / 1e9
            ts_exit = np.sort(ts_exit)
            ts_enter = float(row.tv_sec) + float(row.tv_nsec) / 1e9
            for i in range(ts_exit.size):
                delta = ts_exit[i] - ts_enter
                if delta < 0:
                    continue
                elif delta > 1.:
                    break
                else:
                    ttls.append(delta)
                    break
    return np.array(ttls)


def read_ttlbls(base_dir, onvm_mode='cp', workload='average', num_pgms: List[int]=None) -> List[Dict[str, Any]]:
    assert onvm_mode in ['cp', 'cs'] # cp = core-pinning, cs = core-sharing.
    assert workload in ['average', 'high-pps', 'high-fps']
    if num_pgms is None:
        num_pgms = [1]
    datas = []
    for d in os.listdir(base_dir):
        trial_dir = os.path.join(base_dir, d)
        if os.path.isdir(trial_dir):
            try:
                config = ttl_config_from_name(d)
                # if config['onvm_mode'] != 'cp':
                #     continue
                # if config['workload'] != 'average':
                #     continue
                # if config['num_pgms'] not in [1]:
                #     continue
                # if config['model_size'] != 'big':
                #     continue
                if config['onvm_mode'] == onvm_mode and config['workload'] == workload and config['num_pgms'] in num_pgms:
                    print(d, end=': ')
                    enter = pd.read_csv(os.path.join(trial_dir, 'time_to_label_tls_filter.csv'), sep=';')
                    exit = pd.read_csv(os.path.join(trial_dir, 'time_to_label_tls_classifier.csv'), sep=';', skiprows=1)
                    print(f"Filter file has {enter.shape[0]}, Classifier has {exit.shape[0]} entries.")
                    ttls = calculate_ttls(enter, exit)
                    config['ttlbls'] = ttls
                    config['median'] = np.median(ttls)
                    config['mean'] = np.mean(ttls)
                    config['95p'] = np.percentile(ttls, 95)
                    config['97.5p'] = np.percentile(ttls, 97.5)
                    config['99p'] = np.percentile(ttls, 99)
                    datas.append(config)
            except Exception as e:
                print(e)
    return datas
