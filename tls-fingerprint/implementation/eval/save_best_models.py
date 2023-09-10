import os
import sys
import numpy as np
import pandas as pd
import json as js
from typing import Dict, Tuple, List, Any

sys.path.append('/mounted-data/code/')
sys.path.append('/mounted-data/code/tls-fingerprint/implementation/')

from data_conversion import data_aggregation
from data_conversion import constants
from data_conversion import data_aggregation_hmm_np
from baum_welch import phmm, learning

def get_best_checkpoint(logdir: str) -> str:
    """
    Given a logdir for a tune trial return the path to the checkpoint with the
    best model associated.
    The best model is retrieved as follows:
        1) Get all the training iterations for which checkpoints exists.
        2) From those training iterations, find the maximum purity.
        3) Select those and sort by NLL ascending.
        4) Take the checkpoint with the highest purity and the lowest NLL by
            selecting the first row.
    Args:
        logdir:

    Returns:

    """
    def _get_checkpoint_nums(logdir: str) -> List[int]:
        return [int(f.split("_")[1]) for f in os.listdir(logdir) if f.startswith('checkpoint')]

    progress = pd.read_csv(os.path.join(logdir, 'progress.csv'))
    nums = _get_checkpoint_nums(logdir)
    progress.set_index('training_iteration', inplace=True)

    progress = progress.loc[nums, ['nll', 'purity']]
    best_purity = progress['purity'].max()
    progress = progress.loc[progress.purity.values >= best_purity - 1e-9]
    best_num = progress.reset_index().sort_values('nll').iloc[0]['training_iteration']
    return os.path.join(logdir, 'checkpoint_{:d}'.format(int(best_num)))

def convert_json_dict_to_class(duration, trans, obs):

    hmm = phmm.basic_phmm(duration, [])
    hmm.p_ij = trans
    hmm.p_o_in_i = obs

    return hmm


def load_model(path: str) -> phmm:

    log_prob = 0

    _, _, files = next(os.walk(path))
    files = [fi for fi in files if fi.endswith('.json')]

    json_file = os.path.join(path, files[0])

    with open(json_file, 'r', encoding='utf-8') as f:

        fingerprint = js.load(f)

    duration = fingerprint['duration']
    # log_prob = fingerprint['log_prob']
    trans = fingerprint['trans']
    obs = fingerprint['obs']

    trans = {eval(k): v for k, v in trans.items()}
    obs = {eval(k): v for k, v in obs.items()}

    hmm = convert_json_dict_to_class(duration, trans, obs)

    return hmm, -log_prob

def get_parameters(path: str):

    param_file = os.path.join(path, 'params.json')

    with open(param_file, 'r', encoding='utf-8') as f:

        params = js.load(f)

    return params

def load_model_params(trial_path):

    result_path = get_best_checkpoint(trial_path)

    hmm, _ = load_model(result_path)

    params = get_parameters(trial_path)

    return hmm, params

def calc_threshold(hmm_best: phmm.Hmm, traces: List[str], opt_threshold: int) -> float:

    log_prob_l = []
    for trace in traces:
        log_prob = learning.calc_log_prob(hmm_best, [trace])
        log_prob_l.append(log_prob)

    threshold = np.percentile(log_prob_l, 100 - opt_threshold)

    return threshold

def save_hmm_params(hmm: phmm.Hmm, threshold: float, params: Dict[str, Any], centers: np.array) -> None:

    data_aggregation_hmm_np.save_phmm(hmm, params['company'], params['browser'], params['trace_length'], centers, threshold)
    data_aggregation_hmm_np.save_params_phmm(
            path=None,
            company=params['company'],
            browser=params['browser'],
            flow=params['included_packets'],
            binning_method=params['binning_method'],
            num_bins=params['num_bins'],
            trace_length=params['trace_length'],
            hmm_duration=params['hmm_duration'],
            init_prior=params['init_prior']
            )


hyper_param_path = '/mounted-data/data/hyper_para_results/best_trials'
_, hyper_param_files, _ = next(os.walk(hyper_param_path))

opt_threshold = {'amazon': 99, 'facebook': 99, 'google': 100, 'google_drive': 99, 'google_maps': 99, 'wikipedia': 100, 'youtube': 100}

for grid_result in hyper_param_files:

    trial_path = os.path.join(hyper_param_path, grid_result)
    hmm, params = load_model_params(trial_path)
    pcaps = data_aggregation.acquire_pcap_files(params['company'], params['browser'], 'train')
    binning_method = params['binning_method']
    if binning_method is None:
        binning_method = 'None'
    centers = data_aggregation.get_binning_centers(num_bins=params['num_bins'], binning=binning_method, pcaps=pcaps)
    traces_val = data_aggregation_hmm_np.convert_pcap_to_states(
            company=params['company'],
            browser=params['browser'],
            flow=params['included_packets'],
            trace_length=params['trace_length'],
            centers=centers,
            data_set='val'
        ) 
    threshold = calc_threshold(hmm, traces_val, opt_threshold[params['company']])
    save_hmm_params(hmm, threshold, params, centers)
