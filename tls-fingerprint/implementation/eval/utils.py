import pandas as pd
import numpy as np
import logging
from ray.tune import Analysis
import os
from typing import List, Any, Dict, Tuple, Union
import itertools as itt
import shutil
import json

from baum_welch import phmm, learning
import implementation.data_conversion.constants as const
import implementation.data_conversion.data_aggregation_hmm_np as data_aggregation
from implementation.phmm_np.cross_val import TrainingConfig, get_all_companies_test,\
    get_all_companies_train, get_all_companies_val, _get_edges


logger = logging.getLogger('eval.utils')
logger.setLevel(level=logging.INFO)


def load_tune_analysis(full_dir_path: str) -> Union[None, Analysis]:
    """
        Loads the analysis output generated with a `tune.run` call.

        Args:
            full_dir_path: Full path to the directory in which the analysis
                result is stored.

        Returns:
            A tune analysis object.
    """
    analysis = None
    try:
        analysis = Analysis(full_dir_path)
    except Exception as e:
        logger.exception(e)
    return analysis


def load_multiple_tune_results(full_dir_path: str, experiment_prefix: str) -> List[Analysis]:
    """
        Search for all folders in the directory that start with the experiment_prefix
        and load the analysis result.

        Args:
            full_dir_path: Path in which experiments are contained.
            experiment_prefix: Prefix that all experiments have in common.

        Returns:
            results: List of analysis results.
    """
    results = []
    for f in os.listdir(full_dir_path):
        if f.startswith(experiment_prefix):
            results.append(load_tune_analysis(os.path.join(full_dir_path, f)))
        else:
            continue
    return results


def load_experiments(dir_path: str) -> Dict[str, List[Analysis]]:
    """
        Iterate over all folders in the given directory and load all experiments
        in there.
    """
    returns = {}
    for exp_dir in os.listdir(dir_path):
        returns[exp_dir] = load_multiple_tune_results(os.path.join(dir_path, exp_dir), '')
    return returns


def rename_columns(df: pd.DataFrame):
    """
    Remove the prefix "config/" from the column names.
    Args:
        df:

    Returns:

    """
    df.columns = [c[7:] if c.startswith("config/") else c for c in df.columns]
    return df


def evaluate_configs(analysises):
    """
    Take in a list of analysis objects and return statistics regarding log likelihood
    and purity over the runs.
    Args:
        analysises:

    Returns:

    """
    df = pd.concat([analysis.dataframe('purity', 'max') for analysis in analysises])
    df = rename_columns(df)
    df = df.drop(
        ['iter', 'done', 'timesteps_total', 'episodes_total', 'training_iteration',
         'experiment_id', 'date', 'timestamp', 'time_this_iter_s', 'time_total_s',
         'pid', 'hostname', 'node_ip', 'time_since_restore', 'timesteps_since_restore',
         'iterations_since_restore', 'trial_id', 'experiment_tag', 'seed',
         'trace_changes', 'logdir', 'change', 'average'], axis=1)
    df = df.drop(['max_val_geo_bin', 'included_packets'], axis=1)

    stats = df.groupby(['company', 'browser', 'binning_method', 'hmm_duration',
                'init_prior', 'num_bins', 'trace_length']).agg(
        [np.min, np.median, np.mean, np.max, lambda x: x.shape[0]]
    )
    return stats


def evaluate_configs2(analysises, metric='nll'):
    res = []
    mode = {'nll': 'min', 'purity': 'max'}[metric]
    for ana in analysises:
        if ana is None:
            continue
        config = ana.get_best_config(metric, mode)
        if config is None:
            continue
        df = ana.dataframe(metric=metric, mode=mode)
        config['purity'] = df['purity'][0]
        config['nll'] = df['nll'][0]
        if config['binning_method'] is None:
            config['binning_method'] = 'None'
        res.append(config)
    stats = pd.DataFrame.from_dict(res)
    stats.drop('seed', axis=1, inplace=True)
    return stats


def analyze_stats(stats2, browser, company, top=20):
    part = stats2.loc[pd.IndexSlice[:, browser, company, :, :, :, :, :, :, :], :]
    max_purity = part.loc[:, pd.IndexSlice['purity', 'mean']].max()
    part = part.loc[part.loc[:, pd.IndexSlice['purity', 'mean']] >= max_purity - 0.001]
    print("Has {} entries with purity = {}".format(part.shape[0], max_purity))
    return part.sort_values(by=('nll', 'mean'), ascending=True).iloc[:top]


def group_stats(stats):
    cs = ['binning_method', 'browser', 'company', 'hmm_duration', 'included_packets',
          'init_prior', 'max_val_geo_bin', 'num_bins', 'trace_changes', 'trace_length']
    return stats.groupby(cs).agg([np.mean, np.median, np.min, np.max, lambda x: x.shape[0]])


def get_ana_best_model(analysises: List[Analysis], configs: List[TrainingConfig],
                       browser: str, company: str) -> Analysis:
    """
    Get the analysis object of the best model.

    Best is defined like this:
        1) Select all analysis results with highest purity.
        2) From those, select the ones with minimum number of parameters.
        3) From those, select the ones requiring the smallest amount of packets.
        4) From those, select the ones with the smallest log-likelihood.
        5) From those, select the first one.
    Args:
        analysises:
        browser:
        company:

    Returns:

    """
    max_purity = 0
    for i, ana in enumerate(analysises):
        config = configs[i]
        if config is None:
            continue
        if config.browser != browser:
            continue
        if config.company != company:
            continue
        df = ana.dataframe(metric='purity', mode='max')
        config.purity = df['purity'][0]
        config.nll = df['nll'][0]
        config.ana = ana

        if config.purity > max_purity:
            max_purity = config.purity

        if config.binning_method == const.BINNING_NONE:
            factor = 1500
        elif config.binning_method == const.BINNING_SINGLE:
            factor = 1
        else:
            factor = config.num_bins
        config.num_params = 3 * (3 * config.hmm_duration + 1) + 3 \
            + (2 * config.hmm_duration + 1) * factor
        # configs.append(config)

    print(len(configs), end=' -> ')
    configs = [c for c in configs if c.purity >= max_purity - 1e-6]
    print(len(configs), end=' -> ')

    min_params = np.min([c.num_params for c in configs])
    configs = [c for c in configs if c.num_params <= min_params + 0.1]
    print(len(configs), end=' -> ')

    min_packets = np.min([c.trace_length for c in configs])
    configs = [c for c in configs if c.trace_length <= min_packets + 0.1]
    print(len(configs), end=' -> ')

    min_nll = np.min([c.nll for c in configs])
    configs = [c for c in configs if c.nll <= min_nll + 1e-9]
    print(len(configs))

    config = configs[0]
    return config.ana


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


def get_logdir_best_model(analysises: List[Analysis], binning_method: str,
                          num_bins: int, hmm_duration: int, init_prior: str,
                          trace_length: int, company: str, browser: str) -> str:
    template = TrainingConfig(
        binning_method=binning_method,
        num_bins=num_bins,
        hmm_duration=hmm_duration,
        trace_length=trace_length,
        company=company,
        browser=browser,
        init_prior=init_prior,
        max_val_geo_bin=1500,
        included_packets=const.FLOW_MAIN,
        trace_changes=False,
        seed=1
    )
    return get_logdir_best_model_t(analysises, template)


def export_best_models():
    analysises = load_multiple_tune_results('./data/existing_tune_results/HmmGridSearch/',
                                                'HmmGridSearchTrainable_launch_')
    analysises = [ana for ana in analysises if ana is not None]
    configs = [ana.get_best_config('purity', 'max') for ana in analysises]
    configs = [None if c is None else TrainingConfig.from_dict(c) for c in configs]
    browsers = [const.BROWSER_CHROME, const.BROWSER_MOZILLA, const.BROWSER_NOT_WGET]
    logdirs = {c: {} for c in const.applications}
    for company, browser in itt.product(const.applications, browsers):
        print(company, browser)
        logdirs[company][browser] = get_ana_best_model(analysises, configs, browser, company)
    for cmp in logdirs.values():
        for ana in cmp.values():
            p = ana.get_best_logdir('purity', 'max')
            shutil.copytree(p, os.path.join('./data/best_trials/', os.path.split(p)[1]))
    # analysises = evu.load_multiple_tune_results('./data/existing_tune_result/HmmGridSearch/', 'HmmGridSearchTrainable_launch_')


def load_config(trial_path: str) -> TrainingConfig:
    with open(os.path.join(trial_path, 'params.json'), 'r') as fh:
        d = json.load(fh)
    config = TrainingConfig.from_dict(d)
    return config


def phmm_from_trial(trial_path: str) -> phmm.Hmm:
    """
    Load a HMM from the saved one in the trial.
    Args:
        config:

    Returns:

    """
    config = load_config(trial_path)
    checkpoint = get_best_checkpoint(trial_path)

    json_file = None
    for f in os.listdir(checkpoint):
        if f.endswith(".json"):
            json_file = f
    assert json_file is not None, "Did not find a json file in the path {}".format(checkpoint)

    with open(os.path.join(checkpoint, json_file), 'r') as fh:
        data = json.load(fh)

    hmm = phmm.basic_phmm(config.hmm_duration, [])
    p_ij = {eval(k): v for k, v in data['trans'].items()}
    p_o = {eval(k): v for k, v in data['obs'].items()}
    hmm.p_o_in_i = p_o
    hmm.p_ij = p_ij
    return hmm


def evaluate_binary(hmm: phmm.Hmm, config: TrainingConfig, data: str) -> Dict[str, List[float]]:
    if data == 'train':
        all_pcaps = get_all_companies_train(config.browser)
    elif data == 'val':
        all_pcaps = get_all_companies_val(config.browser)
    elif data == 'test':
        all_pcaps = get_all_companies_test(config.browser)
    else:
        KeyError("Unknown data set name {} - Use train, val or test.".format(data))
    edges = _get_edges(config)
    print(config.to_dict())
    all_traces_val = {
        company: data_aggregation._convert_pcap_to_states(
            trace_length=config.trace_length,
            pcap_files=files,
            centers=edges,
            flow=config.included_packets
        ) for company, files in all_pcaps.items()
    }
    nlls = {}
    for k, sequences in all_traces_val.items():
        if k not in nlls:
            nlls[k] = []
        for sequence in sequences:
            nlls[k].append(-1. * learning.calc_log_prob(hmm, [sequence]))
    return nlls


def retrain():
    p = '/opt/project/data/best_trials'
    hmms = {}
    configs = {}
    for f in os.listdir(p):
        config = load_config(os.path.join(p, f))
        hmm = phmm_from_trial(os.path.join(p, f))
        if config.company not in hmms:
            hmms[config.company] = {}
            configs[config.company] = {}
        hmms[config.company][config.browser] = hmm
        configs[config.company][config.browser] = config

    scores = {k: {} for k in hmms.keys()}
    for company in hmms.keys():
        for browser in hmms[company].keys():
            print(company, browser)
            config = configs[company][browser]
            hmm = hmms[company][browser]
            pcaps = get_all_companies_train(browser)[company]
            pcaps.extend(get_all_companies_val(browser)[company])
            edges = _get_edges(config)
            sequences = data_aggregation._convert_pcap_to_states(
                trace_length=config.trace_length,
                pcap_files=pcaps,
                centers=edges,
                flow=config.included_packets
            )
            with open(os.path.join('tmp-res', f'{company}-{browser}-seq.csv'), 'w') as fh:
                fh.write('\n'.join(['\t'.join(s) for s in sequences]))
            obs = np.unique(np.concatenate(sequences)).tolist()
            # hmm = phmm.basic_phmm(hmm.duration, observation_space=obs)
            hmm = phmm.hmm_from_sequences(sequences, seed=1)
            lprobs = [[-1. * learning.calc_log_prob(hmm, s) for s in sequences]]
            for i in range(0):
                hmm, _, lprob = learning.baum_welch(hmm, sequences)
                lprobs.append(lprob)
                print(lprob)
            scores[company][browser] = np.max(-1 * np.array(lprobs[-1]))
            hmms[company][browser] = hmm
            with open(os.path.join('tmp-res', f'{company}-{browser}-hmm.json'), 'w') as fh:
                json.dump({
                    'trans': {str(k): v for k, v in hmm.p_ij.items()},
                    'obs': {str(k): v for k, v in hmm.p_o_in_i.items()}
                }, fh, indent=1)
    print("Initialize conf mat")
    conf_mat = {}
    for c in hmms.keys():
        conf_mat['unknown'] = {'unknown': 0.}
        conf_mat[c] = {"unknown": 0.}
        for c2 in hmms.keys():
            conf_mat[c][c2] = 0.
            conf_mat["unknown"][c2] = 0.

    print("Classify traces")
    browser = 'not_wget'
    for company in hmms.keys():
        print(f'\t{browser} {company}')
        config = configs[company][browser]
        pcaps = get_all_companies_test(browser)[company]
        edges = _get_edges(config)
        sequences = data_aggregation._convert_pcap_to_states(
            trace_length=config.trace_length,
            pcap_files=pcaps,
            centers=edges,
            flow=config.included_packets
        )
        for s in sequences:
            names = []
            lls = []
            anos = []
            for c2 in hmms.keys():
                hmm = hmms[c2][browser]
                nll = -1. * learning.calc_log_prob(hmm, s)
                ano = nll / scores[c2][browser]
                if ano <= 1:
                    lls.append(nll)
                    anos.append(ano)
                    names.append(c2)
            if len(names) == 0:
                pred = 'unknown'
            else:
                pred = names[np.argmin(anos)]
            conf_mat[company][pred] += 1

    print("Classify Background")
    all_pcaps = get_all_companies_test(browser)
    for company in hmms.keys():
        print(f"\t{company}")
        config = configs[company][browser]
        # pcaps = all_pcaps[company][:7]
        pcaps = ['defaul'] * 10
        edges = _get_edges(config)
        sequences = data_aggregation._convert_pcap_to_states(
            trace_length=config.trace_length,
            pcap_files=pcaps,
            centers=edges,
            flow='co_flows'
        )
        for s in sequences:
            names = []
            lls = []
            anos = []
            for c2 in hmms.keys():
                hmm = hmms[c2][browser]
                nll = -1. * learning.calc_log_prob(hmm, s)
                ano = nll / scores[c2][browser]
                if ano <= 1:
                    lls.append(nll)
                    anos.append(ano)
                    names.append(c2)
            if len(names) == 0:
                pred = 'unknown'
            else:
                pred = names[np.argmin(anos)]
            conf_mat['unknown'][pred] += 1
    return conf_mat


def classify_all():
    p = '/opt/project/data/best_trials'
    hmms = {}
    configs = {}
    for f in os.listdir(p):
        config = load_config(os.path.join(p, f))
        hmm = phmm_from_trial(os.path.join(p, f))
        if config.company not in hmms:
            hmms[config.company] = {}
            configs[config.company] = {}
        hmms[config.company][config.browser] = hmm
        configs[config.company][config.browser] = config

    scores = {k: {} for k in hmms.keys()}
    for company in hmms.keys():
        for browser in hmms[company].keys():
            hmm = hmms[company][browser]
            config = configs[company][browser]
            pcaps = get_all_companies_train(browser)[company]
            pcaps.extend(get_all_companies_val(browser)[company])
            edges = _get_edges(config)
            sequences = data_aggregation._convert_pcap_to_states(
                trace_length=config.trace_length,
                pcap_files=pcaps,
                centers=edges,
                flow=config.included_packets
            )
            scores[company][browser] = np.max([
                -1 * learning.calc_log_prob(hmm, [s]) for s in sequences
            ])
    print("Initialize conf mat")
    conf_mat = {}
    for c in hmms.keys():
        conf_mat['unknown'] = {'unknown': 0.}
        conf_mat[c] = {"unknown": 0.}
        for c2 in hmms.keys():
            conf_mat[c][c2] = 0.
            conf_mat["unknown"][c2] = 0.

    browser = 'not_wget'
    for company in hmms.keys():
        config = configs[company][browser]
        pcaps = get_all_companies_test(browser)[company]
        edges = _get_edges(config)
        sequences = data_aggregation._convert_pcap_to_states(
            trace_length=config.trace_length,
            pcap_files=pcaps,
            centers=edges,
            flow=config.included_packets
        )
        for s in sequences:
            names = []
            lls = []
            anos = []
            for c2 in hmms.keys():
                hmm = hmms[c2][browser]
                nll = -1. * learning.calc_log_prob(hmm, [s])
                ano = nll / scores[c2][browser]
                if ano <= 1:
                    lls.append(nll)
                    anos.append(ano)
                    names.append(c2)
            if len(names) == 0:
                pred = 'unknown'
            else:
                pred = names[np.argmin(anos)]
            conf_mat[company][pred] += 1

    all_pcaps = get_all_companies_test(browser)
    for company in hmms.keys():
        config = configs[company][browser]
        # pcaps = all_pcaps[company][:7]
        pcaps = ['defaul'] * 1000
        edges = _get_edges(config)
        sequences = data_aggregation._convert_pcap_to_states(
            trace_length=config.trace_length,
            pcap_files=pcaps,
            centers=edges,
            flow='co_flows'
        )
        for s in sequences:
            names = []
            lls = []
            anos = []
            for c2 in hmms.keys():
                hmm = hmms[c2][browser]
                nll = -1. * learning.calc_log_prob(hmm, [s])
                ano = nll / scores[c2][browser]
                if ano <= 1:
                    lls.append(nll)
                    anos.append(ano)
                    names.append(c2)
            if len(names) == 0:
                pred = 'unknown'
            else:
                pred = names[np.argmin(anos)]
            conf_mat['unknown'][pred] += 1


def eval_cross_val_properties():
    # import subprocess
    # subprocess.run("python -m pip install sklearn", shell=True)
    # from sklearn.ensemble import RandomForestRegressor
    # from plots.utils import get_fig
    # import matplotlib.pyplot as plt

    def plot(importances, company, browser):
        fig, ax = get_fig(1, 1)
        ax.bar(np.arange(importances.size),
               importances)
        ax.set_xticks(np.arange(importances.size))
        ax.set_xticklabels(['Binning Method',
                            'HMM Length', 'Initialization',
                            'Num Bins', 'Trace Length'], rotation=90)
        ax.set_ylabel("Feature Importance")
        plt.tight_layout()
        plt.savefig("/opt/project/{}-{}.pdf".format(company, browser))
        plt.close()

    stats = pd.read_hdf('./data/stats.h5', key='stats')
    browsers = ['Chromium', 'Mozilla', 'not_wget']
    companies = ['amazon', 'facebook', 'google', 'google_drive', 'google_maps',
       'wikipedia', 'youtube']

    for browser, company in itt.product(browsers, companies):
        print(browser, company)
        df = stats.loc[pd.IndexSlice[company, browser]]
        df.reset_index(inplace=True)
        X = pd.concat([df.binning_method.astype('category').cat.codes, df.hmm_duration,
                       df.init_prior.astype('category').cat.codes, df.num_bins,
                       df.trace_length], axis=1)
        tree = RandomForestRegressor()
        tree.fit(X.values, df.loc[:, ('purity', 'median')])
        plot(tree.feature_importances_, company, browser)


def plot_hmm():
    def get_pos(name, length):
        parts = name.split('_')
        if parts[0] == 'i':
            y = 0
        elif parts[0] == 'd':
            y = 3
        elif parts[0] == 'm':
            y = -3
        else:
            y = 0
        if name == 'start':
            x = -3
        elif name == 'end':
            x = ((length + 1) * 3)
        elif parts[0] == 'i':
            x = int(parts[1]) * 3 + 1
        else:
            x = int(parts[1]) * 3
        return x, y

    import matplotlib.pyplot as plt
    import networkx as nx
    p = '/opt/project/tmp-res'
    for f in os.listdir(p):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(p, f), 'r') as fh:
            trans = json.load(fh)['trans']
            trans = {eval(k): v for k, v in trans.items()}
        g = nx.DiGraph()
        for (u, v), w in trans.items():
            g.add_edge(u, v, probability=w)
        l = (g.number_of_nodes() - 3.) / 3
        ax = plt.subplot()
        fig = plt.gcf()
        fig.set_figwidth(l * 3)
        #pos = nx.spring_layout(g)
        pos = {n: get_pos(n, l) for n in g.nodes()}
        nx.draw_networkx_nodes(g, pos=pos, ax=ax)
        nx.draw_networkx_labels(g, pos=pos, ax=ax, font_size=8)
        nx.draw_networkx_edges(g, pos, width=[4 * d['probability'] for _, _, d in g.edges(data=True)], ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(p, f"{f[:-4]}.pdf"))
        plt.close()


