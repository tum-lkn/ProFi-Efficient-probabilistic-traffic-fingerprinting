import os
import sys
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
import ray
from ray import tune
import itertools as itt
import json

#   Files from this Project

from baum_welch import phmm, learning
from implementation.data_conversion import data_aggregation
import implementation.data_conversion.constants as const
from implementation.data_conversion.constants import applications


def read_dataset(company: str, browser: str, dataset: str) -> List[str]:
    data_file = os.path.join(const.data_dir, 'benedikts-dataset.json')
    with open(data_file, 'r') as fh:
        data = json.load(fh)
    return data[company][browser][dataset]


def get_all_companies_val(browser: str) -> Dict[str, List[str]]:
    data_file = os.path.join(const.data_dir, 'benedikts-dataset.json')
    with open(data_file, 'r') as fh:
        data = json.load(fh)
    all_companies_val = {cmp: data[cmp][browser]['val'] for cmp in applications}
    return all_companies_val


def get_all_companies_train(browser: str) -> Dict[str, List[str]]:
    data_file = os.path.join(const.data_dir, 'benedikts-dataset.json')
    with open(data_file, 'r') as fh:
        data = json.load(fh)
    all_companies_val = {cmp: data[cmp][browser]['train'] for cmp in applications}
    return all_companies_val


def get_all_companies_test(browser: str) -> Dict[str, List[str]]:
    data_file = os.path.join(const.data_dir, 'benedikts-dataset.json')
    with open(data_file, 'r') as fh:
        data = json.load(fh)
    all_companies_val = {cmp: data[cmp][browser]['test'] for cmp in applications}
    return all_companies_val


def read_datasets(company: str, browser: str) -> Tuple[List[str], List[str], Dict[str, List[str]], List[str]]:
    data_file = os.path.join(const.data_dir, 'benedikts-dataset.json')
    with open(data_file, 'r') as fh:
        data = json.load(fh)
    all_companies_val = {cmp: data[cmp][browser]['val'] for cmp in applications}
    return data[company][browser]['train'], data[company][browser]['val'],\
           all_companies_val, data[company][browser]['test']


def _seq_to_csv(s: str, seq: List[List[str]]):
    tmp = ['\t'.join(sq) for sq in seq]
    s += '\n' + '\n'.join(tmp)
    return s


def _save_traces(s):
    with open('traces.csv', 'w') as fh:
        fh.write(s)


def _get_edges(user_config):
    if user_config.binning_method == const.BINNING_FREQ:
        print("Get sizes...")
        sizes = data_aggregation._get_packet_sizes(read_dataset(
            company=user_config.company,
            browser=user_config.browser,
            dataset='train'
        ))
        edges = data_aggregation.equal_frequency_binning(
            x=sizes,
            num_bins=user_config.num_bins
        )
    elif user_config.binning_method == const.BINNING_GEOM:
        edges = data_aggregation.log_binning(
            max_val=user_config.max_val_geo_bin,
            num_bins=user_config.num_bins
        )
    elif user_config.binning_method == const.BINNING_EQ_WIDTH:
        edges = data_aggregation.equal_width_binning(
            max_val=user_config.max_val_geo_bin,
            num_bins=user_config.num_bins
        )
    elif user_config.binning_method == const.BINNING_SINGLE:
        edges = np.array([0.])
    else:
        edges = None
    return edges


class TrainingConfig(object):

    @classmethod
    def from_dict(cls, dc: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**dc)

    def __init__(self, binning_method: str, num_bins: int, hmm_duration: int,
                 included_packets: str, trace_length: int, seed: int,
                 company: str, browser: str, max_val_geo_bin: int=None,
                 trace_changes=False, init_prior='uniform'):
        if binning_method is None:
            self.binning_method = const.BINNING_NONE
        else:
            self.binning_method = binning_method
        self.num_bins = num_bins
        self.hmm_duration = hmm_duration
        self.included_packets = included_packets
        self.trace_length = trace_length
        self.seed = seed
        self.company = company
        self.browser = browser
        self.max_val_geo_bin = max_val_geo_bin
        self.trace_changes = trace_changes
        self.init_prior = init_prior

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binning_method": self.binning_method,
            "num_bins": self.num_bins,
            "hmm_duration": self.hmm_duration,
            "included_packets": self.included_packets,
            "trace_length": self.trace_length,
            "seed": self.seed,
            "company": self.company,
            "browser": self.browser,
            "max_val_geo_bin": self.max_val_geo_bin,
            'trace_changes': self.trace_changes,
            'init_prior': self.init_prior
        }

    def equals(self, config: 'TrainingConfig', strict: bool=False) -> bool:
        ret = self.binning_method == config.binning_method and \
              self.num_bins == config.num_bins and \
              self.hmm_duration == config.hmm_duration and \
              self.included_packets == config.included_packets and \
              self.trace_length == config.trace_length and \
              self.company == config.company and \
              self.browser == config.browser and \
              self.max_val_geo_bin == config.max_val_geo_bin and \
              self.init_prior == config.init_prior
        if strict:
            ret = ret and self.seed == config.seed
        return ret


class HmmTrainable(tune.Trainable):

    def _traces_from_db(self):
        self.traces_train = data_aggregation.convert_pcap_to_states_mc(
            company=self.user_config.company,
            trace_length=self.user_config.trace_length,
            browser=self.user_config.browser,
            centers=self.edges,
            flow=self.user_config.included_packets,
            data_set='train'
        )
        self.traces_val = data_aggregation.convert_pcap_to_states_mc(
            company=self.user_config.company,
            trace_length=self.user_config.trace_length,
            browser=self.user_config.browser,
            centers=self.edges,
            flow=self.user_config.included_packets,
            data_set='val'
        )
        self.all_traces_val = {
            company: data_aggregation.convert_pcap_to_states_mc(
                company=company,
                trace_length=self.user_config.trace_length,
                browser=self.user_config.browser,
                centers=self.edges,
                flow=self.user_config.included_packets,
                data_set='val'
            ) for company in applications
        }
        self.traces_test = data_aggregation.convert_pcap_to_states_mc(
            company=self.user_config.company,
            trace_length=self.user_config.trace_length,
            browser=self.user_config.browser,
            centers=self.edges,
            flow=self.user_config.included_packets,
            data_set='test'
        )

    def _traces_from_json(self):
        pcaps_train, pcaps_val, pcaps_all_val, pcaps_test = read_datasets(
            company=self.user_config.company,
            browser=self.user_config.browser
        )
        self.traces_train = data_aggregation._convert_pcap_to_states_mc(
            trace_length=self.user_config.trace_length,
            pcap_files=pcaps_train,
            centers=self.edges,
            flow=self.user_config.included_packets
        )
        self.traces_val = data_aggregation._convert_pcap_to_states_mc(
            trace_length=self.user_config.trace_length,
            pcap_files=pcaps_val,
            centers=self.edges,
            flow=self.user_config.included_packets
        )
        self.all_traces_val = {
            company: data_aggregation._convert_pcap_to_states_mc(
                trace_length=self.user_config.trace_length,
                pcap_files=files,
                centers=self.edges,
                flow=self.user_config.included_packets
            ) for company, files in pcaps_all_val.items()
        }
        self.traces_test = data_aggregation._convert_pcap_to_states_mc(
            trace_length=self.user_config.trace_length,
            pcap_files=pcaps_test,
            centers=self.edges,
            flow=self.user_config.included_packets
        )

    def setup(self, config: Dict[str, Any]):
        self.user_config = TrainingConfig.from_dict(config)
        self.changes = []
        self.edges = _get_edges(self.user_config)

        print("get traces...")
        # _save_traces(_seq_to_csv(_seq_to_csv(_seq_to_csv('', self.traces_train), self.traces_val), self.traces_test))
        self._traces_from_json()
        print("average length: ", np.mean([len(x) for x in self.traces_train]))
        print("Get observation space...")
        observation_space = data_aggregation.get_observation_space([
            data_aggregation.get_observation_space(x) for x in [
                self.traces_train,
                self.traces_val,
                self.traces_test]
        ])
        self.hmm = phmm.basic_phmm(
            duration=self.user_config.hmm_duration,
            observation_space=observation_space,
            init_prior=self.user_config.init_prior,
            seed=config['seed']
        )

    def _calc_purity(self):
        count = []
        nlls = []
        for k, sequences in self.all_traces_val.items():
            for sequence in sequences:
                nlls.append(-1. * learning.calc_log_prob(self.hmm, [sequence]))
                count.append(k == self.user_config.company)
        count = np.array(count)[np.argsort(nlls)]
        n_traces = len(self.all_traces_val[self.user_config.company])
        return np.sum(count[:n_traces]) / n_traces

    def step(self):
        self.hmm, changes, nll = learning.baum_welch(
            hmm=self.hmm,
            observations_l=self.traces_train,
            iter=self.training_iteration
        )
        log_prob_l = -1. * learning.calc_log_prob(self.hmm, self.traces_val)
        self.log_prob = log_prob_l
        change = None
        average = None
        if self.user_config.trace_changes:
            self.changes.append(pd.DataFrame.from_dict(changes, orient='columns'))
            change = np.sum(np.abs(self.changes[-1].change.values))
            average = np.mean(np.abs(self.changes[-1].change.values))
            print(self.changes[-1].sort_values(by='change', ascending=False).head())
        return {
            'change': change,
            'average': average,
            'nll_train': nll,
            'nll': log_prob_l,
            "iter": self.training_iteration,
            "purity": self._calc_purity()
        }

    def save_checkpoint(self, tmp_checkpoint_dir):
        name = self.user_config.company
        if self.user_config.browser is not None:
            name += '-{}'.format(self.user_config.browser)
        data_aggregation.save_hmm_np(hmm=self.hmm, company=name,
                                     log_prob=self.log_prob, path=tmp_checkpoint_dir)

        if self.user_config.trace_changes:
            df = pd.concat(self.changes)
            df.to_hdf(os.path.join(tmp_checkpoint_dir, 'parameter-changes.h5'), key='changes')
        return tmp_checkpoint_dir

    def restore(self, checkpoint_path):
        pass


class ExistingConfigs(object):
    def __init__(self):
        self.configs = []
        base_path = '/opt/project/data/existing_tune_results/HmmGridSearch'
        num_files = len(os.listdir(base_path))
        for i, f in enumerate(os.listdir(base_path)):
            if os.path.isdir(os.path.join(base_path, f)) and os.path.exists(os.path.join(base_path, f, 'params.json')):
                with open(os.path.join(base_path, f, 'params.json'), 'r') as fh:
                    self.configs.append(TrainingConfig.from_dict(json.load(fh)))
            if i % 1000 == 0:
                print("checked {:7d} of {:7d} files.".format(i, num_files))
        print("Retrieved {} configs".format(len(self.configs)))

    def __call__(self, config: TrainingConfig) -> bool:
        ret = False
        for c in self.configs:
            ret = config.equals(c)
            if ret:
                break
        return ret


def get_launches_to_skip(offset, upper_limit=1e12) -> int:
    """
    Get the launch numbers that should be skipped initially. The attributes
    offset and uppe_limit can be used to control the returned number. Only a
    value between those two thresholds is returned, i.e.,
    offset <= max_launch_num < upper_limit.

    After the first experiments failed due to an SQL error, I started another
    experiment sequence with a higher offset value. To continue on either the
    first or the second sequence of trainings I needed this control of the
    range in which the offset value is returned.

    Args:
        offset:
        upper_limit:

    Returns:
        max_launch_num
    """
    # HmmGridSearchTrainable_launch_1000306_0928c_00029
    # experiment_state-2020-10-30_19-55-44.json
    path = '/opt/project/data/existing_tune_results/HmmGridSearch'
    max_launch_num = offset
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            # Identifies the trainable result directory.
            try:
                tmp = int(f.split('_')[2])
                if max_launch_num < tmp < upper_limit:
                    max_launch_num = tmp
            except Exception as e:
                print("failed for ", f)
                raise e
    return max_launch_num


if __name__ == '__main__':
    ray.init()
    # binnings = [const.BINNING_FREQ, const.BINNING_GEOM, const.BINNING_EQ_WIDTH]
    binnings = [const.BINNING_SINGLE, const.BINNING_NONE]
    # nbinss = list(range(30, 101, 10))
    nbinss = [0]
    durs = list(range(8, 33, 4))
    lengths = list(range(8, 33, 4))
    browsers = [const.BROWSER_MOZILLA, const.BROWSER_CHROME, const.BROWSER_NOT_WGET]
    companies = ['google', 'amazon', 'facebook', 'wikipedia', 'youtube', 'google_drive', 'google_maps']
    priors = ["prefer_match", "uniform"]
    launch = 1000000
    # to_skip = get_launches_to_skip(launch)
    exists_config = ExistingConfigs()
    for binning, nbins, dur, length, browser, prior in itt.product(binnings, nbinss, durs, lengths, browsers, priors):
        tmp = TrainingConfig(
            binning_method=binning,
            num_bins=nbins,
            hmm_duration=dur,
            included_packets=const.FLOW_MAIN,
            trace_length=length,
            company='google',
            browser=browser,
            max_val_geo_bin=1500,
            init_prior=prior,
            seed=1
        )
        print("Config {} equals: {}".format(launch, exists_config(tmp)))
        if exists_config(tmp):
            print("skipped launch number {:d}".format(launch))
        else:
            search_space = {
                "binning_method": binning,
                "num_bins": nbins,
                "hmm_duration": dur,
                "included_packets": const.FLOW_MAIN,
                "trace_length": length,
                "seed": tune.randint(1, int(2**32 - 1)),
                "company": tune.grid_search(companies),
                "browser": browser,
                "max_val_geo_bin": 1500,
                "init_prior": prior,
                "trace_changes": False
            }
            print("Start launch number {:d}. Config: {}".format(launch, json.dumps(search_space)))
            tune.run(
                HmmTrainable,
                trial_dirname_creator=lambda trial: 'HmmGridSearchTrainable_launch_{:d}_{:s}'.format(launch, trial.trial_id),
                config=search_space,
                name='HmmGridSearch',
                metric='nll',
                mode='min',
                checkpoint_at_end=True,
                checkpoint_freq=1,
                num_samples=10,
                keep_checkpoints_num=5,
                stop=lambda _, result: result['iter'] >= 20,
                local_dir='/opt/project/data/tune_results/',
                checkpoint_score_attr='min-nll',
                verbose=1
            )
        launch += 1
    # search_space = {
    #     "binning_method": tune.grid_search([const.BINNING_SINGLE, const.BINNING_GEOM,
    #                                         const.BINNING_FREQ, const.BINNING_NONE]),
    #     "num_bins": 50,
    #     "hmm_duration": 10,
    #     "included_packets": const.FLOW_MAIN,
    #     "trace_length": 10,
    #     "seed": tune.randint(1, int(2**32 - 1)),
    #     "company": tune.grid_search(['google', 'amazon', 'facebook', 'wikipedia',
    #                                  'youtube', 'google_drive', 'google_maps']),
    #     "browser": tune.grid_search([const.BROWSER_MOZILLA, const.BROWSER_CHROME,
    #                                  None, const.BROWSER_WGET]),
    #     "max_val_geo_bin": 1500
    # }
    # search_space = {
    #     "binning_method": const.BINNING_NONE,
    #     "num_bins": 50,
    #     "hmm_duration": 15,
    #     "included_packets": const.FLOW_MAIN,
    #     "trace_length": 20,
    #     "seed": tune.randint(1, int(2**32 - 1)),
    #     "company": tune.grid_search(['google', 'amazon', 'facebook', 'wikipedia', 'youtube', 'google_drive', 'google_maps']),
    #     "browser": const.BROWSER_NOT_WGET,
    #     "max_val_geo_bin": 1500,
    #     "trace_changes": True,
    #     "init_prior": "prefer_match"
    # }
    # tune.run(
    #     HmmTrainable,
    #     trial_dirname_creator=lambda trial: 'HmmGridSearchTrainable_launch_{:d}_{:s}'.format(0, trial.trial_id),
    #     config=search_space,
    #     name='HmmTrainingTest',
    #     metric='nll',
    #     mode='min',
    #     checkpoint_at_end=True,
    #     checkpoint_freq=1,
    #     num_samples=1,
    #     keep_checkpoints_num=5,
    #     stop=lambda _, result: result['iter'] >= 20,
    #     local_dir='/opt/project/data/tune_results/',
    #     checkpoint_score_attr='min-nll'
    # )
