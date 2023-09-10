import itertools as itt
import json
import sys

import redis
from scripts.knn_train_eval import Config
import implementation.data_conversion.constants as constants
from implementation.data_conversion.dataprep import get_tags


REDIS_HOST = 'tueilkn-swc06.forschung.lkn.ei.tum.de'


def populate_phmm():
    n_samples = 10
    exp_dir = '/opt/project/data/grid-search-results/'
    seq_lengths = list(range(5, 31, 5))
    hmm_lengths = list(range(5, 31, 5))
    binning_methods = [
        # Done
        # constants.BINNING_SINGLE,
        # constants.BINNING_NONE,

        # pending
        constants.BINNING_EQ_WIDTH,
        constants.BINNING_FREQ,
        constants.BINNING_GEOM,
    ]
    seq_types = [constants.SEQ_TYPE_RECORD, constants.SEQ_TYPE_FRAME]
    # labels = [t for t in get_tags() if t not in ['cloudflare', 'amazon aws',
    #                                            'adult', 'akamai', 'google cloud',
    #                                            'instagram.com', 'cint.com', 'www.youtube.com', 'www.facebook.com']]
    with open('/opt/project/closed-world-labels.json', 'r') as fh:
        labels = json.load(fh)
    redis_db = redis.StrictRedis(host=REDIS_HOST)

    count = 0
    for seq_length, hmm_length in zip(seq_lengths, hmm_lengths):
        for binning_method, s_type in itt.product(binning_methods, seq_types):
            if binning_method == constants.BINNING_NONE:
                n_bins = [None]
            elif binning_method == constants.BINNING_SINGLE:
                n_bins = [1]
            else:
                n_bins = list(range(10, 101, 10))
            for n_bin in n_bins:
                for label in labels:
                    count += 1
                    config = Config(
                        classifier='phmm',
                        binning_method=binning_method,
                        num_bins=n_bin,
                        seq_length=seq_length,
                        seq_element_type=s_type,
                        hmm_length=hmm_length,
                        day_train_start=0,
                        day_train_end=30,
                        knn_num_neighbors=None,
                        hmm_init_prior='uniform',
                        hmm_num_iter=30,
                        ano_density_estimator=None,
                        max_bin_size=1500 if s_type == constants.SEQ_TYPE_FRAME else int(2**14),
                        seed=1,
                        label=label
                    )
                    params = config.to_dict()
                    params['num_samples'] = n_samples
                    params['exp_dir'] = exp_dir
                    redis_db.lpush(REDIS_QUEUE, json.dumps(params))
                print(f"Pushed {count} configs.")


def populate_mc():
    exp_dir = '/opt/project/data/grid-search-results/'
    seq_lengths = list(range(5, 31, 5))
    binning_methods = [
        # Done
        constants.BINNING_SINGLE,
        constants.BINNING_NONE,
        constants.BINNING_EQ_WIDTH,
        constants.BINNING_FREQ,
        constants.BINNING_GEOM,
    ]
    seq_types = [constants.SEQ_TYPE_RECORD, constants.SEQ_TYPE_FRAME]
    with open('/opt/project/closed-world-labels.json', 'r') as fh:
        labels = json.load(fh)
    redis_db = redis.StrictRedis(host=REDIS_HOST)

    count = 0
    for seq_length in seq_lengths:
        for binning_method, s_type in itt.product(binning_methods, seq_types):
            if binning_method == constants.BINNING_NONE:
                n_bins = [None]
            elif binning_method == constants.BINNING_SINGLE:
                n_bins = [1]
            else:
                n_bins = list(range(10, 101, 10))
            for n_bin in n_bins:
                for label in labels:
                    count += 1
                    config = Config(
                        classifier='mc',
                        binning_method=binning_method,
                        num_bins=n_bin,
                        seq_length=seq_length,
                        seq_element_type=s_type,
                        hmm_length=None,
                        day_train_start=0,
                        day_train_end=30,
                        knn_num_neighbors=None,
                        hmm_init_prior=None,
                        hmm_num_iter=None,
                        ano_density_estimator=None,
                        max_bin_size=1500 if s_type == constants.SEQ_TYPE_FRAME else int(2**14),
                        seed=1,
                        label=label
                    )
                    params = config.to_dict()
                    params['num_samples'] = 1
                    params['exp_dir'] = exp_dir
                    redis_db.lpush(REDIS_QUEUE, json.dumps(params))
                print(f"Pushed {count} configs.")


def populate_knn():
    exp_dir = '/opt/project/data/grid-search-results/'
    seq_lengths = list(range(30, 0, -5))
    binning_methods = [
        constants.BINNING_NONE,
        constants.BINNING_SINGLE,
        constants.BINNING_EQ_WIDTH,
        constants.BINNING_FREQ,
        constants.BINNING_GEOM,
    ]
    seq_types = [constants.SEQ_TYPE_RECORD, constants.SEQ_TYPE_FRAME]
    num_neighbors = [9, 6, 3]

    redis_db = redis.StrictRedis(host=REDIS_HOST)
    count = 0
    for nns in num_neighbors:
        for seq_length in seq_lengths:
            for binning_method, s_type in itt.product(binning_methods, seq_types):
                if binning_method == constants.BINNING_NONE:
                    n_bins = [None]
                elif binning_method == constants.BINNING_SINGLE:
                    n_bins = [1]
                else:
                    n_bins = list(range(10, 101, 10))
                for n_bin in n_bins:
                    count += 1
                    config = Config(
                        classifier='knn',
                        binning_method=binning_method,
                        num_bins=n_bin,
                        seq_length=seq_length,
                        seq_element_type=s_type,
                        hmm_length=None,
                        day_train_start=0,
                        day_train_end=30,
                        knn_num_neighbors=nns,
                        hmm_init_prior=None,
                        hmm_num_iter=None,
                        ano_density_estimator=None,
                        max_bin_size=1500 if s_type == constants.SEQ_TYPE_FRAME else int(2**14),
                        seed=1,
                        label=None
                    )
                    params = config.to_dict()
                    params['num_samples'] = 1
                    params['exp_dir'] = exp_dir
                    redis_db.lpush(REDIS_QUEUE, json.dumps(params))
                    print(f"Pushed {count} configs.")


if __name__ == '__main__':
    if sys.argv[1] == 'phmm_+':
        REDIS_QUEUE = 'gridsearch_pgm'
        populate_phmm()
    elif sys.argv[1] == 'mc':
        REDIS_QUEUE = 'gridsearch_pgm'
        populate_mc()
    elif sys.argv[1] == 'knn_':
        REDIS_QUEUE = 'gridsearch'
        populate_knn()
    else:
        raise KeyError(f"Unknown model type {sys.argv[1]}")