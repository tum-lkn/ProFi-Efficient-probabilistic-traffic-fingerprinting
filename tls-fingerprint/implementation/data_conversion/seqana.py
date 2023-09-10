"""
Module used to evaluate sequences.
"""
import pandas as pd
import numpy as np
import logging
import os
import json
import sqlalchemy
import sys
from typing import Dict, List, Tuple, Any, Union, ClassVar
import matplotlib.pyplot as plt
import plots.utils as plutils
import Levenshtein as leven
import itertools as itt
import time
import multiprocessing as mp


import implementation.data_conversion.tls_flow_extraction as tlsex
import implementation.data_conversion.dataprep as dprep


logger = logging.getLogger('seqana')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

with open("CONFIG.json", 'r') as fh:
    CONFIG = json.laod(fh)

COLORS = ['#80b1d3', '#fb8072', '#bebada', '#fdb462', '#8dd3c7']
SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://root:{CONFIG["db_password"]}' + \
                          f'{CONFIG["db_host"]}:3306/gatherer_upscaled'


BIN_EDGES = np.array([0.])
BIN_EDGES = None


def load_sequences(tag: str, seq_element_type: ClassVar, seq_length: int) -> Tuple[List[List[str]], List[str]]:
    """
    Load main flows and convert them to sequences.

    Args:
        tag:
        seq_element_type:
        seq_length:

    Returns:
        sequences: List of List of strings representing sequqences.
        obs_space: Unique symbols over all sequences.
    """
    meta = dprep.get_db_metadata_for_tag(tag)
    keys = np.array(list(meta.keys()))
    np.random.shuffle(keys)
    num_flows = 300
    obs_space = np.array([], dtype=object)
    sequences = []
    count = 0
    for filename in keys:
        if os.path.exists(f'/k8s-json/{filename}.json'):
            if count >= num_flows:
                break
            count += 1
            main_flow = dprep.make_main_flows([dprep.load_flow_dict(f'{filename}.json')])[filename]
            seq = tlsex.main_flow_to_symbol(
                main_flow=main_flow,
                seq_length=seq_length,
                to_symbolize=seq_element_type,
                bin_edges=BIN_EDGES
            )
            sequences.append(seq)
            obs_space = np.unique(np.concatenate([obs_space, np.array(seq)]))
        else:
            meta.pop(filename)
    return sequences, [str(x) for x in obs_space]


def load_sequences_mp(args) -> Tuple[str, List[List[str]], List[str]]:
    element_types = {
        'record': tlsex.TlsRecord,
        'frame': tlsex.Frame
    }
    seqs, obs_space = load_sequences(args['tag'], element_types[args['seq_element_type']], args['seq_length'])
    return args['tag'], seqs, obs_space


def calc_similarity(tag_one: str, tag_two: str, seqs_tag_one: List[List[str]],
                    seqs_tag_two: List[List[str]], obs_space: List[str]) -> Dict[str, Union[str, List[float]]]:
    def to_unicode(seqs: List[List[str]]) -> List[str]:
        seqs_u = []
        for i in range(len(seqs)):
            seqs_u.append(''.join([symbol_to_unicode[s] for s in seqs[i]]))
        return seqs_u

    symbol_to_unicode = {k: chr(i) for i, k in enumerate(obs_space)}
    seqs_tag_one = to_unicode(seqs_tag_one)
    seqs_tag_two = seqs_tag_one if tag_one == tag_two else to_unicode(seqs_tag_two)

    print(f"Calculate similarities for {tag_one} and {tag_two}.")
    t1 = time.perf_counter()
    similarities = []
    for i in range(len(seqs_tag_one)):
        begin = i + 1 if tag_one == tag_two else 0
        for j in range(begin, len(seqs_tag_two)):
            similarities.append(leven.distance(seqs_tag_one[i], seqs_tag_two[j]))
    t = time.perf_counter() - t1
    print(f"Calculation of similarities for {tag_one} and {tag_two} took {t}s.")
    return {'tag_one': tag_one, 'tag_two': tag_two, 'similarities': similarities}


def calc_similarity_mp(args) -> Dict[str, Union[str, List[float]]]:
    values = calc_similarity(**args)
    tag_one = args['tag_one']
    tag_two = args['tag_two']
    d = {
        'tag_one': tag_one,
        'tag_two': tag_two,
        'min': float(np.min(values)) if len(values) > 0 else None,
        'p05': float(np.percentile(values, 5)) if len(values) > 0 else None,
        'mean': float(np.mean(values)) if len(values) > 0 else None,
        'median': float(np.median(values)) if len(values) > 0 else None,
        'p95': float(np.percentile(values, 95)) if len(values) > 0 else None,
        'max': float(np.max(values)) if len(values) > 0 else None
    }
    t1 = tag_one.replace('.', '_')
    t2 = tag_two.replace('.', '_')
    with open(f'./data/tmp-res/{t1}-{t2}.json', 'w') as fh:
        json.dump(d, fh)
    return d


def similarity_tags(tag_one: str, tag_two: str, seq_element_type: str,
                    seq_length: int) -> Dict[str, Union[str, List[float]]]:
    """
    Compare the Levensthein distance of the traces behind two tags.
    Args:
        tag_one:
        tag_two:
        seq_element_type: Must be in {record, frame}.
        seq_length

    Returns:

    """
    element_types = {
        'record': tlsex.TlsRecord,
        'frame': tlsex.Frame
    }
    t1 = time.perf_counter()
    print(f"Load sequences for tag {tag_one}")
    seqs_tag_one, obs_space = load_sequences(
        tag_one,
        element_types[seq_element_type],
        seq_length
    )
    t = time.perf_counter() - t1
    print(f"Took {t} s\nLoad sequences for tag {tag_two}")
    t1 = time.perf_counter()
    if tag_one == tag_two:
        seqs_tag_two = seqs_tag_one
    else:
        seqs_tag_two, obs_space_ = load_sequences(
            tag_two,
            element_types[seq_element_type],
            seq_length
        )
        obs_space = np.unique(np.concatenate([np.array(obs_space), np.array(obs_space_)]))
    t = time.perf_counter() - t1
    print(f"Loading {tag_two} Took {t}s")
    return calc_similarity(tag_one, tag_two, seqs_tag_one, seqs_tag_two, obs_space)


def similarity_tags_aggregated(tag_one: str, tag_two: str, seq_element_type: str,
                               seq_length: int) -> Dict[str, Union[str, float]]:
    values = np.sort(similarity_tags(tag_one, tag_two, seq_element_type, seq_length)['similarities'])
    d = {
        'tag_one': tag_one,
        'tag_two': tag_two,
        'min': float(np.min(values)),
        'p05': float(np.percentile(values, 5)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'max': float(np.max(values))
    }
    t1 = tag_one.replace('.', '_')
    t2 = tag_two.replace('.', '_')
    with open(f'./data/tmp-res/{t1}-{t2}.json', 'w') as fh:
        json.dump(d, fh)
    return d


def similarity_tags_aggregated_mp(args: Dict[str, Union[str, int]]) -> Dict[str, Union[str, float]]:
    num = args.pop('num')
    print("===============================")
    print(f"        Do {num}              ")
    print("===============================")
    return similarity_tags_aggregated(**args)


def analyze_tags(seq_element_type: str, seq_length: int):
    tags = dprep.get_tags()
    tags = [t for t in tags if t not in ['cloudflare', 'amazon aws', 'adult', 'akamai', 'google cloud', '']]

    p = mp.Pool(processes=30)
    summaries = p.map(
        similarity_tags_aggregated_mp,
        [
            {
                'num': i,
                'tag_one': t1,
                'tag_two': t2,
                'seq_element_type': seq_element_type,
                'seq_length': seq_length
            }
        for i, (t1, t2) in enumerate(itt.product(tags, tags))])
    p.close()
    df = pd.DataFrame.from_dict(summaries)
    df.to_hdf(f'./data/seq-summaries-one-bin-{seq_element_type}-len-{seq_length}.h5', key='seq-summaries')


def analyze_tags2(seq_element_type: str, seq_length: int):
    tags = dprep.get_tags()
    tags = [t for t in tags if t not in ['cloudflare', 'amazon aws', 'adult', 'akamai', 'google cloud', '']]
    t1 = time.time()
    pool = mp.Pool(30)
    vals = pool.map(
        load_sequences_mp,
        [{
            'tag': t,
            'seq_element_type': seq_element_type,
            'seq_length': seq_length
        } for t in tags]
    )
    pool.close()
    sequences = {}
    for tag, seqs, obs_space in vals:
        sequences[tag] = {'seqs': seqs, 'obs_space': obs_space}
    t = time.time() - t1
    print(f"\n\nFinished loading files after {t}s.\n\n")
    # pool = mp.Pool(30)
    # vals = pool.map(
    #     calc_similarity_mp,
    #     [{
    #         'tag_one': t1,
    #         'tag_two': t2,
    #         'seqs_tag_one': sequences[t1]['seqs'],
    #         'seqs_tag_two': sequences[t2]['seqs'],
    #         'obs_space': [str(x) for x in np.unique(np.concatenate((sequences[t1]['obs_space'], sequences[t2]['obs_space'])))]
    #     } for t1, t2 in itt.product(tags)]
    # )
    # pool.close()
    vals = []
    x = len(tags) ** 2
    ts1 = time.time()
    for i, (t1, t2) in enumerate(itt.product(tags, tags)):
        if i % 100 == 0:
            ts2 = time.time()
            t = ts2 - ts1
            ts1 = ts2
            print(f"Finished {i:4d} of {x}: {i/x*100:.2f}% in {t}s")
        if len(sequences[t1]['seqs']) == 0 or len(sequences[t2]['seqs']) == 0:
            continue
        vals.append(calc_similarity(
            tag_one=t1,
            tag_two=t2,
            seqs_tag_one=sequences[t1]['seqs'],
            seqs_tag_two=sequences[t2]['seqs'],
            obs_space=[str(x) for x in np.unique(np.concatenate((sequences[t1]['obs_space'], sequences[t2]['obs_space'])))]
        ))
    df = pd.DataFrame.from_dict(vals)
    df.to_hdf(f'/results/seq-summaries-no-bin-{seq_element_type}-len-{seq_length}.h5', key='seq-summaries')


if __name__ == '__main__':
    # analyze_tags2('record', 30)
    # analyze_tags2('frame', 30)

    analyze_tags2('record', 25)
    analyze_tags2('frame', 25)

    analyze_tags2('record', 20)
    analyze_tags2('frame', 20)

    analyze_tags2('record', 15)
    analyze_tags2('frame', 15)

    analyze_tags2('record', 10)
    analyze_tags2('frame', 10)
