"""

The format of the file naming is <label>_(train|val).json.
"""

import json
import os
import sys
from datetime import datetime
import multiprocessing as mp
from typing import List, Dict, Tuple, Any, Union
import logging
import uuid

import implementation.data_conversion.dataprep as dprep
from implementation.seqcache import is_cached, read_cache, add_to_cache
from implementation.logging_factory import produce_logger



logger = produce_logger('create-dsets')


def load_flow_dicts(dset: str, meta_data_day: List[Dict[str, Any]],
                    day_index: int, indicator: Dict[str, str],
                    labels: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str], int]:
    X = []
    y = []
    skipped = 0
    total = 0
    local_logger = logging.getLogger(uuid.uuid4().hex)
    local_logger.addHandler(logging.StreamHandler(sys.stdout))
    local_logger.setLevel(logging.DEBUG)
    local_logger.info(f"Extract {len(meta_data_day)} items.")
    for i, data in enumerate(meta_data_day):
        if int(total / len(meta_data_day) * 100) % 5 == 0:
            perc = total / len(meta_data_day) * 100
            local_logger.info(f"Processed {total} items ({perc}%%), {len(X)} with success, skipped {skipped}")
        total += 1
        url_id = str(data['url_id'])
        if url_id not in indicator or url_id not in labels:
            local_logger.debug("URL ID does not exist in indicator or labels")
            skipped += 1
            continue
        if indicator[url_id] != dset:
            local_logger.debug('URL Id not in correct data set')
            skipped += 1
            continue
        main_flow = dprep.load_flow_dict(f"{data['filename']}.json")
        if main_flow is None:
            local_logger.debug("Main FLow retrieval failed")
            skipped += 1
        elif len(main_flow['frames']) < 6:
            local_logger.debug(f"Main Flow has less than 10 frames: {len(main_flow['frames'])}")
            skipped += 1
        else:
            reduced_main_flow = {
                'frames': main_flow['frames'][:35]
            }
            X.append(reduced_main_flow)
            y.append(labels[url_id])
    return X, y, day_index


def load_flow_dicts_mp(args) -> Tuple[List[Dict[str, Any]], List[str], int]:
    return load_flow_dicts(
        dset=args[0],
        meta_data_day=args[1],
        day_index=args[2],
        indicator=args[3],
        labels=args[4]
    )


def make_train_val(setname: str) -> Dict[str, Dict[int, List[str]]]:
    logger.info(f"Create {setname} data set from raw data")
    pool = mp.Pool(30)
    rets = pool.map(
        load_flow_dicts_mp,
        [(setname, m, i, indicator, labels) for i, m in enumerate(meta_data)]
    )
    # rets.sort(key=lambda x: x[2])
    pool.close()

    all_labels = {}
    for x, y, day in rets:
        for flow, lbl in zip(x, y):
            if lbl not in all_labels:
                all_labels[lbl] = {}
            if day not in all_labels[lbl]:
                all_labels[lbl][day] = []
            all_labels[lbl][day].append(flow)
    return all_labels


if __name__ == '__main__':

    train_start_day = 0
    train_end_day = 35

    if is_cached('indicator.json'):
        indicator = read_cache('indicator.json')
        labels = read_cache('labels.json')
    else:
        logger.info("Retrieve datasets")
        # Create indicator variables for datasets and the labels.
        _, indicator, labels = dprep.create_data_sets()
        add_to_cache('indicator.json', indicator)
        add_to_cache('labels.json', labels)

    if is_cached('meta_data.json'):
        meta_data = read_cache('meta_data.json')
        meta_data = [(datetime.strptime(a, '%Y-%m-%d %H:%M:%S'), b) for a, b in meta_data]
    else:
        logger.info("Retrieve metadata")
        logger.info("Retrieve days")
        days = dprep.get_days()
        logger.info(f"Retrieved {len(days)} days")
        pool = mp.Pool(30)
        meta_data = pool.map(dprep.get_days_metadata, days)
        pool.close()
        meta_data.sort(key=lambda x: x[0])
        add_to_cache('meta_data.json', [(str(a), b) for a, b in meta_data])
    logger.info("Serialize meta data")
    meta_data = [m for _, m in meta_data]

    all_labels = make_train_val('train')
    logger.info("Created x_train, write to file.")
    for label, data in all_labels.items():
        add_to_cache(f"{label}_train.json", data)

    all_labels = make_train_val('val')
    logger.info("Created x_val, write to file.")
    for label, data in all_labels.items():
        add_to_cache(f"{label}_val.json", data)

    all_labels = make_train_val('test')
    logger.info("Created x_test, write to file.")
    for label, data in all_labels.items():
        add_to_cache(f"{label}_test.json", data)
