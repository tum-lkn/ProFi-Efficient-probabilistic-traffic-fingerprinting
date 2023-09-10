import json
import os
import time
import sys
import logging
import gc
from typing import List, Dict, Tuple, Any
print(sys.path)
sys.path.insert(0, '/opt/project')

import implementation.rediswq as rediswq
import implementation.logging_factory as logging_factory
from implementation.phmm_np.grid_search import evaluate_model
import implementation.data_conversion.constants as constants

assert os.path.exists('/opt/project')
assert os.path.exists('/opt/project/data')
assert os.path.exists('/opt/project/data/grid-search-results')
assert os.path.exists('/opt/project/data/grid-search-results/logs')

POD_NAME = os.environ.get("POD_NAME")
if POD_NAME is None:
    POD_NAME = 'default'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filemode='w',
    filename=logging_factory.get_log_file(
        name=POD_NAME,
        log_dir='/opt/project/data/grid-search-results/logs'
    ),
    level=logging.INFO
)


def worker_call(trial_dir: str, closed_world_labels: List[str],
                open_world_labels: List[str]) -> None:
    seq_lengths = list(range(5, 31, 5))
    hmm_lengths = list(range(5, 31, 5))
    n_bins = list(range(10, 101, 10))
    n_bins.append(0)
    n_bins.append(None)

    config_path = os.path.join(trial_dir, 'config.json')
    model_path = os.path.join(trial_dir, 'model.json')
    perf_path = os.path.join(trial_dir, 'perf.h5')
    if os.path.exists(perf_path):
        logger.info(f"Perf results exist already in {trial_dir}.")
        return
    if not os.path.exists(config_path):
        logger.info(f"{config_path} does not exist.")
        return
    if not os.path.exists(model_path):
        logger.info(f"{model_path} does not exist.")
        return
    with open(config_path, "r") as fh:
        config = json.load(fh)
    if config['label'] in closed_world_labels:
        pass
    else:
        return

    if 'tp' in config:
        logger.info(f"Already updated trial {trial_dir}.")
        return
    if config['num_bins'] not in n_bins:
        logger.info(f"Trial {trial_dir} not in configured hyperparams.")
        return
    elif config['classifier'] != 'phmm':
        logger.info(f"Wrong classifier for trial {trial_dir}.")
        return
    # elif config['seq_length'] not in seq_lengths:
    #     logger.info(f"Trial {trial_dir} not in configured hyperparams.")
    #     return
    # elif config['hmm_length'] not in hmm_lengths:
    #     logger.info(f"Trial {trial_dir} not in configured hyperparams.")
    #     return
    # elif config['hmm_length'] != config['seq_length']:
    #     logger.info(f"Trial {trial_dir} not in configured hyperparams.")
    #     return
    else:
        pass

    with open(model_path, "r") as fh:
        model_d = json.load(fh)
    model_d['init_prior'] = config['hmm_init_prior']
    model_d['label'] = config['label']
    model_d['num_iter'] = config['hmm_num_iter']

    try:
        updates = evaluate_model(
            config=config,
            model_params=model_d,
            closed_world_labels=closed_world_labels,
            logger=logger
        )
        config.update(updates)
        with open(config_path, "w") as fh:
            json.dump(config, fh)
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    logger = logging_factory.produce_logger(
        name=f'{POD_NAME}',
        log_dir='/opt/project/data/grid-search-results/logs'
    )

    # closed_world_labels = [
    #     'sxyprn.com',
    #     'www.kompas.com',
    #     'txxx.com',
    #     'www.bola.net'
    # ]
    # trial_dir = '/opt/project/data/grid-search-results/devel-model'
    # worker_call(trial_dir, closed_world_labels)

    wait = 10
    reported_queue_empty = False
    with open("/opt/project/closed-world-labels.json", "r") as fh:
        closed_world_labels = json.load(fh)
    with open("/opt/project/open-world-labels.json", "r") as fh:
        open_world_labels = json.load(fh)

    redis_q = rediswq.RedisWQ(name='modelperf', host='tueilkn-swc06.forschung.lkn.ei.tum.de')
    count = 0
    while count < 50:
        try:
            if redis_q.empty():
                if not reported_queue_empty:
                    logger.info(f"Queue empty, wait for new jobs...")
                    reported_queue_empty = True
                time.sleep(wait)
            else:
                logger.debug("Try to retrieve item.")
                reported_queue_empty = False
                item = redis_q.lease(lease_secs=30, block=True, timeout=2)
                if item is not None:
                    logger.debug("Item retrieved, load config.")
                    redis_q.complete(item)
                    trial_dir = item.decode('utf-8')
                    count += 1
                    sep = '\n\t====================================================='
                    logger.info(f"{sep}\n\tEvaluate Model in: {trial_dir}{sep}")
                    t1 = time.perf_counter()
                    worker_call(trial_dir, closed_world_labels, open_world_labels)
                    if time.perf_counter() - t1 < 1:
                        count -= 1
                    logger.info(f"------> Took {time.perf_counter() - t1}s\n")
                    gc.collect()
        except Exception as e:
            logger.error("Unexpected error occured.")
            logger.exception(e)
    logger.info("Stop after evaluating 100 trained HMMs.")
