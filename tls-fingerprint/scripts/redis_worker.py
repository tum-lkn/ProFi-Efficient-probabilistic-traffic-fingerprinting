import json
import os
import time
import sys
import logging
print(sys.path)
sys.path.insert(0, '/opt/project')

import implementation.rediswq as rediswq
import implementation.logging_factory as logging_factory
from implementation.phmm_np.grid_search import train_phmm_model
from scripts.knn_train_eval import run_redis as train_knn_model
import gc

assert os.path.exists('/opt/project')
assert os.path.exists('/opt/project/data')
assert os.path.exists('/opt/project/data/grid-search-results')
assert os.path.exists('/opt/project/data/grid-search-results/logs')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filemode='w',
    filename=logging_factory.get_log_file(
        name=f'{os.environ.get("POD_NAME")}',
        log_dir='/opt/project/data/grid-search-results/logs'
    ),
    level=logging.INFO
)


if __name__ == '__main__':
    pod_name = os.environ.get('POD_NAME')
    wait = 10
    reported_queue_empty = False
    logger = logging_factory.produce_logger(
        name=f'{pod_name}',
        log_dir='/opt/project/data/grid-search-results/logs'
    )
    with open("/opt/project/closed-world-labels.json", "r") as fh:
        closed_world_labels = json.load(fh)

    redis_q = rediswq.RedisWQ(name='gridsearch_pgm', host='tueilkn-swc06.forschung.lkn.ei.tum.de')
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
                    config = json.loads(item.decode("utf-8"))
                    count += 1
                    msg = f"{config['classifier']}-{config['seq_element_type']}-" +\
                          f"{config['binning_method']}-{config['num_bins']}-" +\
                          f"{config['seq_length']}-{config['hmm_length']}"
                    sep = '\n\t====================================================='
                    logger.info(f"{sep}\n\tTrain new model: {msg}{sep}")
                    t1 = time.perf_counter()
                    if config['classifier'] in ['phmm', 'mc']:
                        train_phmm_model(config, closed_world_labels, logger)
                    elif config['classifier'] == 'knn':
                        train_knn_model(logger)
                    logger.info(f"Trained model number {count} {msg} in {time.perf_counter() - t1}s")
                    gc.collect()
        except Exception as e:
            logger.error("Unexpected error occured.")
            logger.exception(e)
    logger.info("Stop after 100 trained HMMs.")
