import logging
import multiprocessing as mp

import implementation.data_conversion.cumul as dpcumul

logger = logging.getLogger('base-logger')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


if __name__ == '__main__':
    logger.info("Get metadata")
    meta_data = dpcumul.get_metadata()
    args_list = [[] for _ in range(mp.cpu_count() - 1)]
    logger.info("Map to jobs")
    for i, args in enumerate(meta_data):
        args_list[i % len(args_list)].append(args)
    logger.info("Extract Features")
    pool = mp.Pool()
    data = pool.map(dpcumul.mp_driver, args_list)
    pool.close()
