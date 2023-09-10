import logging
import multiprocessing as mp
import pandas as pd

import implementation.data_conversion.cumul as dpcumul

logger = logging.getLogger('script-logger')
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())


if __name__ == '__main__':
    logger.info("Get metadata")
    meta_data = dpcumul.get_metadata()
    args_list = [[] for _ in range(mp.cpu_count() - 1)]
    logger.info("Map to jobs")
    for i, args in enumerate(meta_data):
        args_list[i % len(args_list)].append(args)
    logger.info("Extract Features")
    pool = mp.Pool(processes=64)
    data = pool.map(dpcumul.mp_make_dataset, args_list)
    pool.close()

    linear_data = []
    total_fails = 0
    for x, n_fails in data:
        linear_data.extend(x)
        total_fails += n_fails
        logger.info(f"No features or errored for {n_fails}")
    logger.info(f"No features or errored for {total_fails}")
    logger.info("Save Features")
    df = pd.DataFrame(linear_data)
    df.to_hdf('/opt/project/data/cumul-interpolations/n100.h5', key='interpolation')
