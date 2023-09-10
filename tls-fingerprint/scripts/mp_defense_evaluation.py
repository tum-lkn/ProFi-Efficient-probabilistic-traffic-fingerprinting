import multiprocessing as mp
from typing import Dict, Any, List, Tuple
import logging

import scripts.evaluate_multi_binary as emb


def run_trial(parsed_args: Dict[str, Any]) -> None:
    emb.run_from_cmd(
        model=parsed_args['model'],
        scenario=parsed_args['scenario'],
        train_for_days=parsed_args['train_for_days'],
        defense=parsed_args['defense'],
        # direction is indicated with -1/1 in the data, thus cast settings
        # to this range. If value is not in -1/1, then no filtering is applied.
        direction_to_filter=parsed_args['direction_to_filter']
    )


if __name__ == '__main__':
    logger = logging.getLogger('eval-defense')
    logger.addHandler(logging.StreamHandler())
    configs = []
    max_sizes = list(range(100, 2**14, 1000))
    max_sizes.append(2**14)
    for max_size in max_sizes:
        for scenario in ['closed', 'open']:
            for model in ['mc', 'phmm']:
                configs.append({
                    'model': model,
                    'scenario': scenario,
                    'train_for_days': 70,
                    'defense': {
                        'name': 'RandomRecordSizeDefense',
                        'min_record_size': 50,
                        'max_record_size': max_size,
                        'seed': 1
                    },
                    'direction_to_filter': 0
                })
    for direction in [-1, 1]:
        for scenario in ['closed', 'open']:
            for model in ['mc', 'phmm']:
                configs.append({
                    'model': model,
                    'scenario': scenario,
                    'train_for_days': 70,
                    'defense': None,
                    'direction_to_filter': direction
                })
    pool = mp.Pool(30)
    try:
        pool.map(run_trial, configs)
    except Exception as e:
        logger.exception(e)
    finally:
        pool.close()
