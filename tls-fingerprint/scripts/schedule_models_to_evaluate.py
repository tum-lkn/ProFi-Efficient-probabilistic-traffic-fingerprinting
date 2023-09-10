import os
import redis


REDIS_HOST = 'tueilkn-swc06.forschung.lkn.ei.tum.de'
REDIS_QUEUE = 'modelperf'


def populate():
    exp_dir = '/opt/project/data/grid-search-results/'
    redis_db = redis.StrictRedis(host=REDIS_HOST)

    count = 0
    for trial_dir_name in os.listdir(exp_dir):
        count += 1
        trial_dir = os.path.join(exp_dir, trial_dir_name)
        redis_db.lpush(REDIS_QUEUE, trial_dir)
        if count % 100 == 0:
            print(f"Scheduled {count:6d} trials.")


if __name__ == '__main__':
    populate()