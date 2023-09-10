import logging
import time

import docker
import uuid
import sys
import psutil
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    hostname = sys.argv[1]
    expected_num_container = int(sys.argv[2])
    script = {
        'train': 'redis_worker.py',
        'eval': 'result_updater.py',
        'train_knn': 'knn_train_eval.py redis',
        'pgm_time': 'evaluate_multi_binary.py redis'
    }[sys.argv[3]]
    name_part = {'train': 'tune-pgm', 'eval': 'eval-pgm', 'train_knn': 'tune-knn'}[sys.argv[3]]
    env = docker.from_env()
    n_cores = psutil.cpu_count()
    counter = 0

    while True:
        running_container = None
        num_fails = 0
        while num_fails < 10 and running_container is None:
            try:
                running_container = env.containers.list()
            except Exception as e:
                time.sleep(1)
                num_fails += 1
                if num_fails >= 10:
                    logging.exception(e)
                    raise e
        assert num_fails < 10

        num_workers = len([container.name for container in running_container if container.name.startswith('grid-search-worker-')])
        if num_workers < expected_num_container:
            time.sleep(1)
            logging.info(f"Detected {num_workers} instead of {expected_num_container}. Start missing...")
            for _ in range(expected_num_container - num_workers):
                try:
                    cuuid = uuid.uuid4().hex
                    cid = f'{counter % expected_num_container}'
                    core = counter % n_cores
                    name = f'{name_part}-worker-{cuuid}'
                    env.containers.run(
                        image='gitlab.lrz.de:5005/swc/tls-fingerprint:latest',
                        command=f'python3 /opt/project/scripts/{script}',
                        # cpuset_cpus=f'{core}',
                        detach=True,
                        environment={
                            "POD_NAME": f"{hostname}-worker-{cid}",
                            "PYTHONPATH": "/opt/project"
                        },
                        name=name,
                        remove=True,
                        volumes={
                            '/mnt/code/tls-fingerprint': {'bind': '/opt/project', 'mode': 'rw'},
                            '/mnt/cache': {'bind': '/opt/project/data/cache', 'mode': 'rw'},
                            '/mnt/nfs/k8s-grid-search/': {'bind': '/opt/project/data/grid-search-results', 'mode': 'rw'}
                        }
                    )
                    counter += 1
                    print(f"Started Container {name} on core {core}")
                except Exception as e:
                    logging.exception(e)


