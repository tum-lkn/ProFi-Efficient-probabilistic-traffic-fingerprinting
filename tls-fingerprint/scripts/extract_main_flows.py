import multiprocessing as mp
import time
import logging
import queue
import sqlalchemy
import os
import shutil
import json
import traceback


import implementation.data_conversion.tls_flow_extraction as tlsex


if __name__ == '__main__':
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:FKmk2kRzWFFdX8@pk-swc01.forschung.lkn.ei.tum.de:3306/gatherer_upscaled'
    pcap_base_path = '/k8s-traces/'
    json_base_path = '/k8s-json/'
    tmp_pcap_base_path = '/tmp'

    def worker():
        name = mp.current_process().name
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f"/opt/project/data/{name}.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info('Starting')
        tmp_pcap_path = 'does not exist'
        while not work_queue.empty() and not exit_gracefully.is_set():
            try:
                url, filename = work_queue.get_nowait()
                pcap_path = os.path.join(pcap_base_path, f'{filename}.pcapng')
                tmp_pcap_path = os.path.join(tmp_pcap_base_path, f'{filename}.pcapng')
                json_path = os.path.join(json_base_path, f'{filename}.json')
                hostname = tlsex.get_hostname(url)
                logger.info(f"Extract mainflow for url {url} with hostname {hostname}, {pcap_path} -> {json_path}")
                if not os.path.exists(pcap_path):
                    logger.error(f"Trace {pcap_path} does not exist")
                    continue
                elif os.path.exists(json_path):
                    logger.info(f"Trace {pcap_path} already converted: {json_path} exists.")
                    continue
                else:
                    shutil.copy(pcap_path, tmp_pcap_path)
                    flow = tlsex.extract_tls_records(tmp_pcap_path, url, f'/tmp/{name}.bin')
                    with open(json_path, 'w') as fh:
                        json.dump(flow.to_dict(), fh)
            except queue.Empty:
                logger.info("Queue Empty, prepare to close.")
            except AssertionError as e:
                logger.error("Assertion Error occurred.")
                logger.exception((e, tb, name, f'{filename}.pcapng'))
            except Exception as e:
                logger.error("Unexpected exception occurred.")
                tb = traceback.format_exc()
                error_queue.put((e, tb, name, f'{filename}.pcapng'))
                logger.exception(e)
            finally:
                if os.path.exists(tmp_pcap_path):
                    os.remove(tmp_pcap_path)
        logger.info('Exiting')

    num_workers = 30
    workers = []
    work_queue = mp.Queue()
    error_queue = mp.Queue()
    exit_gracefully = mp.Event()

    logger = logging.getLogger('root-logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"/opt/project/data/root.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    error_logger = logging.getLogger('errorLogger')
    error_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"/opt/project/data/errors.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    error_logger.addHandler(fh)

    logger.info("Retrieve data from Database.")
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    t1 = time.perf_counter()
    with engine.connect() as connection:
        query = "SELECT urls.url, traces_metadata.filename FROM " + \
                "gatherer_upscaled.traces_metadata INNER JOIN urls " + \
                "ON urls.id = traces_metadata.url"
        data = connection.execute(query).fetchall()
    t2 = time.perf_counter()
    logger.info(f"Retrieved {len(data)} URLs - took {t2-t1}s.")

    logger.info(f"Fill Queue.")
    for i, (url, filename) in enumerate(data):
        work_queue.put((url, filename))
        if i > 1e9:
            break

    logger.info("Start workers")
    for i in range(num_workers):
        w = mp.Process(name=f'worker-{i:03d}', target=worker)
        w.start()
        workers.append(w)

    t1 = time.perf_counter()
    num_errors = 0
    error_info = []
    while not work_queue.empty() and not exit_gracefully.is_set():
        try:
            finished = 100 * (1. - work_queue.qsize() / float(len(data)))
            took = time.perf_counter() - t1
            p_err = num_errors / float(len(data)) * 100
            logger.info(f"Finished {finished:.2f} % {took}s, {num_errors} errors {p_err:.2f} %")
            while not error_queue.empty():
                try:
                    error, tb, worker, pcap = error_queue.get()
                    num_errors += 1
                    error_logger.error(f"Error in worker {worker} during processing trace {pcap}:")
                    error_logger.error(error)
                    error_logger.error(tb)
                    if type(error) == tlsex.TraceFailure:
                        error_info.append(f'{error.url};{error.reason}')
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error("Unexpected failure occured")
                    logger.exception(e)
            time.sleep(60)
            if len(error_info) > 0:
                try:
                    with open('error-info.csv', 'w') as fh:
                        fh.write(os.linesep.join(error_info))
                except Exception as e:
                    logger.error("Could not write error-info.")
        except KeyboardInterrupt:
            logger.info("Received Keyboard Interrupt, exit gracefully.")
            exit_gracefully.set()
    logger.info("Wait for workers to exit.")
    for w in workers:
        w.join()
