import time
from typing import List, Tuple, Dict, Any
import ctypes
import numpy as np
import multiprocessing as mp
import gc
import queue

from implementation.classification.seq_classifier import SeqClassifier
from baum_welch.phmm import PhmmC, Hmm, basic_phmm, basic_phmm_c
from baum_welch.learning import baum_welch as baum_welch, calc_log_prob
import implementation.data_conversion.data_aggregation_hmm_np as dahn
import implementation.Phmm_CPhmm as cphmm
import implementation.logging_factory as logfac

NPROCS = 4

_baum_welch, _baum_welch_step, _free = cphmm.load_clib()
_c_log_prob, _free_single = cphmm.load_logprob2()


class Phmm(object):

    @classmethod
    def from_dict(cls, dc):
        wrapper = Phmm(
            duration=dc['duration'],
            init_prior=dc['init_prior'],
            seed=dc['seed'],
            label=dc['label'],
            num_iter=dc['label']
        )
        wrapper.phmm_p = Hmm.from_dict(dc)
        return wrapper

    def __init__(self, duration: int, init_prior: str,
                 seed: int, label: str, num_iter: int, **kwargs):
        self.label = label
        self.num_iter = num_iter
        self.log_prob_seq = []
        self.phmm_p = None
        self.hmm_seq = []
        self.duration = duration
        self.init_prior = init_prior
        self.seed = seed

    def fit(self, X: List[List[str]]) -> 'Phmm':
        observation_space = [str(e) for e in np.unique(np.concatenate([np.unique(x) for x in X]))]
        hmm = basic_phmm(
            duration=self.duration,
            observation_space=observation_space,
            init_prior=self.init_prior,
            seed=self.seed
        )
        best_hmm = None
        best_log_prob = 1e9
        for i in range(self.num_iter):
            hmm, _, log_probs = baum_welch(hmm, X)
            avg_log_prob = -1. * np.mean(log_probs)
            self.log_prob_seq.append(avg_log_prob)
            if best_hmm is None or avg_log_prob < best_log_prob:
                best_hmm = hmm
            gc.collect()
            # self.hmm_seq.append(hmm)
        self.phmm_p = best_hmm
        return self

    def score(self, X: List[List[str]]) -> List[float]:
        log_probs = []
        for x in X:
            try:
                log_probs.append(calc_log_prob(self.phmm_p, x))
            except Exception as e:
                log_probs.append(-1e9)
        return log_probs

    def to_dict(self):
        dc = {
            'label': self.label,
            'num_iter': self.num_iter,
            'duration': self.duration,
            'init_prior': self.init_prior,
            'seed': self.seed
        }
        dc.update(self.phmm_p.to_dict())
        return dc


def eval_phmm_mp(args):
    idx = args[0]
    data = args[1]
    params = args[2]
    model = Hmm.from_dict(params)
    log_probs = []
    for x in data:
        try:
            log_probs.append(calc_log_prob(model, x))
        except Exception as e:
            s = ' '.join(x)
            print(f"Failed for sequence {s}")
            print(e)
            log_probs.append(-1e9)
    return idx, log_probs


class CPhmm(object):
    @classmethod
    def from_dict(cls, d) -> 'CPhmm':
        phmm = cls(
            duration=d['duration'],
            init_prior=d.get('init_prior', 'uniform'),
            seed=d['seed'],
            label=d['label'],
            num_iter=d['num_iter']
        )
        phmm.phmm_p = basic_phmm(d['duration'], d['observables'])
        phmm.phmm_p.p_ij = {eval(k): v for k, v in d['p_ij'].items()}
        phmm.phmm_p.p_o_in_i = {eval(k): v for k, v in d['p_o_in_i'].items()}

        phmm.phmm_c = PhmmC(d['duration'], d['observables'], d['init_prior'], d['seed'])
        phmm.phmm_c.add_transitions(phmm.phmm_p)
        phmm.phmm_c.add_emissions(phmm.phmm_p)
        return phmm

    def __init__(self, duration: int, init_prior: str,
                 seed: int, label: str, num_iter: int):
        self.label = label
        self.num_iter = num_iter
        self.duration = duration
        self.init_prior = init_prior
        self.seed = seed
        self.phmm_c = None
        self.log_prob_val = None
        self.log_prob_train = None
        self.log_prob_val_all = None
        self.log_prob_train_all = None
        self.phmm_p = None
        self.baum_welch_res = None

    def fit(self, X: List[List[str]], X_val: List[List[str]]=None) -> 'CPhmm':
        observation_space = [str(e) for e in np.unique(np.concatenate([np.unique(x) for x in X]))]
        self.phmm_c = basic_phmm_c(self.duration, observation_space)
        self.phmm_p = None
        if X_val is None:
            X_val = X
        traces_train, traces_train_lens = dahn.convert_seq_to_int(self.phmm_c, X)
        traces_val, traces_val_lens = dahn.convert_seq_to_int(self.phmm_c, X_val)
        num_traces_train = len(X)
        num_traces_val = len(X_val)
        num_obs = len(observation_space)

        traces_train_p = (ctypes.c_int * len(traces_train))(*traces_train)
        traces_train_lens_p = (ctypes.c_int * len(traces_train_lens))(*traces_train_lens)
        traces_val_p = (ctypes.c_int * len(traces_val))(*traces_val)
        traces_val_lens_p = (ctypes.c_int * len(traces_val_lens))(*traces_val_lens)

        res = _baum_welch(
            self.phmm_c._p_ij_p,
            self.phmm_c._p_o_in_i_p,
            traces_train_p,
            traces_train_lens_p,
            num_traces_train,
            traces_val_p,
            traces_val_lens_p,
            num_traces_val,
            num_obs,
            self.phmm_c.duration,
            self.num_iter
        )

        self.phmm_c.update_p_ij(res[0])
        self.phmm_c.update_p_o_in_i(res[1])
        self.phmm_c._p_ij_p = res[0]
        self.phmm_c._p_o_in_i_p = res[1]

        log_prob_train_all, log_prob_train = cphmm.convert_log_prob(res[2], num_traces_train)
        log_prob_val_all, log_prob_val = cphmm.convert_log_prob(res[3], num_traces_val)

        self.log_prob_train = log_prob_train
        self.log_prob_val = log_prob_val
        self.log_prob_train_all = log_prob_train_all
        self.log_prob_val_all = log_prob_val_all

        # _free(res)
        self.baum_welch_res = res

        self.phmm_p = basic_phmm(self.phmm_c.duration, observation_space)
        self.phmm_p.p_ij = dict(zip(self.phmm_c._p_ij_keys, self.phmm_c._p_ij))
        self.phmm_p.p_o_in_i = dict(zip(self.phmm_c._p_o_in_i_keys, self.phmm_c._p_o_in_i))
        return self

    def score(self, X: List[List[str]]) -> List[float]:
        log_probs = []
        for x in X:
            log_probs.append(calc_log_prob(self.phmm_p, x))
            try:
                log_probs.append(calc_log_prob(self.phmm_p, x))
            except Exception as e:
                log_probs.append(-1e9)
        return log_probs

    def to_json_dict(self) -> Dict[str, Any]:
        d = self.phmm_p.to_dict()
        d['p_ij'] = {str(k): v for k, v in d['p_ij'].items()}
        d['p_o_in_i'] = {str(k): v for k, v in d['p_o_in_i'].items()}
        d['label'] = self.label
        d['init_prior'] = self.init_prior
        d['num_iter'] = self.num_iter
        return d

    def score_c(self, X: List[List[str]]) -> List[float]:
        traces, traces_lens = dahn.convert_seq_to_int(self.phmm_c, X)
        traces_p = (ctypes.c_int * len(traces))(*traces)
        traces_lens_p = (ctypes.c_int * len(traces_lens))(*traces_lens)
        num_traces = len(X)
        log_probs_ = _c_log_prob(
            self.phmm_c._p_ij_p,
            self.phmm_c._p_o_in_i_p,
            traces_p,
            traces_lens_p,
            num_traces,
            self.phmm_c.duration
        )
        log_probs = cphmm.convert_log_prob(log_probs_, len(X))
        _free_single(log_probs_)
        return log_probs[0]

    def __del__(self):
        if self.baum_welch_res is not None:
            _free(self.baum_welch_res)


def mymap(jobs: List[Any], target: callable) -> List[Any]:
    logger = logfac.produce_logger('phmm.mymap')
    job_queue = mp.Queue()
    res_queue = mp.Queue()
    shutdown = mp.Event()
    logger.debug("Create jobs")
    procs = [mp.Process(
        target=target,
        args=(job_queue, res_queue, shutdown, i)) for i in range(NPROCS)]
    job_duration = 1e9
    job_duration_unset = True
    start_queueing = time.perf_counter()
    last_dequeue = time.perf_counter()
    logger.debug("Queue items")
    for job in jobs:
        job_queue.put(job)
    logger.debug("Start jobs")
    for j in procs:
        j.start()

    logger.debug("Add results.")
    results = []
    while len(results) < len(jobs) and time.perf_counter() - last_dequeue < 5 * job_duration:
        logger.debug(f"\tGot {len(results)} of {len(jobs)} results.")
        if res_queue.empty():
            time.sleep(1)
        else:
            if job_duration_unset:
                job_duration_unset = False
                job_duration = time.perf_counter() - start_queueing
            results.append(res_queue.get())
            last_dequeue = time.perf_counter()
    logger.debug("Shutdown jobs.")
    shutdown.set()
    for i, j in enumerate(procs):
        j.join()
        logger.debug(f"process {i} joined.")
        # j.close()
        # print(f"process {i} closed.")
    logger.debug("All jobs shut down.")
    return results


class PhmmClassifier(SeqClassifier):

    def __init__(self, duration: int, init_prior: str, seed: int, num_iter: int, **kwargs):
        super(PhmmClassifier, self).__init__()
        self.duration = duration
        self.init_prior = init_prior
        self.seed = seed
        self.num_iter = num_iter
        self.phmms = {}
        self.trial_dir = None

    def fit(self, X: Dict[str, List[List[str]]], y: List[List[str]]) -> 'PhmmClassifier':
        def job_iterator():
            for label, X_label in X.items():
                yield {
                    'duration': self.duration,
                    'init_prior': self.init_prior,
                    'seed': self.seed,
                    'label': label,
                    'num_iter': self.num_iter,
                    'X': X_label,
                    'trial_dir': self.trial_dir
                }

        hmm_params = mymap([args for args in job_iterator()], fit_phmm_mp_queue_decorator)

        for label, exported_phmm in hmm_params:
            wrapper = Phmm(
                duration=self.duration,
                init_prior=self.init_prior,
                seed=self.seed,
                label=label,
                num_iter=self.num_iter
            )
            wrapper.phmm_p = Hmm.from_dict(exported_phmm)
            self.phmms[label] = wrapper
        return self

    def predict(self, X: List[List[str]]) -> List[str]:

        rets = mymap([{
            "label": l,
            "params": phmm.to_dict(),
            "X": X
        } for l, phmm in self.phmms.items()], score_multi_phmm_mp_queue_decorator)

        # rets = []
        # for l, phmm in self.phmms.items():
        #     rets.append(score_multi_phmm_mp((l, phmm.to_dict(), X)))

        scores = {l: s for l, s in rets}

        # scores = {}
        # for l, mc in self.phmms.items():
        #     scores[l] = mc.score(X)

        labels = []
        for i in range(len(X)):
            best_label = None
            best_likelihood = -1e9
            for l, lls in scores.items():
                if lls[i] > best_likelihood:
                    best_likelihood = lls[i]
                    best_label = l
                else:
                    continue
            labels.append(best_label)
        return labels


def score_multi_phmm_mp_queue_decorator(job_queue: mp.Queue, result_queue: mp.Queue,
                                        shutdown: mp.Event, worker_id: int):
    logger = logfac.produce_logger(f"mymap-worker-{worker_id}")
    failues = 0
    while not shutdown.is_set() and failues < 10:
        try:
            args = job_queue.get(block=False)
            results = score_multi_phmm_mp((args["label"], args['params'], args["X"]))
            result_queue.put(results)
        except queue.Empty:
            failues += 1
            time.sleep(1)
        except Exception as e:
            logger.error("Unexpected error ocured.")
            logger.exception(e)
    logger.info(f"Worker {worker_id} is exiting.")


def score_multi_phmm_mp(args: Tuple[str, Dict[str, Any], List[List[str]]]) -> Tuple[str, List[float]]:
    label = args[0]
    params = args[1]
    X = args[2]
    scores = Phmm.from_dict(params).score(X)
    del args
    gc.collect()
    return label, scores


def fit_phmm_mp_queue_decorator(job_queue: mp.Queue, result_queue: mp.Queue,
                                shutdown: mp.Event, worker_id: int):
    failures = 0
    logger = logfac.produce_logger(f"mymap-worker-{worker_id}")
    while not shutdown.is_set() and failures < 10:
        try:
            args = job_queue.get(block=False)
            failures = 0
            results = fit_phmm_mp(args)
            result_queue.put(results)
        except queue.Empty:
            failures += 1
            time.sleep(1)
        except Exception as e:
            logger.error("Unexpected error ocured.")
            logger.exception(e)
    logger.info(f"Worker {worker_id} is exiting.")


def fit_phmm_mp(args: Dict[str, any]) -> Tuple[str, Dict[str, Any]]:
    logger = logfac.produce_logger('phmm.fit_phmm_mp')
    label = args['label']
    logger.debug(f"Fit HMM for {args['label']} with {len(args['X'])} sequences.")
    wrapper = CPhmm(
        duration=args['duration'],
        init_prior=args['init_prior'],
        seed=args['seed'],
        label=args['label'],
        num_iter=args['num_iter']
    ).fit(args['X'])
    d = wrapper.phmm_p.to_dict()
    logger.debug(f"{label}: Transition matrix has {len(wrapper.phmm_p.p_ij)} elements.")
    logger.debug(f"{label}: Observation matrix has {len(wrapper.phmm_p.p_o_in_i)} elements.")
    # if args['trial_dir'] is not None:
    #     with open(os.path.join(args['trial_dir'], f'hmm-{label}.json'), 'w') as fh:
    #         json.dump(d, fh)
    del args
    del wrapper
    gc.collect()
    return label, d


if __name__ == '__main__':
    sequences = [
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'c', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
    ]
    phmm = CPhmm(duration=4, init_prior='uniform', seed=1, label='tmp', num_iter=20).fit(sequences)
    log_probs = phmm.score_c(sequences)
    print("Log Probs are ", log_probs)
    d = phmm.to_json_dict()
    phmm_r = CPhmm.from_dict(d)
    log_probs = phmm_r.score_c(sequences)
    print("Log Probs are ", log_probs)
    print(phmm.phmm_p.p_ij)
    print(phmm.phmm_p.p_o_in_i)
    print(phmm.score(sequences))

    check_trans = {}
    min_p = 1.
    for (tail, head), value in phmm.phmm_p.p_ij.items():
        min_p = value if value < min_p else min_p
        if tail not in check_trans:
            check_trans[tail] = 0
        check_trans[tail] += value
    print("min_p is", min_p)
    print("Transitions summed up", check_trans)
    check_trans = {}
    min_p = 1.
    for (tail, head), value in phmm.phmm_p.p_o_in_i.items():
        min_p = value if value < min_p else min_p
        if head not in check_trans:
            check_trans[head] = 0
        check_trans[head] += value
    print("min_p is", min_p)
    print("Observations summed up", check_trans)

