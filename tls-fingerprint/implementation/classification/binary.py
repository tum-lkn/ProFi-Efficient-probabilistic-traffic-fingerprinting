"""
Implements a binary classifier. The binary classifier decorates a density
based classifier. The density based classifier is supposed to have a score
function.
The binary classifier uses an anomaly score based on the probability of the
density based classifier.
"""
import time
from typing import List, Dict, Tuple, Any, Union
import numpy as np
import multiprocessing as mp

import implementation.classification.mc as mc
import implementation.classification.phmm as phmm
import implementation.data_conversion.tls_flow_extraction as tlsex
from implementation.classification.seq_classifier import SeqClassifier


FACTORY = {
    'mc': mc.MarkovChain,
    'phmm': phmm.CPhmm
}


class BinaryClassifier(object):

    def __init__(self, ano_density_estimator: Union[None, str],
                 seq_length: int, seq_element_type: str, bin_edges: Union[None, np.array],
                 defense=None, direction_to_filter=0, **kwargs):
        self.seq_length = seq_length
        self.seq_element_type = seq_element_type
        self.bin_edges = bin_edges
        self.score_time = None
        self.skip_tls_handshake = False
        if ano_density_estimator is None:
            self.density_estimator = None
        else:
            self.density_estimator = FACTORY[ano_density_estimator](**kwargs)
        self.threshold = None
        self.defense = defense
        self.main_flow_to_symbol = tlsex.MainFlowToSymbol(
            seq_length=self.seq_length,
            to_symbolize=self.seq_element_type,
            bin_edges=self.bin_edges,
            skip_handshake=self.skip_tls_handshake,
            direction_to_filter=direction_to_filter
        )

    @property
    def label(self) -> str:
        return self.density_estimator.label

    def to_dict(self):
        return self.density_estimator.to_dict()

    def fit(self, X: List[List[str]]) -> 'BinaryClassifier':
        self.density_estimator.fit(X)
        values = self.density_estimator.score(X)
        self.threshold = np.max(np.abs(values))
        return self

    def score(self, X: List[Dict[str, Any]]) -> List[float]:
        """
        Input is a list of main flows.

        Args:
            X:

        Returns:

        """
        if self.defense is not None:
            X = [self.defense(m) for m in X]
        X = [self.main_flow_to_symbol(m) for m in X]
        t1 = time.perf_counter()
        scores = np.abs(self.density_estimator.score_c(X))
        scores = [s / self.threshold for s in scores]
        self.score_time = (time.perf_counter() - t1) / len(X)
        return scores

    def _predict(self, scores: List[float]) -> List[bool]:
        predictions = [True if s <= 1. else False for s in scores]
        return predictions

    def predict(self, X: List[Dict[str, Any]]) -> List[bool]:
        """
        Input is a list of main flows.

        Args:
            X:

        Returns:

        """
        scores = self.score(X)
        return self._predict(scores)


class MultiBinaryClassifier(object):
    """
    Uses multiple Binary classifiers to perform a classification of sequences.
    """
    def __init__(self, configuration: Union[None, Dict[str, Any], Dict[str, Dict[str, Any]]],
                 scenario: str='open'):
        super(MultiBinaryClassifier, self).__init__()
        self.configuration = configuration
        self.bcs = {}
        self.unknown_label = 'unknown'
        self.scenario = scenario
        self.inference_times = []
        self.classification_times = []
        self.training_times = []
        self._skip_tls_handshake = False

    @property
    def skip_tls_handshake(self) -> bool:
        return self._skip_tls_handshake

    @skip_tls_handshake.setter
    def skip_tls_handshake(self, val: bool) -> None:
        self._skip_tls_handshake = val
        for v in self.bcs.values():
            v.skip_tls_handshake = val

    def fit(self, X: Dict[str, List[str]], y: List[str]) -> 'MultiBinaryClassifier':
        def job_iterator():
            for label, X_label in X.items():
                cfg = {k.lstrip('hmm_').lstrip('mc_'): v for k, v in self.configuration.items()}
                cfg['label'] = label
                cfg['X'] = X_label
                yield cfg

        pool = mp.Pool(30)
        hmm_params = pool.map(fit_multi_ano_mp, [j for j in job_iterator()])
        pool.close()
        for args, exported_params in hmm_params:
            wrapper = BinaryClassifier(**args)
            wrapper.density_estimator.from_dict(**exported_params)
            wrapper.skip_tls_handshake = self.skip_tls_handshake
            self.bcs[args['label']] = wrapper
        return self

    def predict(self, X: List[Dict[str, Any]]) -> List[str]:
        """
        Input are the full main flows and not the converted sequences.

        Args:
            X:

        Returns:

        """
        t1 = time.perf_counter()
        scores = {l: bc.score(X) for l, bc in self.bcs.items()}
        predictions = {l: bc._predict(scores[l]) for l, bc in self.bcs.items()}

        predicted_lbls = []
        for i in range(len(X)):
            best_score = 1e9
            best_label = self.unknown_label
            for l, bc in self.bcs.items():
                # In case of open scenario evaluate based on whether model
                # detected class and the anomaly score. In case of other scenarios
                # Select the model with the smallest anomaly score.
                if self.scenario == 'open':
                    if predictions[l][i] and best_score > scores[l][i]:
                        best_label = l
                        best_score = scores[l][i]
                else:
                    if best_score > scores[l][i]:
                        best_label = l
                        best_score = scores[l][i]
            predicted_lbls.append(best_label)
        self.classification_times.append((time.perf_counter() - t1) / len(X))
        self.inference_times.extend([bc.score_time for bc in self.bcs.values()])
        return predicted_lbls


def fit_multi_ano_mp(args: Dict[str, Any]):
    X_label = args.pop('X')
    classifier = BinaryClassifier(**args).fit(X_label)
    return args, classifier.to_dict()


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
    bc = BinaryClassifier('mc', label='mc_test').fit(sequences)
    print(bc.score([['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]))
    print(bc.predict([['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]))
    bc = BinaryClassifier('phmm', label='test', duration=4,
                          init_prior='uniform', seed=1, num_iter=20).fit(sequences)
    print(bc.score([['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]))
    print(bc.predict([['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]))

    mbc = MultiBinaryClassifier({
        "mclbl": {'ano_density_estimator': 'mc'},
        "phlnl": {
            "ano_density_estimator": "phmm",
            "duration": 4,
            "seed": 1,
            "init_prior": "uniform",
            "num_iter": 20
        }
    }).fit(sequences, ['mclbl','mclbl','mclbl','mclbl','mclbl','mclbl','mclbl','mclbl','mclbl','mclbl',
                       'phlnl','phlnl','phlnl','phlnl','phlnl','phlnl','phlnl','phlnl','phlnl','phlnl'])
    print(mbc.predict([['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]))
