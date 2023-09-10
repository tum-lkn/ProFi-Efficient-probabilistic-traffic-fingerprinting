"""
Implements a simple KNN classifier. Follows roughly the sklearn API.
"""
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
import Levenshtein as leven
from typing import List, Tuple, Dict, Union, Any
from implementation.classification.seq_classifier import SeqClassifier


class KnnClassifier(SeqClassifier):
    def __init__(self, num_neighbors: int, fraction: float=0.):
        super(KnnClassifier, self).__init__()
        self.training_sequences = None
        self.training_sequences_labels = None
        self.symbol_to_unicode = {}
        self.num_neighbors = num_neighbors
        self.fraction = fraction
        if self.fraction is None:
            self.fraction = 0.
        self.unknown_label = 'unknown'
        self.votes = {}

    def fit(self, X: List[List[str]], y: List[str]) -> 'KnnClassifier':
        obs_space = np.array([], dtype=object)
        for x in X:
            obs_space = np.unique(np.concatenate([obs_space, np.array(x)]))
        self.symbol_to_unicode = {k: chr(i) for i, k in enumerate(obs_space)}
        self.training_sequences = [''.join([self.symbol_to_unicode[s] for s in seq]) for seq in X]
        self.training_sequences_labels = np.array(y)
        return self

    def predict_mp(self, X: List[List[List[str]]]) -> List[List[str]]:
        n_procs = mp.cpu_count() - 2
        pool = mp.Pool(n_procs)
        rets = pool.map(self.predict, [(x, i) for i, x in enumerate(X)])
        pool.close()
        rets.sort(key=lambda x: x[-1])
        predicted_labels = []
        for plbls, aggrs, times, day in rets:
            predicted_labels.append(plbls)
            self.inference_times.extend(times)
            for plbl, aggr in zip(plbls, aggrs):
                if plbl not in self.votes:
                    self.votes[plbl] = []
                self.votes[plbl].append(float(aggr))
        return predicted_labels

    # def predict(self, X: List[List[str]], idx: int=None) -> Tuple[List[str], int]:
    def predict(self, args) -> Tuple[List[str], List[float], List[float], int]:
        X = args[0]
        idx = args[1]
        predicted_labels = []
        agreements = []
        times = []
        for seq in X:
            seq_ = []
            for s in seq:
                try:
                    seq_.append(self.symbol_to_unicode[s])
                except KeyError:
                    self.symbol_to_unicode[s] = chr(len(self.symbol_to_unicode))
                    seq_.append(self.symbol_to_unicode[s])
            seq = ''.join(seq_)
            t1 = time.perf_counter()
            distances = [leven.distance(seq, x) for x in self.training_sequences]
            sorted_idx = np.argsort(distances)
            labels = self.training_sequences_labels[sorted_idx[:self.num_neighbors]]

            vote = pd.Series(labels).value_counts(normalize=True).reset_index()
            label = vote.loc[0, 'index']
            agreement = vote.loc[0, 0]
            if agreement < self.fraction:
                label = self.unknown_label
            times.append(time.perf_counter() - t1)
            agreements.append(agreement)
            predicted_labels.append(label)

        return predicted_labels, agreements, times, idx


if __name__ == '__main__':
    seqs = [
        ['a', 'b', 'c'],
        ['a', 'b', 'c'],
        ['a', 'b', 'c'],
        ['a', 'b', 'c'],
        ['e', 'f', 'g'],
        ['e', 'f', 'g'],
        ['e', 'f', 'g'],
        ['e', 'f', 'g'],
    ]
    labels = ['one', 'one', 'one', 'one', 'two', 'two', 'two', 'two']
    test = [['a', 'b', 'c'], ['e', 'f', 'g'], ['a', 'b', 'e'], ['x', 'y', 'z']]
    knn_c = KnnClassifier(3).fit(seqs, labels)
    print(knn_c.predict(test))
