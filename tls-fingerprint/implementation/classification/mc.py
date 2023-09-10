"""
Implements a markov chain classifier.
"""
from typing import List, Union, Any, Dict, Tuple
import networkx as nx
import numpy as np

from implementation.classification.seq_classifier import SeqClassifier
import implementation.data_conversion.onvm_model_conversion as omc


class MarkovChain(object):

    @classmethod
    def from_dict(cls, dictionary):
        mc = cls(dictionary['label'])
        for u, v, d in dictionary['edges']:
            mc.mc.add_edge(u, v, **d)
        return mc

    START_NODE = 'START'

    def __init__(self, label: str, **kwargs):
        self.label = label
        self.mc = nx.DiGraph()
        self.default_p = 1e-6
        self.log_prob_train_all = None
        self.log_prob_val_all = None

    def fit(self, X: List[List[str]], X_val: List[List[str]]=None) -> 'MarkovChain':
        for x in X:
            if not self.mc.has_edge(self.START_NODE, x[0]):
                self.mc.add_edge(self.START_NODE, x[0], count=0)
            self.mc.edges[(self.START_NODE, x[0])]['count'] += 1

            for u, v in zip(x[:-1], x[1:]):
                if not self.mc.has_edge(u, v):
                    self.mc.add_edge(u, v, count=0)
                self.mc.edges[(u, v)]['count'] += 1
        for state in self.mc.nodes():
            missing = self.mc.number_of_nodes() - self.mc.out_degree[state]
            count = 0
            for successor in nx.neighbors(self.mc, state):
                count += self.mc.edges[(state, successor)]['count']
            for successor in nx.neighbors(self.mc, state):
                c = self.mc.edges[(state, successor)]['count']
                self.mc.edges[(state, successor)]['probability'] = c / count * (1 - missing * self.default_p)
        self.log_prob_train_all = np.array(self.score(X))
        if X_val is not None:
            self.log_prob_val_all = np.array(self.score(X_val))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "edges": [(u, v, d) for u, v, d in self.mc.edges(data=True)]
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    def score(self, X: List[List[str]]) -> List[float]:
        log_probs = []
        for x in X:
            if self.mc.has_edge(self.START_NODE, x[0]):
                p = self.mc.edges[(self.START_NODE, x[0])]['probability']
            else:
                p = self.default_p
            log_prob = np.log(p)
            for u, v in zip(x[:-1], x[1:]):
                if self.mc.has_edge(u, v):
                    p = self.mc.edges[(u, v)]['probability']
                else:
                    p = self.default_p
                log_prob += np.log(p)
            log_probs.append(log_prob)
        return log_probs

    def score_c(self, X: List[List[str]]) -> List[float]:
        return self.score(X)

    def c_export(self) -> Dict[str, Any]:
        symbols = list(self.mc.nodes())
        int_symbols = [1 if s == self.START_NODE else omc.compute_hash(s) for s in symbols]
        int_sybols = np.array(
            int_symbols,
            dtype=np.uint64
        )
        order = np.argsort(int_sybols)
        symbols = [symbols[i] for i in order]
        int_sybols = int_sybols[order]
        int_neighbors = np.array([])
        log_probs = []
        offsets = [0]
        for s in symbols:
            neighbors = list(nx.neighbors(self.mc, s))
            tmp = np.array(
                [1 if n == self.START_NODE else omc.compute_hash(n) for n in neighbors],
                dtype=np.uint64
            )
            order = np.argsort(tmp)
            tmp = tmp[order]
            offsets.append(offsets[-1] + tmp.size)
            int_neighbors = np.concatenate([int_neighbors, tmp])
            log_probs.extend([np.log(self.mc.edges[s, neighbors[i]]['probability']) for i in order])
        export = {
            'num_nodes': int(self.mc.number_of_nodes()),
            'num_edges': len(int_neighbors),
            'offsets': [int(x) for x in offsets],
            'tails': [int(x) for x in int_sybols],
            'heads': [int(x) for x in int_neighbors],
            'log_probs': [float(x) for x in log_probs]
        }
        return export


class MarkovChainClassifier(SeqClassifier):
    def __init__(self, **kwargs):
        super(MarkovChainClassifier, self).__init__()
        self.mcs = {}

    def fit(self, X: List[List[str]], y: List[str]) -> 'MarkovChainClassifier':
        for x in X:
            x.insert(0, 'START')
            x.append("END")

        reordered = {i: [] for i in y}
        for l, x in zip(y, X):
            reordered[l].append(x)
        for l, X_ in reordered.items():
            self.mcs[l] = MarkovChain(l).fit(X_)
        for x in X:
            x.pop(0)
            x.pop(-1)
        return self

    def predict(self, X: List[List[str]]) -> List[str]:
        for x in X:
            x.insert(0, 'START')
            x.append("END")
        scores = {}
        for l, mc in self.mcs.items():
            scores[l] = mc.score(X)

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
        for x in X:
            x.pop(0)
            x.pop(-1)
        return labels


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
    knn_c = MarkovChainClassifier().fit(seqs, labels)
    print(knn_c.predict(test))
