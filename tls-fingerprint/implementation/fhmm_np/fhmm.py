import numpy as np
from typing import List, Dict, Tuple, Any

class Hmm(object):

    def __init__(self, duration, seed: int = 1):

        self.duration = duration
        self.hiddens = []
        self.observables = []
        self.init = {}
        self.p_ij = {}
        self.p_o_in_i = {}
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)

    def __len__(self):
        return self.duration

    def add_nodes(self):
        for n in range(self.duration):
            self.hiddens.append(n)

    def add_init(self, init_prior: str='uniform'):
        prior = [100.] * self.duration
        if init_prior == 'uniform':
            pass
        elif init_prior == 'prefer_match':
            prior[0] = 500.
        else:
            raise KeyError
        probs = self.random.dirichlet(prior)
        for i in range(self.duration):
            self.init[i] = probs[i]

    def add_transitions(self):
        for i in self.hiddens:
            probs = self.random.dirichlet([100.] * self.duration)
            for j in self.hiddens:
                self.p_ij[(i, j)] = probs[j]

    def add_observtions(self):
        for i in self.hiddens:
            probs = self.random.dirichlet([100.] * len(self.observables))
            for j in range(len(self.observables)):
                self.p_o_in_i[(self.observables[j], i)] = probs[j]

def basic_hmm(duration: int, observation_space: List[int], seed: int = 1, init_prior: str='uniform') -> Hmm:

    hmm = Hmm(duration, seed)
    hmm.observables = observation_space
    hmm.add_nodes()
    hmm.add_init(init_prior)
    hmm.add_transitions()
    hmm.add_observtions()

    return hmm