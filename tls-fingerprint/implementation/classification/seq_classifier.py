from typing import List


class SeqClassifier(object):

    def __init__(self):
        self.inference_times = []

    def fit(self, X: List[List[str]], y: List[str]) -> 'SeqClassifier':
        raise NotImplemented

    def predict(self, X: List[List[str]]) -> List[str]:
        raise NotImplemented

    def score(self, X: List[List[str]]) -> List[float]:
        raise NotImplemented

    def score_c(self, X: List[List[str]]) -> List[float]:
        return self.score(X)
