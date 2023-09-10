from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import os
import json


class AggregateCommonPrefixes(object):
    def __init__(self, common_prefixes: List[str]) -> 'AggregateCommonPrefixes':
        self.common_prefixes = common_prefixes

    def __call__(self, conf_mat: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, str]]:
        """
            Aggregates labels in the `conf_mat` to a common prefix. Add values of the
            confusion matrix up for entries that have the same prefix.

            Args:
                common_prefixes: Prefixes to aggregate URLs with.
                conf_mat: A confusion matrix.

            Returns:
                Confusion matrix with aggregated entries.
        """
        new_conf_mat = {}
        for actual, results in conf_mat.items():
            new_actual = aggregate_label(self.common_prefixes, actual)
            if new_actual not in new_conf_mat: new_conf_mat[new_actual] = {}
            for predicted, value in results.items():
                new_predicted = aggregate_label(self.common_prefixes, predicted)
                if new_predicted not in new_conf_mat[new_actual]: new_conf_mat[new_actual][new_predicted] = 0
                new_conf_mat[new_actual][new_predicted] += value
        return new_conf_mat


def aggregate_label(common_prefixes: List[str], label: str) -> str:
    """
    Convert a label to a common prefix. Iterates over the `common_prefixes` and
    checks if the passed label starts with one of the prefixes. If it does,
    the prefix is returned. Else, the label is returned unchanged.

    Args:
        common_prefixes:
        label:

    Returns:

    """
    new_label = label
    for prefix in common_prefixes:
        if label.startswith(prefix):
            new_label = prefix
            break
    return new_label


def aggregate_common_prefixes(common_prefixes: List[str], conf_mat: Dict[str, Dict[str, float]]):
    """
    Aggregates labels in the `conf_mat` to a common prefix. Add values of the
    confusion matrix up for entries that have the same prefix.

    Args:
        common_prefixes: Prefixes to aggregate URLs with.
        conf_mat: A confusion matrix.

    Returns:
        Confusion matrix with aggregated entries.
    """
    new_conf_mat = {}
    for actual, results in conf_mat.items():
        new_actual = aggregate_label(common_prefixes, actual)
        if new_actual not in new_conf_mat: new_conf_mat[new_actual] = {}
        for predicted, value in results.items():
            new_predicted = aggregate_label(common_prefixes, predicted)
            if new_predicted not in new_conf_mat[new_actual]: new_conf_mat[new_actual][new_predicted] = 0
            new_conf_mat[new_actual][new_predicted] += value
    return new_conf_mat


def get_accuracy(conf_mat: Dict[str, Dict[str, float]], labels: List[str] = None) -> Dict[str, float]:
    """
    Calculate the accuracy from a confusion matrix. Calculate the accuracy by
    transforming the multi-class problem into multiple one-versus-all binary problems.

    Args:
        conf_mat: Confusion matrix, the outer dictionary contains the target label,
            the inner dictionary contains the predictions
        labels: Restrict the evaluation to a specific set of labels. Ignore the
            target labels in the confusion matrix that are not in this list.

    Returns:

    """
    assert type(conf_mat) == dict, 'conf_mat is not dict: ' + str(conf_mat)
    accuracies = {}
    for expected, results in conf_mat.items():
        if labels is not None:
            if expected not in labels:
                continue
        accuracies[expected] = 0.
        total = 0
        for l1, r1 in conf_mat.items():
            for l2, r2 in r1.items():
                if expected == l1 and l1 == l2:
                    # TP
                    accuracies[expected] += r2
                elif expected != l1 and expected != l2:
                    # TN
                    accuracies[expected] += r2
                else:
                    pass
                total += r2
        accuracies[expected] /= total
    return accuracies


def get_precision(conf_mat: Dict[str, Dict[str, float]], labels: List[str] = None) -> Dict[str, float]:
    """
    Calculate the precision matrix from a confusion matrix. The precision is the
    number of true positives divided by the sum of true positives and false
    positives.

    Args:
        conf_mat: Confusion matrix, the outer dictionary contains the target label,
            the inner dictionary contains the predictions
        labels: Restrict the evaluation to a specific set of labels. Ignore the
            target labels in the confusion matrix that are not in this list.

    Returns:

    """
    tps = {}
    fps = {}
    for lbl, results in conf_mat.items():
        if labels is not None:
            if lbl not in labels:
                continue
        if lbl not in tps: tps[lbl] = 0
        if lbl not in fps: fps[lbl] = 0
        for actual, number in results.items():
            if actual not in tps: tps[actual] = 0
            if actual not in fps: fps[actual] = 0
            if actual == lbl:
                # Increase true positives.
                tps[actual] += number
            else:
                # `number` instances of lbl are labeled as actual --> increase
                # the number of false positives.
                fps[actual] += number
    precision = {}
    for k in tps.keys():
        if tps[k] + fps[k] == 0:
            precision[k] = 0.
        else:
            precision[k] = tps[k] / (tps[k] + fps[k])
    return precision


def get_recall(conf_mat: Dict[str, Dict[str, float]], labels: List[str] = None) -> Dict[str, float]:
    """
    Calculate the recall matrix from a confusion matrix. The recall is the
    number of true positives divided by the sum of true positives and false
    negatives. Essentially interates over one line of the confusion matrix
    and divides the true positives by the total number of entries in the line.

    Args:
        conf_mat: Confusion matrix, the outer dictionary contains the target label,
            the inner dictionary contains the predictions
        labels: Restrict the evaluation to a specific set of labels. Ignore the
            target labels in the confusion matrix that are not in this list.

    Returns:

    """
    recall = {}
    for lbl, results in conf_mat.items():
        if labels is not None:
            if lbl not in labels:
                continue
        if lbl in results:
            recall[lbl] = results[lbl] / np.sum(list(results.values()))
        else:
            recall[lbl] = 0
    return recall


def get_accuracy_open(conf_mat: Dict[str, Dict[str, float]], closed: List[str]) -> Dict[str, float]:
    """
    Calculate the accuracy in the open world scenario. All labels in the confusion
    matrix that are not part of the `closed` list are treated as belonging to
    the `unknown` label. The `unknown` label is used as label in the inner
    dictionary of the confusion matrix, i.e., the predictions of the classifier.

    Principle the same as in `get_accuracy`

    Args:
        conf_mat:
        closed:

    Returns:

    """
    assert type(conf_mat) == dict, 'conf_mat is not dict: ' + str(conf_mat)
    accuracies = {}
    totals = {}
    for expected, results in conf_mat.items():
        expected_ = expected if expected in closed else 'unknown'
        if expected_ not in accuracies: accuracies[expected_] = 0.
        if expected_ not in totals: totals[expected_] = 0.
        for l1, r1 in conf_mat.items():
            l1 = l1 if l1 in closed else 'unknown'
            for l2, r2 in r1.items():
                l2 = l2 if l2 in closed else 'unknown'
                if expected_ == l1 and expected_ == l2:
                    # True positives.
                    accuracies[expected_] += r2
                elif expected_ != l1 and expected_ != l2:
                    # True negatives, i.e., the outer label L1 (ground truth)
                    # is not the expected_ label, and the prediction is also
                    # not the expected_ label.
                    accuracies[expected_] += r2
                else:
                    pass
                totals[expected_] += r2
    for lbl in accuracies.keys():
        accuracies[lbl] /= totals[lbl]
    return accuracies


def get_precision_open(conf_mat: Dict[str, Dict[str, float]], closed: List[str]) -> Dict[str, float]:
    """
    See `get_precision` and `get_accuracy_open'.

    Args:
        conf_mat:
        closed:

    Returns:

    """
    tps = {}
    fps = {}
    for lbl, results in conf_mat.items():
        expected_ = lbl if lbl in closed else 'unknown'
        if expected_ not in tps: tps[lbl] = 0
        if expected_ not in fps: fps[lbl] = 0
        for actual, number in results.items():
            if actual not in tps: tps[actual] = 0
            if actual not in fps: fps[actual] = 0
            if actual == expected_:
                tps[expected_] += number
            else:
                fps[actual] += number
    precision = {}
    for k in tps.keys():
        if tps[k] + fps[k] == 0:
            precision[k] = 0.
        else:
            precision[k] = tps[k] / (tps[k] + fps[k])
    return precision


def get_recall_open(conf_mat: Dict[str, Dict[str, float]], closed: List[str]) -> Dict[str, float]:
    """
    See `get_precision` and `get_accuracy_open'.

    Args:
        conf_mat:
        closed:

    Returns:

    """
    recall = {'unknown': 0}
    relevant_unknown = 0
    retrieved_unknown = 0
    for lbl, results in conf_mat.items():
        if lbl in closed:
            # The same as `get_accuracy_open`.
            if lbl in results:
                recall[lbl] = results[lbl] / np.sum(list(results.values()))
            else:
                recall[lbl] = 0
        else:
            # The outer label belongs to the class unknown. Since `conf_mat`
            # has multiple labels that are cast to unknown, aggregate the numbers.
            if 'unknown' in results:
                # Check if `unknown` has been predicted for the label lbl that is
                # not in the closed world labels.
                retrieved_unknown += results['unknown']
            # Increase the total number of unknown labels.
            relevant_unknown += np.sum(list(results.values()))
    if retrieved_unknown == relevant_unknown == 0:
        recall['unknown'] = 0
    elif retrieved_unknown > 0 and relevant_unknown == 0:
        raise ValueError(f"retrieved unknown is {retrieved_unknown} and relevant unknown is zero.")
    else:
        recall['unknown'] = retrieved_unknown / relevant_unknown
    return recall


def _load_conf_mats(trial_dir: str, is_cumul: bool = False) -> List[Dict[str, Dict[str, float]]]:
    if is_cumul:
        with open(os.path.join(trial_dir, 'results.json'), 'r') as fh:
            results = json.load(fh)
        conf_mats: List[Dict[str, Dict[str, float]]] = results['test']['confusion_matrix']
    else:
        conf_mats = None
        for c in ['conf_mats.json', 'conf-mats.json']:
            p = os.path.join(trial_dir, c)
            if os.path.exists(p):
                with open(p, 'r') as fh:
                    tmp = json.load(fh)
                    if type(tmp) == list:
                        conf_mats: List[Dict[str, Dict[str, float]]] = [v for v in tmp if type(v) == dict and len(v) > 0]
                    else:
                        conf_mats: List[Dict[str, Dict[str, float]]] = [v for v in tmp.values() if type(v) == dict]
        if conf_mats == None:
            raise FileNotFoundError(f"Result file {trial_dir}/({' | '.join(['conf_mats.json', 'conf-mats.json'])}) does not exist.")
    return conf_mats


def eval_open_world(trial_dir: str, closed_world_labels: List[str], conf_mat_mut: callable,
                    is_cumul=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conf_mats: List[Dict[str, Dict[str, float]]] = [conf_mat_mut(v) for v in _load_conf_mats(trial_dir, is_cumul)]
    recall = pd.DataFrame.from_dict([get_recall_open(s, closed_world_labels) for s in conf_mats])
    precision = pd.DataFrame.from_dict([get_precision_open(s, closed_world_labels) for s in conf_mats])
    accuracy = pd.DataFrame.from_dict([get_accuracy_open(s, closed_world_labels) for s in conf_mats])
    return recall, precision, accuracy


def eval_closed_world(trial_dir: str, closed_world_labels: List[str], conf_mat_mut: callable,
                      is_cumul=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conf_mats: List[Dict[str, Dict[str, float]]] = [conf_mat_mut(v) for v in _load_conf_mats(trial_dir, is_cumul)]
    recall = pd.DataFrame.from_dict([get_recall(s, closed_world_labels) for s in conf_mats])
    precision = pd.DataFrame.from_dict([get_precision(s, closed_world_labels) for s in conf_mats])
    accuracy = pd.DataFrame.from_dict([get_accuracy(s, closed_world_labels) for s in conf_mats])
    return recall, precision, accuracy

