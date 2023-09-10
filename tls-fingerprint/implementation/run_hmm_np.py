import numpy as np
import pandas as pd
import sys
import os

sys.path.append('../')
sys.path.append('../../')

from data_conversion import data_aggregation
from data_conversion import constants
from hmm_np import fingerprint_hmm
from hmm_np import classification_hmm


def set_constants(num_packets):

    constants.number_packets = num_packets


def check_model(num_packets):

    set_constants(num_packets)

    fingerprint_hmm.run_fingerprint_hmm()
    assignment = classification_hmm.run_classification_hmm()
    accuracy = assignment['accuracy']
    precision = assignment['precision']
    recall = assignment['recall']
    f1_score = assignment['f1_score']

    return accuracy, precision, recall, f1_score


def save_dataframe(accuracy_l, precision_l, recall_l, f1_score_l):

    csv_file = os.path.join(constants.result_path, 'model_search_results_hmm.csv')

    columns = ['packets', 'coflows', 'accuracy', 'precision', 'recall', 'f1_score']

    accuracy_pd = pd.DataFrame(columns = columns)
    accuracy_pd['packets'] = range(2, 31)
    accuracy_pd['coflows'] = [constants.co_flows] * len(range(2, 31))
    accuracy_pd['accuracy'] = accuracy_l
    accuracy_pd['precision'] = precision_l
    accuracy_pd['recall'] = recall_l
    accuracy_pd['f1_score'] = f1_score_l

    accuracy_pd.to_csv(csv_file, index = False, mode = 'a', header = False)


def search_opt_model():

    accuracy_l = []
    precision_l = []
    recall_l = []
    f1_score_l = []

    for i in range(2, 31):

        accuracy, precision, recall, f1_score = check_model(i)
        accuracy_l.append(accuracy)
        precision_l.append(precision)
        recall_l.append(recall)
        f1_score_l.append(f1_score)

    save_dataframe(accuracy_l, precision_l, recall_l, f1_score_l)


if __name__ == '__main__' :
    
    search_opt_model()
