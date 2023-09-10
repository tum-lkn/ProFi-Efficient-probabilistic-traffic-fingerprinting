# Imports:
#   External Libraries

import os
import sys
import json as js
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm as mcm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Any, Dict, List, Tuple

#   Files from this Project

sys.path.append('../')

from data_conversion import data_aggregation
from data_conversion import constants


def get_fig(ncols: int, aspect_ratio: float=0.618) -> Tuple[plt.figure, plt.axis]:

    """
    Create figure and axes objects of correct size.

    Args:
        ncols (float): Percentage of one column in paper.
        aspect_ratio (float): Ratio of width to height. Default is golden ratio.

    Returns:
        fig (plt.Figure)
        ax (plt.Axes)
    """
    COLW = 3.45
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figwidth(ncols * COLW)
    fig.set_figheight(ncols * COLW * aspect_ratio)


    return fig, ax


def draw_confusion_matrix(confusion_matrix: np.array, applications: List[str], browser: str) -> None:

    """
    Plots the confusion matrix

    Args:
        confusion_matrix (np.array): 2D array with shape (num_applications, num_applications)
        applications (list): list of applications

    Returns:
        /
    """

    conf_mat_file = os.path.join(constants.result_path, 'conf_mat_no_' + browser + '.png')

    confusion_matrix = confusion_matrix.astype(np.float64)

    for i in range(confusion_matrix.shape[0]):

        if np.sum(confusion_matrix[i, :]) != 0.:
            confusion_matrix[i, :] = np.divide(confusion_matrix[i, :], np.sum(confusion_matrix[i, :]))

    plt.set_cmap('Set2')
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'font.family': 'serif'})

    fig, ax = get_fig(1)
    cax = ax.imshow(confusion_matrix, cmap=mcm.get_cmap("Greens"))
    ax.set_yticks(np.arange(-0.5, confusion_matrix.shape[0] - 1, 1))
    ax.set_xticks(np.arange(-0.5, confusion_matrix.shape[1] - 1, 1))

    ax.set_xticklabels(np.arange(confusion_matrix.shape[1])) 
    ax.set_yticklabels(np.arange(confusion_matrix.shape[0]))

    dx = 10/72. 
    dy = 0/72.  

    offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans) 

    for label in ax.xaxis.get_majorticklabels(): 

        label.set_transform(label.get_transform() + offset) 

    dx = 0 
    dy = -10/72. 

    offset_y = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans) 

    for label in ax.yaxis.get_majorticklabels():

        label.set_transform(label.get_transform() + offset_y)

    ax.set_xticklabels(constants.applications_short)
    ax.set_yticklabels(constants.applications_short)
    plt.title('confusion matrix ' + browser)
    ax.set_xlabel('prediction')
    ax.set_ylabel('ground truth')
    plt.grid(color='black')
    m = mcm.ScalarMappable(cmap=mcm.get_cmap("Greens"))
    m.set_array(confusion_matrix)
    m.set_clim(0., 1.)
    plt.colorbar(m)

    # fig.tight_layout()

    plt.savefig(conf_mat_file, bbox_inches = 'tight')


#########################################################################################################################################


def calculate_quality_measurement(y_true: List[int], y_pred: List[int], applications: List[str]) -> Tuple[float, float, float, float, np.array]:

    """
    Calculates the confusion matrix, then calculates the accuracy, precision, recall and f1_score out of a confusion matrix

    Args:
        y_true (np.array): list of loglikelihoods of the actual assignment
        y_pred (np.array): list of loglikelihoods of the predicted assignment

    Returns:
        accuracy (double): accuracy of the prediction
        precision (double): precision of the prediction
        recall (double): recall of the prediction
        f1_score (double): f1_score of the prediction
    """

    confusion_matrix_true = np.zeros((len(applications) + 1, len(applications) + 1))
    for pred, act in zip(y_pred, y_true):
        confusion_matrix_true[act, pred] += 1

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')


    return accuracy, precision, recall, f1, confusion_matrix_true


def check_assignment(browser) -> Dict[str, Any]:

    """
    Checks the classification of the markov-chains with regards to accuracy, precision, recall, f1_score

    Args:
        applications (list): list of applications

    Returns:
        assignment (dict):  dict of assignment with quality measures as values
    """

    applications = data_aggregation.get_services()

    json_file_output = os.path.join(constants.result_path, 'stats_no_' + browser +'.json')

    json_files = data_aggregation.get_traces_test(browser)

    y_true = []
    y_pred = []

    for json_file in json_files:

        js_data = data_aggregation.load_prediction(os.path.join(constants.classification_path, json_file + '_clas.json'))

        if js_data['actual application'] == 'unknown':
            y_true.append(len(applications))
        else:
            y_true.append(applications.index(js_data['actual application']))
        if js_data['predicted application'] == 'unknown':
            y_pred.append(len(applications))
        else:
            y_pred.append(applications.index(js_data['predicted application']))

    accuracy, precision, recall, f1_score, confusion_matrix = calculate_quality_measurement(y_true, y_pred, applications)

    output = dict()
    output['number of applications'] = len(applications)
    output['applications'] = applications
    output['number of files'] = len(json_files)
    output['accuracy'] = accuracy
    output['precision'] = precision
    output['recall'] = recall
    output['f1_score'] = f1_score
    output['confusion_matrix'] = confusion_matrix.tolist()

    data_aggregation.save_quality_measurements(output, json_file_output)

    draw_confusion_matrix(confusion_matrix, applications, browser)

    return output
