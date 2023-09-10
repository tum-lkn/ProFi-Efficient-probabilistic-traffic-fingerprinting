"""
Creates the bigram clusters
"""

# Imports:
#   External Libraries

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

#   Files from this Project

from data_conversion import data_aggregation


def calculate_bigram_matrix(points, cluster_centers):

    """
    Calculates the bigram matrix out of bigram attributes

    Args:
        points (list): list of lists of tupel containing the 2D points sorted into different applications
        cluster_centers (np.array): 2D array with shape (num_clusters, 2) containing the cluster centers obtained by the kmeans algorithm

    Returns:
        bigram_best (np.array): 2D array with shape (num_applications, num_clusters) is the optimal bigram probability matrix
    """

    bigram_matrix = np.zeros((len(points), cluster_centers.shape[0]))

    for i in range(len(points)):

        points_app = points[i]
        tmp = 0

        for j in range(len(points_app)):

            tmp += np.nan_to_num(np.divide(1, np.linalg.norm(cluster_centers - points_app[j], ord=2, axis=1)), nan=0.0)

        bigram_matrix[i, :] = np.divide(tmp, np.sum(tmp))

    bigram_matrix[bigram_matrix >= 1 - 0.0001] = 1.0
    bigram_matrix[bigram_matrix <= 0.0001] = 0.0


    return bigram_matrix


def calculate_spva(bigram_matrix, weights):

    """
    Calculates spva (sum of probability variance in applications)

    Args:
        bigram_matrix (np.array): 2D array with shape (num_applications, num_clusters) containing bigram probabilities
        weights (list): list of probabilities of bigram observations

    Returns:
        spva (double): sum of probability variance in applications
    """

    spva = np.sum(weights * np.var(bigram_matrix, axis=1))


    return spva


def calculate_spvc(bigram_matrix):

    """
    Calculates spvc (sum of probability variance in clusters) 

    Args:
        bigram_matrix (np.array): 2D array with shape (num_applications, num_clusters) containing bigram probabilities

    Returns:
        spvc (double): sum of probability variance in clusters
    """

    spvc = np.sum(bigram_matrix.shape[0] * np.amax(bigram_matrix, axis=0) - np.sum(bigram_matrix, axis=0))


    return spvc


def calculate_pss(bigram_matrix, weights):

    """
    Calculates pss (product of spva and spvc) by calculating spva and spvc

    Args:
        bigram_matrix (np.array): 2D array with shape (num_applications, num_clusters) containing bigram probabilities
        weights (list): list of probabilities of bigram observations

    Returns:
        pss (double): product of spva and spvc
    """

    spva = calculate_spva(bigram_matrix, weights)

    spvc = calculate_spvc(bigram_matrix)

    pss = spva * spvc


    return pss


def optimize_bigram_matrix(points, weights):

    """
    Calculates the optimal bigram matrix out of attribute bigrams.
    Lets the number of clusters range from 2 to max_clusters.
    For each number of clusters:
    Calculates the centers of clusters with the k-means algorithm and attribute bigrams.
    Calculates the bigram matrix with the attribute bigrams and the cluster centers.
    Calculates the product of spva and spvc (pss).
    Choose the bigram matrix with the highest pss.

    Args:
        points (list): list of lists of tupel containing the 2D points sorted into different applications
        weights (list): list of probabilities of bigram observations

    Returns:
        bigram_best (np.array): 2D array with shape (num_applications, num_clusters) is the optimal bigram probability matrix
        kmeans_best (sklearn.kmeans): model of optimal cluster distribution by kmeans algorithm
    """

    pss = []
    bigram_matrix_list = []
    kmeans_list = []

    for i in range(2, 2 * len(weights)):

        kmeans = KMeans(n_clusters=i).fit(np.concatenate(points))

        bigram_matrix = calculate_bigram_matrix(points, np.array(kmeans.cluster_centers_))

        pss.append(calculate_pss(bigram_matrix, weights))

        bigram_matrix_list.append(bigram_matrix)

        kmeans_list.append(kmeans)


    bigram_best = bigram_matrix_list[np.argmax(pss)]

    cluster_center = kmeans_list[np.argmax(pss)].cluster_centers_


    return bigram_best, cluster_center


def run_bigram_matrix(bigram_list):

    """
    Optimizes the bigram matrix with respect to attribute bigrams, writes the bigram matrix to a file

    Args:
        bigram_list (list): contains the attribute bigrams

    Returns:
        /
    """

    weights = np.zeros(len(bigram_list))

    for i in range(len(bigram_list)):

        weights[i] = len(bigram_list[i])

    weights = np.divide(weights, np.sum(weights))
    
    bigram_matrix, cluster_center = optimize_bigram_matrix(bigram_list, weights)

    data_aggregation.save_bigram_matrix(bigram_matrix, cluster_center)


