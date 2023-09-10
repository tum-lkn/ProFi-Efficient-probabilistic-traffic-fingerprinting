"""
Utility functions to support plotting.
"""
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
from typing import Tuple, Dict, Any, List
import pandas as pd

COLW = 3.45
COLORS = ['#fb8072', '#80b1d3', '#fdb462', '#bebada', '#8dd3c7', '#ffed6f']
MARKERS = ['s', 'o', 'v', '^', 'x', '*']
plt.rc('axes', labelsize=9)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('axes', labelsize=9)
plt.rc('legend', fontsize=9)


def get_fig(ncols: float, aspect_ratio=0.618) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create figure and axes objects of correct size.

    Args:
        ncols (float): Percentage of one column in paper.
        aspect_ratio (float): Ratio of width to height. Default is golden ratio.

    Returns:
        fig (plt.Figure)
        ax (plt.Axes)
    """
    COLORMAP = plt.set_cmap('Set2')
    plt.set_cmap(COLORMAP)
    matplotlib.rcParams.update({'font.size': 8})
    matplotlib.rcParams.update({'font.family': 'serif'})

    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figwidth(ncols * COLW)
    fig.set_figheight(ncols * COLW * aspect_ratio)
    return fig, ax


def violin_compare_metrics(recall_phmm: pd.DataFrame, recall_mc: pd.DataFrame,
                           recall_knn: pd.DataFrame, recall_cumul: pd.DataFrame,
                           precision_phmm: pd.DataFrame, precision_mc: pd.DataFrame,
                           precision_knn: pd.DataFrame, precision_cumul: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = get_fig(0.66)
    matplotlib.rcParams.update({'font.size': 8})
    x = [0, 1, 2, 3]
    y1 = recall_phmm.values.flatten()
    y2 = recall_mc.values.flatten()
    y3 = recall_knn.values.flatten()
    y4 = recall_cumul.values.flatten()
    y1 = y1[np.logical_not(np.isnan(y1))]
    y2 = y2[np.logical_not(np.isnan(y2))]
    y3 = y3[np.logical_not(np.isnan(y3))]
    y4 = y4[np.logical_not(np.isnan(y4))]
    y = [y3, y2, y1, y4]
    parts = ax.violinplot(
        y,
        positions=x,
        showmeans=False,
        showmedians=False,
        widths=1
    )

    ax.plot([-2, -1], [-1, -2], c=COLORS[0], marker=MARKERS[0], linewidth=2, label='kNN', mec='black', mfc='white')
    ax.plot([-2, -1], [-1, -2], c=COLORS[1], marker=MARKERS[1], linewidth=2, label='MC', mec='black', mfc='white')
    ax.plot([-2, -1], [-1, -2], c=COLORS[2], marker=MARKERS[2], linewidth=2, label='PHMM', mec='black', mfc='white')
    ax.plot([-2, -1], [-1, -2], c=COLORS[3], marker=MARKERS[3], linewidth=2, label='CUMUL', mec='black', mfc='white')

    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname].set_edgecolor([COLORS[0], COLORS[1], COLORS[2], COLORS[3]])
    for i, pc in enumerate(parts['bodies']):
        pc.set_color(COLORS[i])
    for i, v in enumerate(y):
        ax.scatter(np.repeat(x[i], 100), v[np.random.randint(0, v.size, 100)], alpha=0.2, marker='_', c=COLORS[i])
    for i, (x_, y_) in enumerate(zip(x, y)):
        ax.scatter([x_], [np.mean(y_)], marker=MARKERS[i], edgecolors='black', zorder=2, facecolor='white',
                   linewidths=1.25)
        ax.scatter([x_], [np.median(y_)], marker="_", facecolor='black', zorder=2)

    x = [5, 6, 7, 8]
    y1 = precision_phmm.values.flatten()
    y2 = precision_mc.values.flatten()
    y3 = precision_knn.values.flatten()
    y4 = precision_cumul.values.flatten()
    y1 = y1[np.logical_not(np.isnan(y1))]
    y2 = y2[np.logical_not(np.isnan(y2))]
    y3 = y3[np.logical_not(np.isnan(y3))]
    y4 = y4[np.logical_not(np.isnan(y4))]
    y = [y3, y2, y1, y4]
    parts = ax.violinplot(
        y,
        positions=x,
        showmeans=False,
        showmedians=False,
        widths=1
    )
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname].set_edgecolor([COLORS[0], COLORS[1], COLORS[2], COLORS[3]])
    for i, pc in enumerate(parts['bodies']):
        pc.set_color(COLORS[i])
    for i, v in enumerate(y):
        ax.scatter(np.repeat(x[i], 100), v[np.random.randint(0, v.size, 100)], alpha=0.3, marker='_', c=COLORS[i])
    for i, (x_, y_) in enumerate(zip(x, y)):
        # ax.scatter([x_], [np.mean(y_)], marker=MARKERS[i], zorder=2, label=labels[i], edgecolors='black', facecolor='white', linewidths=1.25)
        ax.scatter([x_], [np.mean(y_)], marker=MARKERS[i], zorder=2, edgecolors='black', facecolor='white',
                   linewidths=1.25)
        ax.scatter([x_], [np.median(y_)], marker="_", zorder=2, facecolor='black')
    # ax.scatter(x, [np.mean(t) for t in y], marker=[MARKERS[0], MARKERS[1], MARKERS[2]], c=[COLORS[0], COLORS[1], COLORS[2]])

    # ax.scatter([0], [3], label='kNN', c=COLORS[0])
    # ax.scatter([0], [3], label='MC', c=COLORS[1])
    # ax.scatter([0], [3], label='PHMM', c=COLORS[2])
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlim((-1, 8.5))
    ax.set_xticks([1, 6])
    ax.set_xticklabels(['Recall', 'Precision'])
    ax.legend(frameon=False, ncol=2)
    ax.set_ylabel("Fraction")
    return fig, ax


def line_plot_compare_metric(dfs: List[pd.DataFrame], labels: List[str],
                             xlabel: str, ylabel: str, xlim=None, ylim=None,
                             linewidth=1.25, markevery=5, markersize=5,
                             ncols=1, legend_cols=1) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = get_fig(0.66)#, aspect_ratio=0.7)
    for i, df in enumerate(dfs):
        x = []
        y = []
        for j in range(df.shape[0]):
            y.append(df.iloc[j, :].dropna().mean() * 100)
            x.append(j + 1)
        print(labels[i], np.mean(y[:3]), np.mean(y[-3:]))
        ax.plot(x, y,
                c=COLORS[i], label=labels[i], linewidth=linewidth, marker=MARKERS[i % len(MARKERS)],
                markevery=15, markersize=markersize, mec='black', mfc='white')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.arange(10, 70, 25))
    ax.set_xticklabels(np.arange(10, 70, 25))
    ax.legend(frameon=False, ncol=legend_cols)
    return fig, ax










