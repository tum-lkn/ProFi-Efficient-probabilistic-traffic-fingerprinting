import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict, Any, Union

import plots.utils as plutils
import implementation.data_conversion.classification as dcc
import implementation.data_conversion.prototype as dcp


# COLORS = ['#fb8072', '#80b1d3', '#fdb462', '#bebada', '#8dd3c7', '#ffed6f']
STYLE = 'phd'
FONT_SIZES = {'phd': 9.75, 'tnsm': 8}
IMG_DIR = {'phd': '/opt/project/img/phd', 'tnsm': '/opt/project/img/tnsm'}[STYLE]
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)
COLORS = {
    'phd': ['#003f5c', '#ffa600', '#f95d6a', '#ff7c43', '#d45087', '#a05195', '#665191', '#2f4b7c'],
    'tnsm': plutils.COLORS
}[STYLE]
if STYLE == 'phd':
    plutils.COLORS = COLORS
MARKERS = ['s', 'o', 'v', '^', '<', '>']
plt.rc('axes', labelsize=FONT_SIZES[STYLE])
plt.rc('xtick', labelsize=FONT_SIZES[STYLE])
plt.rc('ytick', labelsize=FONT_SIZES[STYLE])
plt.rc('axes', labelsize=FONT_SIZES[STYLE])
plt.rc('legend', fontsize=FONT_SIZES[STYLE])
plt.rc("font", **{'family': 'serif', 'serif': ['palatino']})
plt.rcParams['text.usetex'] = True
# plt.rcParams["savefig.bbox"] = 'tight'
# plt.rcParams["savefig.pad_inches"] = 0.


COMMON_PREFIXES = ['www.amazon', 'www.google', 'www.ebay']
with open('/opt/project/open-world-labels.json', 'r') as fh:
    OPEN_WORLD_LABELS = json.load(fh)
with open('/opt/project/closed-world-labels.json', 'r') as fh:
    CLOSED_WORLD_LABELS = json.load(fh)
with open('/opt/project/data/cache/labels.json', 'r') as fh:
    LABELS = json.load(fh)
with open('/opt/project/data/cache/meta_data.json', 'r') as fh:
    META_DATA = json.load(fh)
edir = '/opt/project/data/grid-search-results/eval-results/'
CLOSED_WORLD_TRIAL_DIRS = {
    'phmm': '/opt/project/data/grid-search-results/closed-world-phmm-results-all-data-loaded',
    'mc': '/opt/project/data/grid-search-results/closed-world-mc-results-all-data-loaded/',
    'knn': '/opt/project/data/grid-search-results/047f435fd54642f4b2674ac21e08d27f-closed-world-eval/',
    'cumul': os.path.join(edir, 'cumul-closed-world/'),
    'popets': os.path.join(edir, 'popets-closed/')
}
CLOSED_WORLD_TRIAL_DIRS_DEFENSE = {
    'phmm': os.path.join(edir, 'closed-world-phmm-training-days-70-RandomRecordSizeDefense-50-100/'),
    'mc': os.path.join(edir, 'closed-world-mc-training-days-70-RandomRecordSizeDefense-50-100'),
    'knn': os.path.join(edir, 'closed-world-knn-days-70---RandomRecordSizeDefense-50-100'),
    'cumul': os.path.join(edir, 'cumul-closed-world-RandomRecordSizeDefense-50-100/')
}
OPEN_WORLD_TRIAL_DIRS = {
    'phmm': '/opt/project/data/grid-search-results/open-world-phmm-results-all-data-loaded/',
    'mc': '/opt/project/data/grid-search-results/open-world-mc-results-all-data-loaded/',
    'knn': '/opt/project/data/grid-search-results/047f435fd54642f4b2674ac21e08d27f-open-world-eval-old-commit/',
    'cumul': os.path.join(edir, 'cumul-open-world'),
    'popets': os.path.join(edir, 'popets-open/')
}
OPEN_WORLD_TRIAL_DIRS_DEFENSE = {
    'phmm': os.path.join(edir, 'open-world-phmm-training-days-70-RandomRecordSizeDefense-50-100/'),
    'mc': os.path.join(edir, 'open-world-mc-training-days-70-RandomRecordSizeDefense-50-100/'),
    'knn': os.path.join(edir, 'open-world-knn-days-70---RandomRecordSizeDefense-50-100'),
    'cumul': os.path.join(edir, 'cumul-open-world-RandomRecordSizeDefense-50-100')
}


def get_record_sizes(label: str, num_frames=30):
    all_sizes = []
    for day, traces in META_DATA:
        for meta_record in traces:
            url_id = str(meta_record['url_id'])
            if url_id not in LABELS:
                continue
            if LABELS[url_id] != label:
                continue
            if not os.path.exists(f'/opt/project/data/k8s-json/{meta_record["filename"]}.json'):
                continue
            with open(f'/opt/project/data/k8s-json/{meta_record["filename"]}.json', 'r') as fh:
                flow_dict = json.load(fh)
            sizes = []
            for packet in flow_dict['frames'][:num_frames]:
                current_record_id = None
                for record in packet['tls_records']:
                    if record['record_number'] == current_record_id:
                        continue
                    else:
                        current_record_id = record['record_number']
                        sizes.append(record['length'] * record['direction'])
            all_sizes.append(sizes)
    return all_sizes


def get_frame_sizes(label: str, num_frames=30):
    all_sizes = []
    for day, traces in META_DATA:
        for meta_record in traces:
            url_id = str(meta_record['url_id'])
            if url_id not in LABELS:
                continue
            if LABELS[url_id] != label:
                continue
            if not os.path.exists(f'/opt/project/data/k8s-json/{meta_record["filename"]}.json'):
                continue
            with open(f'/opt/project/data/k8s-json/{meta_record["filename"]}.json', 'r') as fh:
                flow_dict = json.load(fh)
            sizes = []
            for packet in flow_dict['frames'][:num_frames]:
                sizes.append(packet['tcp_length'] * packet['direction'])
                if packet['tcp_length'] > 1600:
                    print(f"Trace contains packets larger MTU for {meta_record['filename']}.json")
                    sizes = None
                    break
            if sizes is not None:
                all_sizes.append(sizes)
    return all_sizes


def plot_timings():
    with open('/opt/project/data/grid-search-results/timings-phmm/inference-timings.json', 'r') as fh:
        timings_phmm = json.load(fh)
    with open('/opt/project/data/grid-search-results/timings-mc/inference-timings.json', 'r') as fh:
        timings_mc = json.load(fh)
    with open('/opt/project/data/grid-search-results/timings-knn/inference-timings.json', 'r') as fh:
        timings_knn = json.load(fh)
    with open('/opt/project/data/grid-search-results/eval-results/cumul-closed-world/result.json', 'r') as fh:
        timings_cumul = json.load(fh)['evaluation_time_per_sample']

    print("Timings:", [1 / np.mean(timings_phmm), 1 / np.mean(timings_mc), 1 / np.mean(timings_knn), 1 / np.mean(timings_cumul)])
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 1)
    fig.set_figheight(fig.get_figheight() * 0.7)
    x = [0, 1, 3, 4]
    ax.barh(
        x,
        [1 / np.mean(timings_phmm), 1 / np.mean(timings_mc), 1 / np.mean(timings_knn), 1 / np.mean(timings_cumul)],
        color=[COLORS[0], COLORS[0], COLORS[1], COLORS[1]]
    )
    ax.scatter([-1], [-1], color=COLORS[0], label='\\textsc{{ProFi}}', marker='s')
    ax.scatter([-1], [-1], color=COLORS[1], label='SoA', marker='s')
    ax.set_xscale('log')
    ax.set_yticks(x)
    ax.set_yticklabels(['PHMM', 'MC', 'kNN', 'CUMUL'])
    ax.set_xlabel("Classifications per second")
    # ax.set_xlim(3.5e-1, 2e2)
    ax.set_ylim(-0.5, 4.6)
    ax.legend(frameon=False, handletextpad=0.5)
    plt.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.27)
    plutils.save_fig(os.path.join(IMG_DIR, 'inference-times.pdf'), ax)
    # plt.savefig(os.path.join(IMG_DIR, 'inference-times.pdf'), pad_inches=0, bbox_inches='tight')
    plt.close('all')


def plot_violin_open_world(trial_dirs: Dict[str, str], aggregate: bool, suffix: str):
    if aggregate:
        mutate_conf_mat = dcc.AggregateCommonPrefixes(COMMON_PREFIXES)
    else:
        mutate_conf_mat = lambda x: x
    recall_phmm, precision_phmm, accuracy_phmm = dcc.eval_open_world(
        trial_dir=trial_dirs['phmm'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=False
    )
    recall_mc, precision_mc, accuracy_mc = dcc.eval_open_world(
        trial_dir=trial_dirs['mc'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=False
    )
    recall_knn, precision_knn, accuracy_knn = dcc.eval_open_world(
        trial_dir=trial_dirs['knn'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=False
    )
    recall_cumul, precision_cumul, accuracy_cumul = dcc.eval_open_world(
        trial_dir=trial_dirs['cumul'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=True
    )
    recall_popets, precision_popets, accuracy_popets = dcc.eval_open_world(
        trial_dir=trial_dirs['popets'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=True
    )
    fig, ax = plutils.violin_compare_metrics(recall_phmm, recall_mc, recall_knn,
                                             recall_cumul, recall_popets, precision_phmm,
                                             precision_mc, precision_knn, precision_cumul,
                                             precision_popets)
    # fig.set_figheight(fig.get_figheight() * 1.2)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2, frameon=False, handletextpad=0.5)
    fig.subplots_adjust(left=.22, bottom=.15, right=.99, top=0.7)
    plutils.save_fig(os.path.join(IMG_DIR, f'violin-open-world-recall-precision{suffix}.pdf'), ax)
    plt.close('all')


def plot_violin_closed_world(trial_dirs: Dict[str, str], aggregate: bool, suffix: str):
    if aggregate:
        mutate_conf_mat = dcc.AggregateCommonPrefixes(COMMON_PREFIXES)
    else:
        mutate_conf_mat = lambda x: x
    recall_phmm, precision_phmm, accuracy_phmm = dcc.eval_closed_world(
        trial_dir=trial_dirs['phmm'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=False
    )
    recall_mc, precision_mc, accuracy_mc = dcc.eval_closed_world(
        trial_dir=trial_dirs['mc'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=False
    )
    recall_knn, precision_knn, accuracy_knn = dcc.eval_closed_world(
        trial_dir=trial_dirs['knn'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=False
    )
    recall_cumul, precision_cumul, accuracy_cumul = dcc.eval_closed_world(
        trial_dir=trial_dirs['cumul'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=True
    )
    recall_popets, precision_popets, accuracy_popets = dcc.eval_closed_world(
        trial_dir=trial_dirs['popets'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=True
    )
    fig, ax = plutils.violin_compare_metrics(recall_phmm, recall_mc, recall_knn,
                                             recall_cumul, recall_popets, precision_phmm,
                                             precision_mc, precision_knn, precision_cumul,
                                             precision_popets)
    # fig.set_figheight(fig.get_figheight() * 1.2)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2, frameon=False, handletextpad=0.5)
    fig.subplots_adjust(left=.22, bottom=.15, right=.99, top=0.7)
    plutils.save_fig(os.path.join(IMG_DIR, f'violin-closed-world-recall-precision{suffix}.pdf'), ax)


def plot_line_compare_metrics(scenario: str, trial_dirs: Dict[str, str], metric: str,
                              suffix=''):
    mutate_conf_mat = lambda x: x
    evaluator = {
        'open': dcc.eval_open_world,
        'closed': dcc.eval_closed_world
    }[scenario]
    recall_phmm, precision_phmm, accuracy_phmm = evaluator(
        trial_dir=trial_dirs['phmm'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=False
    )
    recall_mc, precision_mc, accuracy_mc = evaluator(
        trial_dir=trial_dirs['mc'],
        closed_world_labels=CLOSED_WORLD_LABELS,
        conf_mat_mut=mutate_conf_mat,
        is_cumul=False
    )
    # recall_knn, precision_knn, accuracy_knn = evaluator(
    #     trial_dir=trial_dirs['knn'],
    #     closed_world_labels=CLOSED_WORLD_LABELS,
    #     conf_mat_mut=mutate_conf_mat,
    #     is_cumul=False
    # )
    # recall_cumul, precision_cumul, accuracy_cumul = evaluator(
    #     trial_dir=trial_dirs['cumul'],
    #     closed_world_labels=CLOSED_WORLD_LABELS,
    #     conf_mat_mut=mutate_conf_mat,
    #     is_cumul=True
    # )
    # knn, mc, phmm, cumul = {
    #     'precision': (precision_knn, precision_mc, precision_phmm, precision_cumul),
    #     'recall': (recall_knn, recall_mc, recall_phmm, recall_cumul),
    #     'accuracy': (accuracy_knn, accuracy_mc, accuracy_phmm, accuracy_cumul)
    # }[metric]
    mc, phmm = {
        'precision': ( precision_mc, precision_phmm),
        'recall': ( recall_mc, recall_phmm),
        'accuracy': (accuracy_mc, accuracy_phmm)
    }[metric]

    fig, ax = plutils.line_plot_compare_metric(
        # [knn, mc, phmm, cumul],
        [mc, phmm],
        # ['kNN', 'MC', 'PHMM', 'CUMUL'],
        ['MC', 'PHMM'],
        'Day', f'Avg. {metric[0].upper()}{metric[1:]} [\%]',
        ylim=(35, 101),
        legend_cols=2,
        linewidth=2,
        ncols=0.73 if STYLE == 'phd' else 0.5
    )
    # fig.subplots_adjust(left=.2, bottom=.275, right=.99, top=.98)
    # fig.set_figheight(fig.get_figheight() * 1.3)
    fig.subplots_adjust(left=.238, bottom=.28, right=.99, top=0.97)
    # plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=1, frameon=False)
    plt.legend(mode="expand", ncol=1, frameon=False, handletextpad=0.5)
    plutils.save_fig(os.path.join(IMG_DIR, f"line-{scenario}-world-{metric}-over-time{suffix}.pdf"), ax)
    plt.close('all')


def plot_line_training_over_days():
    # Get the confusion matrices.
    dtt_phmm = {}
    dtt_mc = {}
    dtt_knn = {}
    dtt_cumul = {}
    for days_to_train_on in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]:
        name = f'open-world-phmm-results-days-to-train-{days_to_train_on}'
        if os.path.exists(f'/opt/project/data/grid-search-results/{name}/conf-mats.json'):
            with open(f'/opt/project/data/grid-search-results/{name}/conf-mats.json', 'r') as fh:
                dtt_phmm[days_to_train_on] = json.load(fh)
        if os.path.exists(f'/opt/project/data/grid-search-results/{name}/conf-mats.json'):
            name = f'open-world-mc-results-days-to-train-{days_to_train_on}'
            with open(f'/opt/project/data/grid-search-results/{name}/conf-mats.json', 'r') as fh:
                dtt_mc[days_to_train_on] = json.load(fh)
        name = f'cumul-open-over-days-{days_to_train_on}'
        if os.path.exists(f'/opt/project/data/cumul-interpolations/{name}/results.json'):
            with open(f'/opt/project/data/cumul-interpolations/{name}/results.json', 'r') as fh:
                dtt_cumul[days_to_train_on] = json.load(fh)
        name = f"047f435fd54642f4b2674ac21e08d27f-open-world-eval-over-days-no-symbol-change-{days_to_train_on}"
        if os.path.exists(f'/opt/project/data/grid-search-results/{name}/conf-mats.json'):
            with open(f'/opt/project/data/grid-search-results/{name}/conf-mats.json', 'r') as fh:
                l = json.load(fh)
            d = 0
            tmp = {}
            for x in l:
                if len(x) == 0:
                    continue
                tmp[d] = x
                d += 1
            dtt_knn[days_to_train_on] = tmp

    # Compute the precision data frames.
    dtt_phmm_precision = {}
    dtt_mc_precision = {}
    dtt_knn_precision = {}
    dtt_cumul_precision = {}
    for k, conf_mats in dtt_phmm.items():
        dtt_phmm_precision[k] = pd.DataFrame.from_dict(
            [dcc.get_precision_open(s, CLOSED_WORLD_LABELS) for s in conf_mats.values() if type(s) == dict])
    for k, conf_mats in dtt_mc.items():
        dtt_mc_precision[k] = pd.DataFrame.from_dict(
            [dcc.get_precision_open(s, CLOSED_WORLD_LABELS) for s in conf_mats.values() if type(s) == dict])
    for k, conf_mats in dtt_knn.items():
        dtt_knn_precision[k] = pd.DataFrame.from_dict(
            [dcc.get_precision_open(s, CLOSED_WORLD_LABELS) for s in conf_mats.values() if type(s) == dict])
    for k, conf_mats in dtt_cumul.items():
        dtt_cumul_precision[k] = pd.DataFrame.from_dict(
            [dcc.get_precision_open(s, CLOSED_WORLD_LABELS) for s in conf_mats['test']['confusion_matrix'] if
             type(s) == dict])

    # make the plot.
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.66)
    ax.plot(list(dtt_phmm_precision.keys()), [x.mean().mean() for x in dtt_phmm_precision.values()], c=COLORS[2],
            label='PHMM', marker=MARKERS[1], markevery=3, mec='black', mfc='white', lw=2)
    ax.plot(list(dtt_cumul_precision.keys()), [x.mean().mean() for x in dtt_cumul_precision.values()], c=COLORS[3],
            label='CUMUL', marker=MARKERS[3], markevery=3, mec='black', mfc='white', lw=2)
    ax.plot(list(dtt_mc_precision.keys()), [x.mean().mean() for x in dtt_mc_precision.values()], c=COLORS[1],
            label='MC', marker=MARKERS[2], markevery=3, mec='black', mfc='white', lw=2)
    ax.plot(list(dtt_knn_precision.keys()), [x.mean().mean() for x in dtt_knn_precision.values()], c=COLORS[0],
            label='kNN', marker=MARKERS[0], markevery=3, mec='black', mfc='white', lw=2)
    ax.legend(frameon=False, ncol=2, handletextpad=0.5)
    ax.set_xlabel("Days trained on")
    ax.set_ylabel("Avg. Precision")
    ax.set_ylim(0.14, 1)
    # fig.subplots_adjust(left=.2, bottom=.275, right=.99, top=.965)
    # fig.set_figheight(fig.get_figheight() * 1.3)
    fig.subplots_adjust(left=.22, bottom=.225, right=.99, top=0.75)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2, frameon=False)
    plutils.save_fig(os.path.join(IMG_DIR, f'line-open-days-trained-on-precision.pdf'), ax)
    plt.close('all')


    k = 1
    with open('/opt/project/data/cumul-interpolations/cumul-open-over-days-1/results.json', 'r') as fh:
        cumul_open_over_days = json.load(fh)
    open_precision_cumul_1d = pd.DataFrame.from_dict(
        [dcc.get_precision_open(s, CLOSED_WORLD_LABELS) for s in cumul_open_over_days['test']['confusion_matrix'] if
         type(s) == dict])
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.66)
    ax.plot(dtt_phmm_precision[k].mean(axis=1).values, label='PHMM', c=COLORS[2], marker=MARKERS[1], markevery=10,
            mec='black', mfc='white', lw=2)
    ax.plot(open_precision_cumul_1d.mean(axis=1).values, label='CUMUL', c=COLORS[3], marker=MARKERS[3], markevery=10,
            mec='black', mfc='white', lw=2)
    ax.plot(dtt_mc_precision[k].mean(axis=1).values, label='MC', c=COLORS[1], marker=MARKERS[2], markevery=10,
            mec='black', mfc='white', lw=2)
    ax.plot(dtt_knn_precision[k].mean(axis=1).values, label='kNN', c=COLORS[0], marker=MARKERS[0], markevery=10,
            mec='black', mfc='white', lw=2)
    # ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("Day")
    ax.set_ylabel("Avg. Precision")
    ax.legend(frameon=False, ncol=2)
    ax.set_ylim(0.14, 1)
    # fig.subplots_adjust(left=.2, bottom=.275, right=.99, top=.965)
    # fig.set_figheight(fig.get_figheight() * 1.3)
    fig.subplots_adjust(left=.22, bottom=.225, right=.99, top=0.75)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2, frameon=False)
    plutils.save_fig(os.path.join(IMG_DIR, f'line-open-days-trained-on-precision-day-1.pdf'), ax)
    plt.close('all')


def _make_legend(labels: List[str], width: float, filename: str, ncol: int):
    custom_lines = [
        Line2D([0], [0], color=COLORS[i], marker=MARKERS[i], markeredgecolor='black', markerfacecolor='white')
        for i in range(len(labels))
    ]
    fig, ax = plutils.get_fig(width)
    ax.set_yticklabels([])
    legend = ax.legend(custom_lines, labels, ncol=ncol, frameon=False, mode="expand", handletextpad=0.5)

    f = legend.figure
    f.canvas.draw()
    bbox = legend.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    # fig.subplots_adjust(left=left, bottom=bottom, right=.99, top=top)
    plt.savefig(os.path.join(IMG_DIR, filename), bbox_inches=bbox)
    plt.close('all')


def plot_violin_scenarios(trial_dirs: Dict[str, Dict[str, str]], metric: str):
    colwidth = 0.5
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else colwidth)
    pos = 0
    legend_labels = []
    xlabels = []
    xticks = []
    for j, (model, scenarios_d) in enumerate(trial_dirs.items()):
        xlabels.append(model)
        for i, (scenario, trial_dir) in enumerate(scenarios_d.items()):
            evaluator = {
                'open': dcc.eval_open_world,
                'closed': dcc.eval_closed_world,
                'open-def': dcc.eval_open_world,
                'closed-def': dcc.eval_closed_world,
                'open-fs2c': dcc.eval_open_world,
                'open-fc2s': dcc.eval_open_world
            }[scenario]
            recall, precision, accuracy = evaluator(
                trial_dir=trial_dir,
                closed_world_labels=CLOSED_WORLD_LABELS,
                conf_mat_mut=lambda x: x,
                is_cumul=trial_dir.find('cumul-') >= 0
            )
            y = {
                'recall': recall,
                'precision': precision,
                'accuracy': accuracy
            }[metric]# .values.flatten()
            # y = y[np.logical_not(np.isnan(y))]
            y = np.array([y.iloc[:, k].dropna().mean() for k in range(y.shape[1])]) * 100
            parts = ax.violinplot(
                [y],
                positions=[pos],
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=1
            )
            if scenario not in legend_labels:
                legend_labels.append(scenario)
                ax.plot([-9, -9], [-10, -10], c=COLORS[i], marker=MARKERS[i], linewidth=2,
                        label=scenario, mec='black', mfc='white')
            #for partname in ['cbars', 'cmins', 'cmaxes']:
            #    parts[partname].set_edgecolor([COLORS[i]])
            for _, pc in enumerate(parts['bodies']):
                pc.set_color(COLORS[i])
            print(metric, model, scenario, 'mean', np.mean(y), 'median', np.median(y), ">90", np.sum(y>90)/y.size)
            ax.scatter(np.repeat(pos, 100), y[np.random.randint(0, y.size, 100)], alpha=0.2, marker='o', c=COLORS[i], s=3)
            ax.scatter([pos], [np.mean(y)], marker=MARKERS[i], edgecolors='black', zorder=2, facecolor='white', linewidths=1.25)
            ax.scatter([pos], [np.median(y)], marker="_", facecolor='black', zorder=2)
            pos += 1
        xticks.append((2 * pos - len(scenarios_d)) / 2.)
        pos += 1.5
    # plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3, frameon=False)
    if colwidth == 1:
        fig.subplots_adjust(left=.1555, bottom=.12, right=.99, top=0.8)
    else:
        fig.subplots_adjust(left=.238, bottom=.25, right=.99, top=0.99)
    ax.set_ylim(-10, 110)
    ax.set_xlim(-0.75, pos + 0.75 - 2.5)
    ax.set_ylabel(metric[0].upper() + metric[1:] + " [\%]")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(['0\,\%', '50\,\%', '100\,\%'])
    plutils.save_fig(os.path.join(IMG_DIR, f'violin-compare-scenarios-{metric}.pdf'), ax)
    plt.close('all')
    _make_legend(legend_labels, 0.73 * 1.75 if STYLE == 'phd' else 1, f'cdf-legend.pdf', 3)


def plot_cdf_seq_lengths():
    lengths = {}
    for model in ['phmm', 'mc']:
        lengths[model] = {}
        td = CLOSED_WORLD_TRIAL_DIRS[model]
        for f in os.listdir(td):
            if f.endswith('.json'):
                with open(os.path.join(td, f), 'r') as fh:
                    cfg = json.load(fh)
                if 'seq_length' in cfg:
                    set = cfg['seq_element_type']
                    if set not in lengths[model]:
                        lengths[model][set] = []
                    lengths[model][set].append(cfg['seq_length'])
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else .8)
    pos = 0
    xlabel = []
    xpos = []
    count = 0
    seq_elems = {
        'frame': 'pkt',
        'record': 'rcd'
    }
    for i, (model, values) in enumerate(lengths.items()):
        for j, (seq_t, lts) in enumerate(values.items()):
            cdf = np.cumsum(pd.Series(lts).value_counts(normalize=True).sort_index())
            ax.plot(cdf.index.values, cdf.values, c=COLORS[count], marker=MARKERS[count],
                    label=f"{model.upper()}-{seq_elems[seq_t]}"
                    #label=f"{model.upper()}/{seq_t[0].upper()}{seq_t[1:]}"
                    )
            # parts = ax.violinplot(
            #     dataset=[lts],
            #     positions=[pos],
            #     showextrema=False,
            #     showmeans=False,
            #     showmedians=False,
            #     vert=False
            # )
            # for _, pc in enumerate(parts['bodies']):
            #     pc.set_color(COLORS[j])
            # ax.scatter([np.mean(lts)], [pos], marker=MARKERS[j], facecolor='white', edgecolor='black', zorder=2)
            # ax.scatter([np.median(lts)], [pos], marker='_', facecolor='black', zorder=2)
            # if i == 0:
            #     ax.plot([-2, -1], [-1, -2], c=COLORS[j], marker=MARKERS[j], linewidth=2,
            #             label=seq_t, mec='black', mfc='white')
            kwargs = {
                'record+phmm': {'x': 11.7, 'y': .982},
                'frame+phmm': {'x': 15, 'y': .985},
                'record+mc': {'x': 25, 'y': .91},
                'frame+mc': {'x': 15, 'y': .81}
            }
            kwargs = kwargs[f'{seq_t}+{model}']
            ax.text(horizontalalignment='center', fontdict={'color': COLORS[count]}, s=len(lts), **kwargs)
            pos += 1
            count += 1
        xpos.append((2 * pos - len(values)) / 2.)
        xlabel.append(model)
        pos += 1.5
    # plt.legend(frameon=False)
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #             mode="expand", borderaxespad=0, ncol=2, frameon=False)
    # ax.set_ylim(-0.75, pos - 2.5 + 0.75)
    # ax.set_xlim(-4.5, 35)
    # ax.set_yticks(xpos)
    # ax.set_yticklabels(xlabel)
    ax.set_ylabel("$P(X<x)$")
    ax.set_xlabel("Sequence Length")
    fig.subplots_adjust(left=.17, bottom=.22, right=.99, top=0.99)
    plutils.save_fig(os.path.join(IMG_DIR, f'violine-seq-element.pdf'), ax)
    plt.close('all')


def bar_num_pgms(data: List[Dict[str, Any]]):
    data2 = {}
    for sample in data:
        if sample['workload'] not in data2:
            data2[sample['workload']] = {}
        if sample['model'] not in data2[sample['workload']]:
            data2[sample['workload']][sample['model']] = {}
        data2[sample['workload']][sample['model']][sample['type']] = sample['num_pgms']
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.65)
    # fig.set_figheight(1.3 * fig.get_figheight())
    pos = 0
    yticks = []
    yticklabels = []
    for j, workload in enumerate(['high-fps', 'high-pps', 'average']):
        for i, model in enumerate(['mc', 'phmm']):
            ax.barh(
                y=[pos],
                width=[data2[workload][model]['small']],
                height=1,
                left=[0],
                color=[COLORS[0]],
                edgecolor='black',
                align='edge',
                hatch=['/////', 'xxxxx'][i],
                label=f'{model.upper() if model == "mc" else "PH"}-S' if i == j else None
            )
            ax.barh(
                y=[pos],
                width=[data2[workload][model]['big']],
                height=1,
                left=[data2[workload][model]['small']],
                color=[COLORS[1]],
                edgecolor='black',
                align='edge',
                hatch=['/////', 'xxxxx'][i],
                label=f'{model.upper() if model == "mc" else "PH"}-L' if i == j else None
            )
            pos += 1
        yticks.append((2 * pos - 2) / 2.)
        yticklabels.append(workload)
        pos += 0.75
    ax.set_xlabel("Num PGMs")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_label("Num PGMs")
    plt.legend(bbox_to_anchor=(0, 0.93, 1, 0), frameon=False, ncol=2, handletextpad=0.5)
    fig.subplots_adjust(left=.28, bottom=.20, right=.99, top=0.8)
    plutils.save_fig(os.path.join(IMG_DIR, f'bar-num-pgms.pdf'), ax)
    plt.close('all')


def violin_cost_filter(df: pd.DataFrame):
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.65)
    #fig.set_figheight(1.3 * fig.get_figheight())
    pos = 0
    ypos = []
    ylbl = []
    for i, workload in enumerate(['average', 'high-pps', 'high-fps']):
        for j, size in enumerate(['small', 'big']):
            y = df.loc[pd.IndexSlice[workload, size], 'Filter'].values / 2.2e9 * 1e6
            parts = ax.violinplot(
                [y],
                positions=[pos],
                showmeans=False,
                showmedians=False,
                showextrema=False,
                vert=False
            )
            for _, pc in enumerate(parts['bodies']):
                pc.set_color(COLORS[j])
            if i == 0:
                ax.plot([-2, -1], [-1, -2], c=COLORS[j], marker=MARKERS[j], linewidth=2,
                        label=size, mec='black', mfc='white')
            ax.scatter(y[np.random.randint(0, y.size, 100)], np.repeat(pos, 100), alpha=0.2, marker='o', c=COLORS[j],
                       s=3)
            ax.scatter([np.mean(y)], [pos], marker=MARKERS[j], edgecolors='black', zorder=2, facecolor='white',
                       linewidths=1.25)
            ax.scatter([np.median(y)], [pos], marker="|", facecolor='black', zorder=2)
            pos += 1
        ylbl.append(workload)
        ypos.append((2 * pos - 2) / 2)
        pos += 0.75
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylbl)
    ax.set_xlabel("Processing Time [us]")
    ax.set_xlim(0.190, 0.320)
    ax.set_ylim(-0.75, pos - 1.75 + 0.75)
    plt.legend(frameon=False, handletextpad=0.5)
    fig.subplots_adjust(left=.27, bottom=.23, right=.98, top=0.98)
    plutils.save_fig(os.path.join(IMG_DIR, f'filter-processing-cost.pdf'), ax)
    plt.close('all')


def plot_filter_cost():
    def is_big_model(p: str):
        return p.find('big-mc') >= 0 or p.find('hmm-len-30') >= 0

    BASE_DIR = '/opt/project/data/bigdata/protype-measures-ccs'
    samples = []
    for f in os.listdir(BASE_DIR):
        td = os.path.join(BASE_DIR, f)
        if not os.path.isdir(td):
            continue
        parts = f.split('-')
        data = dcp.slice_experiment_period(dcp._read_result(os.path.join(td, 'VNF_stats.txt')))
        try:
            samples.append({
                'model': parts[0],
                'workload': parts[1] if parts[1] == 'average' else f'{parts[1]}-{parts[2]}',
                'model-size': 'big' if is_big_model(f) else 'small',
                'Filter': np.mean(dcp.calc_cost_pp(data, 1)),
                'RecDet': np.mean(dcp.calc_cost_pp(data, 2))
            })
        except Exception as e:
            print(f"Error for file {td}")
            print(e)
    df = pd.DataFrame.from_dict(samples).set_index(['workload', 'model-size']).sort_index()
    violin_cost_filter(df)


def violin_compare_ttl_average() -> None:
    ttl_data = dcp.read_ttlbls('/opt/project/data/bigdata/protype-measures-ccs')
    data = {}
    for sample in ttl_data:
        model = sample['model']
        model_t = sample['model_size']
        if model_t not in data:
            data[model_t] = {}
        if model not in data[model_t]:
            data[model_t][model] = np.array([])
        data[model_t][model] = np.concatenate((data[model_t][model], sample['ttlbls']))

    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.66)
    #fig.set_figheight(fig.get_figheight() * 1.3)
    pos = 0
    xticks = []
    xlabels = []

    for i, model_t in enumerate(['small', 'big']):
        for j, model in enumerate(['mc', 'phmm']):
            ttls = data[model_t][model] * 1000
            print(model, model_t, np.mean(ttls), np.median(ttls))
            mask = np.logical_and(ttls < np.percentile(ttls, 99), ttls > np.percentile(ttls, 1))
            ttls = ttls[mask]
            parts = ax.violinplot(
                [ttls],
                positions=[pos],
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for pc in parts['bodies']:
                pc.set_color(COLORS[j])
            if i == 0:
                ax.plot([-1, -2], [1, 1], label=model.upper(), marker=MARKERS[j], c=COLORS[j], mfc='white', mec='black')
            tmp = ttls[np.random.randint(0, len(ttls), size=100)]
            ax.scatter([pos] * 100, tmp, marker='o', s=2, alpha=0.2, c=COLORS[j])
            ax.scatter([pos], [np.mean(ttls)], marker=MARKERS[j], edgecolor='black', zorder=2, facecolor='white')
            ax.scatter([pos], [np.median(ttls)], marker='_', facecolor='black', zorder=2)
            pos += 1
        xticks.append((2 * pos - 3) / 2)
        xlabels.append(model_t)
        pos += 0.75
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Time To Label [ms]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_yscale('log')
    ax.set_xlim(-0.75, pos - 1)
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(0, 0.99, 1, 0), handletextpad=0.5)
    fig.subplots_adjust(left=.28, bottom=.13, right=.99, top=0.85)
    # plt.savefig('/opt/project/img/phd/violin-pgm-ttls.pdf', pad_inches=0, bbox_inches='tight')
    plutils.save_fig('/opt/project/img/phd/violin-pgm-ttls.pdf', ax)
    plt.close('all')


def plot_size_sequence(sizes: List[List[float]]) -> Tuple[plt.Figure, plt.Axes]:
    random = np.random.RandomState(seed=1)
    sizes = [sizes[i] for i in random.randint(0, len(sizes), size=200)]
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.66)
    max_len = 0
    for s in sizes:
        ax.plot(s, c=COLORS[0], alpha=0.05)
        max_len = np.max([max_len, len(s)])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.plot([0, max_len], [0, 0], c='black', linestyle='--', linewidth=0.75)
    ax.set_ylabel("Size [Bytes]")
    ax.set_xticks(np.arange(0, max_len + 0.1, 5))
    ax.set_xlim(0, max_len)
    fig.subplots_adjust(left=.20, bottom=.33, right=.97, top=.88)
    return fig, ax


def plot_record_sizes_google():
    all_sizes_google_es = get_record_sizes('www.google.es', 15)
    fig, ax = plot_size_sequence(all_sizes_google_es)
    print("Max google record size", np.max([np.max(f) for f in all_sizes_google_es]))
    ax.set_xlabel("Record number")
    # ax.set_yticks([0, 2000, 9000])
    ax.set_ylim(-1000, 8000)
    ax.set_yticks([0, 5000])
    plutils.save_fig(os.path.join(IMG_DIR, f'record-size-sequence-google.pdf'), ax)
    plt.close('all')


def plot_frame_sizes_google():
    all_sizes_google_es = get_frame_sizes('www.google.es', 15)
    print("Max google frame size", np.max([np.max(f) for f in all_sizes_google_es]))
    fig, ax = plot_size_sequence(all_sizes_google_es)
    ax.set_xlabel("Frame number")
    ax.set_yticks([-1000, 0, 1000])
    plutils.save_fig(os.path.join(IMG_DIR, f'frame-size-sequence-google.pdf'), ax)


def plot_record_sizes_primevideo():
    all_sizes_prime_com = get_record_sizes('www.primevideo.com', 15)
    fig, ax = plot_size_sequence(all_sizes_prime_com)
    ax.set_ylim(-1000, 8000)
    ax.set_yticks([0, 5000])
    ax.set_xlabel("Record number")
    plutils.save_fig(os.path.join(IMG_DIR, f'record-size-sequence-primevideo.pdf'), ax)


def plot_frame_sizes_primevideo():
    all_sizes_prime_com = get_frame_sizes('www.primevideo.com', 15)
    fig, ax = plot_size_sequence(all_sizes_prime_com)
    ax.set_xlabel("Frame number")
    ax.set_yticks([-1000, 0, 1000])
    plutils.save_fig(os.path.join(IMG_DIR, f'frame-size-sequence-primevideo.pdf'), ax)


def _make_values(models: List[str], agg: str) -> np.array:
    settings = ['normal', 'defense']  # ['normal', 'agg', 'defense']
    values = np.zeros((3 * len(settings), len(models) * 2))
    for s, setting in enumerate(settings):  # 'agg', 'defense']):
        for c, scenario in enumerate(['closed', 'open']):
            for m, model in enumerate(models):
                try:
                    trial_dir = {
                        'open': {
                            'normal': OPEN_WORLD_TRIAL_DIRS,
                            'agg': OPEN_WORLD_TRIAL_DIRS,
                            'defense': OPEN_WORLD_TRIAL_DIRS_DEFENSE
                        },
                        'closed': {
                            'normal': CLOSED_WORLD_TRIAL_DIRS,
                            'agg': CLOSED_WORLD_TRIAL_DIRS,
                            'defense': CLOSED_WORLD_TRIAL_DIRS_DEFENSE
                        }
                    }[scenario][setting][model]
                    evaluator = {
                        'open': dcc.eval_open_world,
                        'closed': dcc.eval_closed_world,
                        'open-def': dcc.eval_open_world,
                        'closed-def': dcc.eval_closed_world
                    }[scenario]
                    recall, precision, accuracy = evaluator(
                        trial_dir=trial_dir,
                        closed_world_labels=CLOSED_WORLD_LABELS,
                        conf_mat_mut=dcc.AggregateCommonPrefixes(COMMON_PREFIXES) if setting == 'agg' else lambda x: x,
                        is_cumul=trial_dir.find('cumul-') >= 0
                    )
                    for v, metric in enumerate(['precision', 'recall', 'accuracy']):
                        y = {
                            'recall': recall,
                            'precision': precision,
                            'accuracy': accuracy
                        }[metric]
                        if agg == 'median':
                            y = np.median([y.iloc[:, k].dropna().mean() for k in range(y.shape[1])])
                        else:
                            y = np.mean([y.iloc[:, k].dropna().mean() for k in range(y.shape[1])])
                        values[s * 3 + v, m * 2 + c] = y
                except KeyError:
                    print(f"Combination {scenario} - {setting} - {model} does not exist.")
                    continue
    return values


def print_table(agg: str = 'mean'):
    models = ['phmm', 'mc', 'knn', 'cumul', 'popets']
    values = _make_values(models, agg)
    s = ''
    for i in range(values.shape[0]):
        if i == 3:
            s += '\\textbf{Agg}: &&&&&&&&\\\\\n'
        if i == 6:
            s += '\\textbf{Defense}: &&&&&&&&\\\\\n'
        max_idx_closed = np.argmax(values[i, [0, 2, 4, 6]]) * 2
        max_idx_open = np.argmax(values[i, [1, 3, 5, 7]]) * 2 + 1
        for j in range(values.shape[1]):
            if j == 0:
                s += {0: 'Precision', 1: 'Recall', 2: 'Accuracy'}[i % 3]
            if j in [max_idx_closed, max_idx_open]:
                part1 = "$\\bm{"
                part2 = "}$"
            else:
                part1 = "$"
                part2 = "$"
            s += f'& {part1} {float(np.round(values[i, j] * 100, decimals=1)):.1f} {part2} '
        s += "\\\\\n"
    print(s)


def print_table_90deg(agg: str = 'mean'):
    models = ['phmm', 'mc', 'knn', 'cumul', 'popets']
    values = _make_values(models, agg)
    print_versions = {
        'phmm': 'PHMM',
        'mc': 'MC',
        'knn': 'kNN',
        'cumul': 'CUMUL',
        'popets': 'IPFP'
    }
    s = ''
    for offset in range(2):
        for i in range(offset, values.shape[1], 2):
            full, remainder = divmod(i, 2)
            if offset == 0 and i == 0:
                s += "\\multirow{5}{*}{\\rotatebox[origin=c]{90}{Closed World}}\n"
            if offset == 1 and i == 1:
                s += "&&&&&&\\\\\n\\multirow{5}{*}{\\rotatebox[origin=c]{90}{Open World}}\n"
            color_start = ""
            color_stop = ""
            if models[full] not in ['phmm', 'mc']:
                color_start = "\\textcolor{gray}{"
                color_stop = "}"
            s += f'&{color_start}{print_versions[models[full]]}{color_stop}'
            # max_idx_closed = np.argmax(values[i, [0, 2, 4, 6]]) * 2
            # max_idx_open = np.argmax(values[i, [1, 3, 5, 7]]) * 2 + 1
            for j in range(values.shape[0]):
                part1 = f'{color_start}$'
                part2 = f'${color_stop}'
                middle = '-' if values[j, i] == 0 else f'{float(np.round(values[j, i] * 100, decimals=2)):.2f}'
                s += f'& {part1} {middle} {part2} '
            s += "\\\\\n"
    print(s)


def aggregate_dropping_pgms(samples: List[Dict[str, Any]]):
    data = []
    for idx, grp in pd.DataFrame.from_dict(samples).groupby(['model', 'workload', 'model_size']):
        if np.sum(grp.dropped) == 0:
            print('no drop')
            n_pgms = 10
        elif np.sum(grp.dropped) == grp.shape[0]:
            n_pgms = 1
        else:
            grp = grp.sort_values('num_pgms')
            tmp = np.argmax(grp.dropped) - 1
            if tmp == -1:
                n_pgms = 1
            else:
                n_pgms = grp.iloc[tmp, -2]
        data.append({'model': idx[0], 'type': idx[2], 'workload': idx[1], 'num_pgms': n_pgms})
    return data


def plot_defense_prec_over():
    prec_over = [(100, 0.0934), (1100, 0.097694), (2100, 0.11), (3100, 0.1154),
                 (4100, 0.091028), (5100, 0.088054), (6100, 0.087127), (7100, 0.092868),
                 (8100, 0.091423), (9100, 0.096356),
                 (10100, 0.092106), (11100, 0.091578), (12100, 0.096721), (13100, 0.091353), (14100, 0.092042),
                 (15100, 0.096217)]
    prec_over.sort(key=lambda x: x[0])
    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.5)
    ax.plot(np.array(prec_over)[:, 0], np.array(prec_over)[:, 1] * 100, c=COLORS[0])
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.set_xlabel("Upper [Byte]")
    ax.set_ylabel("Precision [\%]")
    ax.set_xticks([100, 5000, 10000, 15000])
    plt.subplots_adjust(0.27, 0.30, 0.965, 0.87)
    print(ax.bbox.width / ax.bbox.height)
    plutils.save_fig(os.path.join(IMG_DIR, f'line-plot-max-size-vs-precision.pdf'), ax)
    plt.close('all')


def plot_defense_overhead_over():
    import implementation.data_conversion.tls_flow_extraction as tlsex
    prec_over = [(100, 0.0934), (1100, 0.097694), (2100, 0.11), (3100, 0.1154),
                 (4100, 0.091028), (5100, 0.088054), (6100, 0.087127), (7100, 0.092868),
                 (8100, 0.091423), (9100, 0.096356),
                 (10100, 0.092106), (11100, 0.091578), (12100, 0.096721), (13100, 0.091353), (14100, 0.092042),
                 (15100, 0.096217)]
    names = ['bitly.com_test', 'blogger.com_test', 'chaturbate.com_test', 'chouftv.ma_test', 'daftsex.com_test',
             'discord.com_test', 'e-hentai.org_test', 'envato.com_test', 'evernote.com_test', 'google.com_test',
             'gyazo.com_test' ,'medium.com_test' ,'namnak.com_test' ,'smallpdf.com_test' ,'discord.com_test']

    prec_head = []
    head_over = []
    for upper, prec in prec_over:
        rnd = np.random.RandomState(seed=1)
        defense = tlsex.RandomRecordSizeDefense(int(1e9), lambda x: rnd.randint(50, upper))
        for name in names:
            with open(f'/opt/project/data/cache/{name}.json', 'r') as fh:
                tmp = json.load(fh)
            flows = []
            dicts = []
            for val in tmp.values():
                for f in val:
                    dicts.append(f)
                    flows.append(defense(f))
        ratios = []
        for flow_d, flow in zip(dicts, flows):
            s1 = np.sum(
                [f['tcp_length'] + f['ip_header_length'] + f['tcp_header_length'] + 19 for f in flow_d['frames']])
            s2 = np.sum([f.tcp_length + f.ip_header_length + f.tcp_header_length + 19 for f in flow.frames])
            ratios.append(s2 / s1)
        prec_head.append([prec, np.mean(ratios)])
        head_over.append([upper, np.mean(ratios)])
        print(f"Overhead {upper} has overhead of {np.mean(ratios)}")

    fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.5)
    ax.plot(np.array(head_over)[:, 0], np.array(head_over)[:, 1] * 100)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.set_xlabel("Upper [Byte]")
    ax.set_ylabel("Overhead [\%]")
    ax.set_xticks([100, 5000, 10000, 15000])
    plt.subplots_adjust(0.27, 0.30, 0.965, 0.87)
    print(ax.bbox.width / ax.bbox.height)
    plutils.save_fig(os.path.join(IMG_DIR, f'line-plot-max-size-vs-overhead.pdf'), ax)


def plot_uncertainty():
    import bootstrap
    slength = 1
    ci_level = 99
    model = "mc"  # "phmm"
    name = f'open-world-{model}-results-days-to-train-{slength}'
    td_eval = f'/opt/project/data/grid-search-results/{name}/'

    for f in os.listdir(td_eval):
        print(f)
        # if f.find("medium.com") == -1:
        #     continue
        if not f.startswith('config-'):
            continue
        with open(os.path.join(td_eval, f), 'r') as fh:
            config_eval = json.load(fh)
        if config_eval['label'] in ['medium.com', 'www.ebay.co.uk']:
            print(config_eval['label'])
            all_nlls = []
            for x in config_eval['nlls']:
                all_nlls.append(x)
            if len(all_nlls) == 0:
                print(f"No NLLs for {f}")
                continue
            # means = np.array([np.mean(np.array(x) / config_eval['max_nll_train']) for x in config_eval['nlls']])
            # upper = [np.percentile(np.array(x) / config_eval['max_nll_train'], 90) for x in config_eval['nlls']]
            # lower = [np.percentile(np.array(x) / config_eval['max_nll_train'], 10) for x in config_eval['nlls']]
            # mads = np.array([np.mean(np.abs(np.array(x) / config_eval['max_nll_train'] - means[i])) for i, x in enumerate(config_eval['nlls'])])
            means = np.array([np.mean(np.array(x)) for x in all_nlls])
            # p1 = np.array([np.percentile(x, 1) for x in all_nlls])
            # p99 = np.array([np.percentile(x, 99) for x in all_nlls])
            cis = np.column_stack([bootstrap.bootstrap_ci(np.array(x), ci_level, np.mean) for x in all_nlls])
            fig, ax = plutils.get_fig(0.73 if STYLE == 'phd' else 0.5)
            # ax.vlines([slength], np.min(cis[0, :]), np.max(cis[1, :]), color='black', linestyle='--')
            # ax.fill_between(np.arange(len(means)), means - mads, means + mads, alpha=0.3, label='MAD')
            ax.fill_between(np.arange(len(means)), cis[0, :], cis[1, :], alpha=0.3, label=f'CI', color=COLORS[0])
            ax.plot(means, label='Mean', c=COLORS[0])
            ax.set_xlabel("Day")
            ax.set_ylabel("NLL")
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.legend(frameon=False)
            # ax.set_ylim(-0.02, 4)
            # ax.set_yticks([0, 1, 2, 3, 4])
            # ax.set_yticklabels([0, 0.5, 1, 1.5])
            ax.set_xticks([1] + list(range(20, 80, 20)))
            fig.set_figheight(fig.get_figheight() * 1.25)
            fig.subplots_adjust(left=.26, bottom=.29, right=.99, top=.87)
            plutils.save_fig(os.path.join(IMG_DIR, f'{model}-nll-{config_eval["label"]}-days-{slength}-ci-{ci_level}.pdf'), ax)
            plt.close("all")


print("\n===\nPrint Table")
# print_table_90deg('mean')
# print("&&&&&&\\\\")
# print("&&&&&&\\\\")
# print_table_90deg('median')

print("\n===\nPlot record sizes")
# plot_frame_sizes_primevideo()
# plot_frame_sizes_google()
# plot_record_sizes_primevideo()
# plot_record_sizes_google()

print("\n===\nPlot timings")
# plot_timings()

print("\n===\nPlot defense")
print("\t\tPlot prevision over upper bytes")
# plot_defense_prec_over()
print("\t\tPlot overhead over")
plot_defense_overhead_over()

print("\n===\nPlot uncertainty")
# plot_uncertainty()


print("\n===\nPlot violin plots")
# _make_legend(['closed', 'open', 'closed-def', 'open-def', 'open-fc2s', 'openfs2c'], 0.73 * 1.75 if STYLE == 'phd' else 1, f'violin-legend.pdf', 3)
# for m in ['precision', 'recall']:
#     print(f"Plot comparison {m}")
#     # continue
#     plot_violin_scenarios(
#         trial_dirs={
#             # 'POPETS': {
#             #     'closed': '/opt/project/data/grid-search-results/eval-results/popets-closed',
#             #     'open':   '/opt/project/data/grid-search-results/eval-results/popets-open'
#             # },
#             # 'kNN': {
#             #     'closed': CLOSED_WORLD_TRIAL_DIRS['knn'],
#             #     'open': OPEN_WORLD_TRIAL_DIRS['knn'],
#             #     'closed-def': CLOSED_WORLD_TRIAL_DIRS_DEFENSE['knn'],
#             #     'open-def': OPEN_WORLD_TRIAL_DIRS_DEFENSE['knn']
#             # },
#             # 'CUMUL': {
#             #     'closed': CLOSED_WORLD_TRIAL_DIRS['cumul'],
#             #     'open': OPEN_WORLD_TRIAL_DIRS['cumul'],
#             #     'closed-def': CLOSED_WORLD_TRIAL_DIRS_DEFENSE['cumul'],
#             #     'open-def': OPEN_WORLD_TRIAL_DIRS_DEFENSE['cumul']
#             # }
#             'MC': {
#                 'closed': CLOSED_WORLD_TRIAL_DIRS['mc'],
#                 'open': OPEN_WORLD_TRIAL_DIRS['mc'],
#                 'closed-def': CLOSED_WORLD_TRIAL_DIRS_DEFENSE['mc'],
#                 'open-def': OPEN_WORLD_TRIAL_DIRS_DEFENSE['mc'],
#                 'open-fc2s': '/opt/project/data/grid-search-results/eval-results/open-world-mc-training-days-70-filter-client2server',
#                 'open-fs2c': '/opt/project/data/grid-search-results/eval-results/open-world-mc-training-days-70-filter-server2client'
#             },
#             'PHMM': {
#                 'closed': CLOSED_WORLD_TRIAL_DIRS['phmm'],
#                 'open': OPEN_WORLD_TRIAL_DIRS['phmm'],
#                 'closed-def': CLOSED_WORLD_TRIAL_DIRS_DEFENSE['phmm'],
#                 'open-def': OPEN_WORLD_TRIAL_DIRS_DEFENSE['phmm'],
#                 'open-fc2s': '/opt/project/data/grid-search-results/eval-results/open-world-phmm-training-days-70-filter-client2server',
#                 'open-fs2c': '/opt/project/data/grid-search-results/eval-results/open-world-phmm-training-days-70-filter-server2client'
#             }
#         },
#         metric=m
#     )



print("\n===\nPlot precision over days")
# plot_line_compare_metrics(
#     scenario='open',
#     metric='precision',
#     trial_dirs={
# #         # 'knn': '/opt/project/data/grid-search-results/eval-results/047f435fd54642f4b2674ac21e08d27f-open-world-eval-over-days-no-symbol-change-1',
#         'mc': '/opt/project/data/grid-search-results/eval-results/open-world-mc-results-days-to-train-1',
#         'phmm': '/opt/project/data/grid-search-results/eval-results/open-world-phmm-results-days-to-train-1',
# #         # 'cumul': '/opt/project/data/grid-search-results/eval-results/cumul-open-over-days-1'
#     },
#     suffix='-singleday'
# )
# plot_line_compare_metrics('open', {'phmm': OPEN_WORLD_TRIAL_DIRS['phmm'], 'mc': OPEN_WORLD_TRIAL_DIRS['mc']}, 'precision')
#
print("\n===\nPlot seqyence infos trained models")
# _make_legend(['PHMM-packet', 'PHMM-record', 'MC-packet', 'MC-record'], 1, filename='cdf-legend.pdf', ncol=2)
# plot_cdf_seq_lengths()
#
print("\n===\nPlot num phmms")
# samples = dcp.retrieve_num_pgms()
# data = aggregate_dropping_pgms(samples)
# bar_num_pgms(data)
#
print("\n===\nProcessing cost of TLS Filter")
# plot_filter_cost()
#
print("\n===\nTTL on average workload.")
# violin_compare_ttl_average()