import pandas as pd
import numpy as np
import logging
import os
import json
import sqlalchemy
import sys
from typing import Dict, List, Tuple, Any, Union
import matplotlib.pyplot as plt
import plots.utils as plutils
from datetime import datetime, timedelta


import implementation.data_conversion.tls_flow_extraction as tlsex


logger = logging.getLogger('dataprep')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

with open("CONFIG.json", 'r') as fh:
    CONFIG = json.laod(fh)

COLORS = ['#80b1d3', '#fb8072', '#bebada', '#fdb462', '#8dd3c7']
SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://root:{CONFIG["db_password"]}' + \
                          f'{CONFIG["db_host"]}:3306/gatherer_upscaled'


def get_days_metadata(day: datetime) -> Tuple[datetime, List[Dict[str, Any]]]:
    """
    Get metadata of one trace event, i.e., one set of scheduled URLs. The
    argument `day` is expected to be a return value of the function `get_days`.
    The function then adds five hours to this time point (the measurement process
    takes around 4h). All meta data entries between the resulting time points
    are returned.

    Args:
        day: Start time of the first trace that is gathered.

    Returns:
        List of dictionaries containing:
            - meta_data_id
            - filename of PCAP
            - url that is traced.
    """
    end_time = day + timedelta(hours=5)
    start_date_str = day.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    q = f"""
    SELECT 
        traces_metadata.id, 
        traces_metadata.filename,
        urls.url,
        urls.id
    FROM traces_metadata INNER JOIN urls ON traces_metadata.url = urls.id
    WHERE start_capture >= '{start_date_str}' AND start_capture  < '{end_date_str}'
    """
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        results = connection.execute(q).fetchall()
    return day, [{'meta_data_id': r[0], 'filename': r[1], 'url': r[2], 'url_id': r[3]} for r in results]


def get_days() -> List[datetime]:
    """
    Read the unique days from the job scheduling table. THe time point of the
    scheduled date is exactly the same for each scheduled day. Detect if the
    time point changes to get the start points when stuff is scheduled to be
    gathered.

    Returns:
        A list of datetime objects that correspond to the time stamps at which
            jobs were scheduled for tracing.
    """
    q = "SELECT scheduled_date FROM jobs"
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        results = connection.execute(q).fetchall()
    dates = [results[0][0]]
    d1 = results[0][0]
    for d in results:
        if (d[0] - d1).seconds > 0:
            d1 = d[0]
            dates.append(d1)
    return dates


def get_datasets_for_tag(tag: str) -> Tuple[List[int], List[int], List[int], Dict[int, str], Dict[str, int]]:
    """
    Distribute existing URLs to training, test and validation set for URLs belonging
    to a specific tag.

    Args:
        tag: Tag that identfies a grouping of URLs.

    Returns:
        train: List of url_ids for the training set.
        val: List of url_ids for the validation set.
        test: List of url_ids for the test set.
        indicator: Dictionary that maps an url id to {test, train, val} depending
            on the dataset the url belongs to.
        labels: Maps url_id back to the tag.
    """
    q = f"""
        SELECT urls.id, urls.url
            FROM gatherer_upscaled.urls
                INNER JOIN tag_assignment on tag_assignment.trace_fk = urls.id
                INNER JOIN tags on tags.id = tag_assignment.tag_fk
            WHERE tags.tag like '{tag}';
    """
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        results = connection.execute(q).fetchall()

    urls = {}
    for url_id, url in results:
        if url not in urls:
            urls[url] = []
        urls[url].append(int(url_id))

    train = []
    val = []
    test = []
    url_keys = np.array(list(urls.keys()))
    random = np.random.RandomState(seed=1)
    random.shuffle(url_keys)
    for i, url in enumerate(url_keys):
        if 0 <= i < 10:
            test.extend(urls[url])
        elif 10 <= i < 20:
            val.extend(urls[url])
        else:
            train.extend(urls[url])
    indicator = {}
    labels = {}
    for url_id in train:
        indicator[int(url_id)] = 'train'
        labels[int(url_id)] = tag
    for url_id in val:
        indicator[int(url_id)] = 'val'
        labels[int(url_id)] = tag
    for url_id in test:
        indicator[int(url_id)] = 'test'
        labels[int(url_id)] = tag

    return train, val, test, indicator, labels


def create_data_sets(limit:int=None):
    tags = [t for t in get_tags() if t not in ['cloudflare', 'amazon aws',
                                               'adult', 'akamai', 'google cloud',
                                               'instagram.com', 'cint.com']]
    if limit is not None:
        tags = tags[:limit]
    datasets = {}
    indicators = {}
    labels = {}
    for tag in tags:
        train, val, test, indicators_, labels_ = get_datasets_for_tag(tag)
        indicators.update(indicators_)
        labels.update(labels_)
        datasets[tag] = {
            'train': train,
            'val': val,
            'test': test
        }
    return datasets, indicators, labels


def try_catch(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error("Error occured")
            logger.exception(e)
    return inner


def get_db_metadata_for_tag(tag: str) -> Dict[str, Dict[str, Any]]:
    q = f"""
        SELECT 
            filename,
            browser,
            start_capture,
            tag
        FROM traces_metadata
            INNER JOIN urls ON traces_metadata.url = urls.id
            INNER JOIN tag_assignment ON tag_assignment.trace_fk = urls.id
            INNER JOIN tags ON tags.id = tag_assignment.tag_fk
        WHERE tag like "{tag}" 
    """
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        results = connection.execute(q).fetchall()
    meta = {}
    for filename, browser, start_capture, tag in results:
        if filename not in meta:
            meta[filename] = {
                'filename': filename,
                'browser': browser,
                'start_capture': pd.to_datetime(start_capture),
                'tag': tag
            }
    return meta


def get_db_metadata() -> Dict[str, Dict[str, Any]]:
    q = """
        SELECT 
            filename,
            browser,
            start_capture,
            tag
        FROM traces_metadata
            INNER JOIN urls ON traces_metadata.url = urls.id
            INNER JOIN tag_assignment ON tag_assignment.trace_fk = urls.id
            INNER JOIN tags ON tags.id = tag_assignment.tag_fk
        WHERE tag not like ''
    """
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        results = connection.execute(q).fetchall()
    meta = {}
    for filename, browser, start_capture, tag in results:
        if filename not in meta:
            meta[filename] = {
                'filename': filename,
                'browser': browser,
                'start_capture': pd.to_datetime(start_capture),
                'tags': []
            }
        meta[filename]['tags'].append(tag)
    return meta


def get_tags() -> List[str]:
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:FKmk2kRzWFFdX8@pk-swc01.' + \
                              'forschung.lkn.ei.tum.de:3306/gatherer_upscaled'
    q = """SELECT tag FROM tags WHERE tag not like ''"""
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        results = connection.execute(q).fetchall()
    return [r[0] for r in results]


def load_flow_dict(file_name: str) -> Union[None, Dict[str, Any]]:
    dc = None
    if not os.path.exists(os.path.join('/k8s-json', file_name)):
        print("Does not exist ", os.path.join('/k8s-json', file_name))
        return dc
    try:
        with open(os.path.join('/k8s-json', file_name), 'r') as fh:
            dc = json.load(fh)
            dc['filename'] = file_name[:-5]  # Remove the .json
    except Exception as e:
        logger.exception(e)
    return dc


def load_dicts(json_path: str) -> List[Dict[str, Any]]:
    dicts = []
    num_files = len(os.listdir(json_path))
    prev_val = 5.
    files = np.array(os.listdir(json_path))
    np.random.shuffle(files)
    for i, f in enumerate(files):
        percentage = i / num_files * 100
        logger.info(f"Loaded {percentage} %% json files.")
        if i / num_files > 0.51:
            break
        if percentage - prev_val > 0:
            percentage = i / num_files * 100
            logger.info(f"Loaded {percentage} %% json files.")
            prev_val += 5.
        if f.endswith('json'):
            dicts.append(load_flow_dict(f))
    return dicts


def main_flow_stats(until: float, tags: List[str]=None):
    def reduce_list(meta_entry: Dict[str, Any], metric_key: str):
        l = meta_entry.pop(metric_key)
        no_data = False
        if len(l) == 0:
            no_data = True
            l = [0]
        meta_entry[f'{metric_key}_min'] = float(np.min(l))
        meta_entry[f'{metric_key}_p01'] = float(np.percentile(l, 1))
        meta_entry[f'{metric_key}_p05'] = float(np.percentile(l, 5))
        meta_entry[f'{metric_key}_median'] = float(np.median(l))
        meta_entry[f'{metric_key}_mean'] = float(np.mean(l))
        meta_entry[f'{metric_key}_p95'] = float(np.percentile(l, 95))
        meta_entry[f'{metric_key}_p99'] = float(np.percentile(l, 99))
        meta_entry[f'{metric_key}_max'] = float(np.max(l))

        if not no_data:
            l = np.log2(l)
        meta_entry[f'{metric_key}_log2_min'] = float(np.min(l))
        meta_entry[f'{metric_key}_log2_p01'] = float(np.percentile(l, 1))
        meta_entry[f'{metric_key}_log2_p05'] = float(np.percentile(l, 5))
        meta_entry[f'{metric_key}_log2_median'] = float(np.median(l))
        meta_entry[f'{metric_key}_log2_mean'] = float(np.mean(l))
        meta_entry[f'{metric_key}_log2_p95'] = float(np.percentile(l, 95))
        meta_entry[f'{metric_key}_log2_p99'] = float(np.percentile(l, 99))
        meta_entry[f'{metric_key}_log2_max'] = float(np.max(l))

    def add_handshake_stats(flow: tlsex.MainFlow, meta_entry: Dict[str, Any]):
        num_handshake_frames = 0
        num_handshake_frames_in = 0
        num_handshake_frames_out = 0
        num_handshake_records = 0
        num_handshake_records_in = 0
        num_handshake_records_out = 0
        for frame in flow.frames:
            if frame.time_epoch - flow.frames[0].time_epoch > until:
                break
            for i, record in enumerate(frame.tls_records):
                if record.content_type == 22:
                    num_handshake_frames += 1 * int(i == 0)
                    num_handshake_records += 1
                    if frame.direction > 0:
                        num_handshake_frames_in += 1 * int(i == 0)
                        num_handshake_records_in += 1
                    else:
                        num_handshake_frames_out += 1 * int(i == 0)
                        num_handshake_records_out += 1
                else:
                    continue
        meta_entry['num_handshake_frames'] = num_handshake_frames
        meta_entry['num_handshake_frames_in'] = num_handshake_frames_in
        meta_entry['num_handshake_frames_out'] = num_handshake_frames_out
        meta_entry['num_handshake_records'] = num_handshake_records
        meta_entry['num_handshake_records_in'] = num_handshake_records_in
        meta_entry['num_handshake_records_out'] = num_handshake_records_out

    if tags is None:
        tags = get_tags()

    for i, tag in enumerate(tags):
        if tag in ['cloudflare', 'amazon aws', 'adult', 'akamai', 'google cloud']:
            continue
        print(f"Extract {i:4d} of {len(tags):4d} - {tag}")
        meta = get_db_metadata_for_tag(tag)
        print(f"\tGot {len(meta)} items.")
        does_not_exist = 0
        for filename in meta.keys():
            if os.path.exists(f'/k8s-json/{filename}.json'):
                flow_dc = make_main_flows([load_flow_dict(f'{filename}.json')])
                add_num_frames(meta, flow_dc, until)
                add_num_records(meta, flow_dc, until)
                add_num_tls_record_per_frame(meta, flow_dc, until)
                add_frame_sizes(meta, flow_dc, until)
                for k in ['frame_sizes', 'frame_sizes_in', 'frame_sizes_out']:
                    reduce_list(meta[filename], k)
                add_tls_record_sizes(meta, flow_dc, until)
                for k in ['tls_record_sizes', 'tls_record_sizes_in', 'tls_record_sizes_out']:
                    reduce_list(meta[filename], k)
                add_num_tls_record_per_frame(meta, flow_dc, until)
                for k in ['num_records_in_frame', 'num_records_in_frame_in', 'num_records_in_frame_out']:
                    reduce_list(meta[filename], k)
                add_handshake_stats(flow_dc[filename], meta[filename])
            else:
                does_not_exist +=1
        print(f"\t{does_not_exist} of {len(meta)} files do not exist.")
        #with open(f"data/stats-{tag}.json", 'w') as fh:
        #    json.dump(meta, fh)
        df = pd.DataFrame.from_dict([x for x in meta.values()])
        df.to_hdf('./data/summaries.h5', key=tag.replace('.', '_'))


@try_catch
def make_main_flows(dicts: List[Dict[str, Any]]) -> Dict[str, tlsex.MainFlow]:
    flows = {}
    for d in dicts:
        filename = d.pop('filename')
        flows[filename] = tlsex.MainFlow.from_dict(d)
    return flows


@try_catch
def add_num_frames(meta: Dict[str, Dict[str, Any]], flows: Dict[str, tlsex.MainFlow],
                   until: float) -> None:
    for filename, flow in flows.items():
        num_frames = 0
        num_frames_in = 0
        num_frames_out = 0
        for frame in flow.frames:
            if frame.time_epoch - flow.frames[0].time_epoch > until:
                break
            else:
                num_frames += 1
                if frame.direction > 0:
                    num_frames_in += 1
                else:
                    num_frames_out += 1
        meta[filename]['num_frames'] = num_frames
        meta[filename]['num_frames_in'] = num_frames_in
        meta[filename]['num_frames_out'] = num_frames_out


@try_catch
def add_num_records(meta: Dict[str, Dict[str, Any]], flows: Dict[str, tlsex.MainFlow],
                    until: float) -> None:
    for filename, flow in flows.items():
        num_reocords = 0
        num_reocords_in = 0
        num_reocords_out = 0
        for frame in flow.frames:
            if frame.time_epoch - flow.frames[0].time_epoch > until:
                break
            else:
                num_reocords += len(frame.tls_records)
                if frame.direction > 0:
                    num_reocords_in += len(frame.tls_records)
                else:
                    num_reocords_out += len(frame.tls_records)
        meta[filename]['num_records'] = num_reocords
        meta[filename]['num_records_in'] = num_reocords_in
        meta[filename]['num_records_out'] = num_reocords_out


@try_catch
def add_frame_sizes(meta: Dict[str, Dict[str, Any]], flows: Dict[str, tlsex.MainFlow],
                    until: float) -> None:
    for filename, flow in flows.items():
        meta[filename]['frame_sizes'] = []
        meta[filename]['frame_sizes_in'] = []
        meta[filename]['frame_sizes_out'] = []
        for i, frame in enumerate(flow.frames):
            if frame.time_epoch - flow.frames[0].time_epoch > until:
                break
            else:
                meta[filename]['frame_sizes'].append(frame.tcp_length)
                if frame.direction > 0:
                    meta[filename]['frame_sizes_in'].append(frame.tcp_length)
                else:
                    meta[filename]['frame_sizes_out'].append(frame.tcp_length)


@try_catch
def add_tls_record_sizes(meta: Dict[str, Dict[str, Any]], flows: Dict[str, tlsex.MainFlow],
                         until: float) -> None:
    for filename, flow in flows.items():
        meta[filename]['tls_record_sizes'] = []
        meta[filename]['tls_record_sizes_in'] = []
        meta[filename]['tls_record_sizes_out'] = []
        for frame in flow.frames:
            if frame.time_epoch - flow.frames[0].time_epoch > until:
                break
            else:
                meta[filename]['tls_record_sizes'].extend([r.length for r in frame.tls_records])
                if frame.direction > 0:
                    meta[filename]['tls_record_sizes_in'].extend([r.length for r in frame.tls_records])
                else:
                    meta[filename]['tls_record_sizes_out'].extend([r.length for r in frame.tls_records])


@try_catch
def add_num_tls_record_per_frame(meta: Dict[str, Dict[str, Any]],
                                 flows: Dict[str, tlsex.MainFlow],
                                 until: float) -> None:
    for filename, flow in flows.items():
        meta[filename]['num_records_in_frame'] = []
        meta[filename]['num_records_in_frame_out'] = []
        meta[filename]['num_records_in_frame_in'] = []
        for frame in flow.frames:
            if frame.time_epoch - flow.frames[0].time_epoch > until:
                break
            else:
                meta[filename]['num_records_in_frame'].append(len(frame.tls_records))
                if frame.direction > 0:
                    meta[filename]['num_records_in_frame_in'].append(len(frame.tls_records))
                else:
                    meta[filename]['num_records_in_frame_out'].append(len(frame.tls_records))


@try_catch
def add_record_sizes_by_record_type(meta: Dict[str, Dict[str, Any]],
                                    flows: Dict[str, tlsex.MainFlow],
                                    until: float) -> None:
    for filename, flow in flows.items():
        record_sizes = {}
        for frame in flow.frames:
            if frame.time_epoch - flow.frames[0].time_epoch > until:
                break
            for record in frame.tls_records:
                ct = 'HANDSHAKE_MESSAGE' if record.content_type == 22 else \
                    tlsex.TLS_MAP[record.content_type]
                if ct not in record_sizes:
                    record_sizes[ct] = {} if record.content_type == 22 else {'in': [], 'out': []}
                if record.content_type == 22:
                    if record.handshake_type in tlsex.TLS_MAP[22]:
                        ht = tlsex.TLS_MAP[22][record.handshake_type]
                    else:
                        ht = 'ENCRYPTED_HANDSHAKE_MESSAGE'
                    if ht not in record_sizes[ct]:
                        record_sizes[ct][ht] = {'in': [], 'out': []}
                    record_sizes[ct][ht]['in' if frame.direction > 0 else 'out'].append(record.length)
                else:
                    record_sizes[ct]['in' if frame.direction > 0 else 'out'].append(record.length)
        meta[filename]['typed_tls_record_sizes'] = record_sizes


def extract_data(until: float, json_path: str) -> Dict[str, Dict[str, Any]]:
    logger.info("\tGet Tags")
    meta = get_db_metadata()
    logger.info("\tGet main flows")
    flows = make_main_flows(load_dicts(json_path))
    to_pop = []
    for k in meta.keys():
        if k not in flows:
            to_pop.append(k)
    logger.info("\tRemoved {} of {} entries from meta".format(len(to_pop), len(meta)))
    for k in to_pop:
        meta.pop(k)
    to_pop = []
    for k in flows.keys():
        if k not in meta:
            to_pop.append(k)
    logger.info("\tRemoved {} of {} entries from flows".format(len(to_pop), len(flows)))
    for k in to_pop:
        flows.pop(k)
    logger.info("\tAdd number of frames")
    add_num_frames(meta, flows, until)
    logger.info("\tAdd number of records")
    add_num_records(meta, flows, until)
    logger.info("\tAdd frame sizes")
    add_frame_sizes(meta, flows, until)
    logger.info("\tAdd record sizes")
    add_tls_record_sizes(meta, flows, until)
    logger.info("\tAdd num records per frame")
    add_num_tls_record_per_frame(meta, flows, until)
    logger.info("\tAdd record sizes per type")
    add_record_sizes_by_record_type(meta, flows, until)
    to_pop = []
    for k in meta.keys():
        if 'frame_sizes' not in meta[k]:
            to_pop.append(k)
    logger.info("\tRemoved {} of {} entries from meta".format(len(to_pop), len(meta)))
    for k in to_pop:
        meta.pop(k)
    return meta


@try_catch
def make_cdf(values) -> pd.Series:
    return np.cumsum(pd.Series(values).value_counts(normalize=True).sort_index())


@try_catch
def compare_cdfs(cdfs, labels, xlabel, ylabel, fig_path, xscale=None):
    fig, ax = plutils.get_fig(1)
    for i, (cdf, label) in enumerate(zip(cdfs, labels)):
        ax.plot(cdf.index.values, cdf.values, c=COLORS[i], label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale is not None:
        ax.set_xscale('log')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)


@try_catch
def compare_pdfs(pdfs, labels, xlabel, ylabel, fig_path, xscale=None):
    fig, ax = plutils.get_fig(1)
    for i, (pdf, label) in enumerate(zip(pdfs, labels)):
        if xscale == 'log':
            if type(pdf) == list:
                pdf = np.array(pdf)
            pdf = np.log(pdf[pdf > 0])
        ax.hist(pdf, density=True, color=COLORS[i], label=label, alpha=0.5, bins=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)


@try_catch
def compare_frame_sizes(meta: Dict[str, Dict[str, Any]], fig_path: str):
    sizes_in = np.array([])
    sizes_out = np.array([])
    for flow_info in meta.values():
        sizes_in = np.concatenate([sizes_in, np.array(flow_info['frame_sizes_in'])])
        sizes_out = np.concatenate([sizes_out, np.array(flow_info['frame_sizes_out'])])
    cdf_in = make_cdf(sizes_in)
    cdf_out = make_cdf(sizes_out)
    xlabel = "Frame Size [Byte]"
    compare_cdfs([cdf_in, cdf_out], ['in', 'out'], xlabel, "P(X < x)",
                 os.path.join(fig_path, 'cdf_frame_sizes.pdf'), 'log')
    compare_pdfs([sizes_in, sizes_out], ['in', 'out'], f'log({xlabel})', "P(x)",
                 os.path.join(fig_path, 'pdf_frame_sizes.pdf'), 'log')


@try_catch
def compare_record_sizes(meta: Dict[str, Dict[str, Any]], fig_path: str):
    sizes_in = np.array([])
    sizes_out = np.array([])
    for flow_info in meta.values():
        sizes_in = np.concatenate([sizes_in, np.array(flow_info['tls_record_sizes_in'])])
        sizes_out = np.concatenate([sizes_out, np.array(flow_info['tls_record_sizes_out'])])
    cdf_in = make_cdf(sizes_in)
    cdf_out = make_cdf(sizes_out)
    xlabel = 'TLS record size [Byte]'
    compare_cdfs([cdf_in, cdf_out], ['in', 'out'], xlabel, "P(X < x)",
                 os.path.join(fig_path, 'cdf_record_sizes.pdf'), 'log')
    compare_pdfs([sizes_in, sizes_out], ['in', 'out'], f'log({xlabel})', "P(X < x)",
                 os.path.join(fig_path, 'pdf_record_sizes.pdf'), 'log')


@try_catch
def compare_num_frames(meta: Dict[str, Dict[str, Any]], fig_path: str):
    sizes_in = []
    sizes_out = []
    for flow_info in meta.values():
        sizes_in.append(flow_info['num_frames_in'])
        sizes_out.append(flow_info['num_frames_out'])
    cdf_in = make_cdf(sizes_in)
    cdf_out = make_cdf(sizes_out)
    xlabel = "Num frames"
    compare_cdfs([cdf_in, cdf_out], ['in', 'out'], xlabel, "P(X < x)",
                 os.path.join(fig_path, 'cdf_num_frames.pdf'), 'log')
    compare_pdfs([sizes_in, sizes_out], ['in', 'out'], f'log({xlabel})', "P(X < x)",
                 os.path.join(fig_path, 'pdf_num_frames.pdf'), 'log')


@try_catch
def compare_num_tls_records(meta: Dict[str, Dict[str, Any]], fig_path: str):
    sizes_in = []
    sizes_out = []
    for flow_info in meta.values():
        sizes_in.append(flow_info['num_records_in'])
        sizes_out.append(flow_info['num_records_out'])
    cdf_in = make_cdf(sizes_in)
    cdf_out = make_cdf(sizes_out)
    xlabel = "Num TLS records"
    compare_cdfs([cdf_in, cdf_out], ['in', 'out'], xlabel, "P(X < x)",
                 os.path.join(fig_path, 'cdf_num_records.pdf'), 'log')
    compare_pdfs([sizes_in, sizes_out], ['in', 'out'], f'log({xlabel})', "P(X < x)",
                 os.path.join(fig_path, 'pdf_num_records.pdf'), 'log')


@try_catch
def compare_num_tls_records_in_frame(meta: Dict[str, Dict[str, Any]], fig_path: str):
    sizes_in = []
    sizes_out = []
    for flow_info in meta.values():
        sizes_in.append(flow_info['num_records_in_frame_in'])
        sizes_out.append(flow_info['num_records_in_frame_out'])
    cdf_in = make_cdf(sizes_in)
    cdf_out = make_cdf(sizes_out)
    xlabel = "Num TLS records in frame"
    compare_cdfs([cdf_in, cdf_out], ['in', 'out'], xlabel, "P(X < x)", fig_path)


@try_catch
def pair_plot(nrows: int, ncols: int, labels: List[str], data: List[np.array],
              bins: np.array, fig_path: str) -> None:
    for row in range(nrows):
        for col in range(row, ncols):
            fig, ax = plutils.get_fig(1)
            ax.set_ylabel("Fraction")
            ax.set_xlabel("log(Metric)")
            if row == col:
                if data[row].size > 0:
                    ax.hist(data[row], bins=bins, density=True)
                else:
                    pass
            else:
                if data[row].size > 0:
                    # cdf = make_cdf(data[row])
                    # ax.plot(cdf.index.values, cdf.values, c=COLORS[0])
                    ax.hist(data[row], bins=bins, density=True, alpha=0.5, color=COLORS[0])
                if data[col].size > 0:
                    # cdf = make_cdf(data[col])
                    # ax.plot(cdf.index.values, cdf.values, c=COLORS[1])
                    ax.hist(data[col], bins=bins, density=True, alpha=0.5, color=COLORS[1])
                # ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(bins[0], bins[-1] * 1.01)
            ax.set_title(f"{labels[row]} vs. {labels[col]}")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_path, f"{row}-{col}.pdf"))
            plt.close(fig)
            plt.close('all')


@try_catch
def pair_plot_attribute(meta: Dict[str, Dict[str, Any]], attribute: str, fig_path: str) -> None:
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    tags = get_tags()
    tag_int = {t: i for i, t in enumerate(tags)}
    values = [np.array([]) for _ in range(len(tags))]
    min = 0
    max = 0
    for flow_dict in meta.values():
        arr = np.array(flow_dict[attribute])
        if arr.size == 0:
            continue
        arr = np.log(arr[arr > 0])
        min = np.min([min, np.min(arr)])
        max = np.max([max, np.max(arr)])
        for tag in flow_dict['tags']:
            values[tag_int[tag]] = np.concatenate([values[tag_int[tag]], arr])
    pair_plot(len(tags), len(tags), tags, values, np.linspace(min, max + 0.1, 20), fig_path)


@try_catch
def pair_plot_handshake(meta: Dict[str, Dict[str, Any]], attribute: Union[str, List[str]],
                        fig_path: str) -> None:
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    tags = get_tags()
    tag_int = {t: i for i, t in enumerate(tags)}
    values = [np.array([]) for _ in range(len(tags))]
    min = 0
    max = 0
    for flow_dict in meta.values():
        if type(attribute) == list:
            x = []
            for a in attribute:
                if a in flow_dict['typed_tls_record_sizes']['HANDSHAKE_MESSAGE']:
                    x.append(flow_dict['typed_tls_record_sizes']['HANDSHAKE_MESSAGE'][a]['in'])
                    x.append(flow_dict['typed_tls_record_sizes']['HANDSHAKE_MESSAGE'][a]['out'])
                else:
                    continue
            if len(x) == 0:
                continue
            elif len(x) == 1:
                arr = np.array(x)
            else:
                arr = np.concatenate(x)
        else:
            if attribute in flow_dict['typed_tls_record_sizes']['HANDSHAKE_MESSAGE']:
                arr = np.concatenate([
                    np.array(flow_dict['typed_tls_record_sizes']['HANDSHAKE_MESSAGE'][attribute]['in']),
                    np.array(flow_dict['typed_tls_record_sizes']['HANDSHAKE_MESSAGE'][attribute]['out'])
                ])
            else:
                continue
        if arr.size == 0:
            continue
        min = np.min([min, np.min(arr)])
        max = np.max([max, np.max(arr)])
        for tag in flow_dict['tags']:
            values[tag_int[tag]] = np.concatenate([values[tag_int[tag]], arr])
    pair_plot(
        nrows=len(tags),
        ncols=len(tags),
        labels=tags,
        data=values,
        bins=np.linspace(min, max + 0.1, 100),
        fig_path=fig_path
    )


def plot_stats():
    img_path = '/opt/project/img'
    join = lambda x: os.path.join(img_path, x)
    logger.info("Load data from json files.")
    meta_data = extract_data(3, '/k8s-json')

    logger.info("Compare frame sizes.")
    compare_frame_sizes(meta_data, img_path)
    logger.info("Compare record sizes.")
    compare_record_sizes(meta_data, img_path)
    logger.info("Compare num frames.")
    compare_num_frames(meta_data, img_path)
    logger.info("Compare num records.")
    compare_num_tls_records(meta_data, img_path)
    logger.info("Evaluate num records in frame.")
    compare_num_tls_records_in_frame(meta_data, img_path)

    logger.info("Pair Plot frame sizes.")
    pair_plot_attribute(meta_data, 'frame_sizes', join('pair_plot_frame_sizes'))
    logger.info("Pair Plot frame sizes in.")
    pair_plot_attribute(meta_data, 'frame_sizes_in', join('pair_plot_frame_sizes_in'))
    logger.info("Pair Plot frame sizes out.")
    pair_plot_attribute(meta_data, 'frame_sizes_out', join('pair_plot_frame_sizes_out'))

    logger.info("Pair Plot record sizes.")
    pair_plot_attribute(meta_data, 'tls_record_sizes', join('pair_plot_tls_record_sizes'))
    logger.info("Pair Plot record sizes in.")
    pair_plot_attribute(meta_data, 'tls_record_sizes_in', join('pair_plot_tls_record_sizes_in'))
    logger.info("Pair Plot record sizes out.")
    pair_plot_attribute(meta_data, 'tls_record_sizes_out', join('pair_plot_tls_record_sizes_out'))

    # logger.info("Pair Plot Client Hello.")
    # pair_plot_handshake(meta_data, 'CLIENT_HELLO', join('pair_plot_client_hello'))
    # logger.info("Pair Plot Server Hello.")
    # pair_plot_handshake(meta_data, 'SERVER_HELLO', join('pair_plot_server_hello'))
    # logger.info("Pair Plot Certificate.")
    # pair_plot_handshake(meta_data, 'CERTIFICATE', join('pair_plot_certificate'))
    # logger.info("Pair Plot Client Key Exchange.")
    # pair_plot_handshake(meta_data, 'CLIENT_KEY_EXCHANGE', join('pair_plot_client_key_exchange'))
    # logger.info("Pair Plot Server Key Exchange.")
    # pair_plot_handshake(meta_data, 'SERVER_KEY_EXCHANGE', join('pair_plot_server_key_exchange'))


def extract_stats():
    main_flow_stats(3)


if __name__ == '__main__':
    plot_stats()
    # extract_stats()
