from __future__ import annotations
import json
import os
import numpy as np
import time
import sys
import logging
import redis
import gc
from typing import List, Dict, Tuple, Any, Union
print(sys.path)
sys.path.insert(0, '/opt/project')
import argparse

import implementation.rediswq as rediswq
import implementation.data_conversion.tls_flow_extraction as tlsex
import implementation.logging_factory as logging_factory
from implementation.phmm_np.grid_search import unpack, _get_edges
from implementation.seqcache import read_cache, is_cached
import implementation.classification.binary as bcmod
from implementation.classification.seq_classifier import SeqClassifier
from implementation.classification.phmm import CPhmm
from implementation.classification.mc import MarkovChain
from scripts.knn_train_eval import Config
from implementation.phmm_np.grid_search import train_phmm_model

pod_name = os.environ.get('POD_NAME')
logger = logging_factory.produce_logger(
    name=f'{pod_name}',
    log_dir='/opt/project/data/grid-search-results/logs'
)

with open("/opt/project/data/closed-world-labels.json", "r") as fh:
    closed_world_labels = json.load(fh)
with open("/opt/project/data/open-world-labels.json", "r") as fh:
    open_world_labels = json.load(fh)


def calc_max_nll(model: SeqClassifier, config_d: Dict[str, Any]) -> Tuple[float, np.array]:
    lengths, x_train = unpack(read_cache(f"{model.label}_train.json"))
    config = Config.from_dict(config_d)
    edges = _get_edges(
        main_flows=x_train,
        seq_element=config.seq_element_type,
        binning_method=config.binning_method,
        num_bins=config.num_bins,
        max_val_geo_bin=config.max_bin_size,
        seq_length=config.seq_length
    )
    main_flow_to_symbol = tlsex.MainFlowToSymbol(
        seq_length=config.seq_length,
        to_symbolize=config.seq_element_type,
        bin_edges=edges
    )
    x_train = [main_flow_to_symbol(m) for m in x_train]
    lls = model.score_c(x_train)
    max_ll = np.max(np.abs(lls))
    return max_ll, edges


def train_model(trial_dir: str, dst_trial_dir: str,
                updates: Dict[str, Any]=None, defense: None | Dict[str, Any] = None,
                direction_to_filter=0) -> Tuple[float, np.array, Config, Union[CPhmm, MarkovChain]]:
    with open(os.path.join(trial_dir, 'config.json'), 'r') as fh:
        config = json.load(fh)
    with open("/opt/project/closed-world-labels.json", "r") as fh:
        closed_world_labels = json.load(fh)
    logger = logging.getLogger('tmp-logger')
    logger.setLevel(logging.DEBUG)
    config['exp_dir'] = '/opt/project/data/grid-search-results'
    config['num_samples'] = 10 if config['classifier'] == 'phmm' else 1
    config['trial_dir'] = dst_trial_dir
    if updates is not None:
        for k, v in updates.items():
            config[k] = v
    model = train_phmm_model(
        params=config,
        closed_world_labels=closed_world_labels,
        logger=logger,
        eval_dset='test',
        defense=defense,
        direction_to_filter=direction_to_filter
    )
    return model.max_nll, model.edges, Config.from_dict(config), model


def load_model(trial_dir: str, prefix=None) -> Tuple[float, np.array, Config, Union[CPhmm, MarkovChain]]:
    logger.info(f"Retrieve model from trial dir {trial_dir}.")
    if prefix is None:
        config_name = 'config.json'
        model_name = 'model.json'
    else:
        config_name = f'config-{prefix}.json'
        model_name = f'model-{prefix}.json'
    with open(os.path.join(trial_dir, config_name), 'r') as fh:
        config = json.load(fh)
    with open(os.path.join(trial_dir, model_name), 'r') as fh:
        params = json.load(fh)
        params['init_prior'] = config['hmm_init_prior']
        params['label'] = config['label']
        params['num_iter'] = config['hmm_num_iter']

    model = {'phmm': CPhmm, 'mc': MarkovChain}[config['classifier']].from_dict(params)
    logger.info("Calculate negative log likelihood")
    nll, edges = calc_max_nll(model, config)
    config = Config.from_dict(config)
    # if 'max_nll_train' in config:
    #     nll = config['max_nll_train']
    # else:
    #     nll = calc_max_nll(model, config)
    #     # config['max_nll_train'] = float(nll)
    #     # with open(os.path.join(trial_dir, 'config_.json'), 'w') as fh:
    #     #     json.dump(config, fh)
    return nll, edges, config, model


def make_multi_binary_classifier(trial_dirs: List[str], dset: str, dst_trial_dir: str,
                                 updates: Dict[str, Any], defense: None | Dict[str, Any] = None,
                                 direction_to_filter=0) -> bcmod.MultiBinaryClassifier:
    if dset == 'test':
        binary_classifiers = []
        for td in trial_dirs:
            binary_classifiers.append(train_model(td, dst_trial_dir, updates, defense, direction_to_filter))
    else:
        binary_classifiers = [load_model(td) for td in trial_dirs]
    mbc = bcmod.MultiBinaryClassifier(None)
    if defense is None:
        defense_ = None
    else:
        random = np.random.RandomState(seed=defense['seed'])
        defense_ = tlsex.RandomRecordSizeDefense(
            max_seq_length=30,
            get_random_length=lambda x: random.randint(
                defense['min_record_size'],
                defense['max_record_size']
            )
        )
    for nll, edges, config, bc in binary_classifiers:
        wrapper = bcmod.BinaryClassifier(
            ano_density_estimator=None,
            seq_length=config.seq_length,
            seq_element_type=config.seq_element_type,
            bin_edges=edges,
            defense=defense_,
            direction_to_filter=direction_to_filter
        )
        wrapper.density_estimator = bc
        wrapper.threshold = nll
        mbc.bcs[bc.label] = wrapper
    return mbc


def evaluate(closed_world_labels: List[str], open_world_labels: List[str],
             trial_dirs: List[str], result_dir: str, updates: Dict[str, Any],
             scenario: str, dset='test', defense: None | Dict[str, Any] = None,
             direction_to_filter=0) -> Dict[int, Dict[str, Dict[str, float]]]:
    logger.info("Make binary classifiers")
    mbc = make_multi_binary_classifier(trial_dirs, dset, result_dir, updates, defense, direction_to_filter)
    mbc.scenario = scenario
    conf_mats = {}
    web_sites = [l for l in closed_world_labels]
    web_sites.extend(open_world_labels)
    for i, web_site in enumerate(web_sites):
        true_label = web_site
        logger.info(f"Evaluate traces of {web_site} - {i} of {len(web_sites)}.")
        if not is_cached(f"{web_site}_{dset}.json"):
            logger.info(f"test data for {web_site} does not exist.")
            continue
        x_test = read_cache(f"{web_site}_{dset}.json")
        logger.debug(f"\tRetrieved {len(x_test)} days for {web_site}.")
        for day, x_day in x_test.items():
            logger.info(f"Evaluate day {day} - has {len(x_day)} main flows.")
            day = int(day)
            if day not in conf_mats:
                conf_mats[day] = {}
            conf_mats[day][true_label] = {}
            predicted_labels = mbc.predict(x_day)
            for predicted_label in predicted_labels:
                if predicted_label not in conf_mats[day][true_label]:
                    conf_mats[day][true_label][predicted_label] = 0
                conf_mats[day][true_label][predicted_label] += 1
    with open(os.path.join(result_dir, 'timings.json'), 'w') as fh:
        json.dump(mbc.inference_times, fh)
    return conf_mats


def phmm_models() -> Dict[str, str]:
    """
    Models with best precision over all configurations.
    Returns:

    """
    return {
        "chaturbate.com": "/opt/project/data/grid-search-results/072e258ac4594aa7b2b249394b09ee49_0",
        "chouftv.ma": "/opt/project/data/grid-search-results/711b5764e85c487d9c51d7181c917ced_0",
        "daftsex.com": "/opt/project/data/grid-search-results/93a0167a12a04c83be137bc6f9972631_0",
        "medium.com": "/opt/project/data/grid-search-results/4d233d1254a347b2add37f1816eaa7fd_0",
        "namnak.com": "/opt/project/data/grid-search-results/9473fa5535644feba762f3ca5db56ebd_0",
        "namu.wiki": "/opt/project/data/grid-search-results/a03a3e9c63b349a6b8b20636caa40fcc_0",
        "softonic.com": "/opt/project/data/grid-search-results/6c97ce6eff314b5fbb880834ff0ff449_0",
        "spankbang.com": "/opt/project/data/grid-search-results/beeb86279118405db61d118f5bf2b79a_0",
        "txxx.com": "/opt/project/data/grid-search-results/7fa4d9e1c356465ea0d984caa2f3852c_0",
        "www.amazon.co.jp": "/opt/project/data/grid-search-results/ff2ae7321299423785d9dca83b83ddc6_0",
        "www.amazon.com": "/opt/project/data/grid-search-results/2bb3c74186ce44f4959653234830c68f_0",
        "www.amazon.in": "/opt/project/data/grid-search-results/be48d6ab35b74cb0b94f26593f908877_0",
        "www.bola.net": "/opt/project/data/grid-search-results/a3f8d0b2133e47a09b87c7570c6285c6_0",
        "www.brilio.net": "/opt/project/data/grid-search-results/34cb8033bda741aa8fe70f1cd4fa5bf6_0",
        "www.cnet.com": "/opt/project/data/grid-search-results/d4487c75d31e49ffabcb4e54a6e995d3_0",
        "www.ebay-kleinanzeigen.de": "/opt/project/data/grid-search-results/94a6616495d447c4a4799bbe473fb08c_0",
        "www.ebay.ca": "/opt/project/data/grid-search-results/2122f414d1bb4831a1873e0a644c2dbe_0",
        "www.ebay.co.uk": "/opt/project/data/grid-search-results/22bef486461040ec8b4f26dc7d6c1d6a_0",
        "www.ebay.com.au": "/opt/project/data/grid-search-results/976a20ccaa9042d7adcbc6e00907af81_0",
        "www.elbalad.news": "/opt/project/data/grid-search-results/db73132175494147a7e2fa01057490e6_0",
        "www.espn.com": "/opt/project/data/grid-search-results/18ab4506da364682ad9920b88b8129c2_0",
        "www.flashscore.com": "/opt/project/data/grid-search-results/499f0075333a4c53b2a738ad2c234a2b_0",
        "www.google.com.br": "/opt/project/data/grid-search-results/4821f25bad094d8283d9e60768050225_0",
        "www.google.com.hk": "/opt/project/data/grid-search-results/2b6ac40ea2f442658e1bf7b5bb3ddaa3_0",
        "www.google.de": "/opt/project/data/grid-search-results/f7978381cb01461ba568b1207edf2349_0",
        "www.google.es": "/opt/project/data/grid-search-results/219d249d397f43c081e71ec4ba75bb88_0",
        "www.google.fr": "/opt/project/data/grid-search-results/de21686df96b41b89a89211163244d3b_0",
        "www.google.ru": "/opt/project/data/grid-search-results/8aa977be291a4adfbaaf2e8ed85c8870_0",
        "www.grammarly.com": "/opt/project/data/grid-search-results/71b8eba37cd44a69931d54758c71f40a_0",
        "www.healthline.com": "/opt/project/data/grid-search-results/ad1836edf82b4ad594eac8518722e986_0",
        "www.imdb.com": "/opt/project/data/grid-search-results/13033f7c8028415f9bee4f5c507ed731_0",
        "www.inquirer.net": "/opt/project/data/grid-search-results/bf2b2469407c4151b5576d9a77ee9dfe_0",
        "www.instagram.com": "/opt/project/data/grid-search-results/4ab58d01ce2c4583bf6b1a478f98a0f9_0",
        "www.kompas.com": "/opt/project/data/grid-search-results/e2e0a7ddf85f4b849076b70bb384ec60_0",
        "www.ladbible.com": "/opt/project/data/grid-search-results/a57bc3f24de4423084858723002f6b08_0",
        "www.metropoles.com": "/opt/project/data/grid-search-results/c1b8c4f8f0234ee5916302fd3f08153f_0",
        "www.nih.gov": "/opt/project/data/grid-search-results/2283f9b2f29e4113bd8bfd91188dea53_0",
        "www.okta.com": "/opt/project/data/grid-search-results/f1d0f905480746dcb737a4435acc58a8_0",
        "www.primevideo.com": "/opt/project/data/grid-search-results/dbca67c04c154f2e9f26fa5217b1236f_0",
        "www.shutterstock.com": "/opt/project/data/grid-search-results/81ccb81904024bf8b1c970e69ebc26eb_0",
        "www.tahiamasr.com": "/opt/project/data/grid-search-results/68d10fcb853144d19e16d1d6e981f4f6_0",
        "www.theepochtimes.com": "/opt/project/data/grid-search-results/0ebdb7166c9b49e2985ef344c33920f3_0",
        "www.trendyol.com": "/opt/project/data/grid-search-results/d8762055816343a6907c95d9b633b39b_0",
        "www.uber.com": "/opt/project/data/grid-search-results/f9134de110ec471887af220a194ecec7_0",
        "www.vidio.com": "/opt/project/data/grid-search-results/ed898d87f82b4f218bf686c61498f034_0",
        "www.worldometers.info": "/opt/project/data/grid-search-results/c414382abe89461a880ea42e4e5df72c_0",
        "www.youm7.com": "/opt/project/data/grid-search-results/bdbf9e4c5f53459f82bb464bb401547b_0",
        "xhamster.com": "/opt/project/data/grid-search-results/e4fa3a84fd3146dfaf708184867ffead_0",
        "zoom.us": "/opt/project/data/grid-search-results/2b4b14c3474244788cc75392229dfc3c_0"
    }


def phmm_models_by_seq_length() -> Dict[int, Dict[str, str]]:
    with open("/opt/project/phmm-configs-by-seq-length.json", 'r') as fh:
        d = json.load(fh)
    return d


def mc_models_by_seq_length() -> Dict[int, Dict[str, str]]:
    with open("/opt/project/mc-configs-by-seq-length.json", 'r') as fh:
        d = json.load(fh)
    return d


def mc_models() -> Dict[str, str]:
    """
    Models with best precision over all configurations.
    Returns:

    """
    return {
        "chaturbate.com": "/opt/project/data/grid-search-results/cba2dd76a9ed4a17ae3e993ba2f19867_0",
        "chouftv.ma": "/opt/project/data/grid-search-results/792dc92db7b64007bc7214d863f361e1_0",
        "daftsex.com": "/opt/project/data/grid-search-results/0b50e07d6bb84f018d5a5f0f808b9fee_0",
        "medium.com": "/opt/project/data/grid-search-results/805ea3840846486980b60660c2612494_0",
        "namnak.com": "/opt/project/data/grid-search-results/968dfed675674b04bfa2305923a27246_0",
        "namu.wiki": "/opt/project/data/grid-search-results/c07441e6ddd94d999844d2e0dec206f7_0",
        "softonic.com": "/opt/project/data/grid-search-results/1144ad68fafd4c90b02608803491d9e7_0",
        "spankbang.com": "/opt/project/data/grid-search-results/21bb6bca944b458da4f637e10113f9e1_0",
        "txxx.com": "/opt/project/data/grid-search-results/70635a3d5a8b45c9b715e34a0d3ed21c_0",
        "www.amazon.co.jp": "/opt/project/data/grid-search-results/4db4b089b03b4d429cc5e803d14e4450_0",
        "www.amazon.com": "/opt/project/data/grid-search-results/3587466ad65e40d584863ca2924cb4d0_0",
        "www.amazon.in": "/opt/project/data/grid-search-results/9d8e591d214a4d22a98dd1ee81854dbb_0",
        "www.bola.net": "/opt/project/data/grid-search-results/42674fb0d5e14d99bd49b57670ef62b1_0",
        "www.brilio.net": "/opt/project/data/grid-search-results/074a13b292cb4f88a8e7cac069cc1371_0",
        "www.cnet.com": "/opt/project/data/grid-search-results/bc71c628967d488cba442f325b2d4b01_0",
        "www.ebay-kleinanzeigen.de": "/opt/project/data/grid-search-results/b96ad79c20484160b4fd7116b43675ae_0",
        "www.ebay.ca": "/opt/project/data/grid-search-results/317106e85cce4b77b3b5a986170727ca_0",
        "www.ebay.co.uk": "/opt/project/data/grid-search-results/8e9a0b5f982e40be8d78c6ebe215ed0f_0",
        "www.ebay.com.au": "/opt/project/data/grid-search-results/e4893c5a792e4f6db120de609214f309_0",
        "www.elbalad.news": "/opt/project/data/grid-search-results/ce1e29c10160495ba23c9239d6baa4c6_0",
        "www.espn.com": "/opt/project/data/grid-search-results/87129b70e5f94dd5b8e4e8da9bfd6f9e_0",
        "www.flashscore.com": "/opt/project/data/grid-search-results/764863ce71894902acbda1828256144c_0",
        "www.google.com.br": "/opt/project/data/grid-search-results/035d44a5f9644f358d6379f022f55ee0_0",
        "www.google.com.hk": "/opt/project/data/grid-search-results/60c53056fdca4a18b6d17f3d999f5905_0",
        "www.google.de": "/opt/project/data/grid-search-results/f82595a65f914012bddb988754b59aca_0",
        "www.google.es": "/opt/project/data/grid-search-results/d8d689289e60414bb3df8ab661fb07fe_0",
        "www.google.fr": "/opt/project/data/grid-search-results/f3a09e8f0b744114b69986e443ed30f3_0",
        "www.google.ru": "/opt/project/data/grid-search-results/69c88466fabb4822915a0fd92f3c4158_0",
        "www.grammarly.com": "/opt/project/data/grid-search-results/8a170bb98950452ab3327581521157fb_0",
        "www.healthline.com": "/opt/project/data/grid-search-results/4ce49b40f18c4b7abc5cb93a4d75652d_0",
        "www.imdb.com": "/opt/project/data/grid-search-results/c88e64ee2dd04112aea227b4a894b2a0_0",
        "www.inquirer.net": "/opt/project/data/grid-search-results/27b0f0a6585741feb355482839daba16_0",
        "www.instagram.com": "/opt/project/data/grid-search-results/2046691e4431421f848b2982d071c421_0",
        "www.kompas.com": "/opt/project/data/grid-search-results/79a5b510ac2f4d12a93374e0b9124847_0",
        "www.ladbible.com": "/opt/project/data/grid-search-results/2d20331bc5864e1daa7c7899eecf2b36_0",
        "www.metropoles.com": "/opt/project/data/grid-search-results/ed9e8be3d2b84b579b8f845f6a5dd5af_0",
        "www.nih.gov": "/opt/project/data/grid-search-results/f9044dd86e2e407090a08664e88962a3_0",
        "www.okta.com": "/opt/project/data/grid-search-results/1519917e3b684dcf84a8599ab97c8fed_0",
        "www.primevideo.com": "/opt/project/data/grid-search-results/f18075b8be71451d829bc7165558054e_0",
        "www.shutterstock.com": "/opt/project/data/grid-search-results/f2d8d6d74a3a4aaabf3953bb1dce96c6_0",
        "www.tahiamasr.com": "/opt/project/data/grid-search-results/72d6bfe89e11439395c0c6986d3507f1_0",
        "www.theepochtimes.com": "/opt/project/data/grid-search-results/7d4933d87c43446386f8153a180d0691_0",
        "www.trendyol.com": "/opt/project/data/grid-search-results/860e1c4fdd754f5e99155aa96b29fe00_0",
        "www.uber.com": "/opt/project/data/grid-search-results/36d14213e5ad49fb952eb49c657ca171_0",
        "www.vidio.com": "/opt/project/data/grid-search-results/eccf9399111d441e969791e489587ef8_0",
        "www.worldometers.info": "/opt/project/data/grid-search-results/90e9e9ea275649b1a4ab9b2c9390fa55_0",
        "www.youm7.com": "/opt/project/data/grid-search-results/e2882b12a2da4c78ae30070804947460_0",
        "xhamster.com": "/opt/project/data/grid-search-results/58f931cdf5654b0bafb1b0e368560eb3_0",
        "zoom.us": "/opt/project/data/grid-search-results/f90df39961d54752a888c317f5a23c9a_0"
    }


def from_redis():
    wait = 10
    redis_q = rediswq.RedisWQ(name='pgm_eval', host='tueilkn-swc06.forschung.lkn.ei.tum.de')
    reported_queue_empty = False
    count = 0
    while count <= 10:
        try:
            if redis_q.empty():
                if not reported_queue_empty:
                    logger.info(f"Queue empty, wait for new jobs...")
                    reported_queue_empty = True
                time.sleep(wait)
            else:
                reported_queue_empty = False
                item = redis_q.lease(lease_secs=30, block=True, timeout=2)
                if item is not None:
                    redis_q.complete(item)
                    config = json.loads(item.decode("utf-8"))
                    count += 1
                    msg = "-".join([f"{k}-{v}" for k, v in config.items()])
                    sep = '\n\t====================================================='
                    logger.info("\n", sep, "\n", msg, sep)

                    t1 = time.perf_counter()
                    result_dir = config['result_dir']
                    if not os.path.exists(result_dir):
                        os.mkdir(result_dir)
                    with open(os.path.join(result_dir, 'meta-config.json'), 'w') as fh:
                        json.dump(config, fh)

                    scenario = config['scenario']
                    tds = config['trial_dirs']
                    conf_mats = evaluate(
                        closed_world_labels=closed_world_labels,
                        open_world_labels=open_world_labels if scenario == 'open' else [],
                        trial_dirs=list(tds.values()),
                        result_dir=result_dir,
                        updates={'day_train_end': config['day_train_end']},
                        scenario=scenario,
                        dset='test'
                    )
                    with open(os.path.join(result_dir, 'conf-mats.json'), 'w') as fh:
                        json.dump(conf_mats, fh)
                    logger.info(f"Trained model number {count} {msg} in {time.perf_counter() - t1}s")
                    gc.collect()
        except Exception as e:
            logger.exception(e)


def make_list(items: List[Any]) -> str:
    s = ''
    for i, item in enumerate(items):
        s += f"\n\t{i}. {item}"
    return s


def run_from_cmd(model: str | None, scenario: str | None, train_for_days: int,
                 defense: Dict[str, Any] | None, direction_to_filter: int):
    if model is None:
        models = ['phmm', 'mc']
        idx = -1
        while not (0 <= idx < len(models)):
            idx = int(input("Enter number of the model to evaluate:" + make_list(models) + "\nYour input:"))
        model = models[idx]
    if scenario is None:
        scenarios = ['closed', 'open']
        idx = -1
        while not (0 <= idx < len(scenarios)):
            idx = int(input("Enter number of the scenario to evaluate:" + make_list(scenarios) + "\nYour input:"))
        scenario = scenarios[idx]
    if train_for_days == -1:
        train_for_days = -1
        while not (0 <= train_for_days <= 75):
            train_for_days = int(input("Enter number of trains from in {{1, ..., 75}}: "))
    assert model in ['phmm', 'mc']
    assert scenario in ['closed', 'open']
    assert 0 <= train_for_days <= 75
    print(f"Evaluate model {model} in scenario {scenario} with data from {train_for_days} days.")

    labels = [s for s in closed_world_labels]
    labels.extend(open_world_labels)

    # with open('/opt/project/open-world-phmm-models.json', 'r') as fh:
    #     trial_dirs = json.load(fh)
    # trial_dirs = [#"/opt/project/data/grid-search-results/000ee76daf2e4c4f8dc078aecd7ba73a_0",
    #               "/opt/project/data/grid-search-results/devel-model"]
    # trial_dirs = {'phmm': phmm_models_by_seq_length, 'mc': mc_models_by_seq_length}[model]()
    # trial_dirs = {'unnecessary-first-level-for-for-loop': {'phmm': phmm_models, 'mc': mc_models}[model]()}
    # trial_dirs = {'unnecessary-first-level-for-for-loop': {
    #     'www.kompas.com': '/opt/project/data/grid-search-results/devel-model',
    #     'zoom.us': '/opt/project/data/grid-search-results/devel-model2'
    # }}
    trial_dirs = {'phmm': phmm_models, 'mc': mc_models}[model]()
    # trial_dirs = {
    #     'zoom.us': '/opt/project/data/grid-search-results/devel-model2',
    #     'www.kompas.com': '/opt/project/data/grid-search-results/devel-model'
    # }
    # defense = {
    #     'name': 'RandomRecordSizeDefense',
    #     'min_record_size': 500,
    #     'max_record_size': 8000,
    #     'seed': 1
    # }
    if defense is None:
        dfstr = ''
    else:
        dfstr = f'-{defense["name"]}-{defense["min_record_size"]}-{defense["max_record_size"]}'
    fstr = ''
    if direction_to_filter == -1:
        fstr = '-filter-client2server'
    if direction_to_filter == 1:
        fstr = '-filter-server2client'
    # for seq_length, tds in trial_dirs.items():
    # result_dir = f"/opt/project/data/grid-search-results/{scenario}-world-{model}-results-all-data-seqlength-{seq_length}"
    # result_dir = f"/opt/project/data/grid-search-results/{scenario}-world-{model}-results-all-data-loaded"
    result_dir = f"/opt/project/data/grid-search-results/eval-results/{scenario}-world-{model}-" \
                 f"training-days-{train_for_days}{dfstr}{fstr}"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    with open(os.path.join(result_dir, 'meta-config.json'), 'w') as fh:
        json.dump(
            {
                'trial_dirs': trial_dirs,
                'scenario': scenario,
                'day_train_end': train_for_days,
                'defense': defense,
                'direction_to_filter': direction_to_filter
            },
            fh
        )
    conf_mats = evaluate(
        closed_world_labels,
        open_world_labels if scenario == 'open' else [],
        list(trial_dirs.values()),
        result_dir,
        {'day_train_end': train_for_days},
        scenario,
        'test',
        defense,
        direction_to_filter
    )
    with open(os.path.join(result_dir, 'conf-mats.json'), 'w') as fh:
        json.dump(conf_mats, fh)


def run_overdays():
    run_from_cmd(int(sys.argv[4]))


def populate():
    models = ['phmm', 'mc']
    scenario = 'open'
    rdir_prefix = f"/opt/project/data/grid-search-results/"
    count = 0
    redis_db = redis.StrictRedis(host='tueilkn-swc06.forschung.lkn.ei.tum.de')
    for model in models:
        trial_dirs = {'phmm': phmm_models, 'mc': mc_models}[model]()
        for days_to_train_on in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]:
            count += 1
            meta_config = {
                'day_train_end': days_to_train_on,
                'scenario': scenario,
                'trial_dirs': trial_dirs,
                'result_dir': os.path.join(rdir_prefix, f"{scenario}-world-{model}-results-days-to-train-{days_to_train_on}")
            }
            redis_db.lpush('pgm_eval', json.dumps(meta_config))
            print(f"Pushed {count} configs.")


if __name__ == '__main__':
    if sys.argv[1] == 'redis':
        from_redis()
    elif sys.argv[1] == 'populate':
        populate()
    elif sys.argv[1] == 'overdays':
        run_overdays()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model",
            help="Model that should be evaluated, must be in {mc, phmm}.",
            default="None"
        )
        parser.add_argument(
            "--scenario",
            help="Scenario that should be evaluated, must be in {open, closed}.",
            default='None'
        )
        parser.add_argument(
            "--train-for-days",
            help="Number of days that should be used for training.",
            default=70,  # Use all days.
            type=int
        )
        parser.add_argument(
            "--filter-client-to-server",
            help="Remove all packets/frames travelling from the client to the server",
            action="store_true"
        )
        parser.add_argument(
            "--filter-server-to-client",
            help="Remove all packets/frames travelling from the server to the client",
            action="store_true"
        )
        parser.add_argument(
            "--use-defense",
            help="Randomly change the size of record lengths. If set, specify min-record-size and max-record-size",
            action="store_true"
        )
        parser.add_argument(
            "--min-record-size",
            help="Minimum Record size for defense.",
            type=int,
            default=-1
        )
        parser.add_argument(
            '--max-record-size',
            help="Maximum record size for defense, must be smaller 2^14.",
            type=int,
            default=-1
        )

        parsed_args, _ = parser.parse_known_args(sys.argv[2:])
        if parsed_args.use_defense:
            defense = {
                'name': 'RandomRecordSizeDefense',
                'min_record_size': int(parsed_args.min_record_size),
                'max_record_size': int(parsed_args.max_record_size),
                'seed': 1
            }
        else:
            defense = None
        run_from_cmd(
            model=parsed_args.model,
            scenario=parsed_args.scenario,
            train_for_days=parsed_args.train_for_days,
            defense=defense,
            # direction is indicated with -1/1 in the data, thus cast settings
            # to this range. If value is not in -1/1, then no filtering is applied.
            direction_to_filter=-1 * int(parsed_args.filter_client_to_server) + int(parsed_args.filter_server_to_client)
        )
