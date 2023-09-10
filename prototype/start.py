import threading
import psutil
import shlex
import os
import sys
import time
import subprocess
import logging
#import pandas as pd
#import numpy as np
import json as js
import argparse
from typing import List, Tuple, Dict, Any, Union

threads = []
processes = []
pidList = []

PKT_GEN = 15
TLS_FILTER = 1
TLS_RECORD_DETECTOR = 2
CLASSIFIER = 3
DATA_AGG = 4
HMM = 5

nf_path = '/home/nfv/openNetVM/NF'
onvm_mgr_path = '/home/nfv/ma_onvm/onvm'
data_agg_config_path = '/home/nfv/openNetVM/q_data_aggs'
hmm_config_path = '/home/nfv/openNetVM/hmms'
run_manager_cmd = 'sudo /home/nfv/ma_onvm/onvm/onvm_mgr/' +\
    'x86_64-native-linuxapp-gcc/onvm_mgr -l 0,1,{tx_cores:s} -n 16 ' +\
    '--proc-type=primary -- -p 3 -n 0xff -s stdout -v 1 -a {num_threads:d} -q /home/nfv/SWC_Test/'


#nf_path = '/home/sim/code/openNetVM/NF'
#onvm_mgr_path = '/home/sim/code/openNetVM/onvm'
#data_agg_config_path = '/home/sim/data/openNetVM/q_data_aggs'
#hmm_config_path = '/home/sim/data/openNetVM/hmms'
#run_manager_cmd = 'sudo /home/sim/code/openNetVM/onvm/onvm_mgr/' +\
#    'x86_64-native-linuxapp-gcc/onvm_mgr -l 0,1,{tx_cores:s} -n 8 ' +\
#    '--proc-type=primary -- -p 3 -n 0xff -s stdout -v 1'

class Run(object):

    def __init__(self,
        worker_commands: List[str],
        sfcs: List[List[int]],
        num_tx_threads: int=2,
        with_yielding: bool=True,
        path_to_nf_control_plane: str=None
        ):

        self.worker_commands = worker_commands
        self.sfcs = sfcs
        self.num_tx_threads = num_tx_threads
        self.with_yielding = with_yielding
        self.path_to_nf_control_plane = path_to_nf_control_plane

    def run_manager_cmd(self):
        """
            Build the run manager cmd from the passed arguments.
        """
        cores = ",".join(
            ["{:d}".format(i) for i in range(2, self.num_tx_threads + 2)]
        )
        return run_manager_cmd.format(tx_cores=cores, num_threads=self.num_tx_threads)


def get_manager_cmd_with_sfcs(run_config: Run):
    """
    Sets the manager command out of a run config

    Args:
       run_config (Run): run config
    Returns:
        cmd (str): command to run the manager
    """

    cmd = run_config.run_manager_cmd()

    if len(run_config.sfcs) > 1:
        cmd += ' -i '
        for sfc in run_config.sfcs:
            for nf in sfc:
                cmd += str(nf) + ','
            cmd = cmd[:-1]
            cmd += ':'
        cmd = cmd[:-1]

    return cmd


def manager(run_manager_cmd):
    """
    Spawns the openNetVM manager with run_config

    Args:
        run_manager_cmd (str): command to spawn the manager
    Returns:
        /
    """

    os.chdir(onvm_mgr_path)
    args = shlex.split(run_manager_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    # Wait till the manager has spawned its threads
    time.sleep(10)
    parent = psutil.Process(proc.pid)
    children = parent.children()
    child_pids = [p.pid for p in children]
    for p in child_pids:
        pidList.append(p)
    return


def spawn_manager(num_tx_threads: int):
    """
    Spawns the openNetVM manager with run_config

    Args:
        /
    Returns:
        /
    """

    run_config = Run(worker_commands=[], sfcs=[], num_tx_threads=num_tx_threads)
    manager_cmd = get_manager_cmd_with_sfcs(run_config)
    print(manager_cmd)
    tm = threading.Thread(target=manager, args=(manager_cmd,))
    threads.append(tm)
    tm.start()
    time.sleep(10)


###############################################################################################################

def tls_filter(tls_filter_cmd):
    """
    Spawns a tls_filter in a new thread

    Args:
        tls_filter_cmd (str): command to start the tls_filter
    Returns:
        /
    """

    os.chdir(nf_path)
    args = shlex.split(tls_filter_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    return


def spawn_tls_filter(req_packets, core):
    """
    Spawns the tls_filter in a new thread

    Args:
        req_packets (int): maximum number of packets required for classification
    Returns:
        /
    """

    tls_filter_cmd = f'./start_nf.sh tls_filter {TLS_FILTER} {core} -d {TLS_RECORD_DETECTOR} -m {req_packets} > /home/nfv/SWC_Test/tls-filter.log'
    tm = threading.Thread(target=tls_filter, args=(tls_filter_cmd,))
    threads.append(tm)
    tm.start()
    # time.sleep(5)

###############################################################################################################

def tls_record_detector(tls_record_det_cmd):
    """
    Spawns a tls_record_detector in a new thread

    Args:
        tls_record_det_cmd (str): command to start the tls_record_detector
    Returns:
        /
    """

    os.chdir(nf_path)
    args = shlex.split(tls_record_det_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    return


def spawn_tls_record_detector(req_packets, num_hmms, core):
    """
    Spawns the tls_record_detector in a new thread

    Args:
        req_packets (int): maximum number of packets required for classification
        num_hmms (int): number of hmms
    Returns:
        /
    """

    tls_record_det_cmd = f'./start_nf.sh tls_record_det {TLS_RECORD_DETECTOR} {core} -d {DATA_AGG} -n {num_hmms} -m {req_packets} > /home/nfv/SWC_Test/tls-record-detector.log'
    tm = threading.Thread(target=tls_record_detector, args=(tls_record_det_cmd,))
    threads.append(tm)
    tm.start()
    # time.sleep(5)


###############################################################################################################

def data_agg(data_agg_cmd):
    """
    Spawns a data_agg in a new thread

    Args:
        data_agg_cmd (str): command to start the data_agg
    Returns:
        /
    """

    os.chdir(nf_path)
    args = shlex.split(data_agg_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    return


def spawn_data_agg(sid, core, dst):
    """
    Spawns a data_agg in a new thread

    Args:
        core (int): core number of the NF
        dst (int): destination to which the NF forwards to
    Returns:
        /
    """

    data_agg_cmd = f'./start_nf.sh data_agg {sid} {core} -d {dst} > /home/nfv/SWC_Test/data-agg-{sid}.log'
    tm = threading.Thread(target=data_agg, args=(data_agg_cmd,))
    threads.append(tm)
    tm.start()
    # time.sleep(5)


###############################################################################################################

def q_data_agg(q_data_agg_cmd):
    """
    Spawns a q_data_agg in a new thread

    Args:
        q_data_agg_cmd (str): command to start the q_data_agg
    Returns:
        /
    """

    os.chdir(nf_path)
    args = shlex.split(q_data_agg_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    return


def spawn_q_data_agg(id, core, dst, config_file):
    """
    Spawns a q_data_agg in a new thread with a given config file

    Args:
        core (int): core number of the NF
        dst (int): destination to which the NF forwards to
        config_file (str): path to config file
    Returns:
        /
    """

    q_data_agg_cmd = f'./start_nf.sh q_data_agg {id} {core} -d {dst} -c {config_file} > /home/nfv/SWC_Test/data-agg-{id}.log'
    tm = threading.Thread(target=q_data_agg, args=(q_data_agg_cmd,))
    threads.append(tm)
    tm.start()
    # time.sleep(5)


###############################################################################################################

def hmm(hmm_cmd):
    """
    Spawns a hmm in a new thread

    Args:
        hmm_cmd (str): command to start the hmm
    Returns:
        /
    """

    os.chdir(nf_path)
    args = shlex.split(hmm_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    return


def spawn_hmm(sid: int, core: int, config_file):
    """
    Spawns a hmm in a new thread with a given config file

    Args:
        core (int): core number of the NF
        config_file (str): path to config file
    Returns:
        /
    """

    hmm_cmd = f'./start_nf.sh hmm {sid} {core} -d {CLASSIFIER} -c {config_file} > /home/nfv/SWC_Test/hmm-{sid}.log'
    tm = threading.Thread(target=hmm, args=(hmm_cmd,))
    threads.append(tm)
    tm.start()


def mc(hmm_cmd):
    """
    Spawns a hmm in a new thread

    Args:
        hmm_cmd (str): command to start the hmm
    Returns:
        /
    """

    os.chdir(nf_path)
    args = shlex.split(hmm_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    return


def spawn_mc(sid: int, core: int, config_file):
    """
    Spawns a hmm in a new thread with a given config file

    Args:
        core (int): core number of the NF
        config_file (str): path to config file
    Returns:
        /
    """

    mc_cmd = f'./start_nf.sh mc {sid} {core} -d {CLASSIFIER} -c {config_file} > /home/nfv/SWC_Test/mc-{sid}.log'
    tm = threading.Thread(target=mc, args=(mc_cmd,))
    threads.append(tm)
    tm.start()


###############################################################################################################

def classifier(classifier_cmd):
    """
    Spawns the classifier in a new thread

    Args:
        classifier_cmd (str): command to start the classifier
    Returns:
        /
    """

    os.chdir(nf_path)
    args = shlex.split(classifier_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    return


def spawn_classifier(num_hmms, core: int):
    """
    Spawns the classifier in a new thread

    Args:
        num_hmms (int): number of hmms in the prototype
    Returns:
        /
    """

    classifier_cmd = f'./start_nf.sh classifier {CLASSIFIER} {core} -n {num_hmms}'
    tm = threading.Thread(target=classifier, args=(classifier_cmd,))
    threads.append(tm)
    tm.start()

    # time.sleep(5)


###############################################################################################################

def pkt_gen(pkt_gen_cmd):
    """
    Spawns the packet generator in a new thread

    Args:
        pkt_gen_cmd (str): command to start the packet generator
    Returns:
        /
    """

    os.chdir(nf_path)
    args = shlex.split(pkt_gen_cmd)
    proc = subprocess.Popen(args)
    processes.append(proc)
    pidList.append(proc.pid)

    return


def spawn_pkt_gen(pcap_file):
    """
    Spawns the packet generator in a new thread

    Args:
        pcap_file (str): path to the pcap file
    Returns:
        /
    """

    pkt_gen_cmd = f'./speed_tester_start_nf.sh speed_tester {PKT_GEN} -d {TLS_FILTER} -o {pcap_file}'
    tm = threading.Thread(target=pkt_gen, args=(pkt_gen_cmd,))
    threads.append(tm)
    tm.start()

    time.sleep(5)


def stop_system():
    """
    Stops the prototype by exiting the processes in the pidList

    Args:
        /
    Returns:
        /
    """

    os.system("sudo kill -2 " + str(pidList[1]))

    for i in range(2, len(pidList)):
        os.system("sudo kill -2 " + str(pidList[i]))
        time.sleep(2)
    for pid in pidList:
        try:
            os.system("sudo kill -9 " + str(pid))
            print('Killin Pid: ' + str(pid))
        except:
            print('Process with Pid: ' + pid + ' was already dead')

    subprocess.call("ps -e | grep tls_ | awk '{print $1}' | xargs sudo kill -9 $1", shell=True)
    subprocess.call("ps -e | grep classifier | awk '{print $1}' | xargs sudo kill -9 $1", shell=True)
    subprocess.call("ps -e | grep hmm | awk '{print $1}' | xargs sudo kill -9 $1", shell=True)
    subprocess.call("ps -e | grep mc | awk '{print $1}' | xargs sudo kill -9 $1", shell=True)
    subprocess.call("ps -e | grep data_agg | awk '{print $1}' | xargs sudo kill -9 $1", shell=True)
    pidList.clear()
    threads.clear()
    processes.clear()
    # time.sleep(2)
    # subprocess.call('clear')


###############################################################################################################

def get_config_files(config_path):
    """
    Collects all config files in spec path

    Args:
        config_path (str): path to configs
    Returns:
        configs (list): list of config files
    """

    _, _, configs = next(os.walk(config_path))

    return configs


def get_configs(data_agg_config_path, hmm_config_path):
    """
    Collects all the config_files used for starting the hmms and data_aggs

    Args:
        data_agg_config_path (str): path to the data_agg config files
        hmm_config_path (str): path to the hmm config files
    Returns:
        configs (list): list containing pairs of hmm and data_aggs config files
    """

    data_agg_configs = get_config_files(data_agg_config_path)
    hmm_configs = get_config_files(hmm_config_path)

    configs = []
    for config in hmm_configs:
        if config.replace('hmm', 'data_agg') in data_agg_configs:
            configs.append((config, config.replace('hmm', 'data_agg')))
        else:
            configs.append((config, 0))

    return configs


def spawn_moongen(pcap_name: str) -> None:
    pcap_path = {
        'high-pps': '/home/nfv/high-pps.pcap',
        'high-fps': '/home/nfv/high-fps.pcap',
        'low-pps': '/home/nfv/high-pps.pcap',
        'low-fps': '/home/nfv/high-fps.pcap',
        'average': '/home/nfv/average.pcap'
    }[pcap_name]
    cmd = f'sudo su nfv -c ' + \
          f'"ssh nfv@rooney.forschung.lkn.ei.tum.de ' + \
          f'\\"sudo /home/nfv/MoonGen/build/MoonGen /home/nfv/MoonGen/examples/pcap/replay-pcap.lua 0 {pcap_path}\\""'
    subprocess.call(
        cmd,
        shell=True
    )


def save_results(archive_name: str) -> None:
    cmd = f'tar -czvf /home/nfv/{archive_name}.tar.gz --directory=/home/nfv/SWC_Test/ ' + \
        '$( find /home/nfv/SWC_Test/ -printf "%f\n" -name "*.txt" -or -name "*.log" -or -name "*.csv" )'
    subprocess.call(cmd, shell=True)
    cmd = 'sudo rm /home/nfv/SWC_Test/*.log'
    subprocess.call(cmd, shell=True)
    cmd = 'sudo rm /home/nfv/SWC_Test/*.txt'
    subprocess.call(cmd, shell=True)
    cmd = 'sudo rm /home/nfv/SWC_Test/*.csv'
    subprocess.call(cmd, shell=True)


def test_mc():
    mc_config = '/home/nfv/openNetVM/NF/mc/config.json'
    mc_config = os.path.join('/home/nfv/openNetVM/mcs', 'www_tahiamasr_com_c_export.json')
    assert os.path.exists(mc_config)
    num_mcs = 2
    req_packets = 5

    spawn_manager()
    first_core = 3
    # spawn tls_filter, tls_record_det and classifier
    spawn_tls_filter(req_packets, first_core)
    spawn_tls_record_detector(req_packets, num_mcs, first_core + 1)
    spawn_classifier(num_mcs, first_core + 2)

    spawn_data_agg(4, 7, 5) # id, real core, dest
    spawn_mc(5, 8, mc_config)

    spawn_data_agg(6, 9, 7) # id, real core, dest
    spawn_mc(7, 10, mc_config)

    time.sleep(10)
    spawn_moongen('high-pps')
    stop_system()


def get_max_req_packets(configs_paths: List[Tuple[str, str]]) -> int:
    """
    Returns the maximum number of packets that are required to classify a flow.
    """
    req_packets = []
    for config_p, _ in configs_paths:
        with open(config_p) as f:
            req_packets.append(js.load(f)['trace_length'])
    req_packets = max(req_packets)
    return req_packets


def get_hmm_configs(phmm_length: int, num_pgms: int) -> List[Tuple[str, str]]:
    configs = [(
        os.path.join('/home/nfv/openNetVM/hmms', f'zoom_us_hmm_config_{phmm_length}_frames.json'),
        os.path.join('/home/nfv/openNetVM/q_data_aggs', f'zoom_us_data_agg_config_{phmm_length}_frames.json')
    ) for _ in range(num_pgms)]
    return configs


def get_mc_configs(which_mc: str, num_pgms: int) -> List[Tuple[str, Union[str, None]]]:
    cfg_p, data_agg = {
        'big-mc': ('/home/nfv/openNetVM/mcs/www_tahiamasr_com_c_export.json', None),
        'small-mc': ('/home/nfv/openNetVM/mcs/www_grammarly_com_c_export.json', '/home/nfv/openNetVM/mcs/www_grammarly_com_data_agg.json')
    }[which_mc]
    configs = [(cfg_p, data_agg) for _ in range(num_pgms)]
    return configs


def do_kompas():
    pgm_t = 'phmm'
    pcap_file = '/home/sim/data/openNetVM/pcap/kompas_firefox_gatherer-01-rwxf6_50991468.pcapng'
    configs = get_configs('/home/nfv/openNetVM/q_data_aggs', hmm_config_path)
    configs = [configs[3], configs[4]]
    # get number of required packets, is maximum over all trace lengths
    req_packets = get_max_req_packets(configs)
    num_pgms = len(configs)

    # Spawn prototype by spawning manager first
    spawn_manager()

    # One Manager, One RX, One TX
    first_core = 3
    # spawn tls_filter, tls_record_det and classifier
    spawn_tls_filter(req_packets, first_core)
    spawn_tls_record_detector(req_packets, num_pgms, first_core + 1)
    spawn_classifier(num_pgms, first_core + 2)
    # spawn data_aggs and pgms in pairs, number determined by number of configs provided
    core_offset = first_core + 3
    sid = CLASSIFIER + 1
    for idx, (pgm_config_path, data_agg_config_path) in enumerate(configs):
        core_pgm = core_offset + 2 * idx + 1
        core_agg = core_offset + 2 * idx
        if core_pgm > 22 or core_agg > 22:
            break
        if data_agg_config_path == None:
            spawn_data_agg(sid, core_agg, sid + 1)
        else:
            spawn_q_data_agg(sid, core_agg, sid + 1, data_agg_config_path) # id, real core, dest
        sid += 1
        if pgm_t == 'phmm':
            spawn_hmm(sid, core_pgm, pgm_config_path)
        else:
            spawn_mc(sid, core_pgm, pgm_config_path)
        sid += 1
    # spawn packet gen
    time.sleep(10)
    spawn_pkt_gen(pcap_file)
    # wait prototype uptime, before stopping and removing system
    time.sleep(60)
    stop_system()
    trace_name = 'kompas'
    core_mode = 'cp'
    if pgm_t == 'phmm':
        result_folder = f"{pgm_t}-{trace_name}-{core_mode}-num-pgms-{num_pgms}-trace-len-{req_packets}"
    else:
        result_folder = f"{pgm_t}-{trace_name}-{core_mode}-num-pgms-{num_pgms}-trace-len-{req_packets}"
    save_results(result_folder)


def main(num_pgms: int, phmm_length: int, trace_name: str, core_mode: str, pgm_t: str,
         which_mc: str, run: str):
    """
    Starts and stops the prototype, gets required constants and config_files

    Args:
        /
    Returns:
        /
    """
    # get configs and set number of hmms
    num_tx_cores = 2
    configs: List[Tuple[str, Union[str, None]]] = {
        'mc': lambda: get_mc_configs(which_mc, num_pgms),
        'phmm': lambda: get_hmm_configs(phmm_length, num_pgms)
    }[pgm_t]()
    print(configs)

    # get number of required packets, is maximum over all trace lengths
    req_packets = get_max_req_packets(configs)

    # Spawn prototype by spawning manager first
    spawn_manager(num_tx_cores)

    # One Manager, One RX, One TX
    first_core = 2 + num_tx_cores
    # spawn tls_filter, tls_record_det and classifier
    spawn_tls_filter(req_packets, first_core)
    spawn_tls_record_detector(req_packets, num_pgms, first_core + 1)
    spawn_classifier(num_pgms, first_core + 2)

    # spawn data_aggs and pgms in pairs, number determined by number of configs provided
    core_offset = first_core + 3
    sid = CLASSIFIER + 1
    for idx, (pgm_config_path, data_agg_config_path) in enumerate(configs):
        if core_mode == "cp":
            core_pgm = core_offset + 2 * idx + 1
            core_agg = core_offset + 2 * idx
            if core_pgm > 22 or core_agg > 22:
                break
        else:
            # Assign pgm and data aggregation to different cores. The data agg is
            # much cheaper than the PGM and will interfere with the execution.
            core_pgm = core_offset
            core_agg = core_offset + 1
            # core_pgm = (2 * idx) % 16 + core_offset
            # core_agg = (2 * idx) % 16 + core_offset
        if data_agg_config_path == None:
            spawn_data_agg(sid, core_agg, sid + 1)
        else:
            spawn_q_data_agg(sid, core_agg, sid + 1, data_agg_config_path) # id, real core, dest
        sid += 1
        if pgm_t == 'phmm':
            spawn_hmm(sid, core_pgm, pgm_config_path)
        else:
            spawn_mc(sid, core_pgm, pgm_config_path)
        sid += 1
        time.sleep(0.5)

    # spawn packet gen
    time.sleep(10)
    spawn_moongen(trace_name)
    time.sleep(1)

    # wait prototype uptime, before stopping and removing system
    # time.sleep(60)

    stop_system()
    if pgm_t == 'phmm':
        result_folder = f"{pgm_t}-{trace_name}-{core_mode}-num-pgms-{num_pgms}-trace-len-{req_packets}-hmm-len-{phmm_length}-{run}"
    else:
        result_folder = f"{pgm_t}-{trace_name}-{core_mode}-num-pgms-{num_pgms}-trace-len-{req_packets}-{which_mc}-{run}"
    save_results(result_folder)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workload',
        help='Workload that should be evaluated. Must be in {high-pps, low-pps, high-fps, low-fps, average}.'
    )
    parser.add_argument(
        '--pgm',
        help="PGM type to launch, must be in {mc, phmm}"
    )
    parser.add_argument(
        '--onvm-mode',
        help='Whehter to use core sharing ore core pinning. Must be in {cs, cp}.'
    )
    parser.add_argument(
        '--num-pgms',
        help='Specifies the number of PGMs the system will run in parallel.',
        type=int
    )
    parser.add_argument(
        '--phmm-length',
        help="Length of the PHMM. Must be set in conjuction with --pgm=phmm.",
        default=None
    )
    parser.add_argument(
        '--which-mc',
        help="Use the largest trained MC with >10 000 parameters, use in conjunction with --pgm-mc. Must be in {big-mc}",
        default=None
    )
    parser.add_argument(
        '--run',
        help="Use the largest trained MC with >10 000 parameters, use in conjunction with --pgm-mc. Must be in {big-mc}",
        default=''
    )
    parsed_args, _ = parser.parse_known_args()
    num_pgms_ = int(parsed_args.num_pgms)
    pgm_t_ = parsed_args.pgm
    onvm_mode_ = parsed_args.onvm_mode
    phmm_length_ = None if parsed_args.phmm_length is None else int(parsed_args.phmm_length)
    which_mc_ = parsed_args.which_mc
    workload_ = parsed_args.workload
    assert pgm_t_ in ['phmm', 'mc']
    assert onvm_mode_ in ['cs', 'cp']
    assert workload_ in ['high-pps', 'low-pps', 'high-fps', 'low-fps', 'average']
    if pgm_t_ == 'phmm':
        assert phmm_length_ in [5, 10, 15, 20, 25, 30]
    if pgm_t_ == 'mc':
        assert which_mc_ in ['big-mc', 'small-mc']
    main(
        num_pgms=num_pgms_,
        phmm_length=phmm_length_,
        trace_name=workload_,
        core_mode=onvm_mode_,
        pgm_t=pgm_t_,
        which_mc=which_mc_,
        run=parsed_args.run
    )
    # test_mc()
    # num_hmms = 1 if len(sys.argv) == 1 else int(sys.argv[1])
    # length = 30 if len(sys.argv) < 3 else int(sys.argv[2])
    # trace_name = 'high-pps' if len(sys.argv) < 4 else sys.argv[3]
    # core_mode = 'cp' if len(sys.argv) < 5 else sys.argv[4]

