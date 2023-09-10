import os
import sys
import subprocess
import json as js
import time
import numpy as np

num_samples = 1000

trace_folder = '/home/sim/K8STRACES'
cp_folder = '/home/sim/data/tmp_folder'

output_file_1 = '/home/sim/data/tmp_folder/merged_traffic_1.pcapng'
output_file_2 = '/home/sim/data/tmp_folder/merged_traffic_2.pcapng'

pcap_filenames = '/home/sim/code/openNetVM/filenames_merged_traffic.json'

f = open(output_file_1, 'w')
f.close()

f = open(output_file_2, 'w')
f.close()

with open(pcap_filenames, 'r', encoding='utf-8') as f:
    traffic = js.load(f)

traffic = np.random.choice(traffic, num_samples, replace=False).tolist()

def convert_pcap(index):

    cp_cmd = 'cp {} {}'.format(os.path.join(trace_folder, traffic[index]), os.path.join(cp_folder, traffic[index]))
    subprocess.check_call(cp_cmd, shell=True)

    timestamp_cmd = 'capinfos -a -S {}'.format(os.path.join(cp_folder, traffic[index]))

    try:
        time_epoch = float(subprocess.check_output(timestamp_cmd, shell=True).decode('utf-8').split('\n')[1].split(':')[1].strip())
        time_epoch -= i*10e-6
    except:
        rm_cmd = 'rm {}'.format(os.path.join(cp_folder, traffic[index]))
        subprocess.check_call(rm_cmd, shell=True)
        return 1

    edit_cmd = 'editcap -t {} {} {}'.format(-time_epoch, os.path.join(cp_folder, traffic[index]), os.path.join(cp_folder, 'tmp.pcapng'))
    subprocess.check_call(edit_cmd, shell=True)

    rm_cmd = 'rm {}'.format(os.path.join(cp_folder, traffic[index]))
    subprocess.check_call(rm_cmd, shell=True)

    return 0

for i in range(len(traffic)):

    if convert_pcap(i) == 1:
        continue

    merge_cmd = 'mergecap {} {} -w {}'.format(os.path.join(cp_folder, 'tmp.pcapng'), output_file_1, output_file_2)
    subprocess.check_call(merge_cmd, shell=True)

    rm_cmd = 'rm {}'.format(os.path.join(cp_folder, 'tmp.pcapng'))
    subprocess.check_call(rm_cmd, shell=True)

    tmp = output_file_1
    output_file_1 = output_file_2
    output_file_2 = tmp

rm_cmd = 'rm {}'.format(output_file_2)
subprocess.check_call(rm_cmd, shell=True)

mv_cmd = 'mv {} {}'.format(output_file_1, '/home/sim/data/tmp_folder/merged_traffic.pcapng')
subprocess.check_call(mv_cmd, shell=True)
