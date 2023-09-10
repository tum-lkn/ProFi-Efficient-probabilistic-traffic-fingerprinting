#!/bin/bash
echo export ONVM_HOME=$(pwd) >> ~/.bashrc
cd dpdk
echo export RTE_SDK=$(pwd) >> ~/.bashrc
echo export RTE_TARGET=x86_64-native-linuxapp-gcc  >> ~/.bashrc
echo export ONVM_NUM_HUGEPAGES=1024 >> ~/.bashrc
export ONVM_NIC_PCI="00:13.0 00:14.0 00:15.0"
source ~/.bashrc
cd ..
cd scripts
./setup_environment.sh