#!/bin/bash
cd ./classifier
make
cd ../data_agg
make
cd ../q_data_agg
make
cd ../hmm
make
cd ../tls_filter
make
cd ../tls_record_det
make
cd ../speed_tester
make
cd ../mc
make

