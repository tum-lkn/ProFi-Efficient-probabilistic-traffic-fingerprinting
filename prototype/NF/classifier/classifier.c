/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2019 George Washington University
 *            2015-2019 University of California Riverside
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * The name of the author may not be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * forward.c - an example using onvm. Forwards packets to a DST NF.
 ********************************************************************/


/*
Structure:
    n input ports
    0 output ports

Functionality:
    Collects the decisions of all connected hmms and makes a final decision

*/

///////////////////////////////////////////////////////////////////////////////////////////////

#include <rte_common.h>
#include <rte_hash.h>
#include <rte_ip.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <time.h>
#include <inttypes.h>
#include <stdio.h>

#include "onvm_flow_dir.h"
#include "onvm_flow_table.h"
#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"

#define NF_TAG "classifier"

#include <stdio.h>

///////////////////////////////////////////////////////////////////////////////////////////////

const uint32_t MAXSIZE = 1024;
struct state_info *state_info;
uint16_t num_hmms;
struct state_info
{
    struct onvm_ft *flow_table;
    uint16_t num_stored;
};

struct classifier_flow_entry 
{
    uint16_t sum;
    uint16_t counter;
};

struct time_to_label_info {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint64_t timestamp;
    uint64_t timestamp_ns;
};

FILE *classification_file;
uint32_t num_ttlbl_info_records = 1000000;
uint32_t ttlbl_idx = 0;
struct time_to_label_info* ttlbl_infos;
uint64_t start_time;

uint8_t core = 99;

static int
parse_app_args(int argc, char *argv[]) {
    /*
    Parses the arguments in the start command of the NF
    Args:
        -c : path to the config_file
    */

    int c;
    while ((c = getopt(argc, argv, "n:y:")) != -1) {
        switch (c) {
            case 'n':
                num_hmms = strtoul(optarg, NULL, 10);
                break;
            case 'y':
                core = strtoul(optarg, NULL, 10);
                break;
        }
    }

    return optind;
}

static int
write_classification_result_to_file(struct classifier_flow_entry *data, struct onvm_ft_ipv4_5tuple *key) {
    /*
    Writes the final decision to debug file
    Args:
        classifier_flow_entry (struct): data structure containing final decision
    Returns:
        0/-1 (bool): indicating if function was successful
    */

    fprintf(classification_file, "Timestamp: %lu, Src IP: %hu.%hu.%hu.%hu, Dst IP: %hu.%hu.%hu.%hu, Src Port: %hu, Dst Port: %hu, Sum: %hu\n",
        (uint64_t)time(NULL) - start_time,
        (key->src_addr) & 0x000000ff, (key->src_addr >> 8) & 0x000000ff, (key->src_addr >> 16) & 0x000000ff, (key->src_addr >> 24) & 0x000000ff, 
        (key->dst_addr) & 0x000000ff, (key->dst_addr >> 8) & 0x000000ff, (key->dst_addr >> 16) & 0x000000ff, (key->dst_addr >> 24) & 0x000000ff,
        key->src_port, key->dst_port, data->sum);

    return 0;
}

static int
table_add_entry(struct onvm_ft_ipv4_5tuple *key, struct state_info *state_info, struct rte_mbuf *pkt) {
    /*
    Adds table entry to the internal flow_table
    Args:
        key (pointer): pointer to the ipv4_5tuple
        state_info (pointer): pointer to the state_info holding info about the flow_table and number of stored flows
        pkt (pointer): pointer to the packet
    Returns:
        0/-1 (bool): indicating bool if actions were successful
    */

    // Check if key exists
    if (unlikely(key == NULL || state_info == NULL))
        return -1;

    // Define a new flow_table entry and check if adding the key is successful
    struct classifier_flow_entry *data = NULL;
    if(onvm_ft_add_key(state_info->flow_table, key, (char **)&data) < 0)
        return -1;

    // Get pointer to begin of ip header and extract ipv4 header len and udp header len
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t*, sizeof(struct rte_ether_hdr));
    uint16_t header_len = ((pkt_data[0] & 0x0f) << 2) + 8;

    data->sum = pkt_data[header_len];
    data->counter = 1;
    // data->sum = 0;
    //data->counter = 0;

    // Update the number of flows in the state_info
    state_info->num_stored++;

    // Writes classification result to file and deletes flow entry
    if(data->counter == num_hmms) {
         if(ttlbl_idx < 1000000) {
             struct timespec entry_ts;
             clock_gettime(CLOCK_REALTIME, &entry_ts);
             ttlbl_infos[ttlbl_idx].src_ip = key->src_addr;
             ttlbl_infos[ttlbl_idx].dst_ip = key->dst_addr;
             ttlbl_infos[ttlbl_idx].src_port = key->src_port;
             ttlbl_infos[ttlbl_idx].dst_port = key->dst_port;
             ttlbl_infos[ttlbl_idx].timestamp_ns = entry_ts.tv_nsec;
             ttlbl_infos[ttlbl_idx].timestamp = entry_ts.tv_sec;
             ttlbl_idx++;
         }
         write_classification_result_to_file(data, key);
         onvm_ft_remove_key(state_info->flow_table, key);
         state_info->num_stored--;
         return 0;
    }

    return 0;
}

static int
table_lookup_entry(struct rte_mbuf *pkt, struct state_info *state_info) {
    /*
    Checks if packet is in flow_table or not
    If packet is in flow_table execute forward pass
    If not checks if packet is new tls_flow, if new tls_flow add flow to flow_table and execute forward pass

    Args:
        pkt (pointer): pointer to current packet
        state_info (pointer): pointer to the state_info holding info about the flow_table and number of stored flows
    Returns:
        0/-1 (bool): indicating bool if actions were successful
    */

    // Check if packet and state_info exist
    if (unlikely(pkt == NULL || state_info == NULL)) {
        return -1;
    }

    // Define key consisting of src ip, port and dst ip, port, check if key is in able to be added to flow_table
    struct onvm_ft_ipv4_5tuple key;
    if (onvm_ft_fill_key_symmetric(&key, pkt) < 0) {
        return -1;
    }

    // Get pointer to begin of ip header and extract ipv4 header len and udp header len
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t*, sizeof(struct rte_ether_hdr));
    uint16_t header_len = ((pkt_data[0] & 0x0f) << 2) + 8;

    // use correct ports
    key.src_port = (pkt_data[20] << 8) | pkt_data[21];
    key.dst_port = (pkt_data[22] << 8) | pkt_data[23];

    if(key.dst_port > key.src_port) {
        uint16_t temp = key.dst_port;
        key.dst_port = key.src_port;
        key.src_port = temp;
    }

    // Define a new flow_table entry
    struct classifier_flow_entry *data = NULL;
    int tbl_index = onvm_ft_lookup_key(state_info->flow_table, &key, (char **)&data);

    // If key is new add key to flow_table
    if (tbl_index == -ENOENT) {
        table_add_entry(&key, state_info, pkt);
        return -1;
    }
    if(tbl_index < 0) {
        return -1;
    }
    else {
        // Add decision to final decision
        data->sum += pkt_data[header_len];
        data->counter++;

        // Writes classification result to file and deletes flow entry
        if(data->counter >= num_hmms) {
            // if(ttlbl_idx < num_ttlbl_info_records) {
            if(ttlbl_idx < 1000000) {
                struct timespec entry_ts;
                clock_gettime(CLOCK_REALTIME, &entry_ts);
                ttlbl_infos[ttlbl_idx].src_ip = key.src_addr;
                ttlbl_infos[ttlbl_idx].dst_ip = key.dst_addr;
                ttlbl_infos[ttlbl_idx].src_port = key.src_port;
                ttlbl_infos[ttlbl_idx].dst_port = key.dst_port;
                ttlbl_infos[ttlbl_idx].timestamp = entry_ts.tv_sec;
                ttlbl_infos[ttlbl_idx].timestamp_ns = entry_ts.tv_nsec;
                ttlbl_idx++;
            }
            write_classification_result_to_file(data, &key);
            onvm_ft_remove_key(state_info->flow_table, &key);
            state_info->num_stored--;
        }
        return -1;
    }
}

static int
packet_handler(struct rte_mbuf *pkt, struct onvm_pkt_meta *meta, __attribute__((unused)) struct onvm_nf_local_ctx *nf_local_ctx) {
    table_lookup_entry(pkt, state_info);
    meta->action = ONVM_NF_ACTION_DROP;
    // rte_pktmbuf_free(pkt);
    return 0;
}


static void write_ttlbl_infos(void) {
    FILE * fh = fopen("/home/nfv/SWC_Test/time_to_label_tls_classifier.csv", "w+");
    // Write header
    fprintf(fh, "Stored %" PRIu32 " records\n", ttlbl_idx);
    fprintf(fh, "src_ip;src_port;dst_ip;dst_port;tv_sec;tv_nsec\n");
    for(uint32_t i = 0; i < ttlbl_idx; i++) {
        fprintf(
          fh
          ,"%" PRIu32 ";%" PRIu16 ";%" PRIu32 ";%" PRIu16 ";%" PRIu64 ";%" PRIu64 "\n"
          ,ttlbl_infos[i].src_ip
          ,ttlbl_infos[i].src_port
          ,ttlbl_infos[i].dst_ip
          ,ttlbl_infos[i].dst_port
          ,ttlbl_infos[i].timestamp
          ,ttlbl_infos[i].timestamp_ns
        );
    }
    fclose(fh);
    free(ttlbl_infos);
}


int
main(int argc, char *argv[]) {
    /*
    Inits the NF, starts and stops the NF, cleans up used variables
    */

    // Inits the NF
    // ttlbl_infos = (struct time_to_label_info*)malloc(num_ttlbl_info_records * sizeof(struct time_to_label_info));
    ttlbl_infos = (struct time_to_label_info*)malloc(1000000 * sizeof(struct time_to_label_info));
    if(ttlbl_infos == NULL) {
        rte_exit(EXIT_FAILURE, "Failed to allocate ttlbl_infos table");
    }

    struct onvm_nf_local_ctx *nf_local_ctx;
    struct onvm_nf_function_table *nf_function_table;
    int arg_offset;

    nf_local_ctx = onvm_nflib_init_nf_local_ctx();
    onvm_nflib_start_signal_handler(nf_local_ctx, NULL);

    nf_function_table = onvm_nflib_init_nf_function_table();
    nf_function_table->pkt_handler = &packet_handler;

    if((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, nf_local_ctx, nf_function_table)) < 0) 
    {
        onvm_nflib_stop(nf_local_ctx);
        if(arg_offset == ONVM_SIGNAL_TERMINATION) 
        {
            return 0;
        } 
        else 
        {
            rte_exit(EXIT_FAILURE, "Failed ONVM init\n");
        }
    }

    argc -= arg_offset;
    argv += arg_offset;

    if (parse_app_args(argc, argv) < 0) {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Invalid command-line arguments\n");
    }

    // Inits the state_info holding the flow_table, if unsuccessful stop NF
    state_info = rte_calloc("state", 1, sizeof(struct state_info), 0);
    if (state_info == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to initialize NF state");
    }

    // Inits the flow_table, if unsuccessful stop NF
    state_info->flow_table = onvm_ft_create(MAXSIZE, sizeof(struct classifier_flow_entry));
    if (state_info->flow_table == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to create flow table");
    }

    classification_file = fopen("/home/nfv/classification-x.txt", "w");

    fprintf(classification_file, "If Sum is 1, flow is from websites\n");

    start_time = (uint64_t)time(NULL);

    if (core != 0)
    {
        nf_local_ctx->nf->thread_info.core = core;
    }

    // Run and stop the NF, release allocated memory
    onvm_nflib_run(nf_local_ctx);
    write_ttlbl_infos();
    onvm_nflib_stop(nf_local_ctx);
    onvm_ft_free(state_info->flow_table);

    fprintf(classification_file, "-------------\n");

    rte_free(state_info);

    fclose(classification_file);

    return 0;
}
