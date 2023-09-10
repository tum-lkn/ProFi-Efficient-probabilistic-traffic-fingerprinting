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
    1 input port: incoming packets port
    2 output ports: 1 port to tls_record_detector, 1 to flow_rule_table

Functionality:
    Checks if packet is ipv4 and tcp, if not forward to flow_rule_table
    Checks if packet is part of existing tls_flow in internal flow_table, tls_flow is identified with the src ip, port and dst ip, port (protocol as 5th identifier is always tcp)
    If packet is part of tls_flow in interal flow_table, forward packet to flow_rule_table and tls_record_detector
    If flow is not in flow_table, check if packet is 'Client Hello' (first message in tls protocol)
    If packet is 'Client Hello', add tls_flow to internal flow_table, forward packet to flow_rule_table and tls_record_detector
    If packet is not 'Client Hello', forward packet to flow_rule_table
    The maximum number of simultaneously tracket tls_flows is 1024
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

#include "onvm_flow_table.h"
#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"

#define NF_TAG "tls_filter"
#define PKTMBUF_POOL_NAME "MProc_pktmbuf_pool"
#define EXPIRE_TIME 5
#define CHECK_TIME 5000

FILE *debug_file;

///////////////////////////////////////////////////////////////////////////////////////////////

uint32_t FRT_out;
uint32_t TLS_Record_det;
uint8_t max_packets;

const uint32_t MAXSIZE = 1024 * 1024 * 10;
struct state_info
{
    struct onvm_ft *flow_table;
    uint16_t num_stored;
    uint64_t elapsed_cycles;
    uint64_t last_cycles;
};

struct tls_filter_flow_entry
{
    uint64_t last_pkt_cycles;
    uint8_t num_packets;
};

struct time_to_label_info {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint64_t timestamp;
    uint64_t timestamp_ns;
};

struct state_info *state_info;
struct rte_mempool *pktmbuf_pool;
uint32_t num_ttlbl_info_records = 1000000;
struct time_to_label_info* ttlbl_infos;
uint32_t ttlbl_idx = 0;

struct timespec entry_ts;

uint8_t core = 99;

static int
parse_app_args(int argc, char *argv[]) {
    /*
    Parses the arguments in the start command of the NF
    Args:
        -d : destination ID of tls_record_detector
        -f : destination ID of flow_rule_table
    */

    int c;
    while ((c = getopt(argc, argv, "d:f:m:y:")) != -1) {
        switch (c) {
            case 'd':
                TLS_Record_det = strtoul(optarg, NULL, 10);
                break;
            case 'f':
                FRT_out = strtoul(optarg, NULL, 10);
                break;
            case 'm':
                max_packets = strtoul(optarg, NULL, 10);
                break;
            case 'y':
                core = strtoul(optarg, NULL, 10);
                break;
        }
    }

    return optind;
}

static int
clear_entries(struct state_info *state_info) {
    /*
    Removes all expired entries of the flow table

    Args:
        state_info (pointer): pointer to the state info of the NF
    Returns:
        0/-1 (bool): indicating bool if actions were successful
    */

    if (unlikely(state_info == NULL)) 
        return -1;

    // define current flow entry, key, next and ret type
    struct tls_filter_flow_entry *data = NULL;
    struct onvm_ft_ipv4_5tuple *key = NULL;
    uint32_t next = 0;
    int ret = 0;

    // iterate over all entries in flow table and update the status
    while(onvm_ft_iterate(state_info->flow_table, (const void **)&key, (void **)&data, &next) > -1) 
    {
        if((state_info->elapsed_cycles - data->last_pkt_cycles) / rte_get_timer_hz() >= EXPIRE_TIME)
        {
            ret = onvm_ft_remove_key(state_info->flow_table, key);
            state_info->num_stored--;
            if(ret < 0) 
                state_info->num_stored++;
        }
    }

    return 0;
}

static int
table_add_entry(struct onvm_ft_ipv4_5tuple *key, struct state_info *state_info) {
    /*
    Adds table entry to the internal flow_table
    Args:
        key (pointer): pointer to the ipv4_5tuple
        state_info (pointer): pointer to the state_info holding info about the flow_table and number of stored flows
    Returns:
        0/-1 (bool): indicating bool if actions were successful
    */

    if(unlikely(key == NULL))
         return -1;

    // Define a new flow_table entry and check if adding the key is successful
    struct tls_filter_flow_entry *data = NULL;
    if (onvm_ft_add_key(state_info->flow_table, key, (char **)&data) < 0)
        return -1;

    // If adding is successful, declare flow as active, increase number of stored flows and set number of seen packets to 1
    data->last_pkt_cycles = state_info->elapsed_cycles;
    data->num_packets = 1;
    state_info->num_stored++;

    return 0;
}

static int
table_lookup_entry(struct rte_mbuf *pkt, struct state_info *state_info, struct timespec* entry_ts) {
    /*
    Checks if packet is in flow_table or not, if not checks if packet is new tls_flow, if new tls_flow add flow to flow_table

    Args:
        pkt (pointer): pointer to current packet
        state_info (pointer): pointer to the state_info holding info about the flow_table and number of stored flows
    Returns:
        0/-1 (bool): indicating bool if actions were successful
    */

    // Check if packet and state_info exist
    if (unlikely(pkt == NULL || state_info == NULL))
        return -1;

    // Get pointer to beginning of ip header and extract ipv4 and tcp header_len
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t* , sizeof(struct rte_ether_hdr));
    uint16_t offset = (pkt_data[0] & 0x0f) << 2;
    offset += (pkt_data[offset + 12] & 0xf0) >> 2;

    // Check if packet is tcp syn/syn-ack or ack, if yes do not add packet to flow_table
    if((pkt_data[2] << 8 | pkt_data[3]) == offset)
        return -1;

    // Define key consisting of src ip, port and dst ip, port, check if key is in able to be added to flow_table
    struct onvm_ft_ipv4_5tuple key;
    if(onvm_ft_fill_key_symmetric(&key, pkt) < 0)
        return -1;

    // use correct ports
    key.src_port = (pkt_data[20] << 8) | pkt_data[21];
    key.dst_port = (pkt_data[22] << 8) | pkt_data[23];

    if(key.dst_port > key.src_port) {
        uint16_t temp = key.dst_port;
        key.dst_port = key.src_port;
        key.src_port = temp;
    }

    // Define a new flow_table entry
    struct tls_filter_flow_entry *data = NULL;
    int tbl_index = onvm_ft_lookup_key(state_info->flow_table, &key, (char **)&data);

    // If key is new and packet is 'Client Hello' add key to flow_table
    if(tbl_index == -ENOENT && pkt_data[offset] == 0x16 && pkt_data[offset + 5] == 0x01) {
        return table_add_entry(&key, state_info);
    }
    if(tbl_index < 0) {
        return -1;
    }

    // update cycles and number of packets
    data->last_pkt_cycles = state_info->elapsed_cycles;
    data->num_packets++;

    // if maximum number of packets is seen, remove entry from flow table
    if(data->num_packets == max_packets) {
        onvm_ft_remove_key(state_info->flow_table, &key);
        if(ttlbl_idx < num_ttlbl_info_records) {
            clock_gettime(CLOCK_REALTIME, entry_ts);
            ttlbl_infos[ttlbl_idx].src_ip = key.src_addr;
            ttlbl_infos[ttlbl_idx].dst_ip = key.dst_addr;
            ttlbl_infos[ttlbl_idx].src_port = key.src_port;
            ttlbl_infos[ttlbl_idx].dst_port = key.dst_port;
            ttlbl_infos[ttlbl_idx].timestamp_ns = entry_ts->tv_nsec;
            ttlbl_infos[ttlbl_idx].timestamp = entry_ts->tv_sec;
            ttlbl_idx++;
        }
        state_info->num_stored--;
    }

    return 0;
}

static void write_ttlbl_infos(void) {
    FILE * fh = fopen("/home/nfv/SWC_Test/time_to_label_tls_filter.csv", "w+");
    // Write header
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


static int
packet_handler(struct rte_mbuf *pkt, struct onvm_pkt_meta *meta, struct onvm_nf_local_ctx *nf_local_ctx) {
    /*
    Handles the incoming packets, forwards all packets to flow_rule_table, if packet is part of currently active tls_flow or a new tls_flow forward packet to tls_record_detector

    Args:
        pkt (pointer): pointer to current packet
        meta (pointer): pointer to the actions of current packet
        nf_local_ctx (pointer): pointer to current NF

    Returns:
        0 (bool): always returns successful handling of packets
    */
    // struct timespec entry_ts;
    // clock_gettime(CLOCK_REALTIME, &entry_ts);
    // Check if packet is ipv4 and tcp, if not forward packet to flow_rule_table and exit function
    if(!onvm_pkt_is_ipv4(pkt) || !onvm_pkt_is_tcp(pkt)) {
        //meta->destination = FRT_out;
        //meta->action = ONVM_NF_ACTION_DROP;
        meta->action = ONVM_NF_ACTION_OUT;
        return 0;
    }

    // Check if packet is part of new or existing tls_flow, if not forward to flow_rule_table
    if(table_lookup_entry(pkt, state_info, &entry_ts) < 0) {
        //meta->destination = FRT_out;
        //meta->action = ONVM_NF_ACTION_DROP;
        meta->action = ONVM_NF_ACTION_OUT;
        return 0;
    }

    // Copy current packet, set destination and action of current and cloned packet accordingly, cloned packet goes to tls_record_detector
    struct rte_mbuf *pkt_new = rte_pktmbuf_alloc(pktmbuf_pool);
    if(pkt_new == NULL) {
        RTE_LOG(INFO, APP, "Failed to allocate new UDP packet\n");
    } else {
        uint8_t *pkt_data_new = rte_pktmbuf_mtod_offset(pkt_new, uint8_t*, 0);
        uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t*, 0);
        rte_pktmbuf_append(pkt_new, (pkt_data[16] << 8) | pkt_data[17]);
        rte_memcpy(pkt_data_new, pkt_data, pkt->pkt_len);
        // for(uint16_t i = 0; i < pkt->pkt_len; i++)
        //     pkt_data_new[i] = pkt_data[i];

        struct onvm_pkt_meta *meta_new = onvm_get_pkt_meta(pkt_new);
        if(meta_new == NULL) {
            RTE_LOG(INFO, APP, "Failed to get new packet's metadata\n");
        } else {
            //meta->destination = FRT_out;
            //meta->action = ONVM_NF_ACTION_DROP;
            meta_new->destination = TLS_Record_det;
            meta_new->action = ONVM_NF_ACTION_TONF;
            // Return cloned packet to NF (important if not called, cloned packet does not get forwarded)
            onvm_nflib_return_pkt(nf_local_ctx->nf, pkt_new);
        }
    }
    meta->action = ONVM_NF_ACTION_OUT;
    
    return 0;
}

static int
callback_handler(__attribute__((unused)) struct onvm_nf_local_ctx *nf_local_ctx) {
    state_info->elapsed_cycles = rte_get_tsc_cycles();

    if((state_info->elapsed_cycles - state_info->last_cycles) / rte_get_timer_hz() >= CHECK_TIME)
    {
        state_info->last_cycles = state_info->elapsed_cycles;
        if(clear_entries(state_info) < 0)
            return -1;
    }
        
    return 0;
}

int
main(int argc, char *argv[]) {
    /*
    Inits the NF, starts and stops the NF, cleans up used variables
    */

    // Inits the NF
    clock_gettime(CLOCK_REALTIME, &entry_ts);
    struct onvm_nf_local_ctx *nf_local_ctx;
    struct onvm_nf_function_table *nf_function_table;
    int arg_offset;

    ttlbl_infos = (struct time_to_label_info*)malloc(num_ttlbl_info_records * sizeof(struct time_to_label_info));
    if(ttlbl_infos == NULL) {
        rte_exit(EXIT_FAILURE, "Failed to allocate ttlbl_infos table");
    }
    nf_local_ctx = onvm_nflib_init_nf_local_ctx();
    onvm_nflib_start_signal_handler(nf_local_ctx, NULL);

    nf_function_table = onvm_nflib_init_nf_function_table();
    nf_function_table->pkt_handler = &packet_handler;
    nf_function_table->user_actions = &callback_handler;

    if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, nf_local_ctx, nf_function_table)) < 0) {
            onvm_nflib_stop(nf_local_ctx);
            if (arg_offset == ONVM_SIGNAL_TERMINATION) {
                    return 0;
            } else {
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
    state_info->flow_table = onvm_ft_create(MAXSIZE, sizeof(struct tls_filter_flow_entry));
    if (state_info->flow_table == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to create flow table");
    }

    // Allocate the memory_pool used to clone packets
    pktmbuf_pool = rte_mempool_lookup(PKTMBUF_POOL_NAME);
    if (pktmbuf_pool == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Cannot find mbuf pool!\n");
    }

    // Init NF Timer
    state_info->elapsed_cycles = rte_get_tsc_cycles();

    if(core != 0)
    {
        nf_local_ctx->nf->thread_info.core = core;
    }

    // debug_file = fopen("/home/benedikt/code/debug_tls_filter.txt", "w");

    // Run and stop the NF, release allocated memory
    onvm_nflib_run(nf_local_ctx);
    write_ttlbl_infos();
    onvm_nflib_stop(nf_local_ctx);
    onvm_ft_free(state_info->flow_table);
    rte_free(state_info);

    // fclose(debug_file);

    return 0;
}
