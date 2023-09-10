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
    1 input port
    1 output port

Functionality:
    Converts 64bit hash into index i.e. 0, 1, ...
    if new flow calculate first iteration of forward pass and drop packet
    if not new flow calculate iteration of forward pass
    if number of packets is reached calculate last iteration of forward pass and take log of it
        compare to threshold and make decision, write decision to packet and forward to classifier
    if number of packets is not reached calculate iteration of forward pass and drop packet
*/

///////////////////////////////////////////////////////////////////////////////////////////////

#include <rte_common.h>
#include <rte_hash.h>
#include <rte_ip.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>

#include "onvm_flow_dir.h"
#include "onvm_flow_table.h"
#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"

#include <math.h>

#define NF_TAG "hmm"

///////////////////////////////////////////////////////////////////////////////////////////////

uint8_t trace_length;
uint8_t hmm_duration;
uint8_t num_obs;
double threshold;
double* trans_matrix;
double* obs_matrix;
double* alphas_init;
uint64_t* sym_lookup;

///////////////////////////////////////////////////////////////////////////////////////////////

char *cfg_filename;
uint32_t classifier;
const uint32_t MAXSIZE = 1024 * 1024;
struct state_info
{
    struct onvm_ft *flow_table;
    uint16_t num_stored;
    uint64_t elapsed_cycles;
    uint64_t last_cycles;
};

struct hmm_flow_entry 
{
    double* alphas_prev;
    double* alphas_curr;
    uint8_t counter;
};

struct state_info *state_info;

uint8_t core = 99;

FILE *debug_file;

///////////////////////////////////////////////////////////////////////////////////////////////

static int 
init_alphas(void) {
    /*
    Initializes the alpha values for all flows, transition matrix and hmm_duration of config is needed
    Args:
        /
    Returns:
        0/-1 (bool): returns 0 if init was successful, if not -1
    */

    // Allocates memory for alphas and set first entry to 1 and second to prob of start state to 1st delete state
    alphas_init = rte_calloc("alphas_init", hmm_duration + 1, sizeof(double), 0);
    alphas_init[0] = 1.0;
    alphas_init[1] = trans_matrix[0];

    // Calculate the remaining alphas
    for(uint8_t i = 1; i < hmm_duration; i++)
    {
        alphas_init[i + 1] = alphas_init[i] * trans_matrix[3 * i];
    }

    return 0;
}

static double 
get_obs(int obs, uint8_t type_tail, uint8_t x_tail) {
    /*
    Returns the corresponding observation matrix entry for a given observation and state indices
        if obs is -1 i.e. not present returns small number
    Args:
        obs (int): observation or symbol
        type_tail (int): type of state
        x_tail (int): state index
    Returns:
        probability of observation (double):
    */

    if(obs == -1)
        return 1e-12;

    // Calculate the offset for the observation matrix, adjust offset if type_tail is 2
    uint16_t offset = obs * (2 * hmm_duration + 1) + x_tail - 1;
    if(type_tail == 2)
        offset += hmm_duration + 1;
    return obs_matrix[offset];
}

static void 
_calc_alphas_t_first(int obs, double* alphas_init, double* alphas_prev) {
    /*
    Calculates the first iteration of the forward pass
    Args:
        obs (int): observation or symbol
        alphas_init (double*): pointer to array of init alphas
        alphas_curr (double*): pointer to array of current alphas
    Returns:
        /
    */

    double a_t;

    // Calculate the first match, insert and delete state
    alphas_prev[2 * hmm_duration] = trans_matrix[2] * get_obs(obs, 2, 0);
    alphas_prev[hmm_duration] = trans_matrix[1] * get_obs(obs, 1, 1);
    alphas_prev[0] = alphas_prev[2 * hmm_duration] * trans_matrix[6 * hmm_duration + 1];
    alphas_prev[3 * hmm_duration] = alphas_init[hmm_duration] * trans_matrix[3 * hmm_duration] * get_obs(obs, 2, hmm_duration);

    // Calculate the remaining states
    for(uint8_t i = 1; i < hmm_duration; i++)
    {
        alphas_prev[hmm_duration + i] = alphas_init[i] * trans_matrix[3 * i + 1] * get_obs(obs, 1, i + 1);
        alphas_prev[2 * hmm_duration + i] = alphas_init[i] * trans_matrix[3 * i + 2] * get_obs(obs, 2, i);
        
        a_t = alphas_prev[i - 1] * trans_matrix[3 * i];
        a_t += alphas_prev[hmm_duration + i - 1] * trans_matrix[3 * (hmm_duration + i) - 1];
        alphas_prev[i] = a_t + alphas_prev[2 * hmm_duration + i] * trans_matrix[6 * hmm_duration + 3 * i + 1];
    }

}

static void 
_calc_alphas_t(int obs, double* alphas_prev, double* alphas_curr) {
    /*
    Calculates an iteration of the forward pass
    Args:
        obs (int): observation or symbol
        alphas_prev (double*): pointer to array of previous alphas
        alphas_curr (double*): pointer to array of current alphas
    Returns:
        /
    */

    double a_t;

    // Calculate the first match, insert and delete state
    alphas_curr[hmm_duration] = alphas_prev[2 * hmm_duration] * trans_matrix[6 * hmm_duration + 2] * get_obs(obs, 1, 1);
    alphas_curr[2 * hmm_duration] = alphas_prev[2 * hmm_duration] * trans_matrix[6 * hmm_duration + 3] * get_obs(obs, 2, 0);
    alphas_curr[0] = alphas_curr[2 * hmm_duration] * trans_matrix[6 * hmm_duration + 1];

    // Calculate the match and insert states
    for(uint8_t i = 1; i < hmm_duration; i++)
    {
        a_t = alphas_prev[i - 1] * trans_matrix[3 * i + 1];
        a_t += alphas_prev[2 * hmm_duration + i] * trans_matrix[6 * hmm_duration + 3 * i + 2];
        a_t += alphas_prev[1 * hmm_duration + i - 1] * trans_matrix[3 * (hmm_duration + i)];
        alphas_curr[hmm_duration + i] = a_t * get_obs(obs, 1, i + 1);

        a_t = alphas_prev[i - 1] * trans_matrix[3 * i + 2];
        a_t += alphas_prev[hmm_duration + i - 1] * trans_matrix[3 * hmm_duration + 3 * i + 1];
        a_t += alphas_prev[2 * hmm_duration + i] * trans_matrix[6 * hmm_duration + 3 * (i + 1)];
        alphas_curr[2 * hmm_duration + i] = a_t * get_obs(obs, 2, i);
    }

    // Calculate the last insert state
    a_t = alphas_prev[hmm_duration - 1] * trans_matrix[3 * hmm_duration];
    a_t += alphas_prev[2 * hmm_duration - 1] * trans_matrix[6 * hmm_duration - 1];
    a_t += alphas_prev[3 * hmm_duration] * trans_matrix[9 * hmm_duration + 1];
    alphas_curr[3 * hmm_duration] = a_t * get_obs(obs, 2, hmm_duration);

    // Calculate the delete states
    for(uint8_t i = 1; i < hmm_duration; i++)
    {
        a_t = alphas_curr[i - 1] * trans_matrix[3 * i];
        a_t += alphas_curr[hmm_duration + i - 1] * trans_matrix[3 * (hmm_duration + i) - 1];
        alphas_curr[i] = a_t + alphas_curr[2 * hmm_duration + i] * trans_matrix[6 * hmm_duration + 3 * i + 1];
    }

    // Write current alphas into previous alphas for next iteration
    for(uint8_t i = 0; i < 3 * hmm_duration + 1; i++)
        alphas_prev[i] = alphas_curr[i];
}

static double 
_calc_alphas_end(double* alphas_prev) {
    /*
    Calculates the last iteration of the forward pass
    Args:
        alphas_prev (double*): pointer to array of previous alphas
    Returns:
        log (double): log_prob of the forward pass
    */

    double a_t;

    // Sum all transitions to end state and take log of it
    a_t = alphas_prev[hmm_duration - 1] * trans_matrix[3 * hmm_duration + 1];
    a_t += alphas_prev[3 * hmm_duration] * trans_matrix[9 * hmm_duration + 2];
    a_t += alphas_prev[2 * hmm_duration - 1] * trans_matrix[6 * hmm_duration];
    return log(a_t);
}

///////////////////////////////////////////////////////////////////////////////////////////////

static int
parse_app_args(int argc, char *argv[]) {
    /*
    Parses the arguments in the start command of the NF
    Args:
        -d : destination ID of classifier
        -c : filename of config file
    */

    int c;
    while ((c = getopt(argc, argv, "d:c:y:")) != -1) {
        switch (c) {
            case 'd':
                classifier = strtoul(optarg, NULL, 10);
                break;
            case 'c':
                cfg_filename = strdup(optarg);
                break;
            case 'y':
                core = strtoul(optarg, NULL, 10);
                break;
        }
    }

    return optind;
}

static int
load_config(void) {
    /*
    Loads the config for a hmm
    Args:
        /
    Returns:
        0/-1 (bool): returns 0 if loading was successful, if not -1
    */

    cJSON* config = onvm_config_parse_file(cfg_filename);

    // Load trace_length, hmm_duration, number of observations and threshold
    trace_length = (uint8_t)cJSON_GetObjectItem(config, "trace_length")->valueint;
    hmm_duration = (uint8_t)cJSON_GetObjectItem(config, "hmm_duration")->valueint;
    num_obs = (uint8_t)cJSON_GetObjectItem(config, "num_obs")->valueint;
    threshold = cJSON_GetObjectItem(config, "threshold")->valuedouble;

    // Allocate memory for trans_matrix, obs_matrix and symbol lookup
    trans_matrix = rte_calloc("trans_matrix", 9 * hmm_duration + 3, sizeof(double), 0);
    obs_matrix = rte_calloc("obs_matrix", num_obs * (2 * hmm_duration + 1), sizeof(double), 0);
    sym_lookup = rte_calloc("sym_lookup", num_obs, sizeof(double), 0);

    // Write trans_matrix into c array
    cJSON* trans = cJSON_GetObjectItem(config, "trans_matrix");
    for(uint16_t i = 0; i < 9 * hmm_duration + 3; i++)
        trans_matrix[i] = cJSON_GetArrayItem(trans, i)->valuedouble;

    // Write obs_matrix into c array
    cJSON* obs = cJSON_GetObjectItem(config, "obs_matrix");
    for(uint16_t i = 0; i < num_obs * (2 * hmm_duration + 1); i++)
        obs_matrix[i] = cJSON_GetArrayItem(obs, i)->valuedouble;

    // Write sym_lookup into c array
    cJSON* sym = cJSON_GetObjectItem(config, "symbol");
    for(uint16_t i = 0; i < num_obs; i++)
        sym_lookup[i] = (uint64_t)cJSON_GetArrayItem(sym, i)->valuedouble;

    return 0;
}

static int
create_symbol(uint8_t* pkt_data, uint16_t header_len) {
    /*
    Finds the 64bit hash of packet in sym_lookup
    Args:
        pkt_data (pointer): pointer to beginning of ip header
        header_len (int): length of ip and udp header
    Returns:
        index (int): index of found symbol else -1 if not found
    */

    // Define variables for binary search
    int first = 0;
    int last = num_obs - 1;
    int mid = 0;

    // Extracts the 64bit hash out of packet
    uint64_t obs = ((uint64_t)pkt_data[header_len] << 56);
    obs |= ((uint64_t)pkt_data[header_len + 1] << 48);
    obs |= ((uint64_t)pkt_data[header_len + 2] << 40);
    obs |= ((uint64_t)pkt_data[header_len + 3] << 32);
    obs |= ((uint64_t)pkt_data[header_len + 4] << 24);
    obs |= ((uint64_t)pkt_data[header_len + 5] << 16);
    obs |= ((uint64_t)pkt_data[header_len + 6] << 8);
    obs |= ((uint64_t)pkt_data[header_len + 7]);

    // Binary search
    while(first <= last)
    {
        mid = (int)((first + last) / 2);
        if(sym_lookup[mid] < obs)
        {
            first = mid + 1;
        }
        else if(sym_lookup[mid] > obs)
        {
            last = mid - 1;
        }
        else
        {
            return mid;
        }   
    }
    return -1;        
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
    {
        return -1;
    }

    // Define a new flow_table entry and check if adding the key is successful
    struct hmm_flow_entry *data = NULL;
    if(onvm_ft_add_key(state_info->flow_table, key, (char **)&data) < 0)
    {
        return -1;
    }

    // Allocate memory for the alphas and update flow entry
    data->alphas_prev = rte_calloc("alphas_prev", 3 * hmm_duration + 1, sizeof(double), 0);
    data->alphas_curr = rte_calloc("alphas_curr", 3 * hmm_duration + 1, sizeof(double), 0);
    data->counter = 1;

    // Get pointer to begin of ip header and extract ipv4 header len and udp header len
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t*, sizeof(struct rte_ether_hdr));
    uint16_t header_len = ((pkt_data[0] & 0x0f) << 2) + 8;

    // Get symbol for forward pass, execute first iteration of forward pass
    int symbol = create_symbol(pkt_data, header_len);
    _calc_alphas_t_first(symbol, alphas_init, data->alphas_prev);

    // Update the number of flows in the state_info
    state_info->num_stored++;
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
    if (unlikely(pkt == NULL || state_info == NULL))
        return -1;

    // Define key consisting of src ip, port and dst ip, port, check if key is in able to be added to flow_table
    struct onvm_ft_ipv4_5tuple key;
    if(onvm_ft_fill_key_symmetric(&key, pkt) < 0)
        return -1;

    // Get pointer to begin of ip header and extract ipv4 header len and udp header len
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t*, sizeof(struct rte_ether_hdr));
    uint16_t header_len = ((pkt_data[0] & 0x0f) << 2);

    // use correct ports
    key.src_port = (pkt_data[20] << 8) | pkt_data[21];
    key.dst_port = (pkt_data[22] << 8) | pkt_data[23];

    if(key.dst_port > key.src_port)
    {
        uint16_t temp = key.dst_port;
        key.dst_port = key.src_port;
        key.src_port = temp;
    }

    // Define a new flow_table entry
    struct hmm_flow_entry *data = NULL;
    int tbl_index = onvm_ft_lookup_key(state_info->flow_table, &key, (char **)&data);

    if((((pkt_data[2] << 8) | (pkt_data[3])) == 42))
    {
        if(data != NULL)
        {
            double log_prob = _calc_alphas_end(data->alphas_prev);
            pkt_data[header_len + 8] = 0;
            if(log_prob >= threshold)
                pkt_data[header_len + 8] = 1;
            onvm_ft_remove_key(state_info->flow_table, &key);
            state_info->num_stored--;
            return 0;
        }

        return -1;
    }

    // If key is new add key to flow_table
    if(tbl_index == -ENOENT && pkt_data[header_len + 7] == 1)
    {
        // fprintf(debug_file, "Src IP: %hu.%hu.%hu.%hu, Dst IP: %hu.%hu.%hu.%hu, Src Port: %hu, Dst Port: %hu, Protokoll: %hu\n",
        //     (key.src_addr) & 0x000000ff, (key.src_addr >> 8) & 0x000000ff, (key.src_addr >> 16) & 0x000000ff, (key.src_addr >>24) & 0x000000ff, 
        //     (key.dst_addr) & 0x000000ff, (key.dst_addr >> 8) & 0x000000ff, (key.dst_addr >> 16) & 0x000000ff, (key.dst_addr >>24) & 0x000000ff,
        //     key.src_port, key.dst_port, key.proto);
        table_add_entry(&key, state_info, pkt);
        return -1;
    }
    if(tbl_index < 0)
    {
        return -1;
    }
    else
    {
        // Get symbol for forward pass, execute an iteration of forward pass
        header_len += 8;
        int symbol = create_symbol(pkt_data, header_len);
        _calc_alphas_t(symbol, data->alphas_prev, data->alphas_curr);
        data->counter++;

        // If number of iterations is reached, make last iteration of forward pass, write decision to packet and remove key out of flow_table
        if(data->counter == trace_length)
        {
            double log_prob = _calc_alphas_end(data->alphas_prev);
            pkt_data[header_len] = 0;
            if(log_prob >= threshold)
                pkt_data[header_len] = 1;
            onvm_ft_remove_key(state_info->flow_table, &key);
            state_info->num_stored--;
            return 0;
        }

        return -1;
    }
}

static int
packet_handler(struct rte_mbuf *pkt, struct onvm_pkt_meta *meta, __attribute__((unused)) struct onvm_nf_local_ctx *nf_local_ctx) {
    /*
    Handles the incoming packets, extracts the features of the tls packet and forwards them to the data_aggs, if the packet is a tcp retransmission nothing is done

    Args:
        pkt (pointer): pointer to current packet
        meta (pointer): pointer to the actions of current packet
        nf_local_ctx (pointer): pointer to current NF

    Returns:
        0 (bool): always returns successful handling of packets
    */

    // Check if packet is part of new or existing tls_flow, if not drop packet
    if(table_lookup_entry(pkt, state_info) < 0) {
        // rte_pktmbuf_free(pkt);
        meta->action = ONVM_NF_ACTION_DROP;
        return 0;
    }

    meta->action = ONVM_NF_ACTION_TONF;
    meta->destination = classifier;
    return 0;
}

// static int
// callback_handler(__attribute__((unused)) struct onvm_nf_local_ctx *nf_local_ctx) 
// {
//     state_info->elapsed_cycles = rte_get_tsc_cycles();

//     if((state_info->elapsed_cycles - state_info->last_cycles) / rte_get_timer_hz() >= 1)
//         state_info->last_cycles = state_info->elapsed_cycles;
        
//     return 0;
// }

int
main(int argc, char *argv[]) {
    /*
    Inits the NF, starts and stops the NF, cleans up used variables
    */

    // Inits the NF
    struct onvm_nf_local_ctx *nf_local_ctx;
    struct onvm_nf_function_table *nf_function_table;
    int arg_offset;

    nf_local_ctx = onvm_nflib_init_nf_local_ctx();
    onvm_nflib_start_signal_handler(nf_local_ctx, NULL);

    nf_function_table = onvm_nflib_init_nf_function_table();
    nf_function_table->pkt_handler = &packet_handler;
    // nf_function_table->user_actions = &callback_handler;

    if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, nf_local_ctx, nf_function_table)) < 0) 
    {
        onvm_nflib_stop(nf_local_ctx);
        if (arg_offset == ONVM_SIGNAL_TERMINATION) 
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
    state_info->flow_table = onvm_ft_create(MAXSIZE, sizeof(struct hmm_flow_entry));
    if (state_info->flow_table == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to create flow table");
    }

    // Load config and init alphas
    load_config();
    init_alphas();

    if (core != 0)
    {
        nf_local_ctx->nf->thread_info.core = core;
    }

    // char buf[48];
    // sprintf(buf, "/home/benedikt/code/debug_hmm_%hu.txt", nf_local_ctx->nf->service_id);
    // debug_file = fopen(buf, "w");

    // Run and stop the NF, release allocated memory
    onvm_nflib_run(nf_local_ctx);
    onvm_nflib_stop(nf_local_ctx);
    onvm_ft_free(state_info->flow_table);
    rte_free(state_info);
    rte_free(trans_matrix);
    rte_free(obs_matrix);
    rte_free(sym_lookup);
    rte_free(alphas_init);

    // fprintf(debug_file, "Num Flows: %hu\n", state_info->num_stored);
    // fclose(debug_file);

    return 0;
}
