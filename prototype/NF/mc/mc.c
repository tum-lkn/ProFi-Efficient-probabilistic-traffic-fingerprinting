#include <rte_common.h>
#include <rte_hash.h>
#include <rte_ip.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <stdint.h>
#include <stdio.h>

#include "onvm_flow_dir.h"
#include "onvm_flow_table.h"
#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"

#include <math.h>

#define NF_TAG "mc"
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
uint64_t* sym_lookup;

const uint64_t START_SYMBOL = 1;
// const uint64_t END_SYMBOL = 18446744073709551615;

struct state_info {
    struct onvm_ft *flow_table;
    uint16_t num_stored;
    uint64_t elapsed_cycles;
    uint64_t last_cycles;
};

struct mc_flow_entry {
    uint64_t prev_symbol;
    uint8_t counter;
    double log_prob;
};

struct markov_chain {
    int num_symbols;
    int * offsets;
    uint64_t * tails;
    uint64_t * heads;
    double * log_probs;
};
typedef struct markov_chain markov_chain_t;
markov_chain_t mc;
static int binary_search(uint64_t * buffer, uint64_t val, int buffer_size) {
    int ret_val = -1;
    int first = 0;
    int last = buffer_size - 1;
    int mid = 0;
    while(first <= last) {
        mid = (int)((first + last) / 2);
        if(buffer[mid] < val) {
            first = mid + 1;
        } else if(buffer[mid] > val) {
            last = mid - 1;
        } else {
            ret_val = mid;
            first = last + 1;
        }
    }
    return ret_val;
}
static double transition_probability(markov_chain_t * mc, uint64_t tail, uint64_t head) {
    int tail_idx = binary_search(mc->tails, tail, mc->num_symbols);
    if(tail_idx == -1) {
        return -1;
    }
    int offset = mc->offsets[tail_idx];
    int num_neighbors = mc->offsets[tail_idx + 1] - offset;
    int head_idx = binary_search(mc->heads + offset, head, num_neighbors);
    if(head_idx == -1) {
        return -13.815510557964274;
    } else {
        return mc->log_probs[offset + head_idx];
    }
}


struct state_info *state_info;

uint8_t core = 99;

FILE *debug_file;


static int parse_app_args(int argc, char *argv[]) {
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


static int load_config(void) {
    cJSON * config = onvm_config_parse_file(cfg_filename);
    // trace_length = (uint32_t)cJSON_GetObjectItem(config, "trace_length")->valueint;
    uint32_t num_nodes = (uint32_t)cJSON_GetObjectItem(config, "num_nodes")->valueint;
    uint32_t num_edges = (uint32_t)cJSON_GetObjectItem(config, "num_edges")->valueint;

    mc.num_symbols = num_nodes;
    mc.tails = rte_calloc("tails", num_nodes, sizeof(uint64_t), 0);
    mc.offsets = rte_calloc("offsets", num_nodes, sizeof(int), 0);
    mc.heads = rte_calloc("heads", num_edges, sizeof(uint64_t), 0);
    mc.log_probs = rte_calloc("log_probs", num_edges, sizeof(double), 0);
    
    cJSON * lps = cJSON_GetObjectItem(config, "log_probs");
    for(uint16_t i = 0; i < num_edges; i++) {
        mc.log_probs[i] = cJSON_GetArrayItem(lps, i)->valuedouble;
    }

    lps = cJSON_GetObjectItem(config, "tails");
    for(uint32_t i = 0; i < num_nodes; i++) {
        mc.tails[i] = (uint64_t)cJSON_GetArrayItem(lps, i)->valueint;
    }

    lps = cJSON_GetObjectItem(config, "offsets");
    for(uint32_t i = 0; i < num_nodes; i++) {
        mc.offsets[i] = (int)cJSON_GetArrayItem(lps, i)->valueint;
    }

    lps = cJSON_GetObjectItem(config, "heads");
    for(uint32_t i = 0; i < num_edges; i++) {
        mc.heads[i] = (uint64_t)cJSON_GetArrayItem(lps, i)->valueint;
    }
    
    return 0;
}


static int create_symbol(uint8_t* pkt_data, uint16_t header_len) {
    /*
    Finds the 64bit hash of packet in sym_lookup
    Args:
        pkt_data (pointer): pointer to beginning of ip header
        header_len (int): length of ip and udp header
    Returns:
        index (int): index of found symbol else -1 if not found
    */

    // Extracts the 64bit hash out of packet
    uint64_t obs = ((uint64_t)pkt_data[header_len] << 56);
    obs |= ((uint64_t)pkt_data[header_len + 1] << 48);
    obs |= ((uint64_t)pkt_data[header_len + 2] << 40);
    obs |= ((uint64_t)pkt_data[header_len + 3] << 32);
    obs |= ((uint64_t)pkt_data[header_len + 4] << 24);
    obs |= ((uint64_t)pkt_data[header_len + 5] << 16);
    obs |= ((uint64_t)pkt_data[header_len + 6] << 8);
    obs |= ((uint64_t)pkt_data[header_len + 7]);
    return obs;
}


static int update_probability(uint64_t head, struct mc_flow_entry * data) {
    // Add the other C code from the tls_Fingerprint repo here. Program creates a
    // MarkovChain globally at the beginning. The Chain is accessed here with
    // the respective parameters.
    double log_prob = 0;
    log_prob = transition_probability(&mc, data->prev_symbol, head);
    data->log_prob += log_prob;
    data->prev_symbol = head;
    data->counter++;
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
    if (unlikely(key == NULL || state_info == NULL)) {
        return -1;
    }

    // Define a new flow_table entry and check if adding the key is successful
    struct mc_flow_entry *data = NULL;
    if(onvm_ft_add_key(state_info->flow_table, key, (char **)&data) < 0) {
        return -1;
    }

    // Allocate memory for the alphas and update flow entry
    data->counter = 0;
    data->log_prob = 0;
    data->prev_symbol = START_SYMBOL;

    // Get pointer to begin of ip header and extract ipv4 header len and udp header len
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t*, sizeof(struct rte_ether_hdr));
    uint16_t header_len = ((pkt_data[0] & 0x0f) << 2) + 8;

    // Get symbol for forward pass, execute first iteration of forward pass
    int symbol = create_symbol(pkt_data, header_len);
    update_probability(symbol, data);

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

    if(key.dst_port > key.src_port) {
        uint16_t temp = key.dst_port;
        key.dst_port = key.src_port;
        key.src_port = temp;
    }

    // Define a new flow_table entry
    struct mc_flow_entry *data = NULL;
    int tbl_index = onvm_ft_lookup_key(state_info->flow_table, &key, (char **)&data);

    // convert pkt_data[2] and pkt_data[3] into one integer and check if this
    // integer is equal to 42, which signals the end of the flow.
    if((((pkt_data[2] << 8) | (pkt_data[3])) == 42)) {
        // Check if something has been returned previously.
        if(data != NULL) {
            // update_probability(END_SYMBOL, data);
            double log_prob = data->log_prob;
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
    if(tbl_index == -ENOENT && pkt_data[header_len + 7] == 1) {
        // fprintf(debug_file, "Src IP: %hu.%hu.%hu.%hu, Dst IP: %hu.%hu.%hu.%hu, Src Port: %hu, Dst Port: %hu, Protokoll: %hu\n",
        //     (key.src_addr) & 0x000000ff, (key.src_addr >> 8) & 0x000000ff, (key.src_addr >> 16) & 0x000000ff, (key.src_addr >>24) & 0x000000ff, 
        //     (key.dst_addr) & 0x000000ff, (key.dst_addr >> 8) & 0x000000ff, (key.dst_addr >> 16) & 0x000000ff, (key.dst_addr >>24) & 0x000000ff,
        //     key.src_port, key.dst_port, key.proto);
        table_add_entry(&key, state_info, pkt);
        return -1;
    }
    if(tbl_index < 0) {
        return -1;
    }
    else {
        // Get symbol for forward pass, execute an iteration of forward pass
        header_len += 8;
        int symbol = create_symbol(pkt_data, header_len);
        update_probability(symbol, data);

        // If number of iterations is reached, make last iteration of forward pass, write decision to packet and remove key out of flow_table
        if(data->counter >= trace_length) {
            // update_probability(END_SYMBOL, data);
            double log_prob = data->log_prob;
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


static int packet_handler(struct rte_mbuf *pkt, struct onvm_pkt_meta *meta, __attribute__((unused)) struct onvm_nf_local_ctx *nf_local_ctx) {
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


int main(int argc, char *argv[]) {
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

    if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, nf_local_ctx, nf_function_table)) < 0) {
        onvm_nflib_stop(nf_local_ctx);
        if (arg_offset == ONVM_SIGNAL_TERMINATION) {
            return 0;
        } 
        else {
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
    if (state_info == NULL) {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to initialize NF state");
    }

    // Inits the flow_table, if unsuccessful stop NF
    state_info->flow_table = onvm_ft_create(MAXSIZE, sizeof(struct mc_flow_entry));
    if (state_info->flow_table == NULL) {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to create flow table");
    }

    // Load config and init alphas
    load_config();

    if (core != 0) {
        nf_local_ctx->nf->thread_info.core = core;
    }

    // Run and stop the NF, release allocated memory
    onvm_nflib_run(nf_local_ctx);
    onvm_nflib_stop(nf_local_ctx);
    onvm_ft_free(state_info->flow_table);
    rte_free(state_info);
    rte_free(mc.heads);
    rte_free(mc.log_probs);
    rte_free(mc.offsets);
    rte_free(mc.tails);
    rte_free(sym_lookup);

    return 0;
}
