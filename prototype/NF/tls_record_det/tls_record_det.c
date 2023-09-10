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
    1 input port: incoming packets port, connection to tls_filter
    n output ports: n number of different data_aggregations

Functionality:
    Extract direction, packet_length and different tls_messages in packet, create udp_packet to forward to data_aggs, drop tcp packets
    Extraction of direction is based on the tcp sequence numbers, direction is 1 if packet is from client to server and 0 from server to client
    Packet length is extracted via ipv4 header
    Extracts tls_meassges out of packets, there are 3 cases
        1. case:
            1 tls_message in 1 packet
        2. case:
            multiple tls_messages in 1 packet
        3. case:
            tls_message continues in next packet
    The continuation of tls_messages can be interrupted by new packets
    The maximum number of simultaneously tracket tls_flows is 1024
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

#define NF_TAG "tls_record_detector"
#define PKTMBUF_POOL_NAME "MProc_pktmbuf_pool"
#define EXPIRE_TIME 1
#define CHECK_TIME 1

///////////////////////////////////////////////////////////////////////////////////////////////

uint32_t *data_aggs;
uint32_t first_data_agg;
uint32_t num_hmms;
uint8_t max_packets;

struct rte_mempool *pktmbuf_pool;
const uint32_t MAXSIZE = 1024 * 1024 * 20;
struct state_info
{
    struct onvm_ft *flow_table;
    uint16_t num_stored;
    uint64_t elapsed_cycles;
    uint64_t last_cycles;
};

struct tls_record_det_flow_entry
{
    uint64_t seq_ctos;
    uint16_t r_bytes_ctos;
    uint64_t seq_stoc;
    uint16_t r_bytes_stoc;
    uint8_t *features_ctos;
    uint8_t num_features_ctos;
    uint8_t *features_stoc;
    uint8_t num_features_stoc;
    uint64_t last_pkt_cycles;
    uint8_t num_packets;
};

struct state_info *state_info;
uint16_t *r_bytes;
uint8_t **features;
uint8_t *num_features;

uint8_t core = 99;

FILE *debug_file;

static int
parse_app_args(int argc, char *argv[]) {
    /*
    Parses the arguments in the start command of the NF
    Args:
        -d : destination ID of data_aggregation
    */

    int c;
    while ((c = getopt(argc, argv, "d:n:m:y:")) != -1) {
        switch (c) {
            case 'd':
                first_data_agg = strtoul(optarg, NULL, 10);
                break;
            case 'n':
                num_hmms = strtoul(optarg, NULL, 10);
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
setup(struct onvm_nf_local_ctx *nf_local_ctx) {
    // Allocate the memory_pool used to send new udp packets
    pktmbuf_pool = rte_mempool_lookup(PKTMBUF_POOL_NAME);
    if(pktmbuf_pool == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Cannot find mbuf pool!\n");
    }

    // Inits the state_info holding the flow_table, if unsuccessful stop NF
    state_info = rte_calloc("state", 1, sizeof(struct state_info), 0);
    if (state_info == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to initialize NF state");
    }

    // Inits the flow_table, if unsuccessful stop NF
    state_info->flow_table = onvm_ft_create(MAXSIZE, sizeof(struct tls_record_det_flow_entry));
    if (state_info->flow_table == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to create flow table");
    }

    data_aggs = rte_calloc("data_aggs", num_hmms, sizeof(uint32_t), 0);
    if(data_aggs == NULL)
    {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Unable to init dst array");
    }

    for(uint8_t i = 0; i < num_hmms; i++)
        data_aggs[i] = 2 * i + first_data_agg;

    return 0;
}

static int
extract_msg_type(uint8_t *pkt_data, uint16_t offset) {
    /*
    Maps a tls_message to an int and writes it to feature array
    Currently supported message types:
        Change Cipher Spec
        Alert
        Application Data
        Client Hello
        Server Hello
        New Session Ticket
        Certificate
        Certificate Request
        Finished
        end_of_early_data
        encrypted_extensions
        certificate_verify
        key_update

    Args:
        pkt_data (pointer): pointer to packet_data
        offset (int):   offset in packet_data, where to extract the message
    Returns:
        0/-1 (bool): returns 0 if extraction was successful, if not -1
    */

    (*num_features)++;
    switch(pkt_data[offset])
    {
        case 20:
            (*features)[*num_features + 2] = 0;
            return 0;
        case 21:
            (*features)[*num_features + 2] = 1;
            return 0;
        case 23:
            (*features)[*num_features + 2] = 2;
            return 0;
        case 22:
            break;
    }

    switch(pkt_data[offset + 5])
    {
        case 1:
            (*features)[*num_features + 2] = 3;
            return 0;
        case 2:
            (*features)[*num_features + 2] = 4;
            return 0;
        case 4:
            (*features)[*num_features + 2] = 5;
            return 0;
        case 11:
            (*features)[*num_features + 2] = 6;
            return 0;
        case 13:
            (*features)[*num_features + 2] = 7;
            return 0;
        case 20:
            (*features)[*num_features + 2] = 8;
            return 0;
        case 5:
            (*features)[*num_features + 2] = 9;
            return 0;
        case 8:
            (*features)[*num_features + 2] = 10;
            return 0;
        case 15:
            (*features)[*num_features + 2] = 11;
            return 0;
        case 24:
            (*features)[*num_features + 2] = 12;
            return 0;
        case 22:
            (*features)[*num_features + 2] = 13;
            return 0;
        case 16:
            (*features)[*num_features + 2] = 14;
            return 0;
        case 0:
            (*features)[*num_features + 2] = 15;
            return 0;
        case 12:
            (*features)[*num_features + 2] = 16;
            return 0;
        case 14:
            (*features)[*num_features + 2] = 17;
            return 0;
    }
    (*num_features)--;
    return -1;
}

static int
send_flow_ending_packet(struct onvm_ft_ipv4_5tuple *key, struct onvm_nf_local_ctx *nf_local_ctx) {
    /*
    Create and forward packet containing the information which flow is finished

    Args:
        key (pointer): pointer to the flow table key
        nf_local_ctx (pointer): pointer to the NF
    Returns:
        0/-1 (bool): indicating bool if actions were successful
    */

    // Define new udp packet, meta and pointer to packet data
    struct rte_mbuf *udp_pkt = rte_pktmbuf_alloc(pktmbuf_pool);
    if(udp_pkt == NULL) {
        RTE_LOG(INFO, APP, "Failed to allocate closing flow packet\n");
        return -1;
    }
    struct onvm_pkt_meta *udp_meta;
    uint8_t *udp_pkt_data = rte_pktmbuf_mtod_offset(udp_pkt, uint8_t*, 0);

    struct rte_mbuf *udp_pkt_new;
    struct onvm_pkt_meta *udp_meta_new;
    uint8_t *udp_pkt_data_new;

    // init udp packet to 0
    for(uint16_t i = 0; i < 42; i++)
        udp_pkt_data[i] = 0;

    // set header info for ipv4 and udp
    udp_pkt_data[14] = 69;
    udp_pkt_data[17] = 42;
    udp_pkt_data[20] = 64;
    udp_pkt_data[23] = 17;
    udp_pkt_data[39] = 42;

    // set src address
    udp_pkt_data[26] = (key->src_addr) & 0x000000ff;
    udp_pkt_data[27] = (key->src_addr >> 8) & 0x000000ff;
    udp_pkt_data[28] = (key->src_addr >> 16) & 0x000000ff;
    udp_pkt_data[29] = (key->src_addr >> 24) & 0x000000ff;

    // set dst address
    udp_pkt_data[30] = (key->dst_addr) & 0x000000ff;
    udp_pkt_data[31] = (key->dst_addr >> 8) & 0x000000ff;
    udp_pkt_data[32] = (key->dst_addr >> 16) & 0x000000ff;
    udp_pkt_data[33] = (key->dst_addr >> 24) & 0x000000ff;

    // set src port
    udp_pkt_data[34] = (key->src_port >> 8) & 0x00ff;
    udp_pkt_data[35] = (key->src_port) & 0x00ff;

    // set dst port
    udp_pkt_data[36] = (key->dst_port >> 8) & 0x00ff;
    udp_pkt_data[37] = (key->dst_port) & 0x00ff;

    // update meta of first packet and set dest
    udp_meta = onvm_get_pkt_meta(udp_pkt);
    udp_meta->action = ONVM_NF_ACTION_TONF;
    udp_meta->destination = data_aggs[0];

    // loop over all dst to forward udp packet to
    for(uint16_t j = 1; j < num_hmms; j++)
    {
        // allocate new udp packet and packet data
        udp_pkt_new = rte_pktmbuf_alloc(pktmbuf_pool);
        if(udp_pkt_new == NULL) {
            RTE_LOG(INFO, APP, "Failed to allocate closing flow packet copy\n");
        } else {
            udp_pkt_data_new = rte_pktmbuf_mtod_offset(udp_pkt_new, uint8_t*, 0);
            // for(uint16_t i = 0; i < header_len + *num_features + 11; i++)
            //     udp_pkt_data_new[i] = udp_pkt_data[i];

            // copy udp packet data to new udp packet data
            rte_memcpy(udp_pkt_data_new, udp_pkt_data, 42);

            // update meta of new udp packet and forward to data_agg
            udp_meta_new = onvm_get_pkt_meta(udp_pkt_new);
            udp_meta_new->action = ONVM_NF_ACTION_TONF;
            udp_meta_new->destination = data_aggs[j];

            onvm_nflib_return_pkt(nf_local_ctx->nf, udp_pkt_new);
        }
    }

    onvm_nflib_return_pkt(nf_local_ctx->nf, udp_pkt);

    return 0;
}

static int
clear_entries(struct state_info *state_info, struct onvm_nf_local_ctx *nf_local_ctx) {
    /*
    Removes all expired entries of the flow table

    Args:
        state_info (pointer): pointer to the state info of the NF
    Returns:
        0/-1 (bool): indicating bool if actions were successful
    */

    if(unlikely(state_info == NULL)) 
        return -1;

    // define current flow entry, key, next and ret type
    struct tls_record_det_flow_entry *data = NULL;
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
            send_flow_ending_packet(key, nf_local_ctx);
        }
    }

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
    if (unlikely(key == NULL)) 
        return -1;

    // Define a new flow_table entry and check if adding the key is successful
    struct tls_record_det_flow_entry *data = NULL;
    int tbl_index = onvm_ft_add_key(state_info->flow_table, key, (char **)&data);
    if (tbl_index < 0)
        return -1;

    // If successful, get pointer to begin of ip header and extract ipv4 header len
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t* , sizeof(struct rte_ether_hdr));
    uint16_t header_len = (pkt_data[0] & 0x0f) << 2;

    data->features_ctos = rte_calloc(NULL, 64, sizeof(uint8_t), 0);
    data->features_stoc = rte_calloc(NULL, 64, sizeof(uint8_t), 0);
    data->num_features_ctos = 0;
    data->num_features_stoc = 0;

    // Get the tcp sequence number from client to server and from server to client
    data->seq_ctos = ((pkt_data[header_len + 4] << 24) & 0xff000000) | ((pkt_data[header_len + 5] << 16) & 0x00ff0000) | ((pkt_data[header_len + 6] << 8) & 0x0000ff00) | ((pkt_data[header_len + 7]) & 0x000000ff);
    data->seq_stoc = ((pkt_data[header_len + 8] << 24) & 0xff000000) | ((pkt_data[header_len + 9] << 16) & 0x00ff0000) | ((pkt_data[header_len + 10] << 8) & 0x0000ff00) | ((pkt_data[header_len + 11]) & 0x000000ff);

    // Extract the tcp header len and pkt_len
    header_len += (pkt_data[header_len + 12] & 0xf0) >> 2;
    uint16_t pkt_len = (pkt_data[2] << 8) | pkt_data[3];

    // update the client to server sequence number to the next expected sequence number 
    data->seq_ctos += pkt_len;
    data->seq_ctos -= header_len;

    // Set the remaining_bytes of client to server and server to client to 0, assign r_bytes pointer to address of remaining bytes of client to server
    data->r_bytes_ctos = 0;
    data->r_bytes_stoc = 0;
    r_bytes = &data->r_bytes_ctos;
    features = &data->features_ctos;
    num_features = &data->num_features_ctos;

    // Calculate the payload_length
    uint16_t payload_len = pkt_len - header_len;

    // Set the direction and payload_len in the features array
    data->features_ctos[0] = 1;
    data->features_ctos[1] = (payload_len >> 8) & 0xff;
    data->features_ctos[2] = (payload_len) & 0xff;

    // Extract the tls_message type
    if(extract_msg_type(pkt_data, header_len) < 0)
        return -1;

    // Update the number of flows in the state_info
    data->num_packets = 1;
    data->last_pkt_cycles = state_info->elapsed_cycles;
    state_info->num_stored++;

    return 0;
}

static int
table_lookup_entry(struct rte_mbuf *pkt, struct state_info *state_info) {
    /*
    Checks if packet is in flow_table or not
    If packet is in flow_table extract features
    If not checks if packet is new tls_flow, if new tls_flow add flow to flow_table and extract features

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
    if (onvm_ft_fill_key_symmetric(&key, pkt) < 0)
        return -1;

    // Get pointer to begin of ipv4 header and extract ipv4 header len
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t* , sizeof(struct rte_ether_hdr));
    uint16_t header_len = (pkt_data[0] & 0x0f) << 2;

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
    struct tls_record_det_flow_entry *data = NULL;
    int tbl_index = onvm_ft_lookup_key(state_info->flow_table, &key, (char **)&data);

    // If key is new add key to flow_table
    if(tbl_index == -ENOENT)
    {
        return table_add_entry(&key, state_info, pkt);
    }
    if(tbl_index < 0)
    {
        return -1;
    }



    // Extract the current sequence number in the tcp header and update the header len by the tcp header len
    uint64_t curr_seq = ((pkt_data[header_len + 4] << 24) & 0xff000000) | ((pkt_data[header_len + 5] << 16) & 0x00ff0000) | ((pkt_data[header_len + 6] << 8) & 0x0000ff00) | ((pkt_data[header_len + 7]) & 0x000000ff);
    header_len += (pkt_data[header_len + 12] & 0xf0) >> 2;

    // Extract packet len in ipv4 header
    uint16_t pkt_len = (pkt_data[2] << 8) | pkt_data[3];

    // Check if packet is from Client to Server or vice versa, update sequence numbers and the pointer to remaining bytes, set direction of packet, if tcp retransmission exit function
    if(curr_seq == data->seq_ctos)
    {
        data->seq_ctos += pkt_len;
        data->seq_ctos -= header_len;
        r_bytes = &data->r_bytes_ctos;
        features = &data->features_ctos;
        num_features = &data->num_features_ctos;
        data->features_ctos[0] = 1;
    }
    else if(curr_seq == data->seq_stoc)
    {
        data->seq_stoc += pkt_len;
        data->seq_stoc -= header_len;
        r_bytes = &data->r_bytes_stoc;
        features = &data->features_stoc;
        num_features = &data->num_features_stoc;
        data->features_stoc[0] = 0;
    }
    else
    {
        return -1;
    }

    data->num_packets++;
    data->last_pkt_cycles = state_info->elapsed_cycles;

    if(data->num_packets == max_packets)
    {
        onvm_ft_remove_key(state_info->flow_table, &key);
        state_info->num_stored--;
    }

    // Calculate payload len and set payload len in feature array
    uint16_t payload_len = pkt_len - header_len;
    (*features)[1] = (payload_len >> 8) & 0xff;
    (*features)[2] = (payload_len) & 0xff;

    // If the payload len is less than the remaining bytes, tls_record continues in next packet, exit function
    if(payload_len <= *r_bytes)
    {
        *r_bytes -= payload_len;
        return 0;
    }

    // tls_record ends in this packet, extract new message in packet
    if(extract_msg_type(pkt_data, *r_bytes + header_len) < 0)
        return -1;

    // Extract the current tls_record len, calculate offset of begin of next tls_record
    uint16_t tls_record_len = (pkt_data[*r_bytes + header_len + 3] << 8) | pkt_data[*r_bytes + header_len + 4];
    uint16_t offset = *r_bytes + tls_record_len + 5;

    // Continuosly extract tls_messages until end of packet is reached
    while(offset < payload_len)
    {
        // Found new tls_record, extract tls_message and update offset
        tls_record_len = (pkt_data[header_len + offset + 3] << 8) | pkt_data[header_len + offset + 4];
        if(extract_msg_type(pkt_data, header_len + offset) < 0)
            return -1;
        offset += tls_record_len + 5;
    }

    // Calculate the remaining bytes of the current direction
    *r_bytes = offset - payload_len;

    return 0;
}

static int
set_features_array(void) {
    /*
    Resets the features array after sending features
    Args:
        /
    Returns:
        0/-1 (bool): indicating bool if actions were successful
    */

    // If the remaining bytes are 0, reset feature array
    if(*r_bytes == 0)
    {
        *num_features = 0;
        return 0;
    }

    // If the record continues in the next record, set the last feature as the first
    if(*num_features > 1)
    {   
        (*features)[3] = (*features)[*num_features + 2];
        *num_features = 1;
        return 0;
    }

    return 0;
}

static int
create_and_send_upd_packet(struct rte_mbuf *pkt, struct onvm_nf_local_ctx *nf_local_ctx) {
    /*
    Creates a udp packet with the features as payload out of the tcp packet

    Args:
        pkt (pointer): pointer to current packet
        nf_local_ctx (pointer): pointer to current NF

    Returns:
        0 (bool): always returns successful handling of packets
    */

    // Allocate a new packet and its actions
    struct rte_mbuf *udp_pkt = rte_pktmbuf_alloc(pktmbuf_pool);
    if(udp_pkt == NULL) {
        RTE_LOG(INFO, APP, "Failed to allocate symbol UDP packet\n");
        return -1;
    }
    struct onvm_pkt_meta *udp_meta;
    
    // Get a pointer to the beginning of the tcp and udp packet
    uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t*, 0);
    uint8_t *udp_pkt_data = rte_pktmbuf_mtod_offset(udp_pkt, uint8_t*, 0);

    // Define temporary packet
    struct rte_mbuf *udp_pkt_new;
    struct onvm_pkt_meta *udp_meta_new;
    uint8_t *udp_pkt_data_new;

    // Calculate the header len, consisting of ethernet and ip headerlen
    uint8_t header_len = 14;
    header_len += (pkt_data[header_len] & 0x0f) << 2;

    // Set the size of the udp packet and fill the headers 
    rte_pktmbuf_append(udp_pkt, header_len + *num_features + 11);
    rte_memcpy(udp_pkt_data, pkt_data, header_len + 4);

    // Set the udp payload len and checksum
    udp_pkt_data[header_len + 4] = 0;
    udp_pkt_data[header_len + 6] = 0;
    udp_pkt_data[header_len + 7] = 0;
    udp_pkt_data[header_len + 5] = *num_features + 11;

    if((*features)[3] == 3 && (*features)[0] == 1)
        udp_pkt_data[header_len + 7] = 1;

    // Update the header len by the udp header len
    header_len += 8;

    // Fill the payload with the features
    for(uint8_t i = 0; i < *num_features + 3; i++)
        udp_pkt_data[header_len + i] = (*features)[i];
    
    udp_pkt_data[23] = 17;
    // update the actions of the udp packet
    udp_meta = onvm_get_pkt_meta(udp_pkt);

    udp_meta->action = ONVM_NF_ACTION_TONF;
    udp_meta->destination = data_aggs[0];

    for(uint8_t i = 1; i < num_hmms; i++)
    {
        udp_pkt_new = rte_pktmbuf_alloc(pktmbuf_pool);
        if(udp_pkt_new == NULL) {
            RTE_LOG(INFO, APP, "Failed to allocate copy of symbol UDP packet\n");
        } else {
            udp_pkt_data_new = rte_pktmbuf_mtod_offset(udp_pkt_new, uint8_t*, 0);
    
            rte_memcpy(udp_pkt_data_new, udp_pkt_data, header_len + *num_features + 11);

            udp_pkt_data_new[23] = 17;
            udp_meta_new = onvm_get_pkt_meta(udp_pkt_new);
            udp_meta_new->action = ONVM_NF_ACTION_TONF;
            udp_meta_new->destination = data_aggs[i];

            onvm_nflib_return_pkt(nf_local_ctx->nf, udp_pkt_new);
        }
    }

    // Send the upd_packet the data_agg
    onvm_nflib_return_pkt(nf_local_ctx->nf, udp_pkt);

    return 0;
}

static int
packet_handler(struct rte_mbuf *pkt, struct onvm_pkt_meta *meta, struct onvm_nf_local_ctx *nf_local_ctx) {
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
    if(table_lookup_entry(pkt, state_info) < 0)
    {
        meta->action = ONVM_NF_ACTION_DROP;
        return 0;
    }

    // Send the features in the array as a udp packet
    create_and_send_upd_packet(pkt, nf_local_ctx);

    // Drop the tcp packet
    meta->action = ONVM_NF_ACTION_DROP;
    // rte_pktmbuf_free(pkt);

    // Set the feature array for new packets
    set_features_array();

    return 0;
}

static int
callback_handler(struct onvm_nf_local_ctx *nf_local_ctx) {
    state_info->elapsed_cycles = rte_get_tsc_cycles();

    if((state_info->elapsed_cycles - state_info->last_cycles) / rte_get_timer_hz() >= CHECK_TIME)
    {
        state_info->last_cycles = state_info->elapsed_cycles;
        if(clear_entries(state_info, nf_local_ctx) < 0)
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
    struct onvm_nf_local_ctx *nf_local_ctx;
    struct onvm_nf_function_table *nf_function_table;
    int arg_offset;

    nf_local_ctx = onvm_nflib_init_nf_local_ctx();
    onvm_nflib_start_signal_handler(nf_local_ctx, NULL);

    nf_function_table = onvm_nflib_init_nf_function_table();
    nf_function_table->pkt_handler = &packet_handler;
    nf_function_table->user_actions = &callback_handler;

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

    if(parse_app_args(argc, argv) < 0) {
        onvm_nflib_stop(nf_local_ctx);
        rte_exit(EXIT_FAILURE, "Invalid command-line arguments\n");
    }

    setup(nf_local_ctx);

    if(core != 0)
    {
        nf_local_ctx->nf->thread_info.core = core;
    }

    // Init NF Timer
    state_info->elapsed_cycles = rte_get_tsc_cycles();

    // debug_file = fopen("/home/benedikt/code/debug_tls_record_det.txt", "w");

    // Run and stop the NF, release allocated memory
    onvm_nflib_run(nf_local_ctx);
    onvm_nflib_stop(nf_local_ctx);
    onvm_ft_free(state_info->flow_table);
    rte_free(state_info);

    // fclose(debug_file);

    return 0;
}
