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
	1 input port: incoming packets port, connection to tls_record_detector
	1 output port: outgoing packets port, connection to hmm

Functionality:
	Quantize the packet length of packet, create 64bit hash of direction, packet length and tls_message_types.
	Write hash into packet, adjust packet length accordingly if necessary.
	Forward packet to hmm NF.
*/


#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/queue.h>
#include <unistd.h>

#include <rte_common.h>
#include <rte_malloc.h>
#include <rte_ip.h>
#include <rte_mbuf.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"

#define NF_TAG "data_agg"

static uint32_t hmm;
uint8_t core = 99;

static int
parse_app_args(int argc, char *argv[])
{
    /*
    Parses the arguments in the start command of the NF
    Args:
	-d : destination ID of hmm
	-c : path to the config_file
    */

    int c;
    while ((c = getopt(argc, argv, "d:y:")) != -1) 
    {
	    switch (c) 
        {
	        case 'd':
		        hmm = strtoul(optarg, NULL, 10);
		        break;
            case 'y':
		        core = strtoul(optarg, NULL, 10);
		        break;
	    }
    }

    return optind;
}

static int
packet_handler(struct rte_mbuf *pkt, struct onvm_pkt_meta *meta, __attribute__((unused)) struct onvm_nf_local_ctx *nf_local_ctx)
{
	/*
	Handles the incoming packets, quantizes the packet length, creates a unique symbol out of the direction, packet length and tls messages and forwards the packet to all hmms
	Args:
		pkt (pointer): pointer to current packet
		meta (pointer): pointer to the actions of current packet
		nf_local_ctx (pointer): pointer to current NF

	Returns:
		0 (bool): always returns successful handling of packets
	*/

	// Get pointer to begin of ipv4 header and extract ipv4 header len and quantize the packet length
	uint8_t *pkt_data = rte_pktmbuf_mtod_offset(pkt, uint8_t* , sizeof(struct rte_ether_hdr));

	if(((pkt_data[2] << 8) | (pkt_data[3])) == 42)
	{
		meta->destination = hmm;
		meta->action = ONVM_NF_ACTION_TONF;
		return 0;
	}

	uint16_t header_len = ((pkt_data[0] & 0x0f) << 2) + 8;

	// Create unique symbol out of direction, packet length and tls messages by applying a hash function to it
	uint64_t hash = 5381;
	for(uint8_t i = 0; i < pkt_data[header_len - 3] - 8; i++)
		hash = ((hash << 5) + hash) + pkt_data[header_len + i];
		
	// Append additional space to packet if not enough features
	if(pkt_data[header_len - 3] < 16)
	{
		rte_pktmbuf_append(pkt, 16 - pkt_data[header_len - 3]);
		pkt_data[header_len - 3] = 16;
	}

	// Write 64bit hash into packet        
	pkt_data[header_len] = (hash >> 56) & 0x00000000000000ff;
	pkt_data[header_len + 1] = (hash >> 48) & 0x00000000000000ff;
	pkt_data[header_len + 2] = (hash >> 40) & 0x00000000000000ff;
	pkt_data[header_len + 3] = (hash >> 32) & 0x00000000000000ff;
	pkt_data[header_len + 4] = (hash >> 24) & 0x00000000000000ff;
	pkt_data[header_len + 5] = (hash >> 16) & 0x00000000000000ff;
	pkt_data[header_len + 6] = (hash >> 8) & 0x00000000000000ff;
	pkt_data[header_len + 7] = (hash) & 0x00000000000000ff;

	// Forward packet to next NF
	meta->destination = hmm;
	meta->action = ONVM_NF_ACTION_TONF;

	return 0;
}

int
main(int argc, char *argv[])
{
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

	if (core != 0)
	{
	    nf_local_ctx->nf->thread_info.core = core;
	}

	// Start and Stop the NF, release the allocated memory
	onvm_nflib_run(nf_local_ctx);
	onvm_nflib_stop(nf_local_ctx);

	return 0;
}
