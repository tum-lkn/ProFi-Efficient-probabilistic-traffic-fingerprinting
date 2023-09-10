#include <stdio.h>
#include <pcap.h>
#include <netinet/in.h>
#include <netinet/if_ether.h>
#include <iostream>
#include <arpa/inet.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <math.h>
#include <iostream>
#include <fstream>


typedef unsigned __int128 uint128_t;


struct tls_header {
  uint32_t record_type=0;
  uint32_t version=0;
  uint32_t length=0;
  uint32_t handshake_type=-1;
};


struct ip_header {
  uint32_t src_ip;
  uint32_t dst_ip;
  uint32_t hdr_length;
};


struct tcp_header {
  uint32_t src_port;
  uint32_t dst_port;
  uint32_t hdr_length;
  uint32_t segment_length;
  uint32_t sequence_number;
};


struct tls_packet {
  struct ip_header * ip_hdr;
  struct tcp_header * tcp_hdr;
  struct tls_header * tls_hdr;
  uint32_t frame_length;
};


struct tls_state {
    uint32_t remaining_bytes=0;
    uint32_t next_seq=0;
    int is_partial=0;
    uint32_t processed_bytes=0;
    struct tls_header * prev_header;
    int valid = 1;
    int num_packets = 1;
};

__time_t first_packet = 0;
int debug = 0;

void flow_hash(struct tls_packet *pkt, uint128_t &hash) {
  hash = 0;
  if(pkt->ip_hdr->src_ip < pkt->ip_hdr->dst_ip) {
    hash += (uint128_t)(pkt->ip_hdr->src_ip) << 64;
    hash += (uint128_t)(pkt->ip_hdr->dst_ip) << 32;
  } else {
    hash += (uint128_t)(pkt->ip_hdr->dst_ip) << 64;
    hash += (uint128_t)(pkt->ip_hdr->src_ip) << 32;
  }
  if(pkt->tcp_hdr->src_port < pkt->tcp_hdr->dst_port) {
    hash += (uint128_t)(pkt->tcp_hdr->src_port) << 16;
    hash += (uint128_t)pkt->tcp_hdr->dst_port;
  } else {
    hash += (uint128_t)(pkt->tcp_hdr->dst_port) << 16;
    hash += (uint128_t)pkt->tcp_hdr->src_port;
  }
}


void directional_hash(struct tls_packet *pkt, uint128_t &hash) {
    hash = 0;
    hash += (uint128_t)(pkt->ip_hdr->src_ip) << 64;
    hash += (uint128_t)(pkt->ip_hdr->dst_ip) << 32;
    hash += (uint128_t)(pkt->tcp_hdr->src_port) << 16;
    hash += (uint128_t)pkt->tcp_hdr->dst_port;
}


void reverse_directional_hash(struct tls_packet *pkt, uint128_t &hash) {
    hash = 0;
    hash += (uint128_t)(pkt->ip_hdr->src_ip) << 32;
    hash += (uint128_t)(pkt->ip_hdr->dst_ip) << 64;
    hash += (uint128_t)pkt->tcp_hdr->src_port;
    hash += (uint128_t)(pkt->tcp_hdr->dst_port) << 16;
}


void print_packet_info(const u_char *packet, struct pcap_pkthdr packet_header) {
    printf("Packet capture length: %d\n", packet_header.caplen);
    printf("Packet total length %d\n", packet_header.len);
}


void print_ip(uint32_t ip) {
    // Extract octetts in the reverse order they are stored in the integer.
    printf("%d.%d.%d.%d", (ip >> 24) & 0xff, (ip >> 16) & 0xff, (ip >> 8) & 0xff, ip & 0xff);
}


void print_tls_packet_struct(struct tls_packet * tls_pkt) {
    print_ip(tls_pkt->ip_hdr->src_ip);
    printf(":%" PRIu32 " -> ", tls_pkt->tcp_hdr->src_port);
    print_ip(tls_pkt->ip_hdr->dst_ip);
    printf(":%" PRIu32 ", Seq-Nr: %" PRIu32 ", Frame Size: %" PRIu32 ", Payload Size %" PRIu32 ": ",
           tls_pkt->tcp_hdr->dst_port, tls_pkt->tcp_hdr->sequence_number, tls_pkt->frame_length, tls_pkt->tcp_hdr->segment_length);
    printf("Record Type: %" PRIu32 ", Version: %" PRIu32 ".%" PRIu32 " Length: %" PRIu32 " Handshake Type: %" PRIu32 "\n",
           tls_pkt->tls_hdr->record_type,
           tls_pkt->tls_hdr->version >> 8,
           tls_pkt->tls_hdr->version & 0xf,
           tls_pkt->tls_hdr->length,
           tls_pkt->tls_hdr->handshake_type
           );
}


void print_ip_tcp_header(struct tls_packet * tls_pkt) {
    print_ip(tls_pkt->ip_hdr->src_ip);
    printf(":%" PRIu32 " -> ", tls_pkt->tcp_hdr->src_port);
    print_ip(tls_pkt->ip_hdr->dst_ip);
    printf(":%" PRIu32 ", Seq-Nr: %" PRIu32 ", Frame Size: %" PRIu32 ", Payload Size %" PRIu32 ": ",
           tls_pkt->tcp_hdr->dst_port, tls_pkt->tcp_hdr->sequence_number, tls_pkt->frame_length, tls_pkt->tcp_hdr->segment_length);
}


void fill_ip(const u_char *packet, uint32_t byte_offset, uint32_t &ip) {
  // Zero all bits in the destination container to ensure the correct ip is filled in.
  ip = 0;
  ip += ((uint32_t)(packet[byte_offset + 0])) << 24; // First octett.
  ip += ((uint32_t)(packet[byte_offset + 1])) << 16; // Second octett.
  ip += ((uint32_t)(packet[byte_offset + 2])) << 8;  // Third octett.
  ip += ((uint32_t)(packet[byte_offset + 3]));       // last octett.
}


void fill_ip_header(const u_char *packet, struct ip_header * hdr) {
  fill_ip(packet, + 12, hdr->src_ip);
  fill_ip(packet, 16, hdr->dst_ip);
  hdr->hdr_length = ((*packet) & 0x0F);
  /* The IHL is number of 32-bit segments. Multiply
     by four to get a byte count for pointer arithmetic */
  hdr->hdr_length = hdr->hdr_length * 4;
}


void fill_tcp_header(const u_char *packet, struct tcp_header * hdr) {
    hdr->hdr_length  = ((*(packet + 12)) & 0xF0) >> 4;
    /* The TCP header length stored in those 4 bits represents
       how many 32-bit words there are in the header, just like
       the IP header length. We multiply by four again to get a
       byte count. */
    hdr->hdr_length = hdr->hdr_length * 4;
    hdr->src_port = 0;
    hdr->src_port += (uint32_t)(packet[0]) << 8;
    hdr->src_port += (uint32_t)(packet[1]);

    hdr->dst_port = 0;
    hdr->dst_port += (uint32_t)(packet[2]) << 8;
    hdr->dst_port += (uint32_t)(packet[3]);

    hdr->sequence_number = 0;
    for(int i = 0; i < 4; i++) {
        hdr->sequence_number += ((uint32_t)packet[4 + i]) << (24 - 8 * i);
    }
}


void fill_ip_tcp_headers(const u_char *packet, struct tls_packet * tls_pkt) {
  int ethernet_header_length = 14; /* Doesn't change */
  fill_ip_header(packet + ethernet_header_length, tls_pkt->ip_hdr);
  fill_tcp_header(packet + ethernet_header_length + tls_pkt->ip_hdr->hdr_length, tls_pkt->tcp_hdr);
  tls_pkt->tcp_hdr->segment_length = tls_pkt->frame_length - ethernet_header_length - tls_pkt->ip_hdr->hdr_length - tls_pkt->tcp_hdr->hdr_length;
}


void fill_tls_header(const u_char *packet, struct tls_header *hdr, struct tls_state * state, uint32_t remaining_payload) {
    if ( debug == 1) { printf("\t Parse bytes %x %x %x %x %x %x \t", *packet, *(packet+1), *(packet+2), *(packet+3), *(packet+4), *(packet+5)); }
    if(state->is_partial == 1) {
        if (debug == 1) { printf("State is partial, continue at byte %" PRIu32"\n", state->processed_bytes); }
        hdr->record_type = state->prev_header->record_type;
        hdr->version = state->prev_header->version;
        hdr->length = state->prev_header->length;
        hdr->handshake_type = state->prev_header->handshake_type;
    } else {
        hdr->record_type = (uint32_t) (*packet);
        if (hdr->record_type < 20 || hdr->record_type > 23) {
            return;
        }
        state->processed_bytes += 1;
    }
    if (remaining_payload >= state->processed_bytes + 1) {
        if(state->processed_bytes < 2) {
            hdr->version = 0;
            hdr->version += (uint32_t) (packet[1 - state->is_partial * state->processed_bytes]) << 8;
            // Correct the indexing into the packet payload. If partial is one, the amount of already parsed bytes
            // is subtracted from the position. This corrects for the bytes parsed in the previous packets, i.e.,
            // the pointer is shifted to the first byte in the payload.
            state->processed_bytes += 1;
        }
    } else {
        if (debug == 1) { printf("Payload ended - stopped at byte 1\n"); }
        state->is_partial = 1;
        return;
    }
    if (remaining_payload >= state->processed_bytes + 1) {
        if(state->processed_bytes < 3) {
            hdr->version += (uint32_t) (packet[2 - state->is_partial * state->processed_bytes]);
            state->processed_bytes += 1;
        }
    } else {
        if (debug == 1) { printf("Payload ended - stopped at byte 2\n"); }
        state->is_partial = 1;
        return;
    }
    if (remaining_payload >= state->processed_bytes + 1) {
        if(state->processed_bytes < 4) {
            hdr->length = 0;
            hdr->length += (uint32_t) (packet[3 - state->is_partial * state->processed_bytes]) << 8;
            state->processed_bytes += 1;
        }
    } else {
        if (debug == 1) { printf("Payload ended - stopped at byte 3\n"); }
        state->is_partial = 1;
        return;
    }
    if (remaining_payload >= state->processed_bytes + 1) {
        if(state->processed_bytes < 5) {
            hdr->length += (uint32_t) (packet[4 - state->is_partial * state->processed_bytes]);
            state->processed_bytes += 1;
        }
    } else {
        if (debug == 1) { printf("Payload ended - stopped at byte 4\n"); }
        state->is_partial = 1;
        return;
    }
    if (remaining_payload >= state->processed_bytes + 1 && hdr->record_type == 22) {
        hdr->handshake_type = (uint32_t) (packet[5 - state->is_partial * state->processed_bytes]);
        state->processed_bytes += 1;
    }
    state->is_partial = 0;
    state->processed_bytes = 0;
}


void print_tls_header(struct tls_header *hdr) {
  printf("\tRecord Type: %" PRIu32 ", Version: %" PRIu32 ".%" PRIu32 ", Length: %" PRIu32 ", Handshake Type: %" PRIu32 "\n",
    hdr->record_type,
    hdr->version >> 8,
    hdr->version & 0xf,
    hdr->length,
    hdr->handshake_type);
}


void defense(uint32_t tls_record_size, struct tls_packet * pkt, uint32_t seed, std::vector<int32_t> * frame_lengths) {
    uint32_t generated = 0;
    uint32_t mtu = 1512;
    uint32_t segment_len = mtu - 12 - pkt->tcp_hdr->hdr_length - pkt->ip_hdr->hdr_length;
    uint32_t remaining_len = segment_len;
    uint32_t rand_val;
    srand(seed);
    while (generated < tls_record_size) {
        rand_val = rand() % 50 + 50;
        generated += rand_val;
        remaining_len -= rand_val + 5; // Account for the TLS header.
        if (remaining_len <= 0) {
            frame_lengths->push_back(mtu);
            remaining_len = segment_len + remaining_len; // remaining_len will be zero or negative --> reduces available length.
        } else if(rand() % 2 == 0) {
            frame_lengths->push_back(mtu - remaining_len);
            remaining_len = segment_len;
        }
    }
    if (remaining_len < segment_len) {
        frame_lengths->push_back(mtu - remaining_len);
    }
}


void my_packet_handler2(
        u_char *args,
        const struct pcap_pkthdr *header,
        const u_char *packet,
        std::vector<int32_t> * frame_lengths
) {
    struct ether_header *eth_header;
    eth_header = (struct ether_header *) packet;
    if (ntohs(eth_header->ether_type) != ETHERTYPE_IP) {
        if (debug == 1) { printf("Not an IP packet. Skipping...\n\n"); }
        return;
    }

    /* The total packet length, including all headers
       and the data payload is stored in
       header->len and header->caplen. Caplen is
       the amount actually available, and len is the
       total packet length even if it is larger
       than what we currently have captured. If the snapshot
       length set with pcap_open_live() is too small, you may
       not have the whole packet. */
    if (header == NULL) {
        printf("Null pointer returned\n");
    }
    if (debug == 1) { printf("Total packet available: %d bytes\n", header->caplen); }
    if (debug == 1) { printf("Expected packet size: %d bytes\n", header->len); }

    /* Pointers to start point of various headers */
    const u_char *ip_header;
    const u_char *tcp_header;

    /* Header lengths in bytes */
    int ethernet_header_length = 14; /* Doesn't change */
    int ip_header_length;
    int direction;
    uint32_t prefix = 0;
    prefix += 172 << 24;
    prefix += 17 << 16;
    uint32_t mask = 0xffffff00;

    /* Find start of IP header */
    ip_header = packet + ethernet_header_length;
    /* The second-half of the first byte in ip_header
       contains the IP header length (IHL). */
    ip_header_length = ((*ip_header) & 0x0F);
    /* The IHL is number of 32-bit segments. Multiply
       by four to get a byte count for pointer arithmetic */
    ip_header_length = ip_header_length * 4;
    if (debug == 1) { printf(" IHL in bytes: %d\n", ip_header_length); }

    /* Now that we know where the IP header is, we can
       inspect the IP header for a protocol number to
       make sure it is TCP before going any further.
       Protocol is always the 10th byte of the IP header */
    struct ip_header h;
    fill_ip_header(ip_header, &h);
    u_char protocol = *(ip_header + 9);
    if ((h.src_ip & mask) == (prefix & mask)) {
        direction = -1;
    } else {
        direction = 1;
    }
    if(protocol == IPPROTO_UDP) {
        frame_lengths->push_back(direction * (int)header->len);
        if (debug == 1) { printf("UDP packet, store length and continue\n---\n\n"); }
        return;
    } else if (protocol == IPPROTO_TCP) {
        struct tcp_header tcp_hdr;
        tcp_header = packet + ethernet_header_length + ip_header_length;
        fill_tcp_header(tcp_header, &tcp_hdr);
        //tcp_header_length = ((*(tcp_header + 12)) & 0xF0) >> 4;
        //payload_length = header->caplen - (ethernet_header_length + ip_header_length + tcp_header_length);
        if (tcp_hdr.segment_length > 0) {
            frame_lengths->push_back(direction * (int)header->len);
        }
    }
}


void my_packet_handler(
    u_char *args,
    const struct pcap_pkthdr *header,
    const u_char *packet,
    std::unordered_map<uint128_t, struct tls_packet*> * hash_map,
    std::vector<int32_t> * frame_lengths,
    std::unordered_map<uint128_t, struct tls_state*> * tls_state_map
) {
        /* First, lets make sure we have an IP packet */
    struct ether_header *eth_header;
    eth_header = (struct ether_header *) packet;
    if (ntohs(eth_header->ether_type) != ETHERTYPE_IP) {
        if (debug == 1) { printf("Not an IP packet. Skipping...\n\n"); }
        return;
    }

    /* The total packet length, including all headers
       and the data payload is stored in
       header->len and header->caplen. Caplen is
       the amount actually available, and len is the
       total packet length even if it is larger
       than what we currently have captured. If the snapshot
       length set with pcap_open_live() is too small, you may
       not have the whole packet. */
    if (debug == 1) { printf("Total packet available: %d bytes\n", header->caplen); }
    if (debug == 1) { printf("Expected packet size: %d bytes\n", header->len); }

    /* Pointers to start point of various headers */
    const u_char *ip_header;
    const u_char *tcp_header;
    const u_char *payload;
    const u_char *payload_start;

    /* Header lengths in bytes */
    int ethernet_header_length = 14; /* Doesn't change */
    int ip_header_length;
    int tcp_header_length;
    int payload_length;
    uint128_t hash = 0;
    struct tls_state * s;
    struct tls_packet *tls_pkt = (struct tls_packet*)malloc(sizeof(struct tls_packet));
    tls_pkt->tcp_hdr = (struct tcp_header*)(malloc(sizeof(struct tcp_header)));
    tls_pkt->ip_hdr = (struct ip_header*)(malloc(sizeof(struct ip_header)));
    tls_pkt->tls_hdr = (struct tls_header*)(malloc(sizeof(struct tls_header)));
    tls_pkt->frame_length = header->len;
    int direction;
    uint32_t prefix = 0;
    prefix += 172 << 24;
    prefix += 17 << 16;
    uint32_t mask = 0xffffff00;

    /* Find start of IP header */
    ip_header = packet + ethernet_header_length;
    /* The second-half of the first byte in ip_header
       contains the IP header length (IHL). */
    ip_header_length = ((*ip_header) & 0x0F);
    /* The IHL is number of 32-bit segments. Multiply
       by four to get a byte count for pointer arithmetic */
    ip_header_length = ip_header_length * 4;
    if (debug == 1) { printf(" IHL in bytes: %d\n", ip_header_length); }

    /* Now that we know where the IP header is, we can
       inspect the IP header for a protocol number to
       make sure it is TCP before going any further.
       Protocol is always the 10th byte of the IP header */
    u_char protocol = *(ip_header + 9);
    if(protocol == IPPROTO_UDP) {
        struct ip_header h;
        fill_ip_header(ip_header, &h);
        if ((h.src_ip & mask) == (prefix & mask)) {
            direction = -1;
        } else {
            direction = 1;
        }
        frame_lengths->push_back(direction * (int)header->len);
        if (debug == 1) { printf("UDP packet, store length and continue\n---\n\n"); }
        return;
    }
    if (protocol != IPPROTO_TCP) {
        if (debug == 1) { printf("Not a TCP packet. Skipping...\n\n"); }
        return;
    }

    /* Add the ethernet and ip header length to the start of the packet
       to find the beginning of the TCP header */
    tcp_header = packet + ethernet_header_length + ip_header_length;
    /* TCP header length is stored in the first half
       of the 12th byte in the TCP header. Because we only want
       the value of the top half of the byte, we have to shift it
       down to the bottom half otherwise it is using the most
       significant bits instead of the least significant bits */
    tcp_header_length = ((*(tcp_header + 12)) & 0xF0) >> 4;
    /* The TCP header length stored in those 4 bits represents
       how many 32-bit words there are in the header, just like
       the IP header length. We multiply by four again to get a
       byte count. */
    tcp_header_length = tcp_header_length * 4;
    if (debug == 1) { printf("TCP header length in bytes: %d\n", tcp_header_length); }

    payload_start = packet + ethernet_header_length + ip_header_length + tcp_header_length;
    fill_ip_tcp_headers(packet, tls_pkt);
    if ((tls_pkt->ip_hdr->src_ip & mask) == (prefix & mask)) {
        direction = -1;
    } else {
        direction = 1;
    }
    payload_length = header->caplen - (ethernet_header_length + ip_header_length + tcp_header_length);
    if (debug == 1) { printf("Payload size: %d bytes\n", payload_length); }
    /* Add up all the header sizes to find the payload offset */
    int total_headers_size = ethernet_header_length + ip_header_length + tcp_header_length;
    payload = packet + total_headers_size;
    if (debug == 1) { printf("Memory address where payload begins: %p\n", payload); }

    if (payload_length  == 0) {
        if (debug == 1) { printf("Packet has no payload - skip\n---\n\n"); }
        return;
    }
    directional_hash(tls_pkt, hash);
    if(tls_state_map->find(hash) != tls_state_map->end()) {
        // Hash exists, retrieve state. Packet belongs to a TLS flow.
        s = tls_state_map->at(hash);
        if (s->valid == 0) {
            return;
        } else {
            s->num_packets += 1;
            // printf("%d\n", s->num_packets);
        }
    } else {
        frame_lengths->push_back(direction * (int)header->len);
        // Store the frame length. The packet is either a handshake packet, or a regular TCP packet. In both cases
        // the defense does not apply.
        if (payload_length < 6) {
            if (debug == 1) { printf("Packet has less than %" PRIu32 " < 6 Bytes payload -- no handshake packet\n---\n\n", payload_length); }
            return;
        }
        // Parse a TLS header.
        struct tls_state tmp_s;
        fill_tls_header(payload_start, tls_pkt->tls_hdr, &tmp_s, payload_length);
        if (tls_pkt->tls_hdr->record_type == 22 && tls_pkt->tls_hdr->handshake_type == 1) {
            // Handle client hello and initialize the client-to-server direction.
            flow_hash(tls_pkt, hash);
            hash_map->insert(std::make_pair(hash, tls_pkt));

            s = (struct tls_state *) malloc(sizeof(struct tls_state));
            s->valid = 1;
            s->num_packets = 1;
            s->next_seq = tls_pkt->tcp_hdr->sequence_number;
            s->prev_header = (struct tls_header*) malloc(sizeof(struct tls_header));
            directional_hash(tls_pkt, hash);
            tls_state_map->insert(std::make_pair(hash, s));
            if (debug == 1 ) { printf("Client Hello: "); print_tls_packet_struct(tls_pkt); }
        }
        else if (tls_pkt->tls_hdr->record_type == 22 && tls_pkt->tls_hdr->handshake_type == 2) {
            // Handle Server Hello and initialize the server-to-client direction.
            s = (struct tls_state *) malloc(sizeof(struct tls_state));
            s->valid = 1;
            s->num_packets = 1;
            s->prev_header = (struct tls_header*) malloc(sizeof(struct tls_header));
            s->next_seq = tls_pkt->tcp_hdr->sequence_number;
            directional_hash(tls_pkt, hash);
            tls_state_map->insert(std::make_pair(hash, s));
            if (debug == 1 ) { printf("Server Hello: "); print_tls_packet_struct(tls_pkt); }
        } else {
            if (debug == 1) { printf("Packet %" PRIu32 " is Neither a ClientHello nor a ServerHello\n---\n\n", tls_pkt->tcp_hdr->sequence_number); }
            return;
        }
    }

    uint32_t cache_processed_bytes;
    if (debug == 1) { print_tls_packet_struct(tls_pkt); }
    struct tls_header *tls_hdr;
    if (tls_pkt->tcp_hdr->sequence_number < s->next_seq) {
        // Duplicate packet, or retransmission, do not parse it further.
        if (debug == 1) { printf("Duplicate packet detected, skip\n"); }
        return;
    } else if ( tls_pkt->tcp_hdr->sequence_number > s->next_seq) {
        if (debug == 1) {
            printf("Previous segment missing - stop parsing flow: ");
            print_ip_tcp_header(tls_pkt);
            printf("\n");
        }
        s->valid = 0;
        // free(s->prev_header);
        //free(s);
        tls_state_map->erase(hash);
    } else if (s->remaining_bytes >= tls_pkt->tcp_hdr->segment_length) {
        // The payload carries only a part of record's payload. Decrease the count accordingly. There is not header
        // hidden in the payload.
        if (debug == 1) { printf("\tContinuation data, %" PRIu32 " to %" PRIu32 "\n", s->remaining_bytes, s->remaining_bytes - tls_pkt->tcp_hdr->segment_length); }
        s->remaining_bytes -= tls_pkt->tcp_hdr->segment_length;
        s->next_seq = tls_pkt->tcp_hdr->sequence_number + payload_length;
    } else {
        // The payload has at least one TLS record.
        if (debug == 1) { printf("\tContains new record, starts at %" PRIu32 "\n", s->remaining_bytes); }
        int offset = s->remaining_bytes;
        int remaining_payload = tls_pkt->tcp_hdr->segment_length;
        while(offset < (int)tls_pkt->tcp_hdr->segment_length) {
            remaining_payload -= s->remaining_bytes;
            tls_hdr = (struct tls_header*)malloc(sizeof(struct tls_header));
            if (debug == 1) { printf("\tRecord in %" PRIu32 ", %" PRIu32" | %" PRIu32 "", offset, payload_length, remaining_payload); }
            cache_processed_bytes = s->processed_bytes; // Store this value to account for it later, is in {0, ..., 5}.
            fill_tls_header(payload_start + offset, tls_hdr, s, remaining_payload);
            free(s->prev_header);
            s->prev_header = tls_hdr;
            if(tls_hdr->record_type < 20 || tls_hdr->record_type > 23) {
                if (debug == 1) { printf("\tRecord type not a valid one, is %" PRIu32" for packet %" PRIu32 "\n", tls_hdr->record_type, tls_pkt->tcp_hdr->sequence_number); }
                tls_state_map->erase(hash);
                break;
            }
            if (s->is_partial) {
                if (debug == 1) { printf("Record not fully in this packet \n"); }
                offset = (int)tls_pkt->tcp_hdr->segment_length; // Parsed everything, make sure while loop exits.
                remaining_payload = 0;
                s->remaining_bytes = 0; // Initializes the offset. The parsing of the header will continue at the byte that was left off.
            } else {
                if (debug == 1) { print_tls_header(tls_hdr); }
                offset += tls_hdr->length + 5 - cache_processed_bytes;
                s->remaining_bytes = tls_hdr->length;
                // Account for potential bytes of the header that are in the previous packet.
                remaining_payload -= 5 - cache_processed_bytes;
                if(tls_hdr->record_type == 23) {
                    defense(tls_hdr->length, tls_pkt, frame_lengths->size(), frame_lengths);
                } else {
                    frame_lengths->push_back(direction * (int)header->len);
                }
            }
        }
        s->remaining_bytes -= remaining_payload;
        s->next_seq = tls_pkt->tcp_hdr->sequence_number + payload_length;
    }
    if (debug == 1) { printf("---\n\n"); }
}


void make_features(std::vector<int> * lengths, float * storage, int num_features) {
    float step = (float)lengths->size() / (float)num_features - 0.000000001;
    storage[0] = 0;
    for(int i = 1; i < num_features; i++){
        int lower = std::min((float)lengths->size() - 1, floor(i * step));
        int upper = std::min((float)lengths->size() - 1, ceil(i * step));
        if (lower == upper) {
            storage[i] = storage[i - 1] + lengths->at(lower);
        } else {
            float delta = (float)lengths->at(upper) - (float)lengths->at(lower);
            float until = i * step - lower;
            float val = (float)lengths->at(lower) + until * delta;
            storage[i] = val;
            // storage[i] = storage[i - 1] + (lengths->at(lower) + lengths->at(upper)) * (upper - i * step);
        }
    }
}


void cumul_cumsum(std::vector<int> * lengths, float * storage, int num_features) {
    std::vector<int> cumsum(lengths->size() + 1);
    cumsum[0] = 0;
    int sum = 0;
    for (int i = 0; i < (int)lengths->size(); i++) {
        sum += lengths->at(i);
        cumsum[i + 1] = sum;
    }
    make_features(&cumsum, storage, num_features);
}


void cumul_abs_cumsum(std::vector<int> * lengths, float * storage, int num_features) {
    std::vector<int> cumsum(lengths->size() + 1);
    cumsum[0] = 0;
    for (int i = 0; i < (int)lengths->size(); i++) {
        cumsum[i + 1] = cumsum.at(i) + abs(lengths->at(i));
    }
    make_features(&cumsum, storage, num_features);
}


extern "C"
int cumul_features(float * storage, int num_features, const char* filename) {
    std::vector<int32_t> * frame_lengths;
    char error_buffer[PCAP_ERRBUF_SIZE];
    frame_lengths = new std::vector<int>();
    pcap_t *handle;
    const u_char *packet;
    struct pcap_pkthdr packet_header;
    handle = pcap_open_offline(filename, error_buffer);
    if (handle == NULL) {
        fprintf(stderr, "Could not open files: %s\n", error_buffer);
        return 2;
    }
    //pcap_loop(handle, 0, my_packet_handler, NULL);
    while((packet = pcap_next(handle, &packet_header)) != NULL) {
        if (first_packet == 0) { first_packet = packet_header.ts.tv_sec; }
        if (packet_header.ts.tv_sec - first_packet > 7) { break; }
        my_packet_handler2(NULL, &packet_header, packet, frame_lengths);
    }
    if (debug == 1) { for(auto l : *frame_lengths) { std::cout << l << std::endl; }}
    cumul_cumsum(frame_lengths, storage, num_features);
    cumul_abs_cumsum(frame_lengths, storage + num_features, num_features);
    if (debug == 1) { printf("CUMUL FEATURES:\n"); }
    free(frame_lengths);
    return 0;
}


extern "C"
int cumul_with_defense(float * storage, int num_features, const char * filename) {
    char error_buffer[PCAP_ERRBUF_SIZE];
    std::unordered_map<uint128_t, struct tls_packet*> * hash_map;
    std::vector<int32_t> * frame_lengths;
    std::unordered_map<uint128_t, struct tls_state*> * tls_state_map;
    hash_map = new std::unordered_map<uint128_t, struct tls_packet*>();
    tls_state_map = new std::unordered_map<uint128_t, struct tls_state*>();
    frame_lengths = new std::vector<int>();
    pcap_t *handle;
    const u_char *packet;
    struct pcap_pkthdr packet_header;
    handle = pcap_open_offline(filename, error_buffer);
    if (handle == NULL) {
        fprintf(stderr, "Could not open files: %s\n", error_buffer);
        return 2;
    }
    //pcap_loop(handle, 0, my_packet_handler, NULL);
    while((packet = pcap_next(handle, &packet_header)) != NULL) {
        if (first_packet == 0) {
            first_packet = packet_header.ts.tv_sec;
        }
        if (packet_header.ts.tv_sec - first_packet > 7) {
            break;
        }
        my_packet_handler(NULL, &packet_header, packet, hash_map, frame_lengths, tls_state_map);
    }
    if (debug == 1) { printf("Got %ld tcp packets\n", hash_map->size()); }
    std::unordered_map<uint128_t, int> flow_lengths;
    for(auto it : *tls_state_map) {
        uint32_t src_ip = (uint32_t) (it.first >> 64) & 0xffffffff;
        uint32_t dst_ip = (uint32_t) (it.first >> 32) & 0xffffffff;
        uint32_t src_port = (uint32_t) (it.first >> 16) & 0xffff;
        uint32_t dst_port = (uint32_t) (it.first >> 0) & 0xffff;
        uint32_t tmp;
        uint128_t hash = 0;
        if (dst_ip < src_ip) {
            tmp = src_ip;
            src_ip = dst_ip;
            dst_ip = tmp;
        }
        if (dst_port < src_port) {
            tmp = src_port;
            src_port = dst_port;
            dst_port = tmp;
        }
        hash += ((uint128_t) src_ip) << 64;
        hash += ((uint128_t) dst_ip) << 32;
        hash += ((uint128_t) src_port) << 16;
        hash += ((uint128_t) dst_port);
        if (flow_lengths.find(hash) == flow_lengths.end()) {
            /* printf("New flow ");
            print_ip(src_ip); printf(", ");
            print_ip(dst_ip); printf(", ");
            printf("%" PRIu32 ", %" PRIu32 ",\n",
                   src_port,
                   dst_port
            ); */
            flow_lengths.insert(std::make_pair(hash, it.second->num_packets));
        } else {
            flow_lengths[hash] = flow_lengths.at(hash) + it.second->num_packets;
        }
    }
    int num_flows = 0;
    int num_packets = 0;
    int max_num_packets = 0;
    for(auto it : flow_lengths) {
        num_flows += 1;
        num_packets += it.second;
        max_num_packets = std::max(max_num_packets, it.second);
        uint32_t src_ip = (uint32_t) (it.first >> 64) & 0xffffffff;
        uint32_t dst_ip = (uint32_t) (it.first >> 32) & 0xffffffff;
        uint32_t src_port = (uint32_t) (it.first >> 16) & 0xffff;
        uint32_t dst_port = (uint32_t) (it.first >> 0) & 0xffff;
        /* printf("[");
        print_ip(src_ip); printf(", ");
        print_ip(dst_ip); printf(", ");
        printf("%" PRIu32 ", %" PRIu32 ", %d],\n",
               src_port,
               dst_port,
               it.second
               );*/
    }
    printf("num flows: %d num packets: %d max num packets of flow: %d\n", num_flows, num_packets, max_num_packets);
    for(auto entry : *hash_map) {
        if (debug == 1) { print_tls_packet_struct(entry.second); }
        free(entry.second->tcp_hdr);
        free(entry.second->ip_hdr);
        free(entry.second->tls_hdr);
        free(entry.second);
    }
    if (debug == 1) { for(int i = 0; i < frame_lengths->size(); i++) { printf("%d ", frame_lengths->at(i)); } printf("\n"); }
    cumul_cumsum(frame_lengths, storage, num_features);
    cumul_abs_cumsum(frame_lengths, storage + num_features, num_features);
    if (debug == 1) { printf("CUMUL FEATURES:\n"); }
    free(hash_map);
    free(frame_lengths);
    return 0;
}


void save_results(float * storage, int num_features, char * file_path) {
    std::ofstream out_file;
    out_file.open(file_path, std::ios::binary);
    for (int i = 0; i < 2 * num_features; i++) {
        out_file.write((char*)(storage + i), sizeof(float));
    }
    out_file.close();
}


int main(int argc, char **argv) {
    float * storage = (float*) malloc(sizeof(float) * 200);
    int num_features = 100;
    char *p;
    int mode = (int)strtol(argv[1], &p, 10);
    if (mode == 1) {
        printf("Convert %s to %s - defend traffic\n", argv[2], argv[3]);
        cumul_with_defense(storage, num_features, argv[2]);
    } else {
        printf("Convert %s to %s - do not defend traffic\n", argv[2], argv[3]);
        cumul_features(storage, num_features, argv[2]);
    }
    if (debug == 1) { for(int i = 0; i < 100; i++) { printf("%f, ", storage[i]); }}
    if (debug == 1) { printf("\n"); }
    if (debug == 1) { for(int i = 100; i < 200; i++) { printf("%f, ", storage[i]); }}
    save_results(storage, 100, argv[3]);
    free(storage);
    return 0;
}
