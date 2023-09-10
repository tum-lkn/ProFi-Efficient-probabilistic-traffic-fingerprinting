//
// Created by patrick on 22.06.22.
//
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
#include <string.h>

typedef std::unordered_map<std::string, std::vector<std::string>*> domain_db_t;
typedef unsigned __int128 uint128_t;

struct ip_header {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint32_t hdr_length;
};

void print_ip(uint32_t ip) {
  // Extract octetts in the reverse order they are stored in the integer.
  printf("%d.%d.%d.%d", (ip >> 24) & 0xff, (ip >> 16) & 0xff, (ip >> 8) & 0xff, ip & 0xff);
}


void fill_ip(const u_char *packet, uint32_t byte_offset, uint32_t &ip) {
  // Zero all bits in the destination container to ensure the correct ip is filled in.
  ip = 0;
  ip += ((uint32_t)(packet[byte_offset + 0])) << 24; // First octett.
  ip += ((uint32_t)(packet[byte_offset + 1])) << 16; // Second octett.
  ip += ((uint32_t)(packet[byte_offset + 2])) << 8;  // Third octett.
  ip += ((uint32_t)(packet[byte_offset + 3]));       // last octett.
}


uint32_t parse_two_byte_number(const u_char *packet) {
  uint32_t num = 0;
  num += (uint32_t)(packet[0]) << 8;
  num += (uint32_t)(packet[1]);
  return num;
}


void fill_ip_header(const u_char *packet, struct ip_header * hdr) {
  fill_ip(packet, + 12, hdr->src_ip);
  fill_ip(packet, 16, hdr->dst_ip);
  hdr->hdr_length = ((*packet) & 0x0F);
  /* The IHL is number of 32-bit segments. Multiply
     by four to get a byte count for pointer arithmetic */
  hdr->hdr_length = hdr->hdr_length * 4;
}


bool is_dns(const u_char * udp_header) {
  uint32_t dst_port = parse_two_byte_number(udp_header + 2);
  uint32_t src_port = parse_two_byte_number(udp_header);
  //src_port += (uint32_t)(udp_header[0]) << 8;
  //src_port += (uint32_t)(udp_header[1]);
  //dst_port += (uint32_t)(udp_header[2]) << 8;
  //dst_port += (uint32_t)(udp_header[3]);
  return dst_port == 53 || src_port == 53;
}


bool is_dns_response(const u_char * dns_header) {
  u_char m = 128;
  return (dns_header[2] & m) > 0;
}


bool is_dns_failure(const u_char * dns_header) {
  u_char m = 15;
  return (dns_header[3] & m) != 0;
}


bool uses_dns_compression(const u_char * name) {
  u_char m = 192;
  return (name[0] & m) > 0;
}



uint128_t flow_hash(const u_char * ip_header, const u_char * transport_header) {
  uint32_t src_ip, dst_ip, src_port, dst_port;
  fill_ip(ip_header, 12, src_ip);
  fill_ip(ip_header, 16, dst_ip);
  src_port = parse_two_byte_number(transport_header);
  dst_port = parse_two_byte_number(transport_header + 2);
  uint128_t  hash = 0;
  if(src_ip < dst_ip) {
    hash += (uint128_t)(src_ip) << 64;
    hash += (uint128_t)(dst_ip) << 32;
  } else {
    hash += (uint128_t)(dst_ip) << 64;
    hash += (uint128_t)(src_ip) << 32;
  }
  if(src_port < dst_port) {
    hash += (uint128_t)(src_port) << 16;
    hash += (uint128_t)dst_port;
  } else {
    hash += (uint128_t)(dst_port) << 16;
    hash += (uint128_t)src_port;
  }
  return hash;
}


int parse_name(const u_char * name_start, std::string &name, const u_char * dns_header) {
  int total_length = 1;
  auto label_length = (uint32_t)name_start[0];
  while(label_length > 0) {
    if(total_length > 1) name += '.';
    for(uint i = total_length; i < total_length + label_length; i++) {
       name += name_start[i];
     }
     total_length += label_length;
     if(uses_dns_compression(name_start + total_length)) {
       uint32_t m = 0x3fff;
       int32_t ptr = parse_two_byte_number(name_start + total_length) & m;
       name += ".";
       parse_name(dns_header + ptr, name, dns_header);
       total_length += 2;
       label_length = 0;
     } else {
       label_length = (uint32_t) name_start[total_length];
       total_length++;
     }
  }
  return total_length;
}


int retrieve_flows(const char * file_name, const std::string &result_file, const domain_db_t *domain_db, domain_db_t *flow_db, const int debug) {
  __time_t first_packet = 0;
  char error_buffer[PCAP_ERRBUF_SIZE];
  pcap_t *handle;
  const u_char *packet;
  struct pcap_pkthdr header{};
  std::unordered_map<uint128_t, bool> memory;
  std::ofstream resolutions;
  resolutions.open(result_file, std::ios::out);
  resolutions << "ip;domain" << std::endl;
  handle = pcap_open_offline(file_name, error_buffer);
  if (handle == nullptr) {
    fprintf(stderr, "Could not open files: %s\n", error_buffer);
    return 2;
  }
  int pkt_count = 0;
  if (debug == 2) { std::cout << "Opened file " << file_name << std::endl; }

  while((packet = pcap_next(handle, &header)) != nullptr) {
    pkt_count++;
    if (debug == 1) { std::cout << std::endl << "######## Process Frame " << pkt_count << " ######" << std::endl; }
    if (first_packet == 0) {
      first_packet = header.ts.tv_sec;
    }
    if (header.ts.tv_sec - first_packet > 7) {
      break;
    }
    struct ether_header *eth_header;
    eth_header = (struct ether_header *) packet;
    if (ntohs(eth_header->ether_type) != ETHERTYPE_IP) {
      if (debug == 2) { printf("Not an IP packet. Skipping...\n\n"); }
      continue;
    }

    /* The total packet length, including all headers
       and the data payload is stored in
       header->len and header->caplen. Caplen is
       the amount actually available, and len is the
       total packet length even if it is larger
       than what we currently have captured. If the snapshot
       length set with pcap_open_live() is too small, you may
       not have the whole packet. */
    if (debug == 2) { printf("Total packet available: %d bytes\n", header.caplen); }
    if (debug == 2) { printf("Expected packet size: %d bytes\n", header.len); }

    /* Pointers to start point of various headers */
    const u_char *ip_header;
    const u_char *tcp_header;

    /* Header lengths in bytes */
    int ethernet_header_length = 14; /* Doesn't change */
    int ip_header_length;

    /* Find start of IP header */
    ip_header = packet + ethernet_header_length;
    /* The second-half of the first byte in ip_header
       contains the IP header length (IHL). */
    ip_header_length = ((*ip_header) & 0x0F);
    /* The IHL is number of 32-bit segments. Multiply
       by four to get a byte count for pointer arithmetic */
    ip_header_length = ip_header_length * 4;
    if (debug == 2) { printf(" IHL in bytes: %d\n", ip_header_length); }

    /* Now that we know where the IP header is, we can
       inspect the IP header for a protocol number to
       make sure it is TCP before going any further.
       Protocol is always the 10th byte of the IP header */
    struct ip_header h {};
    fill_ip_header(ip_header, &h);
    u_char protocol = *(ip_header + 9);
    tcp_header = ip_header + ip_header_length;
    if (protocol == IPPROTO_TCP || protocol == IPPROTO_UDP) {
      // src port + dst port + seq number + ack number + header length
      // int offset = 2 + 2 + 4 + 4;
      //uint32_t bitmask = parse_two_byte_number(tcp_header + offset);
      //uint32_t mask_syn = 2;
      //uint32_t mask_ack = 16;
      // if((bitmask & mask_syn) > 0 && (bitmask & mask_ack) == 0) {
      uint128_t hash = flow_hash(ip_header, tcp_header);
      if(memory.find(hash) == memory.end()) {
        memory.insert(std::make_pair(hash, true));
        std::string ip;
        ip = std::to_string(ip_header[16]) + "." + std::to_string(ip_header[17]) + "." +
             std::to_string(ip_header[18]) + "." + std::to_string(ip_header[19]);
        if(debug > 0) { std::cout << "Frame " << pkt_count << " is a syn to destionation " << ip << std::endl; }
        if(domain_db->find(ip) != domain_db->end()) {
          for(const auto it : *domain_db->at(ip)) {
            resolutions << ip << ";" << it << std::endl;
          }
        }
      }
    }
  }
  resolutions.close();
  pcap_close(handle);
  return 0;
}


int retrieve_domain_names(const char * file_name, const std::string &result_file, domain_db_t *domain_db, const int debug) {
  __time_t first_packet = 0;
  char error_buffer[PCAP_ERRBUF_SIZE];
  pcap_t *handle;
  const u_char *packet;
  struct pcap_pkthdr header{};
  std::ofstream resolutions;
  resolutions.open(result_file, std::ios::out);
  resolutions << "type;name;cname" << std::endl;
  handle = pcap_open_offline(file_name, error_buffer);
  if (handle == nullptr) {
    fprintf(stderr, "Could not open files: %s\n", error_buffer);
    return 2;
  }
  int pkt_count = 0;
  if (debug == 2) { std::cout << "Opened file " << file_name << std::endl; }
  while((packet = pcap_next(handle, &header)) != nullptr) {
    pkt_count++;
    if (debug == 1) { std::cout << std::endl << "######## Process Frame " << pkt_count << " ######" << std::endl; }
    if (first_packet == 0) {
      first_packet = header.ts.tv_sec;
    }
    if (header.ts.tv_sec - first_packet > 7) {
      break;
    }
    struct ether_header *eth_header;
    eth_header = (struct ether_header *) packet;
    if (ntohs(eth_header->ether_type) != ETHERTYPE_IP) {
      if (debug == 2) { printf("Not an IP packet. Skipping...\n\n"); }
      continue;
    }

    /* The total packet length, including all headers
       and the data payload is stored in
       header->len and header->caplen. Caplen is
       the amount actually available, and len is the
       total packet length even if it is larger
       than what we currently have captured. If the snapshot
       length set with pcap_open_live() is too small, you may
       not have the whole packet. */
    if (debug == 2) { printf("Total packet available: %d bytes\n", header.caplen); }
    if (debug == 2) { printf("Expected packet size: %d bytes\n", header.len); }

    /* Pointers to start point of various headers */
    const u_char *ip_header;
    const u_char *udp_header;
    const u_char *dns_header;
    const u_char *start_replies;

    /* Header lengths in bytes */
    int ethernet_header_length = 14; /* Doesn't change */
    int ip_header_length;

    /* Find start of IP header */
    ip_header = packet + ethernet_header_length;
    /* The second-half of the first byte in ip_header
       contains the IP header length (IHL). */
    ip_header_length = ((*ip_header) & 0x0F);
    /* The IHL is number of 32-bit segments. Multiply
       by four to get a byte count for pointer arithmetic */
    ip_header_length = ip_header_length * 4;
    if (debug == 2) { printf(" IHL in bytes: %d\n", ip_header_length); }

    /* Now that we know where the IP header is, we can
       inspect the IP header for a protocol number to
       make sure it is TCP before going any further.
       Protocol is always the 10th byte of the IP header */
    struct ip_header h {};
    fill_ip_header(ip_header, &h);
    u_char protocol = *(ip_header + 9);
    udp_header = ip_header + ip_header_length;
    dns_header = udp_header + 8;
    if(protocol == IPPROTO_UDP && is_dns(udp_header) && is_dns_response(dns_header)) {
      uint32_t num_questions = parse_two_byte_number(dns_header + 4);
      uint32_t num_replies = parse_two_byte_number(dns_header + 6);
      std::string qname, name, cname;
      int length = parse_name(dns_header + 12, qname, dns_header);
      if(debug > 0) {
        std::cout << "Packet " << pkt_count << " is a DNS packet " << "and response: " << is_dns_response(dns_header)
                  << " is errorenous: " << is_dns_failure(dns_header);
        std::cout << " With " << num_questions << " Questions and " << num_replies << " Replies" << std::endl;
        std::cout << "Query name is " << qname << " with a total length of " << length << " Bytes";
        std::cout << std::endl;
        std::cout << "Replies are: " << std::endl;
      }
      start_replies = dns_header + 12 + length + 4;
      uint32_t offset = 0;
      uint32_t m = 0x3fff;
      for(uint i = 0; i < num_replies; i++) {
        name.clear();
        if(uses_dns_compression(start_replies + offset)) {
          uint32_t ptr = parse_two_byte_number(start_replies + offset) & m;
          parse_name(dns_header + ptr, name, dns_header);
          length = 2;
        } else {
          length = parse_name(start_replies + offset, name, dns_header);
        }
        uint32_t data_type = parse_two_byte_number(start_replies + offset + length);
        uint32_t data_length = parse_two_byte_number(start_replies + offset + length + 8);
        if(data_type == 5) {
          cname.clear();
          parse_name(start_replies + offset + length + 10, cname, dns_header);
        } else if(data_type == 1) {
          cname = std::to_string(start_replies[offset + length + 10]) + "." +
                  std::to_string(start_replies[offset + length + 10 + 1]) + "." +
                  std::to_string(start_replies[offset + length + 10 + 2]) + "." +
                  std::to_string(start_replies[offset + length + 10 + 3]);

          if (domain_db->find(cname) == domain_db->end()) {
            auto container = new std::vector<std::string>();
            domain_db->emplace(std::make_pair(cname, container));
            domain_db->at(cname)->push_back(qname);
          }
        }
        if(debug > 0) {
          std::cout << "\tReply is " << name << " with type " << data_type << " and length " << data_length
                    << " maps to " << cname << "  offset is " << offset << " + " << length + 10 + data_length
                    << std::endl;
        }
        resolutions << data_type << ";" << name << ";" << cname << std::endl;
        offset += length + 10 + data_length;
      }
    } else if (protocol == IPPROTO_TCP) {
      continue;
    }
  }
  resolutions.close();
  pcap_close(handle);
  return 0;
}


int main(int argc, char **argv) {
  if( argc < 4) {
    std::cerr << "Expected three parameters, got two: <path-to-trace> <path-to-result-dir> <result-file-prefix>" << std::endl;
  }
  //std::cout << "Analyse file " << argv[1] << " and write results to folder " << argv[2] << ", filenames prefixed with " << argv[3] << std::endl;
  domain_db_t domain_db{};
  domain_db_t flow_db{};
  std::string result_file = std::string(argv[2]) + std::string(argv[3]) +"-dns-resolutions.csv";
  //std::cout << "write dns resolutions to " << result_file << std::endl;
  // char file_name[] = "/home/patrick/Documents/GitHub/lkn/tls-fingerprint/data/devel-traces/instagram_chromium_gatherer-01-4tss9_91246581.pcapng";
  retrieve_domain_names(argv[1], result_file, &domain_db, 1);
  //std::cout << "Found " << domain_db.size() << " entries" << std::endl;
  result_file = std::string(argv[2]) + std::string(argv[3]) +"-ip-resolutions.csv";
  //std::cout << "write ip resolutions to " << result_file << std::endl;
  retrieve_flows(argv[1], result_file, &domain_db, &flow_db, 1);
  for(const auto& it : domain_db) {
    free(it.second);
  }for(const auto& it : flow_db) {
    free(it.second);
  }
}
