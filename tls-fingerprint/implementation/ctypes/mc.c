#include <stdio.h>
#include <stdint.h>
#include <math.h>

struct markov_chain {
    int num_symbols;
    int num_edges;
    int * offsets;
    uint64_t * tails;
    uint64_t * heads;
    double * log_probs;
};
typedef struct markov_chain markov_chain_t;


int binary_search(uint64_t * buffer, uint64_t val, int buffer_size) {
    // printf("First element in buffer in Binary Search is %lu\n", buffer[0]);
    printf("Find %lu in buffer of length %d, first element is %lu\n", val, buffer_size, buffer[0]);
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

double transition_probability(markov_chain_t * mc, uint64_t tail, uint64_t head) {
    int tail_idx = binary_search(mc->tails, tail, mc->num_symbols);
    if(tail_idx == -1) {
        return -1;
    }
    int offset = mc->offsets[tail_idx];
    int num_neighbors = mc->offsets[tail_idx + 1] - offset;
    int head_idx = binary_search(mc->heads + offset, head, num_neighbors);
    printf("Offset %d, num_neighbors %d, Tail Index (%lu) %d, Head Index (%lu) %d\n", offset, num_neighbors, tail, tail_idx, head, head_idx);
    printf("\t%lu %lu\n", mc->heads[offset], (mc->heads + offset)[0]);
    if(head_idx == -1) {
        return -13.815510557964274;
    } else {
        return mc->log_probs[offset + head_idx];
    }
}

double calc_log_prob(markov_chain_t * mc, uint64_t * sequence, uint32_t seq_length) {
    double lp = 0;
    for (int i = 0 ; i < seq_length; i++) {
        lp += transition_probability(mc, sequence[i], sequence[i + 1]);
    }
    return lp;
}


void calc_log_probs(markov_chain_t * mc, uint64_t * sequences, uint32_t * seq_lengths, uint32_t num_sequences, double * lps) {
    uint32_t offset = 0;
    for (int i = 0; i < num_sequences; i++) {
        lps[i] = calc_log_prob(mc, sequences + offset, seq_lengths[i]);
        offset += seq_lengths[i];
    }
}


void eval_population(markov_chain_t * mc, uint64_t * sequences, uint32_t * seq_lengths, uint32_t num_sequences, double * log_probs,
                     uint32_t population_size, double * utilities) {
    for(int i = 0; i < population_size; i++) {
        mc->log_probs = log_probs + i * mc->num_edges;
        calc_log_probs(mc, sequences, seq_lengths, num_sequences, utilities + i * mc->num_edges);
    }
}
