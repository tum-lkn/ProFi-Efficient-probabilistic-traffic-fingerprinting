#include "array_func.h"
#include <stdlib.h>

/*
 * Calculates the beta values out of the given sequence and the probability matrices
 *
 * Args:
 *  trans_matrix: pointer to transition matrix of HMM
 *  obs_matrix: pointer to emission matrix of HMM
 *  sequence: pointer to the whole sequence, can contain multiple subsequences
 *  seq_offset: beginning of subsequence
 *  len_seq: length of subsequence
 *  hmm_duration: total length of hmm
 */
double* backward(double* trans_matrix, double* obs_matrix, int* sequence, int seq_offset, int len_seq, int hmm_duration)
{
    double* betas = (double*)malloc((len_seq * (3 * hmm_duration + 1) + hmm_duration + 2) * sizeof(double));
    double p_o;
    double b_t;
    
    betas[len_seq * (3 * hmm_duration + 1) + hmm_duration + 1] = 1.0;
    betas[len_seq * (3 * hmm_duration + 1) - hmm_duration - 1] = trans_matrix[3 * hmm_duration + 1];
    betas[len_seq * (3 * hmm_duration + 1) - 1] = trans_matrix[6 * hmm_duration];
    betas[len_seq * (3 * hmm_duration + 1) + hmm_duration] = trans_matrix[9 * hmm_duration + 2];
    for(int i = hmm_duration - 1; i > 0; i--)
    {
        betas[len_seq * (3 * hmm_duration + 1) + i - 2 * hmm_duration - 1] = betas[len_seq * (3 * hmm_duration + 1) + i - 2 * hmm_duration] * trans_matrix[3 * i];
        betas[len_seq * (3 * hmm_duration + 1) + i - hmm_duration - 1] = betas[len_seq * (3 * hmm_duration + 1) + i - 2 * hmm_duration] * trans_matrix[3 * (hmm_duration + i) - 1];
    }
    for(int i = hmm_duration - 1; i > -1; i--)
    {
        betas[len_seq * (3 * hmm_duration + 1) + i] = betas[len_seq * (3 * hmm_duration + 1) + i - 2 * hmm_duration] * trans_matrix[6 * hmm_duration + 3 * i + 1];
    }
    for(int t = len_seq - 2; t > -1; t--)
    {
        p_o = obs_matrix[sequence[seq_offset + t + 1] * (2 * hmm_duration + 1) + 2 * hmm_duration];
        b_t = betas[t * (3 * hmm_duration + 1) + 7 * hmm_duration + 2];
        betas[t * (3 * hmm_duration + 1) + 2 * hmm_duration] = b_t * trans_matrix[3 * hmm_duration] * p_o;
        betas[t * (3 * hmm_duration + 1) + 4 * hmm_duration + 1] = b_t * trans_matrix[9 * hmm_duration + 1] * p_o;
        betas[t * (3 * hmm_duration + 1) + 3 * hmm_duration] = b_t * trans_matrix[6 * hmm_duration - 1] * p_o;
        for(int i = hmm_duration - 1; i > 0; i--)
        {
            b_t = betas[t * (3 * hmm_duration + 1) + 5 * hmm_duration + i + 2] * trans_matrix[3 * i + 1] * obs_matrix[sequence[seq_offset + t + 1] * (2 * hmm_duration + 1) + i];
            b_t += betas[t * (3 * hmm_duration + 1) + 6 * hmm_duration + i + 2] * trans_matrix[3 * i + 2] * obs_matrix[sequence[seq_offset + t + 1] * (2 * hmm_duration + 1) + hmm_duration + i];
            betas[t * (3 * hmm_duration + 1) + hmm_duration + i] = b_t + betas[t * (3 * hmm_duration + 1) + hmm_duration + i + 1] * trans_matrix[3 * i];
        }
        for(int i = hmm_duration - 1; i > -1; i--)
        {
            b_t = betas[t * (3 * hmm_duration + 1) + 5 * hmm_duration + i + 2] * trans_matrix[6 * hmm_duration + 3 * i + 2] * obs_matrix[sequence[seq_offset + t + 1] * (2 * hmm_duration + 1) + i];
            b_t += betas[t * (3 * hmm_duration + 1) + 6 * hmm_duration + i + 2] * trans_matrix[6 * hmm_duration + 3 * (i + 1)] * obs_matrix[sequence[seq_offset + t + 1] * (2 * hmm_duration + 1) + hmm_duration + i];
            betas[t * (3 * hmm_duration + 1) + 3 * hmm_duration + i + 1] = b_t + betas[t * (3 * hmm_duration + 1) + 4 * hmm_duration + i + 2] * trans_matrix[6 * hmm_duration + 3 * i + 1];
        }
        for(int i = hmm_duration - 1; i > 0; i--)
        {
            b_t = betas[t * (3 * hmm_duration + 1) + 5 * hmm_duration + i + 2] * trans_matrix[3 * (hmm_duration + i)] * obs_matrix[sequence[seq_offset + t + 1] * (2 * hmm_duration + 1) + i];
            b_t += betas[t * (3 * hmm_duration + 1) + 6 * hmm_duration + i + 2] * trans_matrix[3 * (hmm_duration + i) + 1] * obs_matrix[sequence[seq_offset + t + 1] * (2 * hmm_duration + 1) + hmm_duration + i];
            betas[t * (3 * hmm_duration + 1) + 2 * hmm_duration + i] = b_t + betas[t * (3 * hmm_duration + 1) + hmm_duration + i + 1] * trans_matrix[3 * (hmm_duration + i) - 1];
        }
    }
    betas[hmm_duration] = betas[4 * hmm_duration + 1] * trans_matrix[3 * hmm_duration] * obs_matrix[sequence[seq_offset] * (2 * hmm_duration + 1) + 2 * hmm_duration];
    for(int i = hmm_duration - 1; i > 0; i--)
    {
        b_t = betas[2 * hmm_duration + i + 1] * trans_matrix[3 * i + 1] * obs_matrix[sequence[seq_offset] * (2 * hmm_duration + 1) + i];
        b_t += betas[3 * hmm_duration + i + 1] * trans_matrix[3 * i + 2] * obs_matrix[sequence[seq_offset] * (2 * hmm_duration + 1) + hmm_duration + i];
        betas[i] = b_t + betas[i + 1] * trans_matrix[3 * i];
    }
    b_t = betas[1] * trans_matrix[0];
    b_t += betas[2 * hmm_duration + 1] * trans_matrix[1] * obs_matrix[sequence[seq_offset] * (2 * hmm_duration + 1)];
    betas[0] = b_t + betas[3 * hmm_duration + 1] * trans_matrix[2] * obs_matrix[sequence[seq_offset] * (2 * hmm_duration + 1) + hmm_duration];

    return betas;
}
