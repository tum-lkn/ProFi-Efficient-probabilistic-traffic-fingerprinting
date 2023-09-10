#include "array_func.h"
#include <stdlib.h>

/*
 * Calculates the eta values out of the alpha, beta values and probability matrices
 *
 * Args:
 *  trans_matrix: pointer to transition matrix of HMM
 *  obs_matrix: pointer to emission matrix of HMM
 *  alphas: pointer to alpha values
 *  betas: pointer to beta values
 *  sequence: pointer to the whole sequence, can contain multiple subsequences
 *  seq_offset: beginning of subsequence
 *  len_seq: length of subsequence
 *  hmm_duration: total length of hmm 
 */
double* calc_etas(double* trans_matrix, double* obs_matrix, double* alphas, double* betas, int* sequence, int seq_offset, int len_seq, int hmm_duration)
{
    double* etas = (double*)malloc(((len_seq - 1) * (hmm_duration * 9 - 3) + 2 * ((hmm_duration - 1) * 3 + 4)) * sizeof(double));
    double b_t;
    double p_o;

    etas[0] = betas[1] * trans_matrix[0];
    etas[1] = betas[m_(0, 1, hmm_duration)] * trans_matrix[1] * obs_matrix[p_o_in_i(sequence[seq_offset + 0], 1, 1, hmm_duration)];
    etas[2] = betas[i_(0, 0, hmm_duration)] * trans_matrix[2] * obs_matrix[p_o_in_i(sequence[seq_offset + 0], 2, 0, hmm_duration)];
    for(int i = 2; i < hmm_duration + 1; i++)
    {
        etas[e_i(0, i - 1, 0, hmm_duration)] = alphas[i - 1] * betas[i] * trans_matrix[p_ij(0, i - 1, 0, hmm_duration)];
        etas[e_i(0, i - 1, 1, hmm_duration)] = alphas[i - 1] * betas[m_(0, i, hmm_duration)] * trans_matrix[p_ij(0, i - 1, 1, hmm_duration)] * obs_matrix[p_o_in_i(sequence[seq_offset + 0], 1, i, hmm_duration)];
    }
    for(int i = 1; i < hmm_duration + 1; i++)
    {
        etas[e_i(0, i, 2, hmm_duration)] = alphas[i] * betas[i_(0, i, hmm_duration)] * trans_matrix[p_ij(0, i, 2, hmm_duration)] * obs_matrix[p_o_in_i(sequence[seq_offset + 0], 2, i, hmm_duration)];
    }
    for(int t = 0; t < len_seq - 1; t++)
    {
        etas[e_b(t, 2, 0, 0, 1, hmm_duration)] = alphas[i_(t, 0, hmm_duration)] * betas[d_(t, 1, hmm_duration)] * trans_matrix[p_ij(2, 0, 0, hmm_duration)];
        etas[e_b(t, 2, 0, 1, 1, hmm_duration)] = alphas[i_(t, 0, hmm_duration)] * betas[m_(t + 1, 1, hmm_duration)] * trans_matrix[p_ij(2, 0, 1, hmm_duration)] * obs_matrix[p_o_in_i(sequence[seq_offset + t + 1], 1, 1, hmm_duration)];
        etas[e_b(t, 2, 0, 2, 0, hmm_duration)] = alphas[i_(t, 0, hmm_duration)] * betas[i_(t + 1, 0, hmm_duration)] * trans_matrix[p_ij(2, 0, 2, hmm_duration)] * obs_matrix[p_o_in_i(sequence[seq_offset + t + 1], 2, 0, hmm_duration)];
        for(int i = 2; i < hmm_duration + 1; i++)
        {
            b_t = betas[d_(t, i, hmm_duration)];
            etas[e_b(t, 0, i - 1, 0, i, hmm_duration)] = alphas[d_(t, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(0, i - 1, 0, hmm_duration)];
            etas[e_b(t, 1, i - 1, 0, i, hmm_duration)] = alphas[m_(t, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(1, i - 1, 0, hmm_duration)];
            etas[e_b(t, 2, i - 1, 0, i, hmm_duration)] = alphas[i_(t, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(2, i - 1, 0, hmm_duration)];
        }
        for(int i = 2; i < hmm_duration + 1; i++)
        {
            b_t = betas[m_(t + 1, i, hmm_duration)];
            p_o = obs_matrix[p_o_in_i(sequence[seq_offset + t + 1], 1, i, hmm_duration)];
            etas[e_b(t, 0, i - 1, 1, i, hmm_duration)] = alphas[d_(t, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(0, i - 1, 1, hmm_duration)] * p_o;
            etas[e_b(t, 1, i - 1, 1, i, hmm_duration)] = alphas[m_(t, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(1, i - 1, 1, hmm_duration)] * p_o;
            etas[e_b(t, 2, i - 1, 1, i, hmm_duration)] = alphas[i_(t, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(2, i - 1, 1, hmm_duration)] * p_o;
        }
        for(int i = 1; i < hmm_duration + 1; i++)
        {
            b_t = betas[i_(t + 1, i, hmm_duration)];
            p_o = obs_matrix[p_o_in_i(sequence[seq_offset + t + 1], 2, i, hmm_duration)];
            etas[e_b(t, 0, i, 2, i, hmm_duration)] = alphas[d_(t, i, hmm_duration)] * b_t * trans_matrix[p_ij(0, i, 2, hmm_duration)] * p_o;
            etas[e_b(t, 1, i, 2, i, hmm_duration)] = alphas[m_(t, i, hmm_duration)] * b_t * trans_matrix[p_ij(1, i, 2, hmm_duration)] * p_o;
            etas[e_b(t, 2, i, 2, i, hmm_duration)] = alphas[i_(t, i, hmm_duration)] * b_t * trans_matrix[p_ij(2, i, 2, hmm_duration)] * p_o;
        }
    }
    etas[e_e(len_seq - 1, 2, 0, 0, 1, hmm_duration)] = alphas[i_(len_seq - 1, 0, hmm_duration)] * betas[d_(len_seq - 1, 1, hmm_duration)] * trans_matrix[p_ij(2, 0, 0, hmm_duration)];
    for(int i = 2; i < hmm_duration + 1; i++)
    {
        b_t = betas[d_(len_seq - 1, i, hmm_duration)];
        etas[e_e(len_seq - 1, 0, i - 1, 0, i, hmm_duration)] = alphas[d_(len_seq - 1, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(0, i - 1, 0, hmm_duration)];
        etas[e_e(len_seq - 1, 1, i - 1, 0, i, hmm_duration)] = alphas[m_(len_seq - 1, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(1, i - 1, 0, hmm_duration)];
        etas[e_e(len_seq - 1, 2, i - 1, 0, i, hmm_duration)] = alphas[i_(len_seq - 1, i - 1, hmm_duration)] * b_t * trans_matrix[p_ij(2, i - 1, 0, hmm_duration)];
    }
    etas[e_e(len_seq - 1, 0, hmm_duration, 3, 0, hmm_duration)] = alphas[d_(len_seq - 1, hmm_duration, hmm_duration)] * trans_matrix[p_ij(0, hmm_duration, 3, hmm_duration)];
    etas[e_e(len_seq - 1, 1, hmm_duration, 3, 0, hmm_duration)] = alphas[m_(len_seq - 1, hmm_duration, hmm_duration)] * trans_matrix[p_ij(1, hmm_duration, 3, hmm_duration)];
    etas[e_e(len_seq - 1, 2, hmm_duration, 3, 0, hmm_duration)] = alphas[i_(len_seq - 1, hmm_duration, hmm_duration)] * trans_matrix[p_ij(2, hmm_duration, 3, hmm_duration)];

    return etas;
}

/*
 * Calculates the gamma values out of the eta values
 * 
 * etas: pointer to eta values
 * len_seq: length of the sequence
 * hmm_duration: total length of the Hmm
 */
double* calc_gammas(double* etas, int len_seq, int hmm_duration)
{
    double* gammas = (double*)malloc((len_seq * (3 * hmm_duration + 1) + hmm_duration + 1) * sizeof(double));

    gammas[0] = etas[0] + etas[1] + etas[2];

    for(int i = 1; i < hmm_duration + 1; i++)
    {
        gammas[i] = etas[e_i(0, i, 0, hmm_duration)] + etas[e_i(0, i, 1, hmm_duration)] + etas[e_i(0, i, 2, hmm_duration)];
    }
    gammas[hmm_duration] = etas[e_i(0, hmm_duration, 2, hmm_duration)];
    for(int t = 0; t < len_seq - 1; t++)
    {
        for(int i = 1; i < hmm_duration + 1; i++)
        {
            gammas[g_(t, 0, i, hmm_duration)] = etas[e_b(t, 0, i, 0, i + 1, hmm_duration)] + etas[e_b(t, 0, i, 1, i + 1, hmm_duration)] + etas[e_b(t, 0, i, 2, i, hmm_duration)];
            gammas[g_(t, 1, i, hmm_duration)] = etas[e_b(t, 1, i, 0, i + 1, hmm_duration)] + etas[e_b(t, 1, i, 1, i + 1, hmm_duration)] + etas[e_b(t, 1, i, 2, i, hmm_duration)];
        }
        for(int i = 0; i < hmm_duration + 1; i++)
        {
            gammas[g_(t, 2, i, hmm_duration)] = etas[e_b(t, 2, i, 0, i + 1, hmm_duration)] + etas[e_b(t, 2, i, 1, i + 1, hmm_duration)] + etas[e_b(t, 2, i, 2, i, hmm_duration)];
        }
        gammas[g_(t, 0, hmm_duration, hmm_duration)] = etas[e_b(t, 0, hmm_duration, 2, hmm_duration, hmm_duration)];
        gammas[g_(t, 1, hmm_duration, hmm_duration)] = etas[e_b(t, 1, hmm_duration, 2, hmm_duration, hmm_duration)];
        gammas[g_(t, 2, hmm_duration, hmm_duration)] = etas[e_b(t, 2, hmm_duration, 2, hmm_duration, hmm_duration)];
    }
    gammas[g_(len_seq - 1, 2, 0, hmm_duration)] = etas[e_e(len_seq - 1, 2, 0, 0, 1, hmm_duration)];
    for(int i = 1; i < hmm_duration + 1; i++)
    {
        gammas[g_(len_seq - 1, 0, i, hmm_duration)] = etas[e_e(len_seq - 1, 0, i, 0, i + 1, hmm_duration)];
        gammas[g_(len_seq - 1, 1, i, hmm_duration)] = etas[e_e(len_seq - 1, 1, i, 0, i + 1, hmm_duration)];
        gammas[g_(len_seq - 1, 2, i, hmm_duration)] = etas[e_e(len_seq - 1, 2, i, 0, i + 1, hmm_duration)];
    }
    gammas[g_(len_seq - 1, 0, hmm_duration, hmm_duration)] = etas[e_e(len_seq - 1, 0, hmm_duration, 3, 0, hmm_duration)];
    gammas[g_(len_seq - 1, 1, hmm_duration, hmm_duration)] = etas[e_e(len_seq - 1, 1, hmm_duration, 3, 0, hmm_duration)];
    gammas[g_(len_seq - 1, 2, hmm_duration, hmm_duration)] = etas[e_e(len_seq - 1, 2, hmm_duration, 3, 0, hmm_duration)];

    return gammas;
}