#include "array_func.h"
#include <stdlib.h>
#include <math.h>

/*
 * Returns the probability in the emission matrix given the observation and the state, if the obseravtion is not
 *      in the emission matrix returns a small probability i.e. 1e-12
 * 
 * Args:
 *  obs_matrix: pointer to the emission matrix
 *  obs: observation
 *  type_tail: Node type of the tail {0: delete, 1: match, 2: insert}
 *  x_tail: layer index of the tail node, in {1, ..., hmm_duration}
 *  hmm_duration: Total length of HMM
 */
double get_obs(double* obs_matrix, int obs, int type_tail, int x_tail, int hmm_duration)
{
    if(obs == -1)
    {
        return 1e-12;
    }
    else
    {
        return obs_matrix[p_o_in_i(obs, type_tail, x_tail, hmm_duration)];
    }
}

/*
 * Calculates the alpha values out of the given sequence and the probability matrices
 * 
 * Args:
 *  trans_matrix: pointer to transition matrix of HMM
 *  obs_matrix: pointer to emission matrix of HMM
 *  sequence: pointer to the whole sequence, can contain multiple subsequences
 *  seq_offset: beginning of subsequence
 *  len_seq: length of subsequence
 *  hmm_duration: total length of hmm
 */
double* forward(double* trans_matrix, double* obs_matrix, int* sequence, int seq_offset, int len_seq, int hmm_duration)
{
    double* alphas = (double*)malloc((len_seq * (3 * hmm_duration + 1) + hmm_duration + 2) * sizeof(double));
    double a_t;

    alphas[0] = 1.0;
    alphas[1] = trans_matrix[0];

    for(int i = 2; i < hmm_duration + 1; i++)
    {
        alphas[i] = alphas[i - 1] * trans_matrix[3 * (i - 1)];
    }
    alphas[2 * hmm_duration + 1] = trans_matrix[1] * get_obs(obs_matrix, sequence[seq_offset], 1, 1, hmm_duration);
    alphas[3 * hmm_duration + 1] = trans_matrix[2] * get_obs(obs_matrix, sequence[seq_offset], 2, 0, hmm_duration);
    for(int i = 2; i < hmm_duration + 1; i++)
    {
        alphas[2 * hmm_duration + i] = alphas[i - 1] * trans_matrix[3 * i - 2] * get_obs(obs_matrix, sequence[seq_offset], 1, i, hmm_duration);
    }
    for(int i = 1; i < hmm_duration; i++)
    {
        alphas[3 * hmm_duration + i + 1] = alphas[i] * trans_matrix[3 * i + 2] * get_obs(obs_matrix, sequence[seq_offset], 2, i, hmm_duration);
    }
    alphas[4 * hmm_duration + 1] = alphas[hmm_duration] * trans_matrix[3 * hmm_duration] * get_obs(obs_matrix, sequence[seq_offset], 2, hmm_duration, hmm_duration);
    alphas[hmm_duration + 1] = alphas[3 * hmm_duration + 1] * trans_matrix[6 * hmm_duration + 1];
    for(int i = 2; i < hmm_duration + 1; i++)
    {
        a_t = alphas[hmm_duration + i - 1] * trans_matrix[3 * (i - 1)];
        a_t += alphas[2 * hmm_duration + i - 1] * trans_matrix[3 * (hmm_duration + i) - 4];
        alphas[hmm_duration + i] = a_t + alphas[3 * hmm_duration + i] * trans_matrix[6 * hmm_duration + 3 * i - 2];
    }
    for(int t = 1; t < len_seq; t++)
    {
        alphas[t * (3 * hmm_duration + 1) + 2 * hmm_duration + 1] = alphas[t * (3 * hmm_duration + 1)] * trans_matrix[6 * hmm_duration + 2] * get_obs(obs_matrix, sequence[seq_offset + t], 1, 1, hmm_duration);
        alphas[t * (3 * hmm_duration + 1) + 3 * hmm_duration + 1] = alphas[t * (3 * hmm_duration + 1)] * trans_matrix[6 * hmm_duration + 3] * get_obs(obs_matrix, sequence[seq_offset + t], 2, 0, hmm_duration);
        for(int i = 2; i < hmm_duration + 1; i++)
        {
            a_t = alphas[t * (3 * hmm_duration + 1) + i - 2 * hmm_duration - 2] * trans_matrix[3 * (i - 1) + 1];
            a_t += alphas[t * (3 * hmm_duration + 1) + i - 1] * trans_matrix[6 * hmm_duration + 3 * i - 1];
            a_t += alphas[t * (3 * hmm_duration + 1) + i - hmm_duration - 2] * trans_matrix[3 * (hmm_duration + i) - 3];
            alphas[t * (3 * hmm_duration + 1) + 2 * hmm_duration + i] = a_t * get_obs(obs_matrix, sequence[seq_offset + t], 1, i, hmm_duration);
        }
        for(int i = 1; i < hmm_duration; i++)
        {
            a_t = alphas[t * (3 * hmm_duration + 1) + i - 2 * hmm_duration - 1] * trans_matrix[3 * i + 2];
            a_t += alphas[t * (3 * hmm_duration + 1) + i - hmm_duration - 1] * trans_matrix[3 * (hmm_duration + i) + 1];
            a_t += alphas[t * (3 * hmm_duration + 1) + i] * trans_matrix[6 * hmm_duration + 3 * (i + 1)];
            alphas[t * (3 * hmm_duration + 1) + 3 * hmm_duration + i + 1] = a_t * get_obs(obs_matrix, sequence[seq_offset + t], 2, i, hmm_duration);
        }
        a_t = alphas[t * (3 * hmm_duration + 1) - hmm_duration - 1] * trans_matrix[3 * hmm_duration];
        a_t += alphas[t * (3 * hmm_duration + 1) - 1] * trans_matrix[6 * hmm_duration - 1];
        a_t += alphas[t * (3 * hmm_duration + 1) + hmm_duration] * trans_matrix[9 * hmm_duration + 1];
        alphas[t * (3 * hmm_duration + 1) + 4 * hmm_duration + 1] = a_t * get_obs(obs_matrix, sequence[seq_offset + t], 2, hmm_duration, hmm_duration);
        alphas[t * (3 * hmm_duration + 1) + hmm_duration + 1] = alphas[t * (3 * hmm_duration + 1) + 3 * hmm_duration + 1] * trans_matrix[6 * hmm_duration + 1];
        for(int i = 2; i < hmm_duration + 1; i++)
        {
            a_t = alphas[t * (3 * hmm_duration + 1) + hmm_duration + i - 1] * trans_matrix[3 * (i - 1)];
            a_t += alphas[t * (3 * hmm_duration + 1) + 2 * hmm_duration + i - 1] * trans_matrix[3 * (hmm_duration + i) - 4];
            alphas[t * (3 * hmm_duration + 1) + hmm_duration + i] = a_t + alphas[t * (3 * hmm_duration + 1) + 3 * hmm_duration + i] * trans_matrix[6 * hmm_duration + 3 * i - 2];
        }
    }
    a_t = alphas[len_seq * (3 * hmm_duration + 1) - hmm_duration - 1] * trans_matrix[3 * hmm_duration + 1];
    a_t += alphas[len_seq * (3 * hmm_duration + 1) + hmm_duration] * trans_matrix[9 * hmm_duration + 2];
    alphas[len_seq * (3 *  hmm_duration + 1) + hmm_duration + 1] = a_t + alphas[len_seq * (3 * hmm_duration + 1) - 1] * trans_matrix[6 * hmm_duration];

    return alphas;
}

/*
 * Calculates the loglikelihood of a sequence given a HMM
 *
 * Args:
 *  trans_matrix: pointer to transition matrix of HMM
 *  obs_matrix: pointer to emission matrix of HMM
 *  sequence: pointer to the whole sequence, can contain multiple subsequences
 *  seq_offset: beginning of subsequence
 *  len_seq: length of subsequence
 *  hmm_duration: total length of hmm
 */
double calc_log_prob(double* trans_matrix, double* obs_matrix, int* sequence, int seq_offset, int len_seq, int hmm_duration)
{
    double* alphas = forward(trans_matrix, obs_matrix, sequence, seq_offset, len_seq, hmm_duration);
    double log_prob = log(alphas[len_seq * (3 *  hmm_duration + 1) + hmm_duration + 1]);
    free(alphas);
    
    return log_prob;
}
