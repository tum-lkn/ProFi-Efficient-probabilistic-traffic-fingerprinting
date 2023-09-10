#include "array_func.h"
#include <stdlib.h>
#include <math.h>

/*
 * Initializes an array to zero
 *
 * Args:
 *  array_len: length of array
 */
double* init_array(int array_len)
{
    double* array = (double*)malloc(array_len * sizeof(double));
    for(int i = 0; i < array_len; i++)
    {
        array[i] = 0.;
    }
    return array;
}

/*
 * Adds an array 2 to array 1, arrays have to have the same length
 *
 * Args:
 *  array_res: pointer to array 1
 *  array: pointer to array 2
 *  array_len: length of array 1 and 2
 */
void add_arrays(double* array_res, double* array, int array_len)
{
    for(int i = 0; i < array_len; i++)
    {
        array_res[i] += array[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * This function normalizes a probability distribution to be between [p_min, 1).
 * The array is assumed to hold multiple discrete probability distributions.
 * The attributes p_start and p_end are used to identify the start and end
 * of the distribution.
 *
 *  Args:
 *  - array: Pointer to a rectangular matrix.
 *  - p_start: Start index of a probability distribution.
 *  - p_end: End index (exclusive) of the probability distribution.
 *  - p_min: Minimum probability.
 *
 * Example of an array. The superscript identifies the probability distribution,
 * the subscript the ith entry of the distribution. The array holds three
 * distribution, the first with five elements, the second with two elements and
 * the third with seven elements:
 *
 * p^1_1 p^1_2 p^1_3 p^1_4 p^1_5 p^2_1 p^2_2 p^3_1 p^3_2 p^3_3 p^3_4 p^3_5 p^3_6 p^3_7
 */
void normalize_adv(double* array, int p_start, int p_end, double p_min)
{
    double num_zeros = 0;
    double denominator = 0;
    double pseudocount,
           value;
    for(int i = p_start; i < p_end; i++)
    {
        if(array[i] < p_min)
        {
            num_zeros = num_zeros + 1;
        } 
        else
        {
            denominator = denominator + array[i];
        }
    }
    if(num_zeros > 0)
    {
        pseudocount = p_min * denominator / (1 - p_min * num_zeros);
        for(int i = p_start; i < p_end; i++) {
            if(array[i] < p_min)
            {
                value = pseudocount;
            } 
            else
            {
                value = array[i];
            }
            array[i] = value / (denominator + (num_zeros * pseudocount));
        }
    }
}

/*
 * This function normalizes a probability distribution to be between [p_min, 1).
 * Note that the matrix is row-major and the distributions are stored in the
 * columns of the matrix.
 *
 *  Args:
 *  - array: Pointer to a rectangular matrix.
 *  - col: Column index we are at.
 *  - n_rows: Number of rows in the matrix.
 *  - n_cols: Number of columns in the matrix.
 *  - p_min: Minimum probability.
 *
 * Example of a matrix. The superscript identifies the probability distribution,
 * the subscript the ith entry of the distribution.
 *     p^1_1 p^2_1, p^3_1
 *     p^1_2 p^2_2, p^3_2
 *     p^1_3 p^2_3, p^3_3
 *     p^1_4 p^2_4, p^3_4
 *     p^1_5 p^2_5, p^3_5
 * Which is stored as
 *     p^1_1 p^2_1, p^3_1 p^1_2 p^2_2, p^3_2 p^1_3 p^2_3, p^3_3 p^1_4 p^2_4, p^3_4 p^1_5 p^2_5, p^3_5
 */
void normalize_adv2(double* array, int col, int n_rows, int n_cols, double p_min)
{
    double num_zeros = 0.;
    double denominator = 0.;
    double pseudocount,
           value;
    int idx;
    for(int i = 0; i < n_rows; i++)
    {
        // Jump to the column and skip the elements in the row to descend
        // along the rows.
        idx = col + i * n_cols;
        if(array[idx] < p_min)
        {
            num_zeros = num_zeros + 1;
        } 
        else
        {
            denominator = denominator + array[idx];
        }
    }
    if(num_zeros > 0)
    {
        pseudocount = p_min * denominator / (1 - p_min * num_zeros);
        for(int i = 0; i < n_rows; i++)
        {
            idx = col + i * n_cols;
            if(array[idx] < p_min)
            {
                value = pseudocount;
            }
            else
            {
                value = array[idx];
            }
            array[idx] = value / (denominator + (num_zeros * pseudocount));
        }
    }
}

/*
 * Normalize the emission probabilities such that they are in the interval
 * [1e-6, 1), i.e., the minimum probability is 1e-6.
 *
 * Args:
 *  - emissions: Pointer to the memory region in which the emissions reside.
 *  - hmm_length: The length/duration of the HMM.
 *  - obs_space_s: The size of the observation space, i.e., the number of
 *    symbols in the observation space.
 */
void normalize_emissions(double* emissions, int hmm_length, int obs_space_s)
{
    int n_cols = 2 * hmm_length + 1;
    for(int col = 0; col < n_cols; col++)
    {
        normalize_adv2(emissions, col, obs_space_s, n_cols, 1e-6);
    }
}

/*
 * Normalize the transition probabilities such that they are in the interval
 * [1e-6, 1), i.e., the minimum probability is 1e-6.
 *
 * Args:
 *  - transitions: Pointer to the memory region in which the transitions reside.
 *  - hmm_length: The length/duration of the HMM.
 *
 * Note:
 * The format of the transitions is as follows: START | DELETES | MATCHES | INSERTS.
 * The first three entries in the array correspond to the transition probabilities
 * from the START state. The next entries are the transition probabilities from
 * all DELETE states. That is, transitions starting at d_1, d_2, ..., d_T.
 * Each of those states has three transitions, except the last state which has
 * only two transitions. Then all MATCH states follow in the same format. Finally,
 * all insert states follow. Note that there is one more insert state than there
 * are delete and match states.
 */
void normalize_transitions(double* transitions, int hmm_length)
{
    int p_start, p_end;
    int offset = 3;
    double p_min = 1e-6;
    // Normalize the starte state.
    normalize_adv(transitions, 0, 3, p_min);

    // Normalize the delete states.
    for(int t = 0; t < hmm_length - 1; t++)
    {
        p_start = t * 3 + offset;
        p_end = (t + 1) * 3 + offset;
        normalize_adv(transitions, p_start, p_end, p_min);
    }
    // Handle the transition from last delete to End state differently.
    p_start = (hmm_length - 1) * 3 + offset;
    p_end = hmm_length * 3 - 1 + offset;
    normalize_adv(transitions, p_start, p_end, p_min);

    // Normalize the match states.
    offset += (hmm_length - 1) * 3 + 2;
    for(int t = 0; t < hmm_length - 1; t++)
    {
        p_start = t * 3 + offset;
        p_end = (t + 1) * 3 + offset;
        normalize_adv(transitions, p_start, p_end, p_min);
    }
    // Handle the transition from last match to End state differently.
    p_start = (hmm_length - 1) * 3 + offset;
    p_end = hmm_length * 3 - 1 + offset;
    normalize_adv(transitions, p_start, p_end, p_min);

    // Normalize the insert states.
    offset += (hmm_length - 1) * 3 + 2;
    for(int t = 0; t < hmm_length; t++)
    {
        p_start = t * 3 + offset;
        p_end = (t + 1) * 3 + offset;
        normalize_adv(transitions, p_start, p_end, p_min);
    }
    // Handle the transition from last match to End state differently.
    p_start = hmm_length * 3 + offset;
    p_end = (hmm_length + 1) * 3 - 1 + offset;
    normalize_adv(transitions, p_start, p_end, p_min);
}

/*
 * Normalize an array with the number of sequences
 *
 * Args:
 *  array: pointer to the array
 *  array_len: length of the array
 *  num_seq: number of sequences
 */
void normalize(double* array, int array_len, int num_seq)
{
    double norm = (double) num_seq;
    for(int i = 0; i < array_len; i++)
    {
        array[i] /= norm;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Calculates the loglikelihood of each sequence
 *
 * Args:
 *  trans_matrix: pointer to transition matrix of HMM
 *  obs_matrix: pointer to emission matrix of HMM
 *  sequence: pointer to the whole sequence, can contain multiple subsequences
 *  seq_lengths: pointer to the lengths of each sequence
 *  num_seq: number of sequences
 *  hmm_duration: total length of hmm
 */
double* calc_log_prob_seq(double* trans_matrix, double* obs_matrix, int* sequence, int* seq_lengths, int num_seq, int hmm_duration)
{
    double* log_prob = (double*)malloc(num_seq * sizeof(double));
    int seq_offset = 0;
    for(int i = 0; i < num_seq; i++)
    {
        log_prob[i] = calc_log_prob(trans_matrix, obs_matrix, sequence, seq_offset, seq_lengths[i], hmm_duration);
        seq_offset += seq_lengths[i];
    }
    return log_prob;
}

/*
 *  Sums the loglikelihood of each sequence
 *
 * Args:
 *  log_probs: pointer to the log_probs
 *  num_seq: number of sequences
 */
double sum_log_prob(double* log_probs, int num_seq)
{
    double log_prob = 0.;
    for(int i = 0; i < num_seq; i++)
    {
        log_prob += log_probs[i];
    }
    return log_prob;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 *  Calculates the transistion matrix out of the eta values
 *
 * Args:
 *  etas: pointer to eta values
 *  len_seq: length of the sequence
 *  hmm_duration: length of the hmm
 */
double* calc_transitions(double* etas, int len_seq, int hmm_duration)
{
    double* nom = init_array(9 * hmm_duration + 3);
    nom[0] = etas[0];
    nom[1] = etas[1];
    nom[2] = etas[2];

    for(int i = 1; i < hmm_duration; i++)
    {
        nom[p_ij(0, i, 0, hmm_duration)] += etas[e_i(0, i, 0, hmm_duration)];
        nom[p_ij(0, i, 1, hmm_duration)] += etas[e_i(0, i, 1, hmm_duration)];
        nom[p_ij(0, i, 2, hmm_duration)] += etas[e_i(0, i, 2, hmm_duration)];
    }
    nom[p_ij(0, hmm_duration, 2, hmm_duration)] += etas[e_i(0, hmm_duration, 2, hmm_duration)];
    for(int t = 0; t < len_seq - 1; t++)
    {
        nom[p_ij(2, 0, 0, hmm_duration)] += etas[e_b(t, 2, 0, 0, 1, hmm_duration)];
        nom[p_ij(2, 0, 1, hmm_duration)] += etas[e_b(t, 2, 0, 1, 1, hmm_duration)];
        nom[p_ij(2, 0, 2, hmm_duration)] += etas[e_b(t, 2, 0, 2, 0, hmm_duration)];
        for(int i = 2; i < hmm_duration + 1; i++)
        {
            nom[p_ij(0, i - 1, 0, hmm_duration)] += etas[e_b(t, 0, i - 1, 0, i, hmm_duration)];
            nom[p_ij(1, i - 1, 0, hmm_duration)] += etas[e_b(t, 1, i - 1, 0, i, hmm_duration)];
            nom[p_ij(2, i - 1, 0, hmm_duration)] += etas[e_b(t, 2, i - 1, 0, i, hmm_duration)];
            nom[p_ij(0, i - 1, 1, hmm_duration)] += etas[e_b(t, 0, i - 1, 1, i, hmm_duration)];
            nom[p_ij(1, i - 1, 1, hmm_duration)] += etas[e_b(t, 1, i - 1, 1, i, hmm_duration)];
            nom[p_ij(2, i - 1, 1, hmm_duration)] += etas[e_b(t, 2, i - 1, 1, i, hmm_duration)];
        }
        for(int i = 1; i < hmm_duration + 1; i++)
        {
            nom[p_ij(0, i, 2, hmm_duration)] += etas[e_b(t, 0, i, 2, i, hmm_duration)];
            nom[p_ij(1, i, 2, hmm_duration)] += etas[e_b(t, 1, i, 2, i, hmm_duration)];
            nom[p_ij(2, i, 2, hmm_duration)] += etas[e_b(t, 2, i, 2, i, hmm_duration)];
        }
    }
    nom[p_ij(2, 0, 0, hmm_duration)] += etas[e_e(len_seq - 1, 2, 0, 0, 1, hmm_duration)];
    for(int i = 2; i < hmm_duration + 1; i++)
    {
        nom[p_ij(0, i - 1, 0, hmm_duration)] += etas[e_e(len_seq - 1, 0, i - 1, 0, i, hmm_duration)];
        nom[p_ij(1, i - 1, 0, hmm_duration)] += etas[e_e(len_seq - 1, 1, i - 1, 0, i, hmm_duration)];
        nom[p_ij(2, i - 1, 0, hmm_duration)] += etas[e_e(len_seq - 1, 2, i - 1, 0, i, hmm_duration)];
    }
    nom[p_ij(0, hmm_duration, 3, hmm_duration)] += etas[e_e(len_seq - 1, 0, hmm_duration, 3, 0, hmm_duration)];
    nom[p_ij(1, hmm_duration, 3, hmm_duration)] += etas[e_e(len_seq - 1, 1, hmm_duration, 3, 0, hmm_duration)];
    nom[p_ij(2, hmm_duration, 3, hmm_duration)] += etas[e_e(len_seq - 1, 2, hmm_duration, 3, 0, hmm_duration)];

    return nom;
}

/*
 *  Calculates the emission matrix out of the gamma values
 *
 * Args:
 *  gamma: pointer to gamma values
 *  sequence: pointer to the sequence
 *  seq_offset: offset of subsequence in the sequence
 *  num_obs: number of different observation
 *  len_seq: length of the sequence
 *  hmm_duration: length of the hmm
 */
double* calc_emissions(double* gammas, int* sequence, int seq_offset, int num_obs, int len_seq, int hmm_duration)
{
    double* emissions = init_array(((2 * hmm_duration + 1) * num_obs));
    for(int t = 0; t < len_seq; t++)
    {
      emissions[p_o_in_i(sequence[seq_offset + t], 2, 0, hmm_duration)] += gammas[g_(t, 2, 0, hmm_duration)];
      for(int i = 1; i < hmm_duration + 1; i++)
      {
            emissions[p_o_in_i(sequence[seq_offset + t], 1, i, hmm_duration)] += gammas[g_(t, 1, i, hmm_duration)];
            emissions[p_o_in_i(sequence[seq_offset + t], 2, i, hmm_duration)] += gammas[g_(t, 2, i, hmm_duration)];
      }
    }
    return emissions;
}

/*
 * Calculates the denominator out of the gammas
 *
 * Args:
 *  gammas: pointer to the gamma values
 *  len_seq: length of the sequence
 *  hmm_duration: length of the hmm
 */
double* calc_denom(double* gammas, int len_seq, int hmm_duration)
{
    double* denom = init_array(3 * hmm_duration + 2);

    for(int i = 0; i < hmm_duration + 1; i++)
    {
        denom[i] = gammas[i];
    }
    for(int t = 0; t < len_seq; t++)
    {
        denom[2 * hmm_duration + 1] += gammas[g_(t, 2, 0, hmm_duration)];
        for(int i = 1; i < hmm_duration + 1; i++)
        {
            denom[i] += gammas[g_(t, 0, i, hmm_duration)];
            denom[hmm_duration + i] += gammas[g_(t, 1, i, hmm_duration)];
            denom[2 * hmm_duration + i + 1] += gammas[g_(t, 2, i, hmm_duration)];
        }
    }
    return denom;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Normlalizes the transition matrix by the denominator
 *
 * Args:
 *  transistions: pointer to the transisions
 *  denom: pointer to the denominator
 *  hmm_duration: length of hmm
 */
void calc_transitions_independent(double* transitions, double* denom, int hmm_duration)
{
    transitions[0] /= denom[0];
    transitions[1] /= denom[0];
    transitions[2] /= denom[0];
    for(int i = 1; i < hmm_duration; i++)
    {
        transitions[p_ij(0, i, 0, hmm_duration)] /= denom[i];
        transitions[p_ij(0, i, 1, hmm_duration)] /= denom[i];
        transitions[p_ij(0, i, 2, hmm_duration)] /= denom[i];
        transitions[p_ij(1, i, 0, hmm_duration)] /= denom[hmm_duration + i];
        transitions[p_ij(1, i, 1, hmm_duration)] /= denom[hmm_duration + i];
        transitions[p_ij(1, i, 2, hmm_duration)] /= denom[hmm_duration + i];
    }
    for(int i = 0; i < hmm_duration; i++)
    {
        transitions[p_ij(2, i, 0, hmm_duration)] /= denom[2 * hmm_duration + i + 1];
        transitions[p_ij(2, i, 1, hmm_duration)] /= denom[2 * hmm_duration + i + 1];
        transitions[p_ij(2, i, 2, hmm_duration)] /= denom[2 * hmm_duration + i + 1];
    }

    transitions[p_ij(0, hmm_duration, 2, hmm_duration)] /= denom[hmm_duration];
    transitions[p_ij(1, hmm_duration, 2, hmm_duration)] /= denom[2 * hmm_duration];
    transitions[p_ij(2, hmm_duration, 2, hmm_duration)] /= denom[3 * hmm_duration + 1];
    transitions[p_ij(0, hmm_duration, 3, hmm_duration)] /= denom[hmm_duration];
    transitions[p_ij(1, hmm_duration, 3, hmm_duration)] /= denom[2 * hmm_duration];
    transitions[p_ij(2, hmm_duration, 3, hmm_duration)] /= denom[3 * hmm_duration + 1];

}

/*
 * Normlalizes the emission matrix by the denominator
 *
 * Args:
 *  transistions: pointer to the transisions
 *  denom: pointer to the denominator
 *  num_obs: number of different observations
 *  hmm_duration: length of hmm
 */
void calc_emissions_independent(double* emissions, double* denom, int num_obs, int hmm_duration)
{
    for(int t = 0; t < num_obs; t++)
    {
      emissions[p_o_in_i(t, 2, 0, hmm_duration)] /= denom[2 * hmm_duration + 1];
      for(int i = 1; i < hmm_duration + 1; i++)
      {
        emissions[p_o_in_i(t, 1, i, hmm_duration)] /= denom[hmm_duration + i];
        emissions[p_o_in_i(t, 2, i, hmm_duration)] /= denom[2 * hmm_duration + i + 1];
      }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Calculates the baum_welch algorithm for one step
 * 
 * Args:
 *  trans_matrix: pointer to transition matrix of HMM
 *  obs_matrix: pointer to emission matrix of HMM
 *  sequence_train: pointer to the training sequences
 *  len_seq_train: pointer to the lengths of the training sequences
 *  num_seq_train: number of training sequences
 *  sequence_val: pointer to the validation sequences
 *  len_seq_val: pointer to the lengths of the validation sequences
 *  num_seq_val: number of validation sequences
 *  num_obs: number of different observations
 *  hmm_duration: length of the hmm
 */
double** baum_welch_step(double* trans_matrix, double* obs_matrix, int* sequence_train,
        int* len_seq_train, int num_seq_train, int* sequence_val, int* len_seq_val,
        int num_seq_val, int num_obs, int hmm_duration) {
    double* transitions_res = init_array(9 * hmm_duration + 3);
    double* emissions_res = init_array(((2 * hmm_duration + 1) * num_obs));

    int seq_offset = 0;
    for(int i = 0; i < num_seq_train; i++)
    {
        double* alphas = forward(trans_matrix, obs_matrix, sequence_train, seq_offset, len_seq_train[i], hmm_duration);
        double* betas = backward(trans_matrix, obs_matrix, sequence_train, seq_offset, len_seq_train[i], hmm_duration);
        double* etas = calc_etas(trans_matrix, obs_matrix, alphas, betas, sequence_train, seq_offset, len_seq_train[i], hmm_duration);
        free(alphas);
        free(betas);
        double* gammas = calc_gammas(etas, len_seq_train[i], hmm_duration);
        double* denom = calc_denom(gammas, len_seq_train[i], hmm_duration);
        double* transitions = calc_transitions(etas, len_seq_train[i], hmm_duration);
        double* emissions = calc_emissions(gammas, sequence_train, seq_offset, num_obs, len_seq_train[i], hmm_duration);
        calc_transitions_independent(transitions, denom, hmm_duration);
        calc_emissions_independent(emissions, denom, num_obs, hmm_duration);
        free(etas);
        add_arrays(transitions_res, transitions, 9 * hmm_duration + 3);
        add_arrays(emissions_res, emissions, (2 * hmm_duration + 1) * num_obs);
        free(gammas);
        free(transitions);
        free(emissions);
        free(denom);
        seq_offset += len_seq_train[i];
    }

    normalize(transitions_res, 9 * hmm_duration + 3, num_seq_train);
    normalize(emissions_res, (2 * hmm_duration + 1) * num_obs, num_seq_train);
    normalize_emissions(emissions_res, hmm_duration, num_obs);
    normalize_transitions(transitions_res, hmm_duration);

    double** res = (double**)malloc(4 * sizeof(double*));
    res[0] = transitions_res;
    res[1] = emissions_res;
    res[2] = calc_log_prob_seq(transitions_res, emissions_res, sequence_train, len_seq_train, num_seq_train, hmm_duration);
    res[3] = calc_log_prob_seq(transitions_res, emissions_res, sequence_val, len_seq_val, num_seq_val, hmm_duration);

    return res;
}

/*
 * Calculates the baum_welch algorithm for a given number of iterations
 * 
 * Args:
 *  trans_matrix: pointer to transition matrix of HMM
 *  obs_matrix: pointer to emission matrix of HMM
 *  sequence_train: pointer to the training sequences
 *  len_seq_train: pointer to the lengths of the training sequences
 *  num_seq_train: number of training sequences
 *  sequence_val: pointer to the validation sequences
 *  len_seq_val: pointer to the lengths of the validation sequences
 *  num_seq_val: number of validation sequences
 *  num_obs: number of different observations
 *  hmm_duration: length of the hmm
 *  iterations: number of iterations
 */
double** baum_welch(double* trans_matrix, double* obs_matrix, int* sequence_train, int* len_seq_train, int num_seq_train, int* sequence_val, int* len_seq_val, int num_seq_val, int num_obs, int hmm_duration, int iterations)
{
    double** res_best;
    double** res;
    double* transitions = trans_matrix;
    double* emissions = obs_matrix;
    double log_prob;
    double log_prob_best = -HUGE_VAL;

    for(int i = 0; i < iterations; i++)
    {
        res = baum_welch_step(transitions, emissions, sequence_train, len_seq_train, num_seq_train, sequence_val, len_seq_val, num_seq_val, num_obs, hmm_duration);
        transitions = res[0];
        emissions = res[1];
        log_prob = sum_log_prob(res[3], num_seq_val);
        if(log_prob > log_prob_best)
        {
            log_prob_best = log_prob;
            res_best = res;
        }
    }
    return res_best;
}
