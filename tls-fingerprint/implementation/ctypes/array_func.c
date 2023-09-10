#include "array_func.h"
#include <stdlib.h>

/*
 * Returns index of an delete state in the alphas/betas array.
 */
int d_(int t, int x_tail, int hmm_duration){return t * (3 * hmm_duration + 1) + hmm_duration + x_tail;}

/*
 * Returns index of an match state in the alphas/betas array.
 */
int m_(int t, int x_tail, int hmm_duration){return t * (3 * hmm_duration + 1) + 2 * hmm_duration + x_tail;}

/*
 * Returns index of an insert state in the alphas/betas array.
 */
int i_(int t, int x_tail, int hmm_duration){return t * (3 * hmm_duration + 1) + 3 * hmm_duration + x_tail + 1;}

/*
 * Returns the index of an element in the transition matrix.
 *
 * Args:
 *  type_tail: Node type of the tail {0: delete, 1: match, 2: insert}
 *  x_tail: layer index of the tail node.
 *  type_head: Node type of the tail {0: delete, 1: match, 2: insert}
 *  hmm_duration: Total length of HMM.
 *
 * Example:
 *      d_3 to m_4, hmm_duration: 4
 *      input: 0, 3, 1, 4
 *      should return: 10, calc: 3 * 3 + 1 = 10
 */
int p_ij(int type_tail, int x_tail, int type_head, int hmm_duration)
{
    // Calc general offset
    int offset = 3 * x_tail + type_head;

    // if last state to end reduce index by 2
    if(x_tail == hmm_duration)
        offset -= 2;

    switch (type_tail)
    {
        case 0:
            // Case delete state, return general offset
            return offset;
        case 1:
            // Case match state, add the number of transitions of start and delete states to general offset
            return offset + 3 * hmm_duration - 1;
        case 2:
            // Case insert state, add the number of transitions of start, delete and match states to general offset
            return offset + 6 * hmm_duration + 1;
    }

}

/*
 * Returns the index of a symbol given the emitting state.
 *
 * Args:
 *  obs: Symbol represented as an integer. Continuous space of ints,
 *         i.e., if observation space has size 100, obs is in {0, ..., 99}.
 *  type_tail: Node type of the tail {0: delete, 1: match, 2: insert}
 *  x_tail: layer index of the tail node, in {1, ..., hmm_duration}
 *  hmm_duration: Total length of HMM
 * 
 *  Example:
 *      3rd obs in state m_3
 *      input: 2, 1, 3, 4
 *      should return: 20, calc 2 * (2 * 4 + 1) + 3 - 1 = 20
 */
int p_o_in_i(int obs, int type_tail, int x_tail, int hmm_duration)
{
    // Calc general offset
    int offset = obs * (2 * hmm_duration + 1) + x_tail - 1;
    switch(type_tail)
    {
        case 1:
            // Case match state, return general offset
            return offset;
        case 2:
            // Case insert state, add number of match states + 1 to general offset
            return offset + hmm_duration + 1;
    }
}

/*
 * Returns the index of inital etas matrix given a pair of states (valid transistion) i.e. d_3 to m_4
 * 
 * Args:
 *  type_tail: Node of the tail {0: delete, 1: match, 2: insert}
 *  x_tail: layer index of the tail node, in {1, ..., hmm_duration}
 *  type_head: Node of the tail {0: delete, 1: match, 2: insert}
 *  hmm_duration: Total length of HMM
 */
int e_i(int type_tail, int x_tail, int type_head, int hmm_duration)
{
    if(x_tail == hmm_duration)
        return 3 * x_tail;

    return 3 * x_tail + type_head;
}

/*
 * Returns the index of between etas matrix given a pair of states (valid transistion) i.e. d_3 to m_4
 * 
 * Args:
 *  t: time index
 *  type_tail: Node of the tail {0: delete, 1: match, 2: insert}
 *  x_tail: layer index of the tail node, in {1, ..., hmm_duration}
 *  type_head: Node of the tail {0: delete, 1: match, 2: insert}
 *  hmm_duration: Total length of HMM
 */
int e_b(int t, int type_tail, int x_tail, int type_head, int x_head, int hmm_duration)
{
    int init_offset = t * (9 * hmm_duration - 3) +  3 * hmm_duration + 1;

    if(type_tail == 2 && x_tail == 0)
        return init_offset + type_head;

    int offset = init_offset + 3 * x_head + type_tail;

    switch(type_head)
    {
    case 0:
        return offset - 3;
    case 1:
        return offset + 3 * hmm_duration - 6;
    case 2:
        return offset + 6 * hmm_duration - 6;
    }
}

/*
 * Returns the index of end etas matrix given a pair of states (valid transistion) i.e. d_3 to m_4
 * 
 * Args:
 *  t: time index
 *  type_tail: Node of the tail {0: delete, 1: match, 2: insert}
 *  x_tail: layer index of the tail node, in {1, ..., hmm_duration}
 *  type_head: Node of the tail {0: delete, 1: match, 2: insert}
 *  hmm_duration: Total length of HMM
 */
int e_e(int t, int type_tail, int x_tail, int type_head, int x_head, int hmm_duration)
{
    int state_offset = t * (9 * hmm_duration - 3) + 3 * hmm_duration + 2;

    if(type_tail == 2 && x_tail == 0)
    {
        return state_offset - 1;
    }

    if(type_head == 3)
    {
        return state_offset + 3 * hmm_duration + type_tail - 3;
    }
    return state_offset + 3 * x_head + type_tail - 6;
}

/*
 * Returns the index of gamma matrix given a pair of states (valid transistion) i.e. d_3 to m_4
 * 
 * Args:
 *  t: time index
 *  type_tail: Node of the tail {0: delete, 1: match, 2: insert}
 *  x_tail: layer index of the tail node, in {1, ..., hmm_duration}
 *  hmm_duration: Total length of HMM
 */
int g_(int t, int type_tail, int x_tail, int hmm_duration)
{
    switch(type_tail)
    {
    case 0:
        return t * (3 * hmm_duration + 1) + hmm_duration + x_tail;
    case 1:
        return t * (3 * hmm_duration + 1) + 2 * hmm_duration + x_tail;
    case 2:
        return t * (3 * hmm_duration + 1) + 3 * hmm_duration + x_tail + 1;

    }
}

/*
 * frees the return pointer of the baum_welch algorithm
 * 
 * Args:
 *  ptr: pointer to the return list
 */
void free_mem(double** ptr)
{
    free(ptr[0]);
    free(ptr[1]);
    free(ptr[2]);
    free(ptr[3]);

    free(ptr);
}

void free_mem_single(double * ptr) {
    free(ptr);
}
