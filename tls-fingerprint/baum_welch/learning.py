"""
Implents algorithms for learning tasks on HMMs.
"""
import numpy as np
from typing import List, Tuple, Any, Dict, Union
import gc

from . import phmm
from .forward import forward
from .backward import backward
from .update import calc_transitions_independent, calc_emissions, calc_emissions_independent, calc_transitions
from .update import _calc_etas


def _normalize(alphas: Dict[str, float], scaler: float=None) -> float:
    if scaler is None:
        sum_alphas = 0.
        for v in alphas.values():
            sum_alphas += v
        scaler = 1. / sum_alphas
    for k, v in alphas.items():
        alphas[k] = v * scaler
    return scaler


def _prod_scalers(scalers_l: List[Dict[int, float]]) -> List[float]:
    products_of_scalers = []
    for scalers in scalers_l:
        tmp = 1.
        for scaler in scalers.values():
            tmp = np.nan_to_num(tmp * scaler)
        products_of_scalers.append(tmp)
    return products_of_scalers


def _calc_gammas(etas: Dict[int, Dict[Tuple[str, str], float]]) -> Dict[int, Dict[str, float]]:
    gammas = {}
    for t, etas_t in etas.items():
        gammas[t] = {}
        for (i, _), v in etas_t.items():
            if i not in gammas[t]:
                gammas[t][i] = 0.
            gammas[t][i] += v
    return gammas


def _calc_initial_probs(hmm: phmm.Hmm, gammas_sequences: List[Dict[str, float]],
                        prods_c_t: List[float]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities from the START state to the inital
    three states of the HMM.

    For the probability of the delete state, the probabilities of all states
    that are not direct successors of the START state must be summed up.
    Args:
        hmm:
        gammas_0:

    Returns:

    """
    succs = hmm.succs[hmm.START]
    d_1 = [i for i in succs if not phmm.emits_symbol(i)][0]
    transitions = {(hmm.START, j): 0. for j in succs if phmm.emits_symbol(j)}
    transitions[(hmm.START, d_1)] = 0.
    denominator = 0.

    for k, gammas in enumerate(gammas_sequences):
        gammas_0 = gammas[0]
        p_k = 1. / prods_c_t[k]
        for j in succs:
            if phmm.emits_symbol(j):
                transitions[(hmm.START, j)] += p_k * gammas_0[j]
                denominator += p_k * gammas_0[j]
        transitions[(hmm.START, d_1)] += p_k * gammas_0[d_1]
        denominator += p_k * gammas_0[d_1]
        for i, v in gammas_0.items():
            if i not in succs:
                transitions[(hmm.START, d_1)] += p_k * v
                denominator += p_k * v
    for k, v in transitions.items():
        transitions[k] = v / denominator
    return transitions


def renormalize_initials(initials: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    initials_new = {}
    min_prob = 1e-4
    denominator = 0.
    for k, v in initials.items():
        initials_new[k] = float(np.clip(v, min_prob, 1))
        denominator += initials_new[k]
    for k, v in initials_new.items():
        initials_new[k] /= denominator
    return initials_new


def renormalize_transitions(hmm: phmm.Hmm, transitions: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    transitions_new = {}
    min_prob = 1e-4
    for hidden_state in hmm.hiddens:
        # if phmm.is_start_state(hidden_state) or phmm.is_end_state(hidden_state):
        #     continue
        if phmm.is_end_state(hidden_state):
            continue
        denominator = 0.
        for target in hmm.succs[hidden_state]:
            transitions_new[(hidden_state, target)] = float(np.clip(transitions[(hidden_state, target)], min_prob, 1))
            denominator += transitions_new[(hidden_state, target)]
        for target in hmm.succs[hidden_state]:
            transitions_new[(hidden_state, target)] /= denominator
    return transitions_new


def renormalize_emissions(hmm: phmm.Hmm, emissions: Dict[Tuple[Any, str], float]) -> Dict[Tuple[Any, str], float]:
    emissions_new = {}
    min_prob = 1e-4
    for i in hmm.hiddens:
        denominator = 0.
        if not phmm.emits_symbol(i):
            continue
        for o in hmm.observables:
            emissions_new[(o, i)] = float(np.clip(emissions[(o, i)], min_prob, 1))
            denominator += emissions_new[(o, i)]
        for o in hmm.observables:
            emissions_new[(o, i)] /= denominator
    return emissions_new


def _update_parameters(hmm: phmm.Hmm, transitions: Dict[Tuple[str, str], float],
                       emissions: Dict[Tuple[Any, str], float]) -> phmm.Hmm:
    new_hmm = phmm.Hmm(hmm.duration)
    new_hmm.observables = hmm.observables
    new_hmm.add_nodes(hmm.hiddens)
    for (i, j), v in hmm.p_ij.items():
        if phmm.is_start_state(i):
            new_hmm.add_transition(i, j, transitions[(i, j)])
        else:
            new_hmm.add_transition(i, j, transitions[(i, j)])
    for (o, i), v in hmm.p_o_in_i.items():
        new_hmm.p_o_in_i[(o, i)] = emissions[(o, i)]
    new_hmm.finalize()
    return new_hmm


def _parameter_changes(hmm: phmm.Hmm, transitions: Dict[Tuple[str, str], float], iter: int,
                           emissions: Dict[Tuple[Any, str], float]) -> List[Dict[str, float]]:
    changes = []
    for i, j in hmm.p_ij.keys():
        changes.append({
            'parameter_name': '{}->{}'.format(i, j),
            'old': hmm.p_ij[(i, j)],
            'new': None,
            'change': None,
            'iter': iter
        })
        if phmm.is_start_state(i):
            changes[-1]['new'] = transitions[(i, j)]
            changes[-1]['change'] = transitions[(i, j)] - hmm.p_ij[(i, j)]
        else:
            changes[-1]['new'] = transitions[(i, j)]
            changes[-1]['change'] = transitions[(i, j)] - hmm.p_ij[(i, j)]

    for o, i in hmm.p_o_in_i.keys():
        changes.append({
            'parameter_name': '{}|{}'.format(o, i),
            'old': hmm.p_o_in_i[(o, i)],
            'new': emissions[(o, i)],
            'change': emissions[(o, i)] - hmm.p_o_in_i[(o, i)],
            'iter': iter
        })
    return changes


def baum_welch(hmm: phmm.Hmm, observations_l: List[List[Any]],
               iter: int = None) -> Tuple[phmm.Hmm, List[Dict[str, float]], List[float]]:
    alphas_l = []
    scalers_l = []
    betas_l = []
    etas_l = []
    gammas_l = []

    for observations in observations_l:
        alphas, scalers = forward(hmm, observations)
        alphas_l.append(alphas)
        scalers_l.append(scalers)
        betas_l.append(backward(hmm, observations, scalers))
        etas_l.append(_calc_etas(hmm, observations, alphas, betas_l[-1]))
        gammas_l.append(_calc_gammas(etas_l[-1]))

    log_prob_l = [np.log(alphas_l[i][len(observations_l[i])]['end']) for i in range(len(alphas_l))]
    transitions = calc_transitions_independent(hmm, observations_l, etas_l, scalers_l)
    emissions = calc_emissions_independent(hmm, observations_l, gammas_l, scalers_l)
    transitions = renormalize_transitions(hmm, transitions)
    emissions = renormalize_emissions(hmm, emissions)

    # changes = _parameter_changes(hmm, transitions, iter, emissions)
    changes = [{}]
    new_hmm = _update_parameters(hmm, transitions, emissions)
    del alphas_l
    del scalers_l
    del betas_l
    del etas_l
    del gammas_l
    gc.collect()
    return new_hmm, changes, log_prob_l


def calc_log_prob(hmm: phmm.Hmm, observations: List[Any]) -> float:
    """
    The logprob is calculated as:

        log(p(o|m)) = -\sum_{t=1}^T\log(c_t)

    i.e., the negative sum of the logarithm of the scalers calculated during the
    forward pass.

    Args:
        hmm:
        observations:

    Returns:

    """
    alphas, scalers = forward(hmm, observations)
    p = alphas[len(observations)]['end']
    if p == 0:
        log_prob = -1e100
    else:
        log_prob = np.log(p)
    return log_prob


#=================================================================================================


def _get_observables(hmm):

    observables = []
    for obs, state in hmm.p_o_in_i.keys():
        observables.append(obs)
    observables = list(set(observables))

    return observables


def _get_states(hmm):

    states = hmm.hiddens
    states.remove('start')
    
    return states


def _get_str_state_to_int_mapping(states):

    str_state_to_int = {}
    for index, state in enumerate(states):
        str_state_to_int[state] = index

    return str_state_to_int


def _get_str_obs_to_int_mapping(observables):

    str_obs_to_int = {}
    for index, state in enumerate(observables):
        str_obs_to_int[state] = index

    return str_obs_to_int


def _get_distributions(hmm):

    states = _get_states(hmm)
    observables = _get_observables(hmm)

    str_state_to_int = _get_str_state_to_int_mapping(states)
    str_obs_to_int = _get_str_obs_to_int_mapping(observables)

    init = np.zeros(len(states))
    trans = np.zeros((len(states), len(states)))
    emissions = np.zeros((len(states), len(observables)))

    for state in hmm.succs[hmm.START]:
        init[str_state_to_int[state]] = hmm.p_ij[(hmm.START, state)]

    for state in states:
        for succ in hmm.succs[state]:
            trans[str_state_to_int[state], str_state_to_int[succ]] = hmm.p_ij[(state, succ)]

    for state in states:
        for obs in observables:
            if phmm.emits_symbol(state):
                emissions[str_state_to_int[state], str_obs_to_int[obs]] = hmm.p_o_in_i[(obs, state)]

    return init, trans, emissions, str_state_to_int, str_obs_to_int


def _convert_str_seq_to_int(sequence, str_obs_to_int):

    sequence_int = []
    for symbol in sequence:
        if symbol not in str_obs_to_int:
            sequence_int.append(len(str_obs_to_int))
        else:
            sequence_int.append(str_obs_to_int[symbol])

    return sequence_int


def _convert_int_path_to_str(best_path, str_state_to_int):

    best_path_str = []
    for path in best_path:
        tmp = list(str_state_to_int.keys())[list(str_state_to_int.values()).index(path)]
        best_path_str.append(tmp)

    return best_path_str


def viterbi(hmm, observation):

    init, trans, obs, int_map, obs_map = _get_distributions(hmm)
    observation = _convert_str_seq_to_int(observation, obs_map)

    len_obs = len(observation)
    num_states = len(init)
    viterbi = np.zeros((num_states, len_obs))
    psi = np.zeros((num_states, len_obs))
    best_path = np.zeros(len_obs, dtype=np.int32)
    
    obs_tmp = obs[:, observation[0]] if observation[0] != len(obs_map) else 1e-12
    viterbi[:, 0] = init.T * obs_tmp
    viterbi[:, 0] /= np.sum(viterbi[:, 0])
    psi[0] = 0

    for t in range(1, len_obs):
        for s in range (0, num_states):
            trans_p = viterbi[:, t - 1] * trans[:, s]
            psi[s, t] = np.argmax(trans_p)
            obs_tmp = obs[s, observation[t]] if observation[t] != len(obs_map) else 1e-12
            viterbi[s, t] = np.max(trans_p) * obs_tmp

        viterbi[:, t] /= np.sum(viterbi[:, t])

    best_path[len_obs - 1] =  viterbi[:, len_obs - 1].argmax()
    for t in range(len_obs - 1, 0, -1):
        best_path[t - 1] = psi[int(best_path[t]), t]

    best_path = best_path.tolist()
    best_path = _convert_int_path_to_str(best_path, int_map)

    return best_path