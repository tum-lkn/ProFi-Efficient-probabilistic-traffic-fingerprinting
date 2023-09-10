import numpy as np
from typing import List, Tuple, Any, Dict

import fhmm

def _normalize(alphas: Dict[int, float], scaler: float=None) -> float:
    if scaler is None:
        sum_alphas = 0.
        for v in alphas.values():
            sum_alphas += v
        scaler = 1. / sum_alphas
    for k, v in alphas.items():
        alphas[k] = v * scaler
    return scaler

def forward(hmm: fhmm.Hmm, observation: List[Any]) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float]]:

    alphas = {}
    scalers = {}
    alphas[0] = {}
    for i in hmm.hiddens:
        alphas[0][i] = hmm.init[i] * hmm.p_o_in_i.get((observation[0], i), 1e-12)
        scalers[0] = _normalize(alphas[0])
    for t in range(1, len(observation)):
        alphas[t] = {}
        for i in hmm.hiddens:
            a_j = 0.
            for j in hmm.hiddens:
                a_j += alphas[t - 1][j] * hmm.p_ij[(j, i)]
            alphas[t][i] = a_j * hmm.p_o_in_i.get((observation[t], i), 1e-12)
        scalers[t] = _normalize(alphas[t])
    return alphas, scalers
    
def backward(hmm: fhmm.Hmm, observation: List[Any], scalers: Dict[int, float]) -> Dict[int, Dict[int, float]]:

    betas = {len(observation) - 1: {i: scalers[len(observation) - 1] for i in hmm.hiddens}}
    for t in range(len(observation) - 2, -1, -1):
        betas[t] = {}
        for i in hmm.hiddens:
            b_t_of_i = 0.
            for j in hmm.hiddens:
                b_t_of_i += betas[t + 1][i] * hmm.p_o_in_i.get((observation[t + 1], j), 1e-12) * hmm.p_ij[(i, j)]
            betas[t][i] = b_t_of_i
        _normalize(betas[t], scalers[t])
    return betas

def _prod_scalers(scalers_l: List[Dict[int, float]]) -> List[float]:
    products_of_scalers = []
    for scalers in scalers_l:
        tmp = 1.
        for scaler in scalers.values():
            tmp = np.nan_to_num(tmp * scaler)
        products_of_scalers.append(tmp)
    return products_of_scalers

def _calc_etas(hmm: fhmm.Hmm, observation: List[Any], alphas: Dict[int, Dict[int, float]], betas: Dict[int, Dict[int, float]]) -> Dict[int, Dict[Tuple[int, int], float]]:

    etas = {i: {} for i in range(len(observation))}
    for t in range(len(observation)):
        o_t1 = None if len(observation) - 1 == t else observation[t + 1]
        for i, j in hmm.p_ij.keys():
            a_t_of_i = alphas[t][i]
            if o_t1 is None:
                p_o_in_i = 1.
            else:
                
                p_o_in_i = hmm.p_o_in_i.get((o_t1, j), 1e-12)
            if t == len(observation) - 1:
                b_t1_of_j = betas[t][j]
            else:
                b_t1_of_j = betas[t + 1][j] * p_o_in_i
            etas[t][(i, j)] = a_t_of_i * hmm.p_ij[(i, j)] * b_t1_of_j
    return etas

def _calc_gammas(etas: Dict[int, Dict[Tuple[int, int], float]]) -> Dict[int, Dict[int, float]]:

    gammas = {}
    for t, etas_t in etas.items():
        gammas[t] = {}
        for (i, _), v in etas_t.items():
            if i not in gammas[t]:
                gammas[t][i] = 0.
            gammas[t][i] += v
    return gammas


def _calc_initial_probs(hmm: fhmm.Hmm, gammas_sequences: List[Dict[int, float]], prods_c_t: List[float]) -> Dict[Tuple[int, int], float]:

    inits = {i: 0. for i in hmm.hiddens}
    denominator = 0.
    for k, gammas in enumerate(gammas_sequences):
        gammas_0 = gammas[0]
        p_k = 1. / prods_c_t[k]
        for i in hmm.hiddens:
            inits[i] += p_k * gammas_0[i]
            denominator += p_k * gammas_0[i]
    for k, v in inits.items():
        inits[k] = v / denominator
    return inits

def _calc_transition_probs(hmm: fhmm.Hmm,
                            obs_l: List[List[Any]], 
                            alphas_l: List[Dict[int, Dict[int, float]]], 
                            betas_l: List[Dict[int, Dict[int, float]]],
                            prods_c_t: List[float]) -> Dict[Tuple[int, int], float]:

    new_transitions = {k: 0. for k in hmm.p_ij.keys()}
    denoms = {h: 0 for h in hmm.hiddens}
    for obs, alphas, betas, c_k in zip(obs_l, alphas_l, betas_l, prods_c_t):
        for i, j in hmm.p_ij.keys():
            for t in range(len(obs)):
                if t == len(obs) - 1:
                    p_o_t1_in_j = 0.
                    beta_t1_of_j = 1.
                else:
                    p_o_t1_in_j = hmm.p_o_in_i.get((obs[t + 1], j), 1e-12)
                    beta_t1_of_j = betas[t + 1][j]
                val = c_k * alphas[t][i] * hmm.p_ij[(i, j)] * p_o_t1_in_j * beta_t1_of_j
                new_transitions[(i, j)] += val
                denoms[i] += val
    for i, j in new_transitions.keys():
        new_transitions[(i, j)] /= denoms[i]
    return new_transitions

def _calc_emission_probs(hmm: fhmm.Hmm,
                            obs_l: List[List[Any]], 
                            alphas_l: List[Dict[int, Dict[int, float]]], 
                            betas_l: List[Dict[int, Dict[int, float]]],
                            prods_c_t: List[float]) -> Dict[Tuple[int, int], float]:

    emissions = {}
    for i in hmm.hiddens:
        for obs in hmm.observables:
            emissions[(obs, i)] = 0.
        denom = 0.
        for obs, alphas, betas, c_k in zip(obs_l, alphas_l, betas_l, prods_c_t):
            for t, o_t in enumerate(obs):
                emissions[(o_t, i)] += c_k * alphas[t][i] * betas[t][i]
                denom += c_k * alphas[t][i] * betas[t][i]
        for obs in hmm.observables:
            emissions[(obs, i)] /=  denom
    return emissions

def renormalize_initials(initials: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    initials_new = {}
    min_prob = 1e-9
    denominator = 0.
    for k, v in initials.items():
        initials_new[k] = float(np.clip(v, min_prob, 1))
        denominator += initials_new[k]
    for k, v in initials_new.items():
        initials_new[k] /= denominator
    return initials_new


def renormalize_transitions(hmm: fhmm.Hmm, transitions: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    transitions_new = {}
    min_prob = 1e-9
    for hidden_state in hmm.hiddens:
        denominator = 0.
        for target in hmm.hiddens:
            transitions_new[(hidden_state, target)] = float(np.clip(transitions[(hidden_state, target)], min_prob, 1))
            denominator += transitions_new[(hidden_state, target)]
        for target in hmm.hiddens:
            transitions_new[(hidden_state, target)] /= denominator
    return transitions_new


def renormalize_emissions(hmm: fhmm.Hmm, emissions: Dict[Tuple[int, Any], float]) -> Dict[Tuple[int, Any], float]:
    emissions_new = {}
    min_prob = 1e-9
    for i in hmm.hiddens:
        denominator = 0.
        for o in hmm.observables:
            emissions_new[(o, i)] = float(np.clip(emissions[(o, i)], min_prob, 1))
            denominator += emissions_new[(o, i)]
        for o in hmm.observables:
            emissions_new[(o, i)] /= denominator
    return emissions_new

def _update_parameters(old_hmm: fhmm.Hmm, initials: Dict[int, float],
                       transitions: Dict[Tuple[int, int], float],
                       emissions: Dict[Tuple[int, Any], float]) -> fhmm.Hmm:
    new_hmm = fhmm.Hmm(old_hmm.duration)
    new_hmm.observables = old_hmm.observables
    new_hmm.add_nodes()
    for i in old_hmm.init.keys():
        new_hmm.init[i] = initials[i]
    for (i, j), v in old_hmm.p_ij.items():
        new_hmm.p_ij[(i, j)] = transitions[(i, j)]
    for (o, i), v in old_hmm.p_o_in_i.items():
        new_hmm.p_o_in_i[(o, i)] = emissions[(o, i)]
    return new_hmm

def baum_welch(hmm: fhmm.Hmm, observations_l: List[List[int]]) -> fhmm.Hmm:

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

    prod_c_t = _prod_scalers(scalers_l)
    initials = _calc_initial_probs(hmm, gammas_l, prod_c_t)
    transitions = _calc_transition_probs(hmm, observations_l, alphas_l, betas_l, prod_c_t)
    emissions = _calc_emission_probs(hmm, observations_l, alphas_l, betas_l, prod_c_t)
    initials = renormalize_initials(initials)
    transitions = renormalize_transitions(hmm, transitions)
    emissions = renormalize_emissions(hmm, emissions)

    new_hmm = _update_parameters(hmm, initials, transitions, emissions)

    return new_hmm

def calc_log_prob(hmm: fhmm.Hmm, observations: List[List[Any]]) -> float:

    log_prob = 0.
    for observation in observations:
        _, scalers = forward(hmm, observation)
        scalers = list(scalers.values())
        for scaler in scalers:
            log_prob += np.log(scaler) if scaler > 0 else np.log(1e-12)
    return -1. * log_prob / len(observations)

def viterbi_fhmm(hmm, observation):

    len_obs = len(observation)
    duration = hmm.duration

    viterbi = {}
    psi = {}
    best_path = np.zeros(len_obs, dtype=np.int32)
    denom = 0
    for i in range(duration):
        viterbi[(observation[0], i)] = hmm.init[i] * hmm.p_o_in_i.get((observation[0], i), 1e-12)
        denom += viterbi[(observation[0], i)]

    for i in range(duration):
        viterbi[(observation[0], i)] /= denom
    for t in range(1, len_obs):
        denom = 0
        for s in range(duration):
            trans_p = np.zeros(duration)
            for i in range(duration):
                trans_p[i] = viterbi[(observation[t - 1], i)] * hmm.p_ij[(i, s)]
            psi[(observation[t], s)] = np.argmax(trans_p)
            viterbi[(observation[t], s)] = np.max(trans_p) * hmm.p_o_in_i.get((observation[t], s), 1e-12)
            denom += viterbi[(observation[t], s)]
        for i in range(duration):
            viterbi[(observation[t], i)] /= denom

    best_paths_last = []
    for i in range(duration):
        best_paths_last.append(viterbi[(observation[len_obs - 1], i)])
    best_path[len_obs - 1] = np.argmax(best_paths_last)
    for t in range(len_obs - 1, 0 , -1):
        best_path[t - 1] = psi[observation[t], int(best_path[t])]

    best_path = best_path.tolist()

    return best_path