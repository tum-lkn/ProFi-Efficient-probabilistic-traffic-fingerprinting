import numpy as np
from typing import Dict, List, Any, Tuple

from . import phmm
from .forward import d_, i_, m_, normalize


def _etas_start(hmm: phmm.Hmm, o_t1: Any, alphas_t: Dict[Any, float], betas_t: Dict[Any, float],
                betas_t1: Dict[Any, float]) -> Dict[Tuple[Any, Any], float]:

    states = [hmm.START]
    for i in range(1, hmm.duration + 1):
        states.append('d_' + str(i))

    etas_t = {
        (hmm.START, d_(1)): alphas_t[hmm.START] * betas_t[d_(1)] * phmm.p_ij(hmm, hmm.START, d_(1)),
        (hmm.START, m_(1)): alphas_t[hmm.START] * betas_t1[m_(1)]
                            * phmm.p_ij(hmm, hmm.START, m_(1)) * phmm.p_o_in_i(hmm, o_t1, m_(1)),
        (hmm.START, i_(0)): alphas_t[hmm.START] * betas_t1[i_(0)]
                            * phmm.p_ij(hmm, hmm.START, i_(0)) * phmm.p_o_in_i(hmm, o_t1, i_(0))
    }
    for i in range(2, hmm.duration + 1):
        etas_t[(d_(i - 1), d_(i))] = alphas_t[d_(i - 1)] * betas_t[d_(i)] * phmm.p_ij(hmm, d_(i - 1), d_(i))
    for i in range(2, hmm.duration + 1):
        etas_t[(d_(i - 1), m_(i))] = alphas_t[d_(i - 1)] * betas_t1[m_(i)] * phmm.p_ij(hmm, d_(i - 1), m_(i)) * phmm.p_o_in_i(hmm, o_t1, m_(i))
    for i in range(1, hmm.duration + 1):
        etas_t[(d_(i), i_(i))] = alphas_t[d_(i)] * betas_t1[i_(i)] * phmm.p_ij(hmm, d_(i), i_(i)) * phmm.p_o_in_i(hmm, o_t1, i_(i))

    etas_new = {}
    for state in states:
        for k1, k2 in etas_t.keys():
            if k1 == state:
                etas_new[(k1, k2)] = etas_t[(k1, k2)]

    return etas_new


def _etas_o1_ot(hmm: phmm.Hmm, o_t1: Any, alphas_t: Dict[Any, float], betas_t: Dict[Any, float],
                betas_t1: Dict[Any, float]) -> Dict[Tuple[Any, Any], float]:
    etas_t = {
        (i_(0), d_(1)): alphas_t[i_(0)] * betas_t[d_(1)] * phmm.p_ij(hmm, i_(0), d_(1)),
        (i_(0), m_(1)): alphas_t[i_(0)] * betas_t1[m_(1)] * phmm.p_ij(hmm, i_(0), m_(1)) * phmm.p_o_in_i(hmm, o_t1, m_(1)),
        (i_(0), i_(0)): alphas_t[i_(0)] * betas_t1[i_(0)] * phmm.p_ij(hmm, i_(0), i_(0)) * phmm.p_o_in_i(hmm, o_t1, i_(0))
    }
    for i in range(2, hmm.duration + 1):
        b_t = betas_t[d_(i)]
        etas_t[(d_(i - 1), d_(i))] = alphas_t[d_(i - 1)] * b_t * phmm.p_ij(hmm, d_(i - 1), d_(i))
        etas_t[(m_(i - 1), d_(i))] = alphas_t[m_(i - 1)] * b_t * phmm.p_ij(hmm, m_(i - 1), d_(i))
        etas_t[(i_(i - 1), d_(i))] = alphas_t[i_(i - 1)] * b_t * phmm.p_ij(hmm, i_(i - 1), d_(i))
    for i in range(2, hmm.duration + 1):
        p_o = phmm.p_o_in_i(hmm, o_t1, m_(i))
        b_t1 = betas_t1[m_(i)]
        etas_t[(d_(i - 1), m_(i))] = alphas_t[d_(i - 1)] * b_t1 * phmm.p_ij(hmm, d_(i - 1), m_(i)) * p_o
        etas_t[(m_(i - 1), m_(i))] = alphas_t[m_(i - 1)] * b_t1 * phmm.p_ij(hmm, m_(i - 1), m_(i)) * p_o
        etas_t[(i_(i - 1), m_(i))] = alphas_t[i_(i - 1)] * b_t1 * phmm.p_ij(hmm, i_(i - 1), m_(i)) * p_o
    for i in range(1, hmm.duration + 1):
        p_o = phmm.p_o_in_i(hmm, o_t1, i_(i))
        b_t1 = betas_t1[i_(i)]
        etas_t[(d_(i), i_(i))] = alphas_t[d_(i)] * b_t1 * phmm.p_ij(hmm, d_(i), i_(i)) * p_o
        etas_t[(m_(i), i_(i))] = alphas_t[m_(i)] * b_t1 * phmm.p_ij(hmm, m_(i), i_(i)) * p_o
        etas_t[(i_(i), i_(i))] = alphas_t[i_(i)] * b_t1 * phmm.p_ij(hmm, i_(i), i_(i)) * p_o    

    return etas_t


def _etas_end(hmm: phmm.Hmm, alphas_t: Dict[Any, float], betas_t: Dict[Any, float],
              betas_t1: Dict[Any, float]) -> Dict[Tuple[Any, Any], float]:
    etas_t = {
        (i_(0), d_(1)): alphas_t[i_(0)] * betas_t[d_(1)] * phmm.p_ij(hmm, i_(0), d_(1))
    }
    for i in range(2, hmm.duration + 1):
        b_t = betas_t[d_(i)]
        etas_t[(d_(i - 1), d_(i))] = alphas_t[d_(i - 1)] * phmm.p_ij(hmm, d_(i - 1), d_(i)) * b_t
        etas_t[(m_(i - 1), d_(i))] = alphas_t[m_(i - 1)] * phmm.p_ij(hmm, m_(i - 1), d_(i)) * b_t
        etas_t[(i_(i - 1), d_(i))] = alphas_t[i_(i - 1)] * phmm.p_ij(hmm, i_(i - 1), d_(i)) * b_t
    t = hmm.duration
    b_t1 = betas_t1[hmm.END]
    etas_t[(d_(t), hmm.END)] = alphas_t[d_(t)] * phmm.p_ij(hmm, d_(t), hmm.END) * b_t1
    etas_t[(m_(t), hmm.END)] = alphas_t[m_(t)] * phmm.p_ij(hmm, m_(t), hmm.END) * b_t1
    etas_t[(i_(t), hmm.END)] = alphas_t[i_(t)] * phmm.p_ij(hmm, i_(t), hmm.END) * b_t1
    
    return etas_t


def _calc_etas(hmm: phmm.Hmm, observations: List[Any], alphas: Dict[int, Dict[Any, float]],
               betas: Dict[int, Dict[Any, float]]) -> Dict[int, Dict[Tuple[Any, Any], float]]:
    etas = {}
    etas[-1] = _etas_start(hmm, observations[0], alphas[-1], betas[-1], betas[0])
    for t in range(0, len(observations) - 1):
        etas[t] = _etas_o1_ot(hmm, observations[t + 1], alphas[t], betas[t], betas[t + 1])
    t = len(observations) - 1
    etas[t] = _etas_end(hmm, alphas[t], betas[t], betas[t + 1])
    return etas


def _transition_denom(hmm: phmm.Hmm, etas: Dict[int, Dict[Tuple[Any, Any], float]]) -> Dict[Any, float]:
    y_i = {h: 0. for h in hmm.hiddens}
    for t, etas_t in etas.items():
        for (i, _), eta_ij in etas_t.items():
            y_i[i] += eta_ij
    return y_i


def _transition_nom(hmm: phmm.Hmm, etas: Dict[int, Dict[Tuple[Any, Any], float]]) -> Dict[Tuple[Any, Any], float]:
    nom = {(i, j): 0 for i, j in hmm.p_ij.keys()}
    for t, etas_t in etas.items():
        for (i, j), eta_ij in etas_t.items():
            nom[(i, j)] += eta_ij
    return nom


def _cum_product(values: List[float]) -> float:
    ret = 1.
    for x in values:
        ret *= x
    return ret


def _calc_transitions_independent(hmm: phmm.Hmm, etas: Dict[int, Dict[Tuple[Any, Any], float]]
                      ) -> Dict[Tuple[Any, Any], float]:
    a_ij = {(i, j): 0. for i, j in hmm.p_ij.keys()}
    nom = _transition_nom(hmm, etas)
    denom = _transition_denom(hmm, etas)
    for a, b in nom.keys():
        a_ij[(a, b)] = nom[(a, b)] / denom[a]
    return a_ij


def calc_transitions_independent(hmm: phmm.Hmm, observations_l: List[List[Any]],
                     etas_l: List[Dict[int, Dict[Tuple[Any, Any], float]]],
                     scalers_l: List[Dict[int, float]]) -> Dict[Tuple[Any, Any], float]:
    a_ij = {(i, j): 0. for i, j in hmm.p_ij.keys()}
    a_i = {i: 0. for i, _ in hmm.p_ij.keys()}

    for etas in etas_l:
        for (a, b), v in _calc_transitions_independent(hmm, etas).items():
            a_ij[(a, b)] += v
            a_i[a] += v
    for i, j in a_ij.keys():
        a_ij[(i, j)] /= a_i[i]
    return a_ij


def calc_transitions(hmm: phmm.Hmm, observations_l: List[List[Any]],
                     etas_l: List[Dict[int, Dict[Tuple[Any, Any], float]]],
                     scalers_l: List[Dict[int, float]]) -> Dict[Tuple[Any, Any], float]:
    a_ij = {(i, j): 0. for i, j in hmm.p_ij.keys()}
    a_i = {i: 0. for i in hmm.hiddens}

    for i, (etas, observations) in enumerate(zip(etas_l, observations_l)):
        denom = _transition_denom(hmm, etas)
        nom = _transition_nom(hmm, etas)

        scale = _cum_product(list(scalers_l[i].values()))
        for a, b in nom.keys():
            a_ij[(a, b)] += nom[(a, b)] / scale
        for x in denom:
            a_i[x] += denom[x] / scale
    for i, j in a_ij.keys():
        a_ij[(i, j)] /= a_i[i]
    return a_ij


def _calc_emissions_independent(hmm: phmm.Hmm, observations: List[Any],
                               gammas: Dict[int, Dict[Any, float]]) -> Dict[Tuple[Any, Any], float]:
    emissions = {(o, s): 0. for o, s in hmm.p_o_in_i.keys()}
    denom = {s: 0. for s in hmm.hiddens if s not in [hmm.START, hmm.END] and not phmm.is_delete_state(s)}
    for t in range(-1, len(observations)):
        gamma_t = gammas[t]
        for state in gamma_t.keys():
            if state in [hmm.START, hmm.END] or phmm.is_delete_state(state):
                continue
            gamma = gamma_t[state]
            denom[state] += gamma
            if 0 <= t <= len(observations) - 1:
                o_t = observations[t]
                emissions[(o_t, state)] += gamma
    for o, s in emissions.keys():
        emissions[(o, s)] /= denom[s]
    return emissions


def calc_emissions_independent(hmm: phmm.Hmm, observations_l: List[List[Any]],
                   gammas_l: List[Dict[int, Dict[Any, float]]],
                   scalers_l: List[Dict[int, float]]) -> Dict[Tuple[Any, Any], float]:
    emissions = {(o, s): 0. for o, s in hmm.p_o_in_i.keys()}
    denom = {s: 0. for s in hmm.hiddens if s not in [hmm.START, hmm.END] and not phmm.is_delete_state(s)}
    for gammas, observations in zip(gammas_l, observations_l):
        for (o, s), v in _calc_emissions_independent(hmm, observations, gammas).items():
            emissions[(o, s)] += v
            denom[s] += v
    for o, s in emissions.keys():
        emissions[(o, s)] /= denom[s]
    return emissions


def calc_emissions(hmm: phmm.Hmm, observations_l: List[List[Any]],
                   gammas_l: List[Dict[int, Dict[Any, float]]],
                   scalers_l: List[Dict[int, float]]) -> Dict[Tuple[Any, Any], float]:
    emissions = {(o, s): 0. for o, s in hmm.p_o_in_i.keys()}
    denom = {s: 0. for s in hmm.hiddens if s not in [hmm.START, hmm.END] and not phmm.is_delete_state(s)}
    for observations, gammas, scalers in zip(observations_l, gammas_l, scalers_l):
        scaler = _cum_product(list(scalers.values()))
        for t in range(-1, len(observations)):
            gamma_t = gammas[t]
            for state in gamma_t.keys():
                if state in [hmm.START, hmm.END] or phmm.is_delete_state(state):
                    continue
                gamma = gamma_t[state]
                denom[state] += gamma / scaler
                if 0 <= t <= len(observations) - 1:
                    o_t = observations[t]
                    emissions[(o_t, state)] += gamma / scaler
    for o, s in emissions.keys():
        emissions[(o, s)] /= denom[s]
    return emissions


def calc_initials(hmm: phmm.Hmm, gammas_l: List[Dict[int, Dict[Any, float]]],
                  etas_l: List[Dict[int, Dict[Tuple[Any, Any], float]]],
                  scalers_l: List[Dict[int, float]]) -> Dict[Tuple[Any, Any], float]:
    denom = 0.
    initials = {
        (hmm.START, d_(1)): 0,
        (hmm.START, m_(1)): 0.,
        (hmm.START, i_(0)): 0.
    }
    for scalers, gammas, etas in zip(scalers_l, gammas_l, etas_l):
        scaler = _cum_product(list(scalers.values()))
        m_1 = etas[-1][(hmm.START, m_(1))]  # gammas[-1][m_(1)]
        i_0 = etas[-1][(hmm.START, i_(0))]  # gammas[-1][i_(0)]
        d_1 = etas[-1][(hmm.START, d_(1))]  # gammas[-1][d_(1)]
        initials[(hmm.START, d_(1))] += d_1 * scaler
        initials[(hmm.START, m_(1))] += m_1 * scaler
        initials[(hmm.START, i_(0))] += i_0 * scaler
        denom += scaler * (m_1 + i_0 + d_1)

    for a, b in initials.keys():
        initials[(a, b)] /= denom

    return initials
