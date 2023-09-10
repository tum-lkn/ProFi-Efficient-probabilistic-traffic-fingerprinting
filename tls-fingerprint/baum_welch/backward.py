import numpy as np
import logging
from typing import Dict, List, Any, Tuple

from . import phmm
from . import setlogging
from .forward import d_, i_, m_, normalize

logger = logging.getLogger("backward")
setlogging.set_logging(logger)
B = "\u03B2"

def sort_betas(betas, states):
    beta_new = {}
    for k in states:
        beta_new[k] = betas[k]
    return beta_new

def _init(hmm: phmm.Hmm, betas_t1: Dict[Any, float], t_, states: List[str]) -> Dict[Any, float]:
    t = hmm.duration
    beta_t = {
        d_(t): phmm.p_ij(hmm, d_(t), hmm.END) * betas_t1[hmm.END]
    }
    logger.debug(f"{B}_{t_}({d_(t)}) = p_ij({d_(t)}, {hmm.END}){B}_{t_+1}({hmm.END})")
    logger.debug(f"{B}_{t_}({i_(t)}) = p_ij({i_(t)}, {hmm.END}){B}_{t_+1}({hmm.END})")
    logger.debug(f"{B}_{t_}({m_(t)}) = p_ij({m_(t)}, {hmm.END}){B}_{t_+1}({hmm.END})")
    # ranges from t - 1 to 1, i.e., d_3, d_2, d_1 for a HMM with duration 4
    # having states d_1, d_2, d_3, d_4
    for i in range(hmm.duration - 1, 0, -1):
        beta_t[d_(i)] = beta_t[d_(i + 1)] * phmm.p_ij(hmm, d_(i), d_(i + 1))
        logger.debug(f"{B}_{t_}({d_(i)}) = p_ij({d_(i)}, {d_(i+1)}){B}_{t_}({d_(i+1)})")
    beta_t[i_(t)] = phmm.p_ij(hmm, i_(t), hmm.END) * betas_t1[hmm.END]
    for i in range(hmm.duration - 1, -1, -1):
        beta_t[i_(i)] = beta_t[d_(i + 1)] * phmm.p_ij(hmm, i_(i), d_(i + 1))
        logger.debug(f"{B}_{t_}({i_(i)}) = p_ij({i_(i)}, {d_(i + 1)}){B}_{t_}({d_(i + 1)})")
    beta_t[m_(t)] = phmm.p_ij(hmm, m_(t), hmm.END) * betas_t1[hmm.END]
    for i in range(hmm.duration - 1, 0, -1):
        beta_t[m_(i)] = beta_t[d_(i + 1)] * phmm.p_ij(hmm, m_(i), d_(i + 1))
        logger.debug(f"{B}_{t_}({m_(i)}) = p_ij({m_(i)}, {d_(i + 1)}){B}_{t_}({d_(i + 1)})")
    beta_t = sort_betas(beta_t, states)
    return beta_t


def _between(hmm: phmm.Hmm, o_t1: str, beta_t1: Dict[Any, float], t_, states: List[str]) -> Dict[Any, float]:
    beta_t = {}
    t = hmm.duration
    beta_t[d_(t)] = beta_t1[i_(t)] * phmm.p_ij(hmm, d_(t), i_(t)) * phmm.p_o_in_i(hmm, o_t1, i_(t))
    logger.debug(f"{B}_{t_}({d_(t)}) = p_ij({d_(t)}, {i_(t)})p({o_t1}|{i_(t)}){B}_{t_+1}({i_(t)})")
    for i in range(hmm.duration - 1, 0, -1):
        b_d =  beta_t[d_(i + 1)]  * phmm.p_ij(hmm, d_(i), d_(i + 1))
        b_d += beta_t1[i_(i)]     * phmm.p_ij(hmm, d_(i), i_(i))     * phmm.p_o_in_i(hmm, o_t1, i_(i))
        b_d += beta_t1[m_(i + 1)] * phmm.p_ij(hmm, d_(i), m_(i + 1)) * phmm.p_o_in_i(hmm, o_t1, m_(i + 1))
        beta_t[d_(i)] = b_d
        logger.debug(f"{B}_{t_}({d_(i)}) = "f""
                     f"p_ij({d_(i)}, {d_(i+1)}){B}_{t_}({d_(i + 1)}) + "
                     f"p_ij({d_(i)}, {i_(i)})p_o({o_t1}|{i_(i)}){B}_{t_+1}({i_(i)}) + "
                     f"p_ij({d_(i)}, {m_(i+1)})p_o({o_t1}|{m_(i+1)}){B}_{t_+1}({m_(i+1)})")

    beta_t[i_(t)] = beta_t1[i_(t)] * phmm.p_ij(hmm, i_(t), i_(t)) * phmm.p_o_in_i(hmm, o_t1, i_(t))
    logger.debug(f"{B}_{t_}({i_(t)}) = p_ij({i_(t)}, {i_(t)})p({o_t1}|{i_(t)}){B}_{t_+1}({i_(t)})")
    for i in range(hmm.duration - 1, -1, -1):
        b_i  = beta_t1[d_(i + 1)] * phmm.p_ij(hmm, i_(i), d_(i + 1))
        b_i += beta_t1[i_(i)]     * phmm.p_ij(hmm, i_(i), i_(i))     * phmm.p_o_in_i(hmm, o_t1, i_(i))
        b_i += beta_t1[m_(i + 1)] * phmm.p_ij(hmm, i_(i), m_(i + 1)) * phmm.p_o_in_i(hmm, o_t1, m_(i + 1))
        beta_t[i_(i)] = b_i
        logger.debug(f"{B}_{t_}({i_(i)}) = "
                     f"p_ij({i_(i)}, {d_(i+1)}){B}_{t_}({d_(i + 1)}) + "
                     f"p_ij({i_(i)}, {i_(i)})p_o({o_t1}|{i_(i)}){B}_{t_+1}({i_(i)}) + "
                     f"p_ij({i_(i)}, {m_(i+1)})p_o({o_t1}|{m_(i+1)}){B}_{t_+1}({m_(i+1)})")

    beta_t[m_(t)] = beta_t1[i_(t)] * phmm.p_ij(hmm, m_(t), i_(t)) * phmm.p_o_in_i(hmm, o_t1, i_(t))
    logger.debug(f"{B}_{t_}({m_(t)}) = p_ij({m_(t)}, {i_(t)})p({o_t1}|{i_(t)}){B}_{t_+1}({i_(t)})")
    for i in range(hmm.duration - 1, 0, -1):
        b_m =  beta_t[d_(i + 1)]  * phmm.p_ij(hmm, m_(i), d_(i + 1))
        b_m += beta_t1[i_(i)]     * phmm.p_ij(hmm, m_(i), i_(i))     * phmm.p_o_in_i(hmm, o_t1, i_(i))
        b_m += beta_t1[m_(i + 1)] * phmm.p_ij(hmm, m_(i), m_(i + 1)) * phmm.p_o_in_i(hmm, o_t1, m_(i + 1))
        beta_t[m_(i)] = b_m
        logger.debug(f"{B}_{t_}({m_(i)}) = "
                     f"p_ij({m_(i)}, {d_(i+1)}){B}_{t_}({d_(i + 1)}) + "
                     f"p_ij({m_(i)}, {i_(i)})p_o({o_t1}|{i_(i)}){B}_{t_+1}({i_(i)}) + "
                     f"p_ij({m_(i)}, {m_(i+1)})p_o({o_t1}|{m_(i+1)}){B}_{t_+1}({m_(i+1)})")

    beta_t = sort_betas(beta_t, states)

    return beta_t


def _end(hmm: phmm.Hmm, o_t1: str, beta_t1: Dict[Any, float]) -> Dict[Any, float]:
    states = ['start']
    for i in range(1, hmm.duration + 1):
        states.append('d_' + str(i))

    t_ = -1
    beta_t = {}
    t = hmm.duration
    beta_t[d_(t)] = beta_t1[i_(t)] * phmm.p_ij(hmm, d_(t), i_(t)) * phmm.p_o_in_i(hmm, o_t1, i_(t))
    logger.debug(f"{B}_{t_}({d_(t)}) = p_ij({d_(t)}, {i_(t)})p({o_t1}|{i_(t)}){B}_{t_+1}({i_(t)})")

    for i in range(hmm.duration - 1, 0, -1):
        b_d = beta_t[d_(i + 1)] * phmm.p_ij(hmm, d_(i), d_(i + 1))
        b_d += beta_t1[i_(i)] * phmm.p_ij(hmm, d_(i), i_(i)) * phmm.p_o_in_i(hmm, o_t1, i_(i))
        b_d += beta_t1[m_(i + 1)] * phmm.p_ij(hmm, d_(i), m_(i + 1)) * phmm.p_o_in_i(hmm, o_t1, m_(i + 1))
        beta_t[d_(i)] = b_d
        logger.debug(f"{B}_{t_}({d_(i)}) = "
                     f"p_ij({d_(i)}, {d_(i+1)}){B}_{t_}({d_(i + 1)}) + " 
                     f"p_ij({d_(i)}, {i_(i)})p_o({o_t1}|{i_(i)}){B}_{t_+1}({i_(i)}) + "
                     f"p_ij({d_(i)}, {m_(i+1)})p_o({o_t1}|{m_(i+1)}){B}_{t_+1}({m_(i+1)})")

    b_s = beta_t[d_(1)] * phmm.p_ij(hmm, hmm.START, d_(1))
    b_s += beta_t1[m_(1)] * phmm.p_ij(hmm, hmm.START, m_(1)) * phmm.p_o_in_i(hmm, o_t1, m_(1))
    b_s += beta_t1[i_(0)] * phmm.p_ij(hmm, hmm.START, i_(0)) * phmm.p_o_in_i(hmm, o_t1, i_(0))
    logger.debug(f"{B}_{t_}({hmm.START}) = "
                 f"p_ij({hmm.START}, {d_(1)}){B}_{t_}({d_(1)}) + "
                 f"p_ij({hmm.START}, {i_(0)})p_o({o_t1}|{i_(0)}){B}_{t_ + 1}({i_(0)}) + "
                 f"p_ij({hmm.START}, {m_(1)})p_o({o_t1}|{m_(1)}){B}_{t_ + 1}({m_(1)})")
    beta_t[hmm.START] = b_s

    beta_t = sort_betas(beta_t, states)

    return beta_t


def backward(hmm: phmm.Hmm, observations: List[Any], scalers: Dict[int, float]) -> Dict[int, Dict[str, float]]:
    """
    Each beta variable `\beta_t(i)` is the probability of the partial
    observation sequence from `t + 1` to the end, given state s_i at time t.

    Args:
        hmm:
        observations:
        scalers:

    Returns:

    """
    states = hmm.hiddens.copy()
    states.remove('start')
    states.remove('end')

    logger.debug("INITIALIZE")
    betas = {len(observations): {hmm.END: 1.}}
    b_t = _init(hmm, betas[len(observations)], len(observations) - 1, states)
    betas[len(observations) - 1] = normalize(b_t, scalers[len(observations) - 1])

    for t in range(len(observations) - 2, -1, -1):
        logger.debug(f"\nDO STEP t={t}")
        b_t = _between(hmm, observations[t + 1], betas[t + 1], t, states)
        betas[t] = normalize(b_t, scalers[t])

    b_1 = _end(hmm, observations[0], betas[0])
    betas[-1] = normalize(b_1, scalers[-1])

    betas_new = {}
    for i in range(-1, len(betas) - 1):
        betas_new[i] = betas[i]

    return betas_new

