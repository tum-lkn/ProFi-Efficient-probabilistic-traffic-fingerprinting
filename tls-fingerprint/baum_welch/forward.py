import numpy as np
import logging
from typing import Any, List, Tuple, Dict

from . import phmm
from . import setlogging

logger = logging.getLogger("forward")
setlogging.set_logging(logger)
A = "\u03B1"


def d_(t: int) -> str:
    return "{:s}_{:d}".format(phmm.Hmm.DELETE_STATE_PREFIX, t)


def i_(t: int) -> str:
    return "{:s}_{:d}".format(phmm.Hmm.INSERT_STATE_PREFIX, t)


def m_(t: int) -> str:
    return "{:s}_{:d}".format(phmm.Hmm.MATCH_STATE_PREFIX, t)


def get_scaler(alphas_t: Dict[Any, float]) -> float:
    s = float(np.sum(list(alphas_t.values())))
    if s == 0:
        s = 1e-6
    return 1. / s


def normalize(alphas_t: Dict[Any, float], scaler: float) -> Dict[Any, float]:
    return {k: v * scaler for k, v in alphas_t.items()}


def sort_alphas(alphas, states):

    alpha_new = {}
    for k in states:
        alpha_new[k] = alphas[k]
    return alpha_new

def _initial(hmm: phmm.Hmm) -> Dict[str, float]:
    alphas_t1 = {hmm.START: 1., d_(1): phmm.p_ij(hmm, hmm.START, d_(1))}
    logger.debug(f"{A}_{{-1}}({d_(1)}) = p_{{ij}}(START, {d_(1)}) * {A}_{{-1}}(START)")

    for i in range(2, hmm.duration + 1):
        alphas_t1[d_(i)] = alphas_t1[d_(i - 1)] * phmm.p_ij(hmm, d_(i - 1), d_(i))
        logger.debug(f"{A}_{{-1}}({d_(i)}) = p_{{ij}}({d_(i - 1)}, {d_(i)}) * {A}_{{-1}}({d_(i - 1)})")

    return alphas_t1

def _between_o1(hmm: phmm.Hmm, o_t: Any, alphas_t: Dict[Any, float], t1: int, states: List[str]) -> Dict[Any, float]:
    
    alphas_t1 = {
        i_(0): alphas_t[hmm.START] * phmm.p_ij(hmm, hmm.START, i_(0)) * phmm.p_o_in_i(hmm, o_t, i_(0)),
        m_(1): alphas_t[hmm.START] * phmm.p_ij(hmm, hmm.START, m_(1)) * phmm.p_o_in_i(hmm, o_t, m_(1))
    }
    logger.debug(f"{A}_{t1}({i_(0)}) = {A}_{{{t1 - 1}}}({hmm.START}) * p_{{ij}}({hmm.START}, {i_(0)}) * p_o({o_t}|{i_(0)})")
    logger.debug(f"{A}_{t1}({m_(1)}) = {A}_{{{t1 - 1}}}({hmm.START}) * p_{{ij}}({hmm.START}, {m_(1)}) * p_o({o_t}|{m_(1)})")

    # Do the match states
    for i in range(2, hmm.duration + 1):
        s = f"{A}_{t1}({m_(i)}) = ({A}_{{{t1-1}}}({d_(i-1)}) * p_{{ij}}({d_(i-1)}, {m_(i)})"
        alphas_t1[m_(i)] = alphas_t[d_(i - 1)] * phmm.p_ij(hmm, d_(i - 1), m_(i)) * phmm.p_o_in_i(hmm, o_t, m_(i))
        s += f") * p_o({o_t}|{m_(i)})"
        logger.debug(s)

    # Do the insert states
    for i in range(1, hmm.duration + 1):
        s = f"{A}_{t1}({i_(i)}) = ({A}_{{{t1-1}}}({d_(i)}) * p_{{ij}}({d_(i)}, {i_(i)})"
        alphas_t1[i_(i)] = alphas_t[d_(i)] * phmm.p_ij(hmm, d_(i), i_(i)) * phmm.p_o_in_i(hmm, o_t, i_(i))
        s += f") * p_o({o_t}|{i_(i)})"
        logger.debug(s)

    # Do the delete states
    alphas_t1[d_(1)] = alphas_t1[i_(0)] * phmm.p_ij(hmm, i_(0), d_(1))
    logger.debug(f"{A}_{t1}({d_(1)}) = {A}_{t1}({i_(0)}) * p_{{ij}}({i_(0)}, {d_(1)}))")
    for i in range(2, hmm.duration + 1):
        a_d = alphas_t1[d_(i - 1)] * phmm.p_ij(hmm, d_(i - 1), d_(i))
        a_d += alphas_t1[m_(i - 1)] * phmm.p_ij(hmm, m_(i - 1), d_(i))
        alphas_t1[d_(i)] = a_d + alphas_t1[i_(i - 1)] * phmm.p_ij(hmm, i_(i - 1), d_(i))
        logger.debug(f"{A}_{t1}({d_(i)}) = ({A}_{t1}({d_(i-1)}) * p_{{ij}}({d_(i-1)}, {d_(i)}) + "
                     f"{A}_{t1}({m_(i-1)}) * p_{{ij}}({m_(i-1)}, {d_(i)}) + " +
                     f"{A}_{t1}({i_(i-1)}) * p_{{ij}}({i_(i-1)}, {d_(i)}))")

    alphas_t1 = sort_alphas(alphas_t1, states)

    return alphas_t1

def _between(hmm: phmm.Hmm, o_t: Any, alphas_t: Dict[Any, float], t1: int, states: List[str]) -> Dict[Any, float]:
    alphas_t1 = {
        i_(0): alphas_t[i_(0)] * phmm.p_ij(hmm, i_(0), i_(0)) * phmm.p_o_in_i(hmm, o_t, i_(0)),
        m_(1): alphas_t[i_(0)] * phmm.p_ij(hmm, i_(0), m_(1)) * phmm.p_o_in_i(hmm, o_t, m_(1))
    }
    logger.debug(f"{A}_{t1}({i_(0)}) = {A}_{{{t1 - 1}}}({i_(0)}) * p_{{ij}}({i_(0)}, {i_(0)}) * p_o({o_t}|{i_(0)})")
    logger.debug(f"{A}_{t1}({m_(1)}) = {A}_{{{t1 - 1}}}({i_(0)}) * p_{{ij}}({i_(0)}, {m_(1)}) * p_o({o_t}|{m_(1)})")

    # Do the match states
    for i in range(2, hmm.duration + 1):
        s = f"{A}_{t1}({m_(i)}) = ({A}_{{{t1-1}}}({d_(i-1)}) * p_{{ij}}({d_(i-1)}, {m_(i)})"
        a_m = alphas_t[d_(i - 1)] * phmm.p_ij(hmm, d_(i - 1), m_(i))
        a_m += alphas_t[i_(i - 1)] * phmm.p_ij(hmm, i_(i - 1), m_(i))
        a_m += alphas_t[m_(i - 1)] * phmm.p_ij(hmm, m_(i - 1), m_(i))
        s += f" + {A}_{{{t1 - 1}}}({m_(i - 1)}) * p_{{ij}}({m_(i - 1)}, {m_(i)}) + " + \
                f"{A}_{t1 - 1}({i_(i - 1)}) * p_{{ij}}({i_(i - 1)}, {m_(i)})"
        alphas_t1[m_(i)] = a_m * phmm.p_o_in_i(hmm, o_t, m_(i))
        s += f") * p_o({o_t}|{m_(i)})"
        logger.debug(s)

    # Do the insert states
    for i in range(1, hmm.duration + 1):
        a_i = alphas_t[d_(i)] * phmm.p_ij(hmm, d_(i), i_(i))
        s = f"{A}_{t1}({i_(i)}) = ({A}_{{{t1-1}}}({d_(i)}) * p_{{ij}}({d_(i)}, {i_(i)})"
        a_i += alphas_t[m_(i)] * phmm.p_ij(hmm, m_(i), i_(i))
        a_i += alphas_t[i_(i)] * phmm.p_ij(hmm, i_(i), i_(i))
        s += f" + {A}_{t1 - 1}({m_(i)}) * p_{{ij}}({m_(i)}, {i_(i)}) + " + \
                f"{A}_{t1 - 1}({i_(i)}) * p_{{ij}}({i_(i)}, {i_(i)})"
        alphas_t1[i_(i)] = a_i * phmm.p_o_in_i(hmm, o_t, i_(i))
        s += f") * p_o({o_t}|{i_(i)})"
        logger.debug(s)

    # Do the delete states
    alphas_t1[d_(1)] = alphas_t1[i_(0)] * phmm.p_ij(hmm, i_(0), d_(1))
    logger.debug(f"{A}_{t1}({d_(1)}) = {A}_{t1}({i_(0)}) * p_{{ij}}({i_(0)}, {d_(1)}))")
    for i in range(2, hmm.duration + 1):
        a_d = 0
        a_d = alphas_t1[d_(i - 1)] * phmm.p_ij(hmm, d_(i - 1), d_(i))
        a_d += alphas_t1[m_(i - 1)] * phmm.p_ij(hmm, m_(i - 1), d_(i))
        alphas_t1[d_(i)] = a_d + alphas_t1[i_(i - 1)] * phmm.p_ij(hmm, i_(i - 1), d_(i))
        logger.debug(f"{A}_{t1}({d_(i)}) = ({A}_{t1}({d_(i-1)}) * p_{{ij}}({d_(i-1)}, {d_(i)}) + "
                     f"{A}_{t1}({m_(i-1)}) * p_{{ij}}({m_(i-1)}, {d_(i)}) + " +
                     f"{A}_{t1}({i_(i-1)}) * p_{{ij}}({i_(i-1)}, {d_(i)}))")

    alphas_t1 = sort_alphas(alphas_t1, states)

    return alphas_t1


def _end(hmm: phmm.Hmm, alphas_t: Dict[Any, float], T: int) -> Dict[Any, float]:
    alphas_t1 = {}
    i = hmm.duration
    a_e = alphas_t[d_(i)] * phmm.p_ij(hmm, d_(i), hmm.END)
    a_e += alphas_t[i_(i)] * phmm.p_ij(hmm, i_(i), hmm.END)
    a_e += alphas_t[m_(i)] * phmm.p_ij(hmm, m_(i), hmm.END)
    alphas_t1[hmm.END] = a_e
    logger.debug(f"{A}_{T}({hmm.END}) = " +
                 f"{A}_{T-1}({d_(i)}) * p_{{ij}}({d_(i)}, {hmm.END}) + " +
                 f"{A}_{T-1}({i_(i)}) * p_{{ij}}({i_(i)}, {hmm.END}) + " +
                 f"{A}_{T-1}({m_(i)}) * p_{{ij}}({m_(i)}, {hmm.END})"
                 )
    return alphas_t1


def forward(hmm: phmm.Hmm, observations: List[Any]) -> Tuple[Dict[int, Dict[Any, float]], Dict[int, float]]:

    states = hmm.hiddens.copy()
    states.remove('start')
    states.remove('end')

    scalers = {-1: 1.}
    alphas = {}
    logger.debug("INITIAL alphas\n======================================")
    a_0 = _initial(hmm)
    scalers[-1] = get_scaler(a_0)
    scalers[-1] = 1
    alphas[-1] = normalize(a_0, scalers[-1])

    o_t = observations[0]
    alphas_t = _between_o1(hmm, o_t, alphas[-1], 0, states)
    scalers[0] = get_scaler(alphas_t)
    scalers[0] = 1.
    alphas[0] = normalize(alphas_t, scalers[0])

    for t in range(1, len(observations)):
        logger.debug(f"\n{A}_{t}\n======================================")
        o_t = observations[t]
        alphas_t = _between(hmm, o_t, alphas[t - 1], t, states)
        scalers[t] = get_scaler(alphas_t)
        scalers[t] = 1.
        alphas[t] = normalize(alphas_t, scalers[t])

    t = len(observations)
    logger.debug(f"\n{A}_{t}\n======================================")
    a_T = _end(hmm, alphas[t - 1], t)
    scalers[t] = get_scaler(a_T)
    scalers[t] = 1.
    alphas[t] = normalize(a_T, scalers[t])

    return alphas, scalers

