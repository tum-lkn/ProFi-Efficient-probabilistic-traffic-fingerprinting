"""
Implements HMM classes.
"""
import numpy as np
import Levenshtein as leven
import itertools as itt
import ctypes
from typing import List, Dict, Tuple, Any


class Hmm(object):
    """
    Class representing a hidden markov model.

    Attributes:
        hiddens (List[str]): List of hidden states.
        observables (List[str]): List of observable symbols.
        transitions (Dict[Tuple[str, str], float): Transition probabilities
            between states.
        preds (Dict[str, List[str]]): Predecessor states for a specific state.
        succs (Dict[str, List[str]]): Successor states for a specific state.
        p_o_in_i (Dict[Tuple[Any, str], float]): Emission probabilities for symbols
            for each hidden state.
    """
    START = 'start'
    END = 'end'
    DELETE_STATE_PREFIX = 'd'
    INSERT_STATE_PREFIX = 'i'
    MATCH_STATE_PREFIX = 'm'

    @classmethod
    def from_dict(cls, dictionary) -> 'Hmm':
        hmm = cls(dictionary['duration'], dictionary['seed'])
        hmm.p_ij = dictionary['p_ij']
        hmm.p_o_in_i = dictionary['p_o_in_i']
        hmm.hiddens = dictionary['hiddens']
        hmm.observables = dictionary['observables']
        hmm.preds = dictionary['preds']
        hmm.succs = dictionary['succs']
        return hmm

    def __init__(self, duration, seed: int = 1):
        """
        Initializes object.
        """
        self.duration = duration
        self.hiddens = []
        self.observables = []
        self.p_ij = {}
        self.preds = {}
        self.succs = {}
        self.p_o_in_i = {}
        # TODO: Remove the p_o_before_d again makes no sense.
        self.p_o_before_d = {}
        self.p_o_after_d = {}
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'p_ij': self.p_ij,
            'p_o_in_i': self.p_o_in_i,
            'seed': self.seed,
            'hiddens': self.hiddens,
            'observables': self.observables,
            'preds': self.preds,
            'succs': self.succs,
            'duration': self.duration
        }

    def add_nodes(self, nodes: List[str]):
        for n in nodes:
            self.hiddens.append(n)
            self.preds[n] = []
            self.succs[n] = []

    def add_transition(self, src: str, dst: str, p: float):
        self.p_ij[(src, dst)] = p
        self.succs[src].append(dst)
        self.preds[dst].append(src)

    def _finalize_p_after_d(self):
        """
        Cache the probability of observing a specific symbol when being in a
        certain delete state and moving forward.

        Method starts at the last delete state and visits every successor of
        this delete state. Calculatest the probability of observing a symbol
        by iterating over each neighbor and summing up for each symbol the
        probability of observing that symbol in the successor with the probability
        of transitioning into this successor.
        Then moves backwards along the delete states and do the same. Use the
        previously calculated probabilities of successor delete states.

        ```
        p(o|d) = \sum_{x\in\suc(d)}p_{ij}(x|d)p_o(o|x)
        ```

        Returns:

        """
        for t in range(self.duration, 0, -1):
            d_state = "{}_{:d}".format(self.DELETE_STATE_PREFIX, t)
            init = False
            for i, suc in enumerate(self.succs[d_state]):
                if is_end_state(suc) and t == self.duration:
                    self.p_o_after_d[(None, d_state)] = self.p_ij[(d_state, suc)]
                    continue
                for obs in self.observables:
                    if not init:
                        self.p_o_after_d[(obs, d_state)] = 0
                    if is_delete_state(suc):
                        p = self.p_ij[(d_state, suc)] * self.p_o_after_d[(obs, suc)]
                    else:
                        p = self.p_ij[(d_state, suc)] * self.p_o_in_i[(obs, suc)]
                    self.p_o_after_d[(obs, d_state)] += p
                init = True
                if t < self.duration and is_delete_state(suc):
                    self.p_o_after_d[(None, d_state)] = self.p_ij[(d_state, suc)] * self.p_o_after_d[(None, suc)]

    def _finalize_p_before_d(self):
        """
        Analogous to _finalize_p_after_d, just in the forward direction. That is
        start at delete state d_1 and move forwards in time until d_T.

        Returns:

        """

        # some stupid design issue since the incoming probabilities do not sum to one.
        for t in range(1, self.duration + 1):
            d_state = "{}_{:d}".format(self.DELETE_STATE_PREFIX, t)
            init = False
            for pred in self.preds[d_state]:
                if is_start_state(pred) and t == 1:
                    self.p_o_before_d[(None, d_state)] = self.p_ij[(pred, d_state)]
                    continue
                for obs in self.observables:
                    if not init:
                        self.p_o_before_d[(obs, d_state)] = 0
                    if is_delete_state(pred):
                        p = self.p_ij[(pred, d_state)] * self.p_o_before_d[(obs, pred)]
                    else:
                        p = self.p_ij[(pred, d_state)] * self.p_o_in_i[(obs, pred)]
                    self.p_o_before_d[(obs, d_state)] += p
                init = True
                if t > 1 and is_delete_state(pred):
                    self.p_o_before_d[(None, d_state)] = self.p_ij[(pred, d_state)] * self.p_o_before_d[(None, pred)]

    def finalize(self):
        # TODO: remove finalize p_before_d.
        # TODO: sort hidden states: delete, match, insert ascending based on time.
        self._finalize_p_before_d()
        self._finalize_p_after_d()

    def __len__(self):
        return self.duration


class PhmmC(object):
    
    def __init__(self, duration: int, obs_space: List[Any], init_prior: str = 'uniform', seed: int = 1):
        self.duration = duration
        self._p_ij = None
        self._p_ij_keys = None
        self._p_ij_p = None
        self._p_o_in_i = None
        self._p_o_in_i_keys = None
        self._p_o_in_i_p = None
        self._observables = obs_space
        self._sym_to_int = {s: i for i, s in enumerate(obs_space)}
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)
        self.init_params(init_prior, seed)

    def add_transitions(self, hmm: Hmm):
        self._p_ij = np.zeros(9 * hmm.duration + 3, dtype=np.float64)
        for i, v in enumerate(hmm.p_ij.values()):
            self._p_ij[i] = v
        self._p_ij_p = self._p_ij.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._p_ij_keys = hmm.p_ij.keys()

    def add_emissions(self, hmm: Hmm):
        self._p_o_in_i = np.zeros(len(self._observables) * (2 * hmm.duration + 1), dtype=np.float64)
        for i, v in enumerate(hmm.p_o_in_i.values()):
            self._p_o_in_i[i] = v
        self._p_o_in_i_p = self._p_o_in_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._p_o_in_i_keys = hmm.p_o_in_i.keys()

    def init_params(self, init_prior: str = 'uniform', seed: int = 1):
        hmm = basic_phmm(self.duration, self._observables, init_prior, seed)
        self.add_transitions(hmm)
        self.add_emissions(hmm)

    def update_p_ij(self, p_ij_n: ctypes.POINTER(ctypes.c_double)):
        self._p_ij_p = p_ij_n
        for i in range(len(self._p_ij)):
            self._p_ij[i] = p_ij_n[i]
    
    def update_p_o_in_i(self, p_o_in_i_n: ctypes.POINTER(ctypes.c_double)):
        self._p_o_in_i_p = p_o_in_i_n
        for i in range(len(self._p_o_in_i)):
            self._p_o_in_i[i] = p_o_in_i_n[i]

    def convert_to_phmm(self):
        hmm = Hmm(self.duration)
        hmm.observables = self._observables
        hmm.p_ij = dict(zip(self._p_ij_keys, self._p_ij))
        hmm.p_o_in_i = dict(zip(self._p_o_in_i_keys, self._p_o_in_i))
        return hmm

def is_delete_state(state: str) -> bool:
    return state.startswith(Hmm.DELETE_STATE_PREFIX)


def is_insert_state(state: str) -> bool:
    return state.startswith(Hmm.INSERT_STATE_PREFIX)


def is_match_state(state: str) -> bool:
    return state.startswith(Hmm.MATCH_STATE_PREFIX)


def is_start_state(state: str) -> bool:
    return state == Hmm.START


def is_end_state(state: str) -> bool:
    return state == Hmm.END


def emits_symbol(state: str) -> bool:
    if state.startswith(Hmm.MATCH_STATE_PREFIX):
        return True
    elif state.startswith(Hmm.INSERT_STATE_PREFIX):
        return True
    else:
        return False


def is_silent_state(state: str) -> bool:
    return not emits_symbol(state)


def _phmm_get_delete_states(hmm: Hmm) -> List[str]:
    """
    Add the delete states to the hidden states.

    Args:
        hmm:

    Returns:

    """

    return ['{:s}_{:d}'.format(hmm.DELETE_STATE_PREFIX, i + 1) for i in range(len(hmm))]


def _phmm_get_insert_states(hmm: Hmm) -> List[str]:
    """
    Add all insert states to the hidden states.

    Args:
        hmm:

    Returns:

    """
    return ['{:s}_{:d}'.format(hmm.INSERT_STATE_PREFIX, i) for i in range(len(hmm) + 1)]


def _phmm_get_match_states(hmm: Hmm) -> List[str]:
    """
    Add all match states to the HMM.

    Args:
        hmm:

    Returns:

    """
    return ['{:s}_{:d}'.format(hmm.MATCH_STATE_PREFIX, i + 1) for i in range(len(hmm))]


def _phmm_get_transitions_for_start(hmm, init_prior: str) -> List[Tuple[str, str, float]]:
    """
    Make all transitions from the start state to their respective end states.
    Args:
        hmm:

    Returns:

    """
    if init_prior == 'prefer_match':
        probs = hmm.random.dirichlet([500., 100., 100.])
    elif init_prior == 'uniform':
        probs = hmm.random.dirichlet([100., 100., 100.])
    else:
        raise ValueError("Unknown prior for transition init: {}".format(init_prior))
    transitions = [
        (hmm.START, '{:s}_1'.format(hmm.DELETE_STATE_PREFIX), probs[2]),
        (hmm.START, '{:s}_1'.format(hmm.MATCH_STATE_PREFIX), probs[0]),
        (hmm.START, '{:s}_0'.format(hmm.INSERT_STATE_PREFIX), probs[1])
        
    ]
    return transitions


def _phmm_get_transitions_for_end(hmm: Hmm, prefix: str) -> List[Tuple[str, str, float]]:
    t = len(hmm)
    probs = hmm.random.dirichlet([100.] * 2)
    return [
        (
            '{:s}_{:d}'.format(prefix, t),
            '{:s}_{:d}'.format(hmm.INSERT_STATE_PREFIX, t),
            probs[1]
        ),
        ('{:s}_{:d}'.format(prefix, t), hmm.END, probs[0])
    ]


def _phmm_get_transitions(hmm: Hmm, prefix: str, t: int, init_prior: str) -> List[Tuple[str, str, float]]:
    if prefix == hmm.MATCH_STATE_PREFIX and init_prior == 'prefer_match':
        probs = hmm.random.dirichlet([100., 100., 500.])
    else:
        probs = hmm.random.dirichlet([100., 100., 100.])
    return [
        (
            '{:s}_{:d}'.format(prefix, t),
            '{:s}_{:d}'.format(hmm.DELETE_STATE_PREFIX, t + 1),
            probs[0]
        ),
        (
            '{:s}_{:d}'.format(prefix, t),
            '{:s}_{:d}'.format(hmm.MATCH_STATE_PREFIX, t + 1),
            probs[2]
        ),
        (
            '{:s}_{:d}'.format(prefix, t),
            '{:s}_{:d}'.format(hmm.INSERT_STATE_PREFIX, t),
            probs[1]
        )
    ]


def basic_phmm(duration: int, observation_space: List[Any], init_prior='uniform',
               seed: int = 1) -> Hmm:
    """
    Args:
        duration: Length of the chain in the hidden state, i.e., number of
            match or delete states.
        observation_space: Observable symbols.
        init_prior: Prior used to initialize transitions. Can be in {uniform,
            prefer_match}. For uniform, transitions are sampled from a dirichlet
            prior with uniform concentration paramters. In case of prefer_match,
            transitions from match states to match states have higher probability.

    Returns:

    """
    def add_transitions(gen: callable, **kwargs):
        for s, t, p in gen(**kwargs):
            hmm.add_transition(s, t, p)

    hmm = Hmm(duration, seed)
    hmm.observables = observation_space
    hmm.add_nodes([hmm.START])
    # The order of adding the nodes must be like this, this order is exploitet
    # in the forward and backward algorithm. If the order is changed, the
    # algorithm fails.
    hmm.add_nodes(_phmm_get_delete_states(hmm))
    hmm.add_nodes(_phmm_get_match_states(hmm))
    hmm.add_nodes(_phmm_get_insert_states(hmm))
    hmm.add_nodes([hmm.END])

    prefixes = [hmm.DELETE_STATE_PREFIX, hmm.MATCH_STATE_PREFIX, hmm.INSERT_STATE_PREFIX]
    add_transitions(_phmm_get_transitions_for_start, hmm=hmm, init_prior=init_prior)
    for prefix in prefixes:
        if prefix == hmm.INSERT_STATE_PREFIX:
            add_transitions(_phmm_get_transitions, hmm=hmm, prefix=hmm.INSERT_STATE_PREFIX, t=0, init_prior=init_prior)
        for t in range(1, len(hmm)):
            add_transitions(_phmm_get_transitions, hmm=hmm, prefix=prefix, t=t, init_prior=init_prior)
        add_transitions(_phmm_get_transitions_for_end, hmm=hmm, prefix=prefix)

    states = hmm.hiddens[len(hmm) + 1: len(hmm.hiddens) - 1]
    for state in states:
        probs = hmm.random.dirichlet([100.] * len(observation_space))
        for i, obs in enumerate(observation_space):
            hmm.p_o_in_i[(obs, state)] = probs[i]

    emissions = {}
    for i in observation_space:
        emissions.update({(k1, k2): v for (k1, k2), v in hmm.p_o_in_i.items() if k1 == i})
    hmm.p_o_in_i = emissions
    hmm.finalize()
    return hmm


def basic_phmm_c(duration: int, observation_space: List[Any], init_prior='uniform',
               seed: int = 1) -> PhmmC:
    return PhmmC(duration, observation_space, init_prior, seed)


def p_o_in_i(hmm, observation: Any, state: str):
    """
    Get the probability of making the observation `observation` in state
    `state`.

    Args:
        observation:
        state:

    Returns:
        1 if `state` is a delete state, else the emission probability of
            `observation` in state `state`.
    """
    if is_silent_state(state) or not state:
        return 1.
    else:
        return hmm.p_o_in_i.get((observation, state), 1e-12)


def p_ij(hmm: Hmm, state_i: str, state_j: str) -> float:
    """
    Get the probability of transitioning from state_i to state_j.

    Args:
        hmm:
        state_i:
        state_j:

    Returns:

    """
    return hmm.p_ij[state_i, state_j]


def _leven_dist_matrix(sequences: List[List[str]]) -> np.array:
    """
    Calculate the distance matrix between all pairs of sequences.

    Args:
        sequences:

    Returns:

    """
    dist = np.zeros((len(sequences), len(sequences)), dtype=np.float32)
    for i in range(len(sequences) - 1):
        for j in range(i + 1, len(sequences)):
            d = leven.distance(''.join(sequences[i]), ''.join(sequences[j]))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _median_sequence(sequences: List[List[str]]) -> List[str]:
    """
    Get the median sequence out of a list of sequences.
    The returned sequence has the minimum median distance to all other
    distances.

    Args:
        sequnces:

    Returns:

    """
    all_dist = _leven_dist_matrix(sequences)
    medians = np.median(all_dist, axis=1)
    return sequences[int(np.argmin(medians))]


def _state_name(prefix: str, t: int) -> str:
    return '{:s}_{:d}'.format(prefix, t)


def _fast_forward_to_edit_op(sequence: List[str], current_state: str, end_t: int,
                             cur_t: int, transitions: Dict[Tuple[str, str], float],
                             observations: Dict[Tuple[str, str], float]) -> Tuple[str, int]:
    """
    Insert match states for the symbols emitted from `cur_t` between `end_t` in
    `sequence`.

    Args:
        sequence: Sequenec of symbols.
        current_state: Current state in the phmm,
        end_t: Last t included in the sequence of match states.
        cur_t: Time step in which sequence of match states begins.
        transitions: Transitions, updated.
        observations: Observations, updated.

    Returns:
        current_state: The state in which the insertion ends.
        t: The new time step.
    """
    for i in range(cur_t, end_t):
        state = _state_name(Hmm.MATCH_STATE_PREFIX, i + 1)
        transitions[(current_state, state)] += 1
        observations[(sequence[i], state)] += 1
        current_state = state
    return current_state, end_t


def _count_obs_and_trans(median_seq: List[str], sequence: List[str],
                         transitions: Dict[Tuple[str, str], float],
                         observations: Dict[Tuple[str, str], float]) -> None:
    """
    Update the passed dictionaries with counts of the alignments computed from
    the median sequence and all other sequences.
    
    Args:
        median_seq: 
        sequences: 
        transitions: 
        observations: 

    Returns:

    """
    # Get operations that transforms the current sequence to the median sequence.
    # Examples:
    #   - delete, 0, 0: Delete the nth symbol (here zeroth) in the source sequence.
    #   - insert, 3, 2: Insert destination[2] after the fourth symbol (zero based
    #       counting) in the source sequence.
    #   - replace, 3, 3: Replace the fourth symbol in the source with the
    #       fourth symbol in the destination.
    # Use the median sequence as source and the other sequence as destination.
    # If they are the same, then nothing needs to be done. Else, we would
    # need to change the median sequence in order to get the other one.
    edit_ops = leven.editops(''.join(median_seq), ''.join(sequence))
    current_state = Hmm.START
    t = 0
    if len(edit_ops) == 0:
        # No edit operations required --> sequences are identical.
        current_state, t = _fast_forward_to_edit_op(sequence, current_state,
                                                    len(sequence), 0, transitions,
                                                    observations)
    for op, src_idx, dst_idx in edit_ops:
        if t < src_idx:
            # Fast forward position. Since no edit operations are required,
            # the sequences are the same for this stretch and thus match.
            # Pass the median_seq here and *not* the sequence since the indices
            # t and src_idx are relativ to the positions in the median_seq.
            current_state, t = _fast_forward_to_edit_op(
                sequence=median_seq,
                current_state=current_state,
                end_t=src_idx,
                cur_t=t,
                transitions=transitions,
                observations=observations
            )
        if op == 'delete':
            # Insert a transition from the current state to the next delete
            # state. No symbol is emitted.
            state = _state_name(Hmm.DELETE_STATE_PREFIX, t + 1)
            transitions[(current_state, state)] += 1
            current_state = state
            t += 1
        elif op == 'replace':
            # Replace operation corresponds to a delete and then an insert
            # transition. The emitted symbol in the insert state corresponds to
            # the symbol in the destination sequence referenced by `dst_idx`.
            state_d = _state_name(Hmm.DELETE_STATE_PREFIX, t + 1)
            state_i = _state_name(Hmm.INSERT_STATE_PREFIX, t + 1)
            transitions[(current_state, state_d)] += 1
            transitions[(state_d, state_i)] += 1
            observations[(sequence[dst_idx], state_i)] += 1
            current_state = state_i
            t += 1
        else:
            # Last remaining edit operation is insert. Corresponds to a transition
            # from the current state to the insert state. The emitted symbol is
            # the one from the target sequence referenced by `dst_idx`.
            state = _state_name(Hmm.INSERT_STATE_PREFIX, t)
            transitions[(current_state, state)] += 1
            observations[(sequence[dst_idx], state)] += 1
            current_state = state
    if t < len(median_seq):
        # for all symbols to have been generated, t must have the same value
        # as the sequence number of symbols. If this is not the case, then not
        # all elements are generated. Since t is smaller than the median sequence,
        # the end of the median sequence from t on and the other sequence
        # match --> insert sequence of match states.
        current_state, t = _fast_forward_to_edit_op(
            sequence=median_seq,
            current_state=current_state,
            end_t=len(median_seq),
            cur_t=t,
            transitions=transitions,
            observations=observations
        )
    transitions[(current_state, Hmm.END)] += 1


def _count_outgoing(dist: Dict[Tuple[str, str], float], agg_pos: int) -> Dict[str, float]:
    """
    Count the number of outgoing edges and sum them up. Yields denominator for
    each state to get a correct distribution.

    Args:
        dist:
        agg_pos: The symbol to which one should aggregate can either be in the
            first or second position of the tuple serving as key in the passed
            count dictionary. Control with this parameter whether the first or
            the second element should be aggregated to.

    Returns:

    """
    counts = {}
    for pair in dist.keys():
        a = pair[agg_pos]
        if a not in counts:
            counts[a] = 0.
        counts[a] += dist[pair]
    return counts


def _count_zeros(dist: Dict[Tuple[str, str], float], agg_pos: int) -> Dict[str, float]:
    """
    Count the number of zero transitions that occur in the counts distribution.

    Args:
        dist:
        agg_pos:

    Returns:

    """
    counts = {}
    for pair, v in dist.items():
        a = pair[agg_pos]
        if a not in counts:
            counts[a] = 0.
        if v == 0:
            counts[a] += 1
    return counts


def normalize_dist(values: List[float], clip_min: float) -> List[float]:
    num_zeros = 0.
    denominator = 0.
    for v in values:
        if v < clip_min:
            num_zeros += 1
        else:
            denominator += v
    pseudocount = clip_min * denominator / (1. - clip_min * num_zeros)
    clipped_values = []
    for v in values:
        if v < clip_min:
            v = pseudocount
        else:
            pass
        clipped_values.append(v / (denominator + (num_zeros * pseudocount)))
    return clipped_values


def _normalize_counts(dist: Dict[Tuple[str, str], float], agg_pos: int,
                      fanout: callable) -> Dict[Tuple[str, str], float]:
    """
    Create a categorical distribution out of the count values. Add pseudocounts
    to the actual counts in order to give each transition a certain minimal
    probability.
    Note that this minimum probability can be larger than an observed transition
    depending on how often that transition is observed.

    Args:
        dist:
        agg_pos:

    Returns:

    """
    denominators = _count_outgoing(dist, agg_pos)
    zeroes = _count_zeros(dist, agg_pos)
    normalized = {}
    # That value is a reasonalbe guess. The sequences to train on are in the
    # order of hundreds. Thus, even if a transition is occured only once,
    # the probability of 1 / (n * 100) > 1 / 10000.
    p = 0.0001
    for pair, v in dist.items():
        k = pair[agg_pos]
        pseudocount = p * denominators[k] / (1. - p * zeroes[k])
        if v == 0:
            v = p * denominators[k] / (1. - p * zeroes[k])
        if denominators[k] == 0:
            # Nothing leaves this node, i.e., was never visited in the first place.
            # Assign equal probability to each outgoing arc.
            v = 1.
            C = fanout(k)
            pseudocount = 0.
        else:
            C = denominators[k]
        normalized[pair] = v / (C + (zeroes[k] * pseudocount))
    return normalized


def _normalize_transition_counts(transitions: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    return _normalize_counts(transitions, 0, lambda x: 4. if x.startswith(Hmm.INSERT_STATE_PREFIX) else 3.)


def _normalize_observation_counts(observations: Dict[Tuple[str, str], float],
                                  num_observables: int) -> Dict[Tuple[str, str], float]:
    return _normalize_counts(observations, 1, lambda x: num_observables)


def _get_observables(sequences: List[List[Any]]) -> List[Any]:
    """
    Reduce all duplicates.

    Args:
        sequences:

    Returns:

    """
    return np.unique(np.concatenate(sequences)).tolist()


def hmm_from_sequences(sequences: List[List[str]], seed: int=1) -> Hmm:
    """
    Estimate a PHMM from a set of sequences.

    The process is as follows:
    1) From the passed sequences, identify to one with the minimum median
        Levenshtein distance to all other sequences.
    2) Use the median sequence to create an alignment for every other sequence
        using the edit operations necessary to transform the median sequence to
        the other sequence.
    3) Translate the edit operations into transitions of the PHMM and count
        the resulting transitions and emissions.
    4) From the count values, obtain a proper discrete probability distribution
        for emission and transition model.
    To avoid overfitting, pseudo counts are used in the last step. Each event
    in the discrete distributions (i.e., transition from hidden states and
        emissions) has a minimum probability of 0.0001. If a hidden state has
        never been visited, then all outgoing transitions and emissions have
        equal probability.
        The value of p = 0.0001 is motivated by the fact that we have in the order
        of 100s sequences from which the HMM is estimated.  Thus, even if an emission
        or transition is observed only once, this probability would still be larger
        than 0.0001.

    Args:
        hmm:
        sequences:

    Returns:

    """
    # The levensthein distance works with strings. Since symbols in our sequences
    # can be strings themselves, transform them to unicode characters. This
    # adapts our sequences to the interface of the Levenshtein package.
    observables = _get_observables(sequences)
    unicode_to_symbol = {chr(i): k for i, k in enumerate(observables)}
    symbol_to_unicode = {k: chr(i) for i, k in enumerate(observables)}

    # Initialize the prior distributions. Map the symbols of the observation
    # distibution prior into unicode space. The length of the HMM equals the
    # length of the median sequence.
    u_sequences = [[symbol_to_unicode[s] for s in seq] for seq in sequences]
    median_seq = _median_sequence(u_sequences)
    hmm = basic_phmm(len(median_seq), observables, seed=seed)
    trans = {k: 0. for k in hmm.p_ij.keys()}
    obs = {(symbol_to_unicode[symbol], state): 0. for symbol, state in hmm.p_o_in_i.keys()}

    for sequence in u_sequences:
        _count_obs_and_trans(median_seq, sequence, trans, obs)
    trans = _normalize_transition_counts(trans)
    obs = _normalize_observation_counts(obs, len(observables))

    new_hmm = Hmm(hmm.duration, seed=hmm.seed)
    new_hmm.observables = hmm.observables
    new_hmm.hiddens = hmm.hiddens
    new_hmm.p_ij = trans
    new_hmm.preds = hmm.preds
    new_hmm.succs = hmm.succs
    # Map the unicode symbols in the observation prior back to the actual
    # symbols that we use.
    new_hmm.p_o_in_i = {(unicode_to_symbol[u_symbol], state): v
                        for (u_symbol, state), v in obs.items()}
    return new_hmm
