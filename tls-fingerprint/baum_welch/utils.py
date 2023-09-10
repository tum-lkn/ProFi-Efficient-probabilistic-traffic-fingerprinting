import numpy as np
import networkx as nx
import learning
import phmm
import json
from typing import List, Dict, Tuple, Any


def _dummy_transitions() -> nx.DiGraph:
    g = nx.DiGraph()
    # Step 0 - A
    g.add_edge('start', 'm_01', p=0.9998)
    g.add_edge('start', 'i_00', p=0.0001)
    g.add_edge('start', 'd_01', p=0.0001)
    # Step 1 - B
    g.add_edge('i_00', 'm_01', p=0.9998)
    g.add_edge('i_00', 'i_00', p=0.0001)
    g.add_edge('i_00', 'd_01', p=0.0001)

    g.add_edge('m_01', 'm_02', p=0.9998)
    g.add_edge('m_01', 'i_01', p=0.0001)
    g.add_edge('m_01', 'd_02', p=0.0001)

    g.add_edge('d_01', 'm_02', p=0.9998)
    g.add_edge('d_01', 'i_01', p=0.0001)
    g.add_edge('d_01', 'd_01', p=0.0001)
    # Step 2 - C
    g.add_edge('i_01', 'm_02', p=0.9998)
    g.add_edge('i_01', 'i_01', p=0.0001)
    g.add_edge('i_01', 'd_02', p=0.0001)

    g.add_edge('m_02', 'm_03', p=0.9998)
    g.add_edge('m_02', 'i_02', p=0.0001)
    g.add_edge('m_02', 'd_03', p=0.0001)

    g.add_edge('d_02', 'm_03', p=0.9998)
    g.add_edge('d_02', 'i_02', p=0.0001)
    g.add_edge('d_02', 'd_03', p=0.0001)
    # Step 3 - D
    g.add_edge('i_02', 'm_03', p=0.9998)
    g.add_edge('i_02', 'i_02', p=0.0001)
    g.add_edge('i_02', 'd_03', p=0.0001)

    g.add_edge('m_03', 'm_04', p=0.9998)
    g.add_edge('m_03', 'i_03', p=0.0001)
    g.add_edge('m_03', 'd_04', p=0.0001)

    g.add_edge('d_03', 'm_04', p=0.9998)
    g.add_edge('d_03', 'i_03', p=0.0001)
    g.add_edge('d_03', 'd_04', p=0.0001)
    # Step 4 - A
    g.add_edge('i_03', 'm_04', p=0.9998)
    g.add_edge('i_03', 'i_03', p=0.0001)
    g.add_edge('i_03', 'd_04', p=0.0001)

    g.add_edge('m_04', 'm_05', p=0.4)
    g.add_edge('m_04', 'i_04', p=0.3)
    g.add_edge('m_04', 'd_05', p=0.3)

    g.add_edge('d_04', 'm_05', p=0.4)
    g.add_edge('d_04', 'i_04', p=0.3)
    g.add_edge('d_04', 'd_05', p=0.3)
    # Step 5 - B
    g.add_edge('i_04', 'm_05', p=0.4)
    g.add_edge('i_04', 'i_04', p=0.3)
    g.add_edge('i_04', 'd_05', p=0.3)

    g.add_edge('m_05', 'm_06', p=0.4)
    g.add_edge('m_05', 'i_05', p=0.3)
    g.add_edge('m_05', 'd_06', p=0.3)

    g.add_edge('d_05', 'm_06', p=0.4)
    g.add_edge('d_05', 'i_05', p=0.3)
    g.add_edge('d_05', 'd_06', p=0.3)
    # Step 5 - C
    g.add_edge('i_05', 'm_06', p=0.4)
    g.add_edge('i_05', 'i_05', p=0.3)
    g.add_edge('i_05', 'd_06', p=0.3)

    g.add_edge('m_06', 'm_07', p=0.4)
    g.add_edge('m_06', 'i_06', p=0.3)
    g.add_edge('m_06', 'd_07', p=0.3)

    g.add_edge('d_06', 'm_07', p=0.4)
    g.add_edge('d_06', 'i_06', p=0.3)
    g.add_edge('d_06', 'd_07', p=0.3)
    # Step 5 - D
    g.add_edge('i_06', 'm_07', p=0.4)
    g.add_edge('i_06', 'i_06', p=0.3)
    g.add_edge('i_06', 'd_07', p=0.3)

    g.add_edge('m_07', 'm_08', p=0.9998)
    g.add_edge('m_07', 'i_07', p=0.0001)
    g.add_edge('m_07', 'd_08', p=0.0001)

    g.add_edge('d_07', 'm_08', p=0.9998)
    g.add_edge('d_07', 'i_07', p=0.0001)
    g.add_edge('d_07', 'd_08', p=0.0001)
    # Step 5 - A
    g.add_edge('i_07', 'm_08', p=0.9998)
    g.add_edge('i_07', 'i_07', p=0.0001)
    g.add_edge('i_07', 'd_08', p=0.0001)

    g.add_edge('m_08', 'm_09', p=0.9998)
    g.add_edge('m_08', 'i_08', p=0.0001)
    g.add_edge('m_08', 'd_09', p=0.0001)

    g.add_edge('d_08', 'm_09', p=0.9998)
    g.add_edge('d_08', 'i_08', p=0.0001)
    g.add_edge('d_08', 'd_09', p=0.0001)
    # Step 5 - B
    g.add_edge('i_08', 'm_09', p=0.9998)
    g.add_edge('i_08', 'i_08', p=0.0001)
    g.add_edge('i_08', 'd_09', p=0.0001)

    g.add_edge('m_09', 'm_10', p=0.9998)
    g.add_edge('m_09', 'i_09', p=0.0001)
    g.add_edge('m_09', 'd_10', p=0.0001)

    g.add_edge('d_09', 'm_10', p=0.9998)
    g.add_edge('d_09', 'i_09', p=0.0001)
    g.add_edge('d_09', 'd_10', p=0.0001)
    # Step 5 - C
    g.add_edge('i_09', 'm_10', p=0.9998)
    g.add_edge('i_09', 'i_09', p=0.0001)
    g.add_edge('i_09', 'd_10', p=0.0001)

    g.add_edge('m_10', 'm_11', p=0.9998)
    g.add_edge('m_10', 'i_10', p=0.0001)
    g.add_edge('m_10', 'd_11', p=0.0001)

    g.add_edge('d_10', 'm_11', p=0.9998)
    g.add_edge('d_10', 'i_10', p=0.0001)
    g.add_edge('d_10', 'd_11', p=0.0001)
    # Step 5 - D
    g.add_edge('i_10', 'm_11', p=0.9998)
    g.add_edge('i_10', 'i_10', p=0.0001)
    g.add_edge('i_10', 'd_11', p=0.0001)

    g.add_edge('m_11', 'm_12', p=0.9998)
    g.add_edge('m_11', 'i_11', p=0.0001)
    g.add_edge('m_11', 'd_12', p=0.0001)

    g.add_edge('d_11', 'm_12', p=0.9998)
    g.add_edge('d_11', 'i_11', p=0.0001)
    g.add_edge('d_11', 'd_12', p=0.0001)

    g.add_edge('i_11', 'm_12', p=0.9998)
    g.add_edge('i_11', 'i_11', p=0.0001)
    g.add_edge('i_11', 'd_12', p=0.0001)

    g.add_edge('m_12', 'end', p=0.99)
    g.add_edge('d_12', 'end', p=0.99)
    g.add_edge('m_12', 'i_12', p=0.01)
    g.add_edge('d_12', 'i_12', p=0.01)
    g.add_edge('i_12', 'end', p=0.6)
    g.add_edge('i_12', 'i_12', p=0.4)
    return g


def _dummy_emissions() -> Dict[str, List[float]]:
    emissions = {
        'm_01': [0.997, 0.001, 0.001, 0.001, 0.00],
        'm_02': [0.001, 0.997, 0.001, 0.001, 0.00],
        'm_03': [0.001, 0.001, 0.997, 0.001, 0.00],
        'm_04': [0.001, 0.001, 0.001, 0.997, 0.00],
        'm_05': [0.997, 0.001, 0.001, 0.001, 0.00],
        'm_06': [0.001, 0.997, 0.001, 0.001, 0.00],
        'm_07': [0.001, 0.001, 0.997, 0.001, 0.00],
        'm_08': [0.001, 0.001, 0.001, 0.997, 0.00],
        'm_09': [0.997, 0.001, 0.001, 0.001, 0.00],
        'm_10': [0.001, 0.997, 0.001, 0.001, 0.00],
        'm_11': [0.001, 0.001, 0.997, 0.001, 0.00],
        'm_12': [0.001, 0.001, 0.001, 0.997, 0.00],

        'i_00': [0.997, 0.001, 0.001, 0.001, 0.00],
        'i_01': [0.001, 0.997, 0.001, 0.001, 0.00],
        'i_02': [0.001, 0.001, 0.997, 0.001, 0.00],
        'i_03': [0.001, 0.001, 0.001, 0.997, 0.00],
        'i_04': [0.25, 0.25, 0.25, 0.25, 0.00],
        'i_05': [0.25, 0.25, 0.25, 0.25, 0.00],
        'i_06': [0.25, 0.25, 0.25, 0.25, 0.00],
        'i_07': [0.25, 0.25, 0.25, 0.25, 0.00],
        'i_08': [0.997, 0.001, 0.001, 0.001, 0.00],
        'i_09': [0.001, 0.997, 0.001, 0.001, 0.00],
        'i_10': [0.001, 0.001, 0.997, 0.001, 0.00],
        'i_11': [0.001, 0.001, 0.001, 0.997, 0.00],
        'i_12': [0.001, 0.001, 0.001, 0.997, 0.00],

        'd_01': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_02': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_03': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_04': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_05': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_06': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_07': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_08': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_09': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_10': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_11': [0.00, 0.00, 0.00, 0.00, 1.00],
        'd_12': [0.00, 0.00, 0.00, 0.00, 1.00]
    }
    return emissions


def _next_state(current, transitions, random):
    neighbors = []
    p = []
    for i, n in enumerate(nx.neighbors(transitions, current)):
        neighbors.append(n)
        p.append(transitions.edges[current, n]['p'])
    return random.choice(neighbors, p=p)


def _dummy_walk(transitions: nx.DiGraph, emissions: Dict[str, List[float]], seed: int) -> List[np.array]:
    obs = []
    symbols = np.arange(5)
    random = np.random.RandomState(seed=seed)
    s_t = _next_state('start', transitions, random)
    while s_t != 'end':
        o_t = random.choice(symbols, p=emissions[s_t])
        if o_t == 4:
            pass
        else:
            obs.append(o_t)
        s_t = _next_state(s_t, transitions, random)
    return obs


def dummy_phmm_data(num_samples: int, start_seed: int) -> Dict[str, List[Any]]:
    transitions = _dummy_transitions()
    emissions = _dummy_emissions()

    data = {
        'sequences': [],
        'lengths': []
    }
    for i in range(num_samples):
        seq = _dummy_walk(transitions, emissions, start_seed + i)
        data['sequences'].append(seq)
        data['lengths'].append(len(seq))
    return data


def dummy_dummy_data(num_sampels: int) -> List[List[str]]:
    seq = ['a', 'b', 'c', 'a', 'b', 'c']
    return [seq for _ in range(num_sampels)]


if __name__ == "__main__":
    hmm = phmm.basic_phmm(4, ['a', 'b', 'c'], seed=3)
    hmm.p_o_in_i[('c', 'i_0')] = 0.98
    hmm.p_o_in_i[('a', 'i_0')] = 0.01
    hmm.p_o_in_i[('b', 'i_0')] = 0.01
    seqs = []
    for i in range(4):
        if i % 2 == 0:
            seqs.append(['a', 'b', 'a', 'b'])
        else:
            seqs.append(['c', 'a', 'b', 'a', 'b'])
    lprob = 1e9
    for i in range(20):
        hmm, changes, lprob = learning.baum_welch(hmm, seqs, 50)
        print(lprob)

    # seqs = [['A', 'B', 'B', 'A'] for i in range(10)]
    # hmm = phmm.basic_phmm(4, ['A', 'B'], seed=2)
    # lprob = 1e9
    # for i in range(20):
    #     hmm, changes, lprob = learning.baum_welch(hmm, seqs, 50)
    #     print(lprob)
    # print(seqs[0])

    # seqs = dummy_dummy_data(10)
    # hmm = phmm.basic_phmm(6, ['a', 'b', 'c'], seed=2)
    # lprob = 1e9
    # for i in range(20):
    #     hmm, changes, lprob = learning.baum_welch(hmm, seqs, 50)
    #     print(lprob)
    # print(seqs[0])
    m = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D'
    }
    data = dummy_phmm_data(500, 1)
    data_p = []
    for seq in data['sequences']:
        seq_p = []
        for o in seq:
            print(m[o], end=' ')
            seq_p.append(m[o])
        data_p.append(seq_p)
        print()
    hmm2 = phmm.basic_phmm(12, ['A', 'B', 'C', 'D'], seed=4)
    # hmm2 = phmm.hmm_from_sequences(sequences=data_p, seed=1)
    print(json.dumps({str(k): v for k, v in hmm2.p_o_in_i.items()}, indent=1))
    print(json.dumps({str(k): v for k, v in hmm2.p_ij.items()}, indent=1))
    hmm3 = hmm2
    blprob = 1e9
    lprob = blprob + 1
    for i in range(20):
        # blprob = 1e9
        if lprob < blprob:
            blprob = lprob
            # print(json.dumps(changes, indent=1))
            print("NEW BEST", lprob)
            with open(f"hmm-{i}.json", 'w') as fh:
                fh.write(json.dumps({
                    "obs": {str(k): v for k, v in hmm2.p_o_in_i.items()},
                    "trans": {str(k): v for k, v in hmm2.p_ij.items()}
                }, indent=1))
        print(i, lprob)
        hmm2, changes, lprob = learning.baum_welch(hmm2, data_p, 10)
    print(json.dumps({str(k): v for k, v in hmm2.p_o_in_i.items()}, indent=1))
    print(json.dumps({str(k): v for k, v in hmm2.p_ij.items()}, indent=1))
    # print(seqs[0])
