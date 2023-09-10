import unittest
import sys
sys.path.insert(0, '/opt/project/baum_welch')
# import phmm
import baum_welch.phmm as phmm


class TestPhmmCreation(unittest.TestCase):

    def test_get_delete_states(self):
        hmm = phmm.Hmm(3)
        states = phmm._phmm_get_delete_states(hmm)
        self.assertEqual(3, len(states))

        prefix = hmm.DELETE_STATE_PREFIX
        self.assertIn('{:s}_1'.format(prefix), states)
        self.assertIn('{:s}_2'.format(prefix), states)
        self.assertIn('{:s}_3'.format(prefix), states)

    def test_get_insert_states(self):
        hmm = phmm.Hmm(3)
        states = phmm._phmm_get_insert_states(hmm)
        self.assertEqual(4, len(states))

        prefix = hmm.INSERT_STATE_PREFIX
        self.assertIn('{:s}_0'.format(prefix), states)
        self.assertIn('{:s}_1'.format(prefix), states)
        self.assertIn('{:s}_2'.format(prefix), states)
        self.assertIn('{:s}_3'.format(prefix), states)

    def test_get_match_states(self):
        hmm = phmm.Hmm(3)
        states = phmm._phmm_get_match_states(hmm)
        self.assertEqual(3, len(states))

        prefix = hmm.MATCH_STATE_PREFIX
        self.assertIn('{:s}_1'.format(prefix), states)
        self.assertIn('{:s}_2'.format(prefix), states)
        self.assertIn('{:s}_3'.format(prefix), states)

    def test_get_transitions_for_start(self):
        hmm = phmm.Hmm(3)
        transitions = phmm._phmm_get_transitions_for_start(hmm)
        self.assertEqual(3, len(transitions))
        targets = [d for _, d, _ in transitions]
        self.assertIn('{:s}_1'.format(hmm.DELETE_STATE_PREFIX), targets)
        self.assertIn('{:s}_0'.format(hmm.INSERT_STATE_PREFIX), targets)
        self.assertIn('{:s}_1'.format(hmm.MATCH_STATE_PREFIX), targets)

    def test_get_transitions_for_end(self):
        hmm = phmm.Hmm(3)

        transitions = phmm._phmm_get_transitions_for_end(hmm, hmm.DELETE_STATE_PREFIX)
        self.assertEqual(2, len(transitions))
        targets = [t for t, t, _ in transitions]
        self.assertIn('{:s}_3'.format(hmm.INSERT_STATE_PREFIX), targets)
        self.assertIn(hmm.END, targets)
        self.assertEqual('{:s}_3'.format(hmm.DELETE_STATE_PREFIX), transitions[0][0])

        transitions = phmm._phmm_get_transitions_for_end(hmm, hmm.INSERT_STATE_PREFIX)
        self.assertEqual(2, len(transitions))
        targets = [t for t, t, _ in transitions]
        self.assertIn('{:s}_3'.format(hmm.INSERT_STATE_PREFIX), targets)
        self.assertIn(hmm.END, targets)
        self.assertEqual('{:s}_3'.format(hmm.INSERT_STATE_PREFIX), transitions[0][0])

        transitions = phmm._phmm_get_transitions_for_end(hmm, hmm.MATCH_STATE_PREFIX)
        self.assertEqual(2, len(transitions))
        targets = [t for t, t, _ in transitions]
        self.assertIn('{:s}_3'.format(hmm.INSERT_STATE_PREFIX), targets)
        self.assertIn(hmm.END, targets)
        self.assertEqual('{:s}_3'.format(hmm.MATCH_STATE_PREFIX), transitions[0][0])

    def test_get_transitions(self):
        def check_next_states(targets):
            self.assertIn('{:s}_2'.format(hmm.DELETE_STATE_PREFIX), targets)
            self.assertIn('{:s}_2'.format(hmm.MATCH_STATE_PREFIX), targets)
            self.assertIn('{:s}_1'.format(hmm.INSERT_STATE_PREFIX), targets)
        hmm = phmm.Hmm(3)

        transitions = phmm._phmm_get_transitions(hmm, hmm.DELETE_STATE_PREFIX, 1)
        self.assertEqual(3, len(transitions))
        check_next_states([t for _, t, _ in transitions])
        self.assertEqual('{:s}_1'.format(hmm.DELETE_STATE_PREFIX), transitions[0][0])

        transitions = phmm._phmm_get_transitions(hmm, hmm.INSERT_STATE_PREFIX, 1)
        self.assertEqual(3, len(transitions))
        check_next_states([t for _, t, _ in transitions])
        self.assertEqual('{:s}_1'.format(hmm.INSERT_STATE_PREFIX), transitions[0][0])

        transitions = phmm._phmm_get_transitions(hmm, hmm.MATCH_STATE_PREFIX, 1)
        self.assertEqual(3, len(transitions))
        check_next_states([t for _, t, _ in transitions])
        self.assertEqual('{:s}_1'.format(hmm.MATCH_STATE_PREFIX), transitions[0][0])

    def test_phmm_generation(self):
        hmm = phmm.basic_phmm(3, ['A', 'B', 'C', 'D'])
        self.assertEqual(3 + 3 + 4 + 2, len(hmm.hiddens))
        self.assertEqual(3 + 7 * 3 + 3 * 2, len(hmm.p_ij))
        self.assertEqual(7 * 4, len(hmm.p_o_in_i))

    def test_successors(self):
        hmm = phmm.basic_phmm(3, ['A', 'B', 'C', 'D'])

        n = hmm.START
        self.assertIn('{:s}_1'.format(hmm.DELETE_STATE_PREFIX), hmm.succs[n])
        self.assertIn('{:s}_0'.format(hmm.INSERT_STATE_PREFIX), hmm.succs[n])
        self.assertIn('{:s}_1'.format(hmm.MATCH_STATE_PREFIX), hmm.succs[n])

        n = '{:s}_2'.format(hmm.INSERT_STATE_PREFIX)
        self.assertIn('{:s}_3'.format(hmm.DELETE_STATE_PREFIX), hmm.succs[n])
        self.assertIn('{:s}_2'.format(hmm.INSERT_STATE_PREFIX), hmm.succs[n])
        self.assertIn('{:s}_3'.format(hmm.MATCH_STATE_PREFIX), hmm.succs[n])

    def test_predecessors(self):
        hmm = phmm.basic_phmm(3, ['A', 'B', 'C', 'D'])

        n = hmm.END
        self.assertIn('{:s}_3'.format(hmm.DELETE_STATE_PREFIX), hmm.preds[n])
        self.assertIn('{:s}_3'.format(hmm.INSERT_STATE_PREFIX), hmm.preds[n])
        self.assertIn('{:s}_3'.format(hmm.MATCH_STATE_PREFIX), hmm.preds[n])

        n = '{:s}_2'.format(hmm.INSERT_STATE_PREFIX)
        self.assertIn('{:s}_2'.format(hmm.DELETE_STATE_PREFIX), hmm.preds[n])
        self.assertIn('{:s}_2'.format(hmm.INSERT_STATE_PREFIX), hmm.preds[n])
        self.assertIn('{:s}_2'.format(hmm.MATCH_STATE_PREFIX), hmm.preds[n])


class TestLevenshteinPrior(unittest.TestCase):

    def test_aggregation_s1(self):
        med_seq = ['a', 'b', 'c']
        s1 = ['a', 'b', 'c']
        hmm = phmm.basic_phmm(len(med_seq), ['a', 'b', 'c', 'x'])
        trans = {k: 0 for k in hmm.p_ij.keys()}
        obs = {k: 0 for k in hmm.p_o_in_i.keys()}
        phmm._count_obs_and_trans(med_seq, s1, trans, obs)

        p_state = phmm.Hmm.START
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 1)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('a', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 2)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('b', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 3)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('c', state)])

        p_state = state
        state = phmm.Hmm.END
        self.assertEqual(1, trans[(p_state, state)])

    def test_aggregation_s2(self):
        med_seq = ['a', 'b', 'c']
        s2 = ['x', 'x', 'x', 'a', 'b', 'c']
        hmm = phmm.basic_phmm(len(med_seq), ['a', 'b', 'c', 'x'])
        trans = {k: 0 for k in hmm.p_ij.keys()}
        obs = {k: 0 for k in hmm.p_o_in_i.keys()}
        phmm._count_obs_and_trans(med_seq, s2, trans, obs)

        p_state = phmm.Hmm.START
        state = phmm._state_name(phmm.Hmm.INSERT_STATE_PREFIX, 0)
        self.assertEqual(1, trans[(p_state, state)])

        p_state = state
        self.assertEqual(2, trans[(p_state, state)])
        self.assertEqual(3, obs[('x', state)])

        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 1)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('a', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 2)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('b', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 3)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('c', state)])

        p_state = state
        state = phmm.Hmm.END
        self.assertEqual(1, trans[(p_state, state)])

    def test_aggregation_s3(self):
        med_seq = ['a', 'b', 'c']
        s3 = ['a', 'b', 'c', 'x', 'x', 'x']
        hmm = phmm.basic_phmm(len(med_seq), ['a', 'b', 'c', 'x'])
        trans = {k: 0 for k in hmm.p_ij.keys()}
        obs = {k: 0 for k in hmm.p_o_in_i.keys()}
        phmm._count_obs_and_trans(med_seq, s3, trans, obs)

        p_state = phmm.Hmm.START
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 1)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('a', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 2)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('b', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 3)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('c', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.INSERT_STATE_PREFIX, 3)
        self.assertEqual(1, trans[(p_state, state)])

        p_state = state
        self.assertEqual(2, trans[(p_state, state)])
        self.assertEqual(3, obs[('x', state)])

        p_state = state
        state = phmm.Hmm.END
        self.assertEqual(1, trans[(p_state, state)])

    def test_aggregation_s4(self):
        med_seq = ['a', 'b', 'c']
        s4 = ['a', 'x', 'x', 'x', 'x', 'c']
        hmm = phmm.basic_phmm(len(med_seq), ['a', 'b', 'c', 'x'])
        trans = {k: 0 for k in hmm.p_ij.keys()}
        obs = {k: 0 for k in hmm.p_o_in_i.keys()}
        phmm._count_obs_and_trans(med_seq, s4, trans, obs)

        p_state = phmm.Hmm.START
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 1)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('a', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.INSERT_STATE_PREFIX, 1)
        self.assertEqual(1, trans[(p_state, state)])

        p_state = state
        self.assertEqual(2, trans[(p_state, state)])
        self.assertEqual(3, obs[('x', state)])

        state = phmm._state_name(phmm.Hmm.DELETE_STATE_PREFIX, 2)
        self.assertEqual(1, trans[(p_state, state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.INSERT_STATE_PREFIX, 2)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('x', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 3)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('c', state)])

        p_state = state
        state = phmm.Hmm.END
        self.assertEqual(1, trans[(p_state, state)])

    def test_aggregation_s5(self):
        med_seq = ['a', 'b', 'c']
        s5 = ['a']
        hmm = phmm.basic_phmm(len(med_seq), ['a', 'b', 'c', 'x'])
        trans = {k: 0 for k in hmm.p_ij.keys()}
        obs = {k: 0 for k in hmm.p_o_in_i.keys()}
        phmm._count_obs_and_trans(med_seq, s5, trans, obs)

        p_state = phmm.Hmm.START
        state = phmm._state_name(phmm.Hmm.MATCH_STATE_PREFIX, 1)
        self.assertEqual(1, trans[(p_state, state)])
        self.assertEqual(1, obs[('a', state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.DELETE_STATE_PREFIX, 2)
        self.assertEqual(1, trans[(p_state, state)])

        p_state = state
        state = phmm._state_name(phmm.Hmm.DELETE_STATE_PREFIX, 3)
        self.assertEqual(1, trans[(p_state, state)])

        p_state = state
        state = phmm.Hmm.END
        self.assertEqual(1, trans[(p_state, state)])

    def test_median_sequence(self):
        sequences = [
            [s for s in 'abc'],
            [s for s in 'a'],
            [s for s in 'xxxabc'],
            [s for s in 'abcxxx'],
            [s for s in 'axxxc']
        ]
        median_seq = phmm._median_sequence(sequences)
        self.assertListEqual(sequences[0], median_seq)

    def test_all(self):
        sequences = [
            [s for s in 'abc'],
            [s for s in 'a'],
            [s for s in 'xxxabc'],
            [s for s in 'abcxxx'],
            [s for s in 'axxxc']
        ]
        new_hmm = phmm.hmm_from_sequences(sequences)
        totoal_prob = {}
        for k, v in new_hmm.p_ij.items():
            self.assertGreaterEqual(1, v, msg="p = {} for transition {}".format(v, str(k)))
            self.assertGreaterEqual(v, 0.0001 - 1e-9, msg="p = {} a 0.0001 for transition {}".format(v, str(k)))
            if k[0] not in totoal_prob:
                totoal_prob[k[0]] = 0.
            totoal_prob[k[0]] += v
        totoal_prob = {}
        for k, v in totoal_prob.items():
            self.assertAlmostEqual(1, v, msg="Transitions sum not to one for {}, are {}".format(k, v))
        for k, v in new_hmm.p_o_in_i.items():
            self.assertGreaterEqual(1, v, msg="p = {} > 1 for observation {} in state".format(v, k[0], k[1]))
            self.assertGreaterEqual(v, 0.0001 - 1e-9, msg="p = {} < 0.0001 for transition {}".format(v, str(k)))
            if k[1] not in totoal_prob:
                totoal_prob[k[1]] = 0.
            totoal_prob[k[1]] += v
        for k, v in totoal_prob.items():
            self.assertAlmostEqual(1, v, msg="Emissions sum not to one for {}, are {}".format(k, v))
        self.assertEqual(new_hmm.duration, 3)
