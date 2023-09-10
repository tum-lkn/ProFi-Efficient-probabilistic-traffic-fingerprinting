import unittest
import learning
from phmm import basic_phmm, p_ij, p_o_in_i
from utils import dummy_phmm_data


class TestSumPredAlphas(unittest.TestCase):

    def setUp(self) -> None:
        self.obs = ['A', 'B', 'C', 'D']
        self.hmm = basic_phmm(3, self.obs)

    def test_sum_pred_alphas(self):
        self.assertEqual(3, learning._sum_pred_alphas(self.hmm, 1, '', [], {}, [], 3))

    def test_calc_alpha(self):
        self.assertEqual(0, learning.calc_alpha(self.hmm, 3, self.hmm.START, [], {}))
        self.assertEqual(1, learning.calc_alpha(self.hmm, -1, self.hmm.START, [], {}))
        self.assertEqual(3, learning.calc_alpha(self.hmm, 1, 'd_1', [], {(1, 'd_1'): 3}))

        n = 'i_0'
        a_0_i0 = self.hmm.p_ij[(self.hmm.START, n)] * self.hmm.p_o_in_i[('A', n)]
        self.assertAlmostEqual(
            a_0_i0,
            learning.calc_alpha(self.hmm, 0, n, self.obs, {})
        )

        n = 'm_1'
        a_0_m0 = self.hmm.p_ij[(self.hmm.START, n)] * self.hmm.p_o_in_i[('A', n)]
        self.assertAlmostEqual(
            a_0_m0,
            learning.calc_alpha(self.hmm, 0, n, self.obs, {})
        )

        n = 'd_1'
        a_m1_d1 = self.hmm.p_ij[(self.hmm.START, n)]
        self.assertAlmostEqual(
            a_m1_d1,
            learning.calc_alpha(self.hmm, -1, n, self.obs, {})
        )

        a_0_d1 = self.hmm.p_ij[('i_0', 'd_1')] * a_0_i0
        self.assertAlmostEqual(
            a_0_d1,
            learning.calc_alpha(self.hmm, 0, n, self.obs, {})
        )

        n = 'i_1'
        a_0_i1 = self.hmm.p_ij[('d_1', n)] * self.hmm.p_o_in_i[('A', n)] * a_m1_d1
        self.assertAlmostEqual(
            a_0_i1,
            learning.calc_alpha(self.hmm, 0, n, self.obs, {})
        )

        n = 'm_2'
        a_0_m2 = self.hmm.p_ij[('d_1', n)] * self.hmm.p_o_in_i[('A', n)] * a_m1_d1
        self.assertAlmostEqual(
            a_0_m2,
            learning.calc_alpha(self.hmm, 0, n, self.obs, {})
        )

        a_1_m2 = (self.hmm.p_ij[('d_1', n)] * a_0_d1 +
                  self.hmm.p_ij[('i_1', n)] * a_0_i1 +
                  self.hmm.p_ij[('m_1', n)] * a_0_m0) * self.hmm.p_o_in_i[('B', n)]
        self.assertAlmostEqual(
            a_1_m2,
            learning.calc_alpha(self.hmm, 1, n, self.obs, {})
        )


class TestIterativeAlphas(unittest.TestCase):

    def setUp(self) -> None:
        hmm = basic_phmm(3, ['A', 'B', 'C', 'D'])
        alphas = {-1: {hmm.START: 1.}, 0: {}, 1: {}, 2: {}, 3: {}}
        scalers = {}
        observations = ['A', 'B', 'C']
        self.hmm = hmm
        self.alphas = alphas
        self.scalers = scalers
        self.observations = observations

        o_t = observations[0]
        alphas[0]['d_1'] = p_ij(hmm, hmm.START, 'd_1')
        alphas[0]['d_2'] = p_ij(hmm, 'd_1', 'd_2') * alphas[0]['d_1']
        alphas[0]['d_3'] = p_ij(hmm, 'd_2', 'd_3') * alphas[0]['d_2']
        alphas[0]['i_0'] = p_ij(hmm, hmm.START, 'i_0') * p_o_in_i(hmm, o_t, 'i_0')
        alphas[0]['i_1'] = alphas[0]['d_1'] * p_ij(hmm, 'd_1', 'i_1') * p_o_in_i(hmm, o_t, 'i_1')
        alphas[0]['i_2'] = alphas[0]['d_2'] * p_ij(hmm, 'd_2', 'i_2') * p_o_in_i(hmm, o_t, 'i_2')
        alphas[0]['i_3'] = alphas[0]['d_3'] * p_ij(hmm, 'd_3', 'i_3') * p_o_in_i(hmm, o_t, 'i_3')
        alphas[0]['m_1'] = p_ij(hmm, hmm.START, 'm_1') * p_o_in_i(hmm, o_t, 'm_1')
        alphas[0]['m_2'] = alphas[0]['d_1'] * p_ij(hmm, 'd_1', 'm_2') * p_o_in_i(hmm, o_t, 'm_2')
        alphas[0]['m_3'] = alphas[0]['d_2'] * p_ij(hmm, 'd_2', 'm_3') * p_o_in_i(hmm, o_t, 'm_3')
        scalers[0] = learning._normalize(alphas[0])

        o_t = observations[1]
        alphas[1]['d_1'] = alphas[0]['i_0'] * p_ij(hmm, 'i_0', 'd_1')
        alphas[1]['d_2'] = alphas[1]['d_1'] * p_ij(hmm, 'd_1', 'd_2') + \
                           alphas[0]['i_1'] * p_ij(hmm, 'i_1', 'd_2') + \
                           alphas[0]['m_1'] * p_ij(hmm, 'm_1', 'd_2')
        alphas[1]['d_3'] = alphas[1]['d_2'] * p_ij(hmm, 'd_2', 'd_3') + \
                           alphas[0]['i_2'] * p_ij(hmm, 'i_2', 'd_3') + \
                           alphas[0]['m_2'] * p_ij(hmm, 'm_2', 'd_3')
        alphas[1]['i_0'] = alphas[0]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * p_o_in_i(hmm, o_t, 'i_0')
        alphas[1]['i_1'] = (alphas[1]['d_1'] * p_ij(hmm, 'd_1', 'i_1') +
                            alphas[0]['i_1'] * p_ij(hmm, 'i_1', 'i_1') +
                            alphas[0]['m_1'] * p_ij(hmm, 'm_1', 'i_1')) * p_o_in_i(hmm, o_t, 'i_1')
        alphas[1]['i_2'] = (alphas[1]['d_2'] * p_ij(hmm, 'd_2', 'i_2') +
                            alphas[0]['i_2'] * p_ij(hmm, 'i_2', 'i_2') +
                            alphas[0]['m_2'] * p_ij(hmm, 'm_2', 'i_2')) * p_o_in_i(hmm, o_t, 'i_2')
        alphas[1]['i_3'] = (alphas[1]['d_3'] * p_ij(hmm, 'd_3', 'i_3') +
                            alphas[0]['i_3'] * p_ij(hmm, 'i_3', 'i_3') +
                            alphas[0]['m_3'] * p_ij(hmm, 'm_3', 'i_3')) * p_o_in_i(hmm, o_t, 'i_3')
        alphas[1]['m_1'] = alphas[0]['i_0'] * p_ij(hmm, 'i_0', 'm_1') * p_o_in_i(hmm, o_t, 'm_1')
        alphas[1]['m_2'] = (alphas[1]['d_1'] * p_ij(hmm, 'd_1', 'm_2') +
                            alphas[0]['i_1'] * p_ij(hmm, 'i_1', 'm_2') +
                            alphas[0]['m_1'] * p_ij(hmm, 'm_1', 'm_2')) * p_o_in_i(hmm, o_t, 'm_2')
        alphas[1]['m_3'] = (alphas[1]['d_2'] * p_ij(hmm, 'd_2', 'm_3') +
                            alphas[0]['i_2'] * p_ij(hmm, 'i_2', 'm_3') +
                            alphas[0]['m_2'] * p_ij(hmm, 'm_2', 'm_3')) * p_o_in_i(hmm, o_t, 'm_3')
        scalers[1] = learning._normalize(alphas[1])

        o_t = observations[2]
        alphas[2]['d_1'] = alphas[1]['i_0'] * p_ij(hmm, 'i_0', 'd_1')
        alphas[2]['d_2'] = alphas[2]['d_1'] * p_ij(hmm, 'd_1', 'd_2') + \
                           alphas[1]['i_1'] * p_ij(hmm, 'i_1', 'd_2') + \
                           alphas[1]['m_1'] * p_ij(hmm, 'm_1', 'd_2')
        alphas[2]['d_3'] = alphas[2]['d_2'] * p_ij(hmm, 'd_2', 'd_3') + \
                           alphas[1]['i_2'] * p_ij(hmm, 'i_2', 'd_3') + \
                           alphas[1]['m_2'] * p_ij(hmm, 'm_2', 'd_3')
        alphas[2]['i_0'] = alphas[1]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * p_o_in_i(hmm, o_t, 'i_0')
        alphas[2]['i_1'] = (alphas[2]['d_1'] * p_ij(hmm, 'd_1', 'i_1') +
                            alphas[1]['i_1'] * p_ij(hmm, 'i_1', 'i_1') +
                            alphas[1]['m_1'] * p_ij(hmm, 'm_1', 'i_1')) * p_o_in_i(hmm, o_t, 'i_1')
        alphas[2]['i_2'] = (alphas[2]['d_2'] * p_ij(hmm, 'd_2', 'i_2') +
                            alphas[1]['i_2'] * p_ij(hmm, 'i_2', 'i_2') +
                            alphas[1]['m_2'] * p_ij(hmm, 'm_2', 'i_2')) * p_o_in_i(hmm, o_t, 'i_2')
        alphas[2]['i_3'] = (alphas[2]['d_3'] * p_ij(hmm, 'd_3', 'i_3') +
                            alphas[1]['i_3'] * p_ij(hmm, 'i_3', 'i_3') +
                            alphas[1]['m_3'] * p_ij(hmm, 'm_3', 'i_3')) * p_o_in_i(hmm, o_t, 'i_3')
        alphas[2]['m_1'] = alphas[1]['i_0'] * p_ij(hmm, 'i_0', 'm_1') * p_o_in_i(hmm, o_t, 'm_1')
        alphas[2]['m_2'] = (alphas[2]['d_1'] * p_ij(hmm, 'd_1', 'm_2') +
                            alphas[1]['i_1'] * p_ij(hmm, 'i_1', 'm_2') +
                            alphas[1]['m_1'] * p_ij(hmm, 'm_1', 'm_2')) * p_o_in_i(hmm, o_t, 'm_2')
        alphas[2]['m_3'] = (alphas[2]['d_2'] * p_ij(hmm, 'd_2', 'm_3') +
                            alphas[1]['i_2'] * p_ij(hmm, 'i_2', 'm_3') +
                            alphas[1]['m_2'] * p_ij(hmm, 'm_2', 'm_3')) * p_o_in_i(hmm, o_t, 'm_3')
        scalers[2] = learning._normalize(alphas[2])

        alphas[3]['d_1'] = alphas[2]['i_0'] * p_ij(hmm, 'i_0', 'd_1')
        alphas[3]['d_2'] = alphas[3]['d_1'] * p_ij(hmm, 'd_1', 'd_2') + \
                           alphas[2]['i_1'] * p_ij(hmm, 'i_1', 'd_2') + \
                           alphas[2]['m_1'] * p_ij(hmm, 'm_1', 'd_2')
        alphas[3]['d_3'] = alphas[3]['d_2'] * p_ij(hmm, 'd_2', 'd_3') + \
                           alphas[2]['i_2'] * p_ij(hmm, 'i_2', 'd_3') + \
                           alphas[2]['m_2'] * p_ij(hmm, 'm_2', 'd_3')
        alphas[3][hmm.END] = alphas[3]['d_3'] * p_ij(hmm, 'd_3', hmm.END) + \
                             alphas[2]['i_3'] * p_ij(hmm, 'i_3', hmm.END) + \
                             alphas[2]['m_3'] * p_ij(hmm, 'm_3', hmm.END)
        scalers[3] = learning._normalize(alphas[3])

    def test_forward(self):
        alphas, scalers = learning.forward(self.hmm, self.observations)
        self.assertEqual(len(self.alphas), len(alphas))
        self.assertEqual(len(self.scalers), len(scalers))

        for k in self.alphas.keys():
            self.assertIn(k, alphas)
            self.assertEqual(len(self.alphas[k]), len(alphas[k]))
            for kk in self.alphas[k].keys():
                self.assertIn(kk, alphas[k])
                self.assertAlmostEqual(self.alphas[k][kk], alphas[k][kk])

    def test_forward2(self):
        alphas, scalers = learning.forward2(self.hmm, self.observations)
        self.assertEqual(len(alphas), 6)


class TestBackward(unittest.TestCase):

    def setUp(self) -> None:
        hmm = basic_phmm(3, ['A', 'B', 'C', 'D'])
        scalers = {}
        observations = ['A', 'B', 'C']
        self.hmm = hmm
        self.scalers = scalers
        self.observations = observations
        _, scalers = learning.forward(hmm, observations)
        self.scalers = scalers

        betas = {
            3: {i: scalers[3] for i in hmm.hiddens if not i == hmm.START},
            2: {},
            1: {},
            0: {}
        }
        self.betas = betas

        betas[2]['d_3'] = betas[3][hmm.END] * p_ij(hmm, 'd_3', hmm.END)
        betas[2]['d_2'] = betas[2]['d_3'] * p_ij(hmm, 'd_2', 'd_3')
        betas[2]['d_1'] = betas[2]['d_2'] * p_ij(hmm, 'd_1', 'd_2')
        betas[2]['i_3'] = betas[3][hmm.END] * p_ij(hmm, 'i_3', hmm.END)
        betas[2]['i_2'] = betas[2]['d_3'] * p_ij(hmm, 'i_2', 'd_3')
        betas[2]['i_1'] = betas[2]['d_2'] * p_ij(hmm, 'i_1', 'd_2')
        betas[2]['i_0'] = betas[2]['d_1'] * p_ij(hmm, 'i_0', 'd_1')
        betas[2]['m_3'] = betas[3][hmm.END] * p_ij(hmm, 'm_3', hmm.END)
        betas[2]['m_2'] = betas[2]['d_3'] * p_ij(hmm, 'm_2', 'd_3')
        betas[2]['m_1'] = betas[2]['d_2'] * p_ij(hmm, 'm_1', 'd_2')

        betas[2]['d_3'] *= scalers[2]
        betas[2]['d_2'] *= scalers[2]
        betas[2]['d_1'] *= scalers[2]
        betas[2]['i_3'] *= scalers[2]
        betas[2]['i_2'] *= scalers[2]
        betas[2]['i_1'] *= scalers[2]
        betas[2]['i_0'] *= scalers[2]
        betas[2]['m_3'] *= scalers[2]
        betas[2]['m_2'] *= scalers[2]
        betas[2]['m_1'] *= scalers[2]

        o_t = observations[2]
        betas[1]['d_3'] = betas[2]['i_3'] * p_ij(hmm, 'd_3', 'i_3') * p_o_in_i(hmm, o_t, 'i_3')
        betas[1]['d_2'] = betas[1]['d_3'] * p_ij(hmm, 'd_2', 'd_3') + \
                          betas[2]['m_3'] * p_ij(hmm, 'd_2', 'm_3') * p_o_in_i(hmm, o_t, 'm_3') + \
                          betas[2]['i_2'] * p_ij(hmm, 'd_2', 'i_2') * p_o_in_i(hmm, o_t, 'i_2')
        betas[1]['d_1'] = betas[1]['d_2'] * p_ij(hmm, 'd_1', 'd_2') + \
                          betas[2]['i_1'] * p_ij(hmm, 'd_1', 'i_1') * p_o_in_i(hmm, o_t, 'i_1') + \
                          betas[2]['m_2'] * p_ij(hmm, 'd_1', 'm_2') * p_o_in_i(hmm, o_t, 'm_2')
        betas[1]['i_3'] = betas[2]['i_3'] * p_ij(hmm, 'i_3', 'i_3') * p_o_in_i(hmm, o_t, 'i_3')
        betas[1]['i_2'] = betas[1]['d_3'] * p_ij(hmm, 'i_2', 'd_3') + \
                          betas[2]['i_2'] * p_ij(hmm, 'i_2', 'i_2') * p_o_in_i(hmm, o_t, 'i_2') + \
                          betas[2]['m_3'] * p_ij(hmm, 'i_2', 'm_3') * p_o_in_i(hmm, o_t, 'm_3')
        betas[1]['i_1'] = betas[1]['d_2'] * p_ij(hmm, 'i_1', 'd_2') + \
                          betas[2]['i_1'] * p_ij(hmm, 'i_1', 'i_1') * p_o_in_i(hmm, o_t, 'i_1') + \
                          betas[2]['m_2'] * p_ij(hmm, 'i_1', 'm_2') * p_o_in_i(hmm, o_t, 'm_2')
        betas[1]['i_0'] = betas[1]['d_1'] * p_ij(hmm, 'i_0', 'd_1') + \
                          betas[2]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * p_o_in_i(hmm, o_t, 'i_0') + \
                          betas[2]['m_1'] * p_ij(hmm, 'i_0', 'm_1') * p_o_in_i(hmm, o_t, 'm_1')
        betas[1]['m_3'] = betas[2]['i_3'] * p_ij(hmm, 'm_3', 'i_3') * p_o_in_i(hmm, o_t, 'i_3')
        betas[1]['m_2'] = betas[1]['d_3'] * p_ij(hmm, 'm_2', 'd_3') + \
                          betas[2]['i_2'] * p_ij(hmm, 'm_2', 'i_2') * p_o_in_i(hmm, o_t, 'i_2') + \
                          betas[2]['m_3'] * p_ij(hmm, 'm_2', 'm_3') * p_o_in_i(hmm, o_t, 'm_3')
        betas[1]['m_1'] = betas[1]['d_2'] * p_ij(hmm, 'm_1', 'd_2') + \
                          betas[2]['i_1'] * p_ij(hmm, 'm_1', 'i_1') * p_o_in_i(hmm, o_t, 'i_1') + \
                          betas[2]['m_2'] * p_ij(hmm, 'm_1', 'm_2') * p_o_in_i(hmm, o_t, 'm_2')

        betas[1]['d_3'] *= scalers[1]
        betas[1]['d_2'] *= scalers[1]
        betas[1]['d_1'] *= scalers[1]
        betas[1]['i_3'] *= scalers[1]
        betas[1]['i_2'] *= scalers[1]
        betas[1]['i_1'] *= scalers[1]
        betas[1]['i_0'] *= scalers[1]
        betas[1]['m_3'] *= scalers[1]
        betas[1]['m_2'] *= scalers[1]
        betas[1]['m_1'] *= scalers[1]

        o_t = observations[1]
        betas[0]['d_3'] = betas[1]['i_3'] * p_ij(hmm, 'd_3', 'i_3') * p_o_in_i(hmm, o_t, 'i_3')
        betas[0]['d_2'] = betas[0]['d_3'] * p_ij(hmm, 'd_2', 'd_3') + \
                          betas[1]['i_2'] * p_ij(hmm, 'd_2', 'i_2') * p_o_in_i(hmm, o_t, 'i_2') + \
                          betas[1]['m_3'] * p_ij(hmm, 'd_2', 'm_3') * p_o_in_i(hmm, o_t, 'm_3')
        betas[0]['d_1'] = betas[0]['d_2'] * p_ij(hmm, 'd_1', 'd_2') + \
                          betas[1]['i_1'] * p_ij(hmm, 'd_1', 'i_1') * p_o_in_i(hmm, o_t, 'i_1') + \
                          betas[1]['m_2'] * p_ij(hmm, 'd_1', 'm_2') * p_o_in_i(hmm, o_t, 'm_2')
        betas[0]['i_3'] = betas[1]['i_3'] * p_ij(hmm, 'i_3', 'i_3') * p_o_in_i(hmm, o_t, 'i_3')
        betas[0]['i_2'] = betas[0]['d_3'] * p_ij(hmm, 'i_2', 'd_3') + \
                          betas[1]['i_2'] * p_ij(hmm, 'i_2', 'i_2') * p_o_in_i(hmm, o_t, 'i_2') + \
                          betas[1]['m_3'] * p_ij(hmm, 'i_2', 'm_3') * p_o_in_i(hmm, o_t, 'm_3')
        betas[0]['i_1'] = betas[0]['d_2'] * p_ij(hmm, 'i_1', 'd_2') + \
                          betas[1]['i_1'] * p_ij(hmm, 'i_1', 'i_1') * p_o_in_i(hmm, o_t, 'i_1') + \
                          betas[1]['m_2'] * p_ij(hmm, 'i_1', 'm_2') * p_o_in_i(hmm, o_t, 'm_2')
        betas[0]['i_0'] = betas[0]['d_1'] * p_ij(hmm, 'i_0', 'd_1') + \
                          betas[1]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * p_o_in_i(hmm, o_t, 'i_0') + \
                          betas[1]['m_1'] * p_ij(hmm, 'i_0', 'm_1') * p_o_in_i(hmm, o_t, 'm_1')
        betas[0]['m_3'] = betas[1]['i_3'] * p_ij(hmm, 'm_3', 'i_3') * p_o_in_i(hmm, o_t, 'i_3')
        betas[0]['m_2'] = betas[0]['d_3'] * p_ij(hmm, 'm_2', 'd_3') + \
                          betas[1]['i_2'] * p_ij(hmm, 'm_2', 'i_2') * p_o_in_i(hmm, o_t, 'i_2') + \
                          betas[1]['m_3'] * p_ij(hmm, 'm_2', 'm_3') * p_o_in_i(hmm, o_t, 'm_3')
        betas[0]['m_1'] = betas[0]['d_2'] * p_ij(hmm, 'm_1', 'd_2') + \
                          betas[1]['i_1'] * p_ij(hmm, 'm_1', 'i_1') * p_o_in_i(hmm, o_t, 'i_1') + \
                          betas[1]['m_2'] * p_ij(hmm, 'm_1', 'm_2') * p_o_in_i(hmm, o_t, 'm_2')

        betas[0]['d_3'] *= scalers[0]
        betas[0]['d_2'] *= scalers[0]
        betas[0]['d_1'] *= scalers[0]
        betas[0]['i_3'] *= scalers[0]
        betas[0]['i_2'] *= scalers[0]
        betas[0]['i_1'] *= scalers[0]
        betas[0]['i_0'] *= scalers[0]
        betas[0]['m_3'] *= scalers[0]
        betas[0]['m_2'] *= scalers[0]
        betas[0]['m_1'] *= scalers[0]

    def test_backwards(self):
        betas = learning.backward(self.hmm, self.observations, self.scalers)
        self.assertEqual(len(self.betas), len(betas))

        for k in self.betas.keys():
            self.assertIn(k, betas)
            self.assertEqual(len(self.betas[k]), len(betas[k]))
            for kk in self.betas[k].keys():
                self.assertIn(kk, betas[k])
                self.assertAlmostEqual(
                    self.betas[k][kk],
                    betas[k][kk],
                    msg="Expected {} got {} for t={}, state={}".format(
                        self.betas[k][kk],
                        betas[k][kk],
                        k,
                        kk
                    )
                )


class TestCalcEtas(unittest.TestCase):
    def setUp(self) -> None:
        hmm = basic_phmm(3, ['A', 'B', 'C', 'D'])
        scalers = {}
        observations = ['A', 'B', 'C']
        self.hmm = hmm
        self.scalers = scalers
        self.observations = observations
        alphas, scalers = learning.forward(hmm, observations)
        self.scalers = scalers
        self.alphas = alphas
        betas = learning.backward(hmm, observations, scalers)
        self.betas = betas

        etas = {0: {}, 1: {}, 2:{}}
        self.etas = etas
        o_t1 = observations[1]
        etas[0][('i_0', 'd_1')] = alphas[0]['i_0'] * p_ij(hmm, 'i_0', 'd_1') * betas[0]['d_1']
        etas[0][('i_0', 'i_0')] = alphas[0]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * \
                                  p_o_in_i(hmm, o_t1, 'i_0') * betas[1]['i_0']
        etas[0][('i_0', 'm_1')] = alphas[0]['i_0'] * p_ij(hmm, 'i_0', 'm_1') * \
                                  p_o_in_i(hmm, o_t1, 'm_1') * betas[1]['m_1']
        etas[0][('i_1', 'd_2')] = alphas[0]['i_1'] * p_ij(hmm, 'i_1', 'd_2') * betas[0]['d_2']
        etas[0][('i_1', 'i_1')] = alphas[0]['i_1'] * p_ij(hmm, 'i_1', 'i_1') * \
                                  p_o_in_i(hmm, o_t1, 'i_1') * betas[1]['i_1']
        etas[0][('i_1', 'm_2')] = alphas[0]['i_1'] * p_ij(hmm, 'i_1', 'm_2') * \
                                  p_o_in_i(hmm, o_t1, 'm_2') * betas[1]['m_2']
        etas[0][('i_2', 'd_3')] = alphas[0]['i_2'] * p_ij(hmm, 'i_2', 'd_3') * betas[0]['d_3']
        etas[0][('i_2', 'i_2')] = alphas[0]['i_2'] * p_ij(hmm, 'i_2', 'i_2') * \
                                  p_o_in_i(hmm, o_t1, 'i_2') * betas[1]['i_2']
        etas[0][('i_2', 'm_3')] = alphas[0]['i_2'] * p_ij(hmm, 'i_2', 'm_3') * \
                                  p_o_in_i(hmm, o_t1, 'm_3') * betas[1]['m_3']
        etas[0][('i_3', 'i_3')] = alphas[0]['i_3'] * p_ij(hmm, 'i_3', 'i_3') * \
                                  p_o_in_i(hmm, o_t1, 'i_3') * betas[1]['i_3']
        etas[0][('d_1', 'd_2')] = alphas[0]['d_1'] * p_ij(hmm, 'd_1', 'd_2') * betas[0]['d_2']
        etas[0][('d_1', 'i_1')] = alphas[0]['d_1'] * p_ij(hmm, 'd_1', 'i_1') * \
                                  p_o_in_i(hmm, o_t1, 'i_1') * betas[1]['i_1']
        etas[0][('d_1', 'm_2')] = alphas[0]['d_1'] * p_ij(hmm, 'd_1', 'm_2') * \
                                  p_o_in_i(hmm, o_t1, 'm_2') * betas[1]['m_2']
        etas[0][('d_2', 'd_3')] = alphas[0]['d_2'] * p_ij(hmm, 'd_2', 'd_3') * betas[0]['d_3']
        etas[0][('d_2', 'i_2')] = alphas[0]['d_2'] * p_ij(hmm, 'd_2', 'i_2') * \
                                  p_o_in_i(hmm, o_t1, 'i_2') * betas[1]['i_2']
        etas[0][('d_2', 'm_3')] = alphas[0]['d_2'] * p_ij(hmm, 'd_2', 'm_3') * \
                                  p_o_in_i(hmm, o_t1, 'm_3') * betas[1]['m_3']
        etas[0][('d_3', 'i_3')] = alphas[0]['d_3'] * p_ij(hmm, 'd_3', 'i_3') * \
                                  p_o_in_i(hmm, o_t1, 'i_3') * betas[1]['i_3']
        etas[0][('m_1', 'd_2')] = alphas[0]['m_1'] * p_ij(hmm, 'm_1', 'd_2') * betas[0]['d_2']
        etas[0][('m_1', 'i_1')] = alphas[0]['m_1'] * p_ij(hmm, 'm_1', 'i_1') * \
                                  p_o_in_i(hmm, o_t1, 'i_1') * betas[1]['i_1']
        etas[0][('m_1', 'm_2')] = alphas[0]['m_1'] * p_ij(hmm, 'm_1', 'm_2') * \
                                  p_o_in_i(hmm, o_t1, 'm_2') * betas[1]['m_2']
        etas[0][('m_2', 'd_3')] = alphas[0]['m_2'] * p_ij(hmm, 'm_2', 'd_3') * betas[0]['d_3']
        etas[0][('m_2', 'i_2')] = alphas[0]['m_2'] * p_ij(hmm, 'm_2', 'i_2') * \
                                  p_o_in_i(hmm, o_t1, 'i_2') * betas[1]['i_2']
        etas[0][('m_2', 'm_3')] = alphas[0]['m_2'] * p_ij(hmm, 'm_2', 'm_3') * \
                                  p_o_in_i(hmm, o_t1, 'm_3') * betas[1]['m_3']
        etas[0][('m_3', 'i_3')] = alphas[0]['m_3'] * p_ij(hmm, 'm_3', 'i_3') * \
                                  p_o_in_i(hmm, o_t1, 'i_3') * betas[1]['i_3']
        learning._normalize(etas[0])

        o_t1 = observations[2]
        etas[1][('i_0', 'd_1')] = alphas[1]['i_0'] * p_ij(hmm, 'i_0', 'd_1') * betas[1]['d_1']
        etas[1][('i_0', 'i_0')] = alphas[1]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * \
                                  p_o_in_i(hmm, o_t1, 'i_0') * betas[2]['i_0']
        etas[1][('i_0', 'm_1')] = alphas[1]['i_0'] * p_ij(hmm, 'i_0', 'm_1') * \
                                  p_o_in_i(hmm, o_t1, 'm_1') * betas[2]['m_1']
        etas[1][('i_1', 'd_2')] = alphas[1]['i_1'] * p_ij(hmm, 'i_1', 'd_2') * betas[1]['d_2']
        etas[1][('i_1', 'i_1')] = alphas[1]['i_1'] * p_ij(hmm, 'i_1', 'i_1') * \
                                  p_o_in_i(hmm, o_t1, 'i_1') * betas[2]['i_1']
        etas[1][('i_1', 'm_2')] = alphas[1]['i_1'] * p_ij(hmm, 'i_1', 'm_2') * \
                                  p_o_in_i(hmm, o_t1, 'm_2') * betas[2]['m_2']
        etas[1][('i_2', 'd_3')] = alphas[1]['i_2'] * p_ij(hmm, 'i_2', 'd_3') * betas[1]['d_3']
        etas[1][('i_2', 'i_2')] = alphas[1]['i_2'] * p_ij(hmm, 'i_2', 'i_2') * \
                                  p_o_in_i(hmm, o_t1, 'i_2') * betas[2]['i_2']
        etas[1][('i_2', 'm_3')] = alphas[1]['i_2'] * p_ij(hmm, 'i_2', 'm_3') * \
                                  p_o_in_i(hmm, o_t1, 'm_3') * betas[2]['m_3']
        etas[1][('i_3', 'i_3')] = alphas[1]['i_3'] * p_ij(hmm, 'i_3', 'i_3') * \
                                  p_o_in_i(hmm, o_t1, 'i_3') * betas[2]['i_3']
        etas[1][('d_1', 'd_2')] = alphas[1]['d_1'] * p_ij(hmm, 'd_1', 'd_2') * betas[1]['d_2']
        etas[1][('d_1', 'i_1')] = alphas[1]['d_1'] * p_ij(hmm, 'd_1', 'i_1') * \
                                  p_o_in_i(hmm, o_t1, 'i_1') * betas[2]['i_1']
        etas[1][('d_1', 'm_2')] = alphas[1]['d_1'] * p_ij(hmm, 'd_1', 'm_2') * \
                                  p_o_in_i(hmm, o_t1, 'm_2') * betas[2]['m_2']
        etas[1][('d_2', 'd_3')] = alphas[1]['d_2'] * p_ij(hmm, 'd_2', 'd_3') * betas[1]['d_3']
        etas[1][('d_2', 'i_2')] = alphas[1]['d_2'] * p_ij(hmm, 'd_2', 'i_2') * \
                                  p_o_in_i(hmm, o_t1, 'i_2') * betas[2]['i_2']
        etas[1][('d_2', 'm_3')] = alphas[1]['d_2'] * p_ij(hmm, 'd_2', 'm_3') * \
                                  p_o_in_i(hmm, o_t1, 'm_3') * betas[2]['m_3']
        etas[1][('d_3', 'i_3')] = alphas[1]['d_3'] * p_ij(hmm, 'd_3', 'i_3') * \
                                  p_o_in_i(hmm, o_t1, 'i_3') * betas[2]['i_3']
        etas[1][('m_1', 'd_2')] = alphas[1]['m_1'] * p_ij(hmm, 'm_1', 'd_2') * betas[1]['d_2']
        etas[1][('m_1', 'i_1')] = alphas[1]['m_1'] * p_ij(hmm, 'm_1', 'i_1') * \
                                  p_o_in_i(hmm, o_t1, 'i_1') * betas[2]['i_1']
        etas[1][('m_1', 'm_2')] = alphas[1]['m_1'] * p_ij(hmm, 'm_1', 'm_2') * \
                                  p_o_in_i(hmm, o_t1, 'm_2') * betas[2]['m_2']
        etas[1][('m_2', 'd_3')] = alphas[1]['m_2'] * p_ij(hmm, 'm_2', 'd_3') * betas[1]['d_3']
        etas[1][('m_2', 'i_2')] = alphas[1]['m_2'] * p_ij(hmm, 'm_2', 'i_2') * \
                                  p_o_in_i(hmm, o_t1, 'i_2') * betas[2]['i_2']
        etas[1][('m_2', 'm_3')] = alphas[1]['m_2'] * p_ij(hmm, 'm_2', 'm_3') * \
                                  p_o_in_i(hmm, o_t1, 'm_3') * betas[2]['m_3']
        etas[1][('m_3', 'i_3')] = alphas[1]['m_3'] * p_ij(hmm, 'm_3', 'i_3') * \
                                  p_o_in_i(hmm, o_t1, 'i_3') * betas[2]['i_3']
        learning._normalize(etas[1])

        o_t1 = None
        etas[2][('i_0', 'd_1')] = alphas[2]['i_0'] * p_ij(hmm, 'i_0', 'd_1') * betas[2]['d_1']
        etas[2][('i_0', 'i_0')] = alphas[2]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * betas[3]['i_0']
        etas[2][('i_0', 'm_1')] = alphas[2]['i_0'] * p_ij(hmm, 'i_0', 'm_1') * betas[3]['m_1']
        etas[2][('i_1', 'd_2')] = alphas[2]['i_1'] * p_ij(hmm, 'i_1', 'd_2') * betas[2]['d_2']
        etas[2][('i_1', 'i_1')] = alphas[2]['i_1'] * p_ij(hmm, 'i_1', 'i_1') * betas[3]['i_1']
        etas[2][('i_1', 'm_2')] = alphas[2]['i_1'] * p_ij(hmm, 'i_1', 'm_2') * betas[3]['m_2']
        etas[2][('i_2', 'd_3')] = alphas[2]['i_2'] * p_ij(hmm, 'i_2', 'd_3') * betas[2]['d_3']
        etas[2][('i_2', 'i_2')] = alphas[2]['i_2'] * p_ij(hmm, 'i_2', 'i_2') * betas[3]['i_2']
        etas[2][('i_2', 'm_3')] = alphas[2]['i_2'] * p_ij(hmm, 'i_2', 'm_3') * betas[3]['m_3']
        etas[2][('i_3', 'i_3')] = alphas[2]['i_3'] * p_ij(hmm, 'i_3', 'i_3') * betas[3]['i_3']
        etas[2][('i_3', 'end')] = alphas[2]['i_3'] * p_ij(hmm, 'i_3', 'end') * betas[3]['end']
        etas[2][('d_1', 'd_2')] = alphas[2]['d_1'] * p_ij(hmm, 'd_1', 'd_2') * betas[2]['d_2']
        etas[2][('d_1', 'i_1')] = alphas[2]['d_1'] * p_ij(hmm, 'd_1', 'i_1') * betas[3]['i_1']
        etas[2][('d_1', 'm_2')] = alphas[2]['d_1'] * p_ij(hmm, 'd_1', 'm_2') * betas[3]['m_2']
        etas[2][('d_2', 'd_3')] = alphas[2]['d_2'] * p_ij(hmm, 'd_2', 'd_3') * betas[2]['d_3']
        etas[2][('d_2', 'i_2')] = alphas[2]['d_2'] * p_ij(hmm, 'd_2', 'i_2') * betas[3]['i_2']
        etas[2][('d_2', 'm_3')] = alphas[2]['d_2'] * p_ij(hmm, 'd_2', 'm_3') * betas[3]['m_3']
        etas[2][('d_3', 'i_3')] = alphas[2]['d_3'] * p_ij(hmm, 'd_3', 'i_3') * betas[3]['i_3']
        etas[2][('d_3', 'end')] = alphas[2]['d_3'] * p_ij(hmm, 'd_3', 'end') * betas[3]['end']
        etas[2][('m_1', 'd_2')] = alphas[2]['m_1'] * p_ij(hmm, 'm_1', 'd_2') * betas[2]['d_2']
        etas[2][('m_1', 'i_1')] = alphas[2]['m_1'] * p_ij(hmm, 'm_1', 'i_1') * betas[3]['i_1']
        etas[2][('m_1', 'm_2')] = alphas[2]['m_1'] * p_ij(hmm, 'm_1', 'm_2') * betas[3]['m_2']
        etas[2][('m_2', 'd_3')] = alphas[2]['m_2'] * p_ij(hmm, 'm_2', 'd_3') * betas[2]['d_3']
        etas[2][('m_2', 'i_2')] = alphas[2]['m_2'] * p_ij(hmm, 'm_2', 'i_2') * betas[3]['i_2']
        etas[2][('m_2', 'm_3')] = alphas[2]['m_2'] * p_ij(hmm, 'm_2', 'm_3') * betas[3]['m_3']
        etas[2][('m_3', 'i_3')] = alphas[2]['m_3'] * p_ij(hmm, 'm_3', 'i_3') * betas[3]['i_3']
        etas[2][('m_3', 'end')] = alphas[2]['m_3'] * p_ij(hmm, 'm_3', 'end') * betas[3]['end']
        learning._normalize(etas[2])

    def test_calc_etas(self):
        etas = learning._calc_etas(self.hmm, self.observations, self.alphas, self.betas)
        self.assertEqual(len(self.etas), len(etas))
        self.assertEqual(len(self.etas[0]), len(etas[0]))
        self.assertEqual(len(self.etas[1]), len(etas[1]))
        for t, etas_t in self.etas.items():
            s = 0.
            for (i, j), v in etas_t.items():
                self.assertAlmostEqual(v, etas[t][(i, j)])
                s += v
            self.assertAlmostEqual(1, s)


class TestCalcGammas(unittest.TestCase):
    def setUp(self) -> None:
        hmm = basic_phmm(3, ['A', 'B', 'C', 'D'])
        scalers = {}
        observations = ['A', 'B', 'C']
        self.hmm = hmm
        self.scalers = scalers
        self.observations = observations
        alphas, scalers = learning.forward(hmm, observations)
        self.scalers = scalers
        self.alphas = alphas
        betas = learning.backward(hmm, observations, scalers)
        self.betas = betas
        self.etas = learning._calc_etas(hmm, observations, alphas, betas)

    def test_calc_gammas(self):
        gammas = learning._calc_gammas(self.etas)
        self.assertEqual(3, len(gammas))
        self.assertEqual(len(self.hmm.hiddens) - 2, len(gammas[0]))
        self.assertEqual(len(self.hmm.hiddens) - 2, len(gammas[1]))
        self.assertEqual(len(self.hmm.hiddens) - 2, len(gammas[2]))
        for t, gammas_t in gammas.items():
            s = 0.
            for _, v in gammas_t.items():
                s += v
            self.assertAlmostEqual(1, s)


class TestCalcParameterUpdates(unittest.TestCase):
    def setUp(self) -> None:
        hmm = basic_phmm(3, ['A', 'B', 'C', 'D'])
        scalers = {}
        observations = ['A', 'B', 'C']
        self.hmm = hmm
        self.scalers = scalers
        self.observations = observations
        alphas, scalers = learning.forward(hmm, observations)
        self.scalers = scalers
        self.alphas = alphas
        betas = learning.backward(hmm, observations, scalers)
        self.betas = betas
        self.etas = learning._calc_etas(hmm, observations, alphas, betas)
        self.gammas = learning._calc_gammas(self.etas)

    def test_calc_initial_probs(self):
        transitions = learning._calc_initial_probs(self.hmm, [self.gammas[0]])
        self.assertEqual(3, len(transitions))
        self.assertIn((self.hmm.START, 'd_1'), transitions)
        self.assertIn((self.hmm.START, 'm_1'), transitions)
        self.assertIn((self.hmm.START, 'i_0'), transitions)
        s = 0.
        for v in transitions.values():
            s += v
        self.assertAlmostEqual(1, s)

    def test_calc_transition_probs(self):
        p_ks = learning._prod_scalers([self.scalers])
        transitions = learning._calc_transition_probs(
            hmm=self.hmm,
            obs_l=[self.observations],
            alphas_l=[self.alphas],
            betas_l=[self.betas],
            prods_c_t=p_ks
        )
        self.assertEqual(27, len(transitions))
        probs = {}
        for (i, j), v in transitions.items():
            if i not in probs:
                probs[i] = 0
            probs[i] += v
        for k, v in probs.items():
            self.assertAlmostEqual(1, v)

    def test_calc_transition_probs2(self):

        def o(t):
            return self.observations[t]

        p_ks = learning._prod_scalers([self.scalers])
        a = self.alphas
        b = self.betas
        hmm = self.hmm

        x_ij = {
            ('i_0', 'd_1'): (a[0]['i_0'] * p_ij(hmm, 'i_0', 'd_1') * b[0]['d_1'] +
                             a[1]['i_0'] * p_ij(hmm, 'i_0', 'd_1') * b[1]['d_1'] +
                             a[2]['i_0'] * p_ij(hmm, 'i_0', 'd_1') * b[2]['d_1']
                             ) * p_ks[0],
            ('i_0', 'i_0'): (a[0]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * p_o_in_i(hmm, o(1), 'i_0') * b[1]['i_0'] +
                             a[1]['i_0'] * p_ij(hmm, 'i_0', 'i_0') * p_o_in_i(hmm, o(2), 'i_0') * b[2]['i_0']
                             ) * p_ks[0],
            ('i_0', 'm_1'): (a[0]['i_0'] * p_ij(hmm, 'i_0', 'm_1') * p_o_in_i(hmm, o(1), 'm_1') * b[1]['m_1'] +
                             a[1]['i_0'] * p_ij(hmm, 'i_0', 'm_1') * p_o_in_i(hmm, o(2), 'm_1') * b[2]['m_1']
                             ) * p_ks[0],
            ('d_1', 'd_2'): (a[0]['d_1'] * p_ij(hmm, 'd_1', 'd_2') * b[0]['d_2'] +
                             a[1]['d_1'] * p_ij(hmm, 'd_1', 'd_2') * b[1]['d_2'] +
                             a[2]['d_1'] * p_ij(hmm, 'd_1', 'd_2') * b[2]['d_2']
                             ) * p_ks[0],
            ('d_1', 'i_1'): (a[0]['d_1'] * p_ij(hmm, 'd_1', 'i_1') * p_o_in_i(hmm, o(1), 'i_1') * b[1]['i_1'] +
                             a[1]['d_1'] * p_ij(hmm, 'd_1', 'i_1') * p_o_in_i(hmm, o(2), 'i_1') * b[2]['i_1']
                             ) * p_ks[0],
            ('d_1', 'm_2'): (a[0]['d_1'] * p_ij(hmm, 'd_1', 'm_2') * p_o_in_i(hmm, o(1), 'm_2') * b[1]['m_2'] +
                             a[1]['d_1'] * p_ij(hmm, 'd_1', 'm_2') * p_o_in_i(hmm, o(2), 'm_2') * b[2]['m_2']
                             ) * p_ks[0],
            ('m_1', 'd_2'): (a[0]['m_1'] * p_ij(hmm, 'm_1', 'd_2') * b[0]['d_2'] +
                             a[1]['m_1'] * p_ij(hmm, 'm_1', 'd_2') * b[1]['d_2'] +
                             a[2]['m_1'] * p_ij(hmm, 'm_1', 'd_2') * b[2]['d_2']
                             ) * p_ks[0],
            ('m_1', 'i_1'): (a[0]['m_1'] * p_ij(hmm, 'm_1', 'i_1') * p_o_in_i(hmm, o(1), 'i_1') * b[1]['i_1'] +
                             a[1]['m_1'] * p_ij(hmm, 'm_1', 'i_1') * p_o_in_i(hmm, o(2), 'i_1') * b[2]['i_1']
                             ) * p_ks[0],
            ('m_1', 'm_2'): (a[0]['m_1'] * p_ij(hmm, 'm_1', 'm_2') * p_o_in_i(hmm, o(1), 'm_2') * b[1]['m_2'] +
                             a[1]['m_1'] * p_ij(hmm, 'm_1', 'm_2') * p_o_in_i(hmm, o(2), 'm_2') * b[2]['m_2']
                             ) * p_ks[0],
            ('i_1', 'd_2'): (a[0]['i_1'] * p_ij(hmm, 'i_1', 'd_2') * b[0]['d_2'] +
                             a[1]['i_1'] * p_ij(hmm, 'i_1', 'd_2') * b[1]['d_2'] +
                             a[2]['i_1'] * p_ij(hmm, 'i_1', 'd_2') * b[2]['d_2']
                             ) * p_ks[0],
            ('i_1', 'i_1'): (a[0]['i_1'] * p_ij(hmm, 'i_1', 'i_1') * p_o_in_i(hmm, o(1), 'i_1') * b[1]['i_1'] +
                             a[1]['i_1'] * p_ij(hmm, 'i_1', 'i_1') * p_o_in_i(hmm, o(2), 'i_1') * b[2]['i_1']
                             ) * p_ks[0],
            ('i_1', 'm_2'): (a[0]['i_1'] * p_ij(hmm, 'i_1', 'm_2') * p_o_in_i(hmm, o(1), 'm_2') * b[1]['m_2'] +
                             a[1]['i_1'] * p_ij(hmm, 'i_1', 'm_2') * p_o_in_i(hmm, o(2), 'm_2') * b[2]['m_2']
                             ) * p_ks[0],
            ('d_2', 'd_3'): (a[0]['d_2'] * p_ij(hmm, 'd_2', 'd_3') * b[0]['d_3'] +
                             a[1]['d_2'] * p_ij(hmm, 'd_2', 'd_3') * b[1]['d_3'] +
                             a[2]['d_2'] * p_ij(hmm, 'd_2', 'd_3') * b[2]['d_3']
                             ) * p_ks[0],
            ('d_2', 'i_2'): (a[0]['d_2'] * p_ij(hmm, 'd_2', 'i_2') * p_o_in_i(hmm, o(1), 'i_2') * b[1]['i_2'] +
                             a[1]['d_2'] * p_ij(hmm, 'd_2', 'i_2') * p_o_in_i(hmm, o(2), 'i_2') * b[2]['i_2']
                             ) * p_ks[0],
            ('d_2', 'm_3'): (a[0]['d_2'] * p_ij(hmm, 'd_2', 'm_3') * p_o_in_i(hmm, o(1), 'm_3') * b[1]['m_3'] +
                             a[1]['d_2'] * p_ij(hmm, 'd_2', 'm_3') * p_o_in_i(hmm, o(2), 'm_3') * b[2]['m_3']
                             ) * p_ks[0],
            ('m_2', 'd_3'): (a[0]['m_2'] * p_ij(hmm, 'm_2', 'd_3') * b[0]['d_3'] +
                             a[1]['m_2'] * p_ij(hmm, 'm_2', 'd_3') * b[1]['d_3'] +
                             a[2]['m_2'] * p_ij(hmm, 'm_2', 'd_3') * b[2]['d_3']
                             ) * p_ks[0],
            ('m_2', 'i_2'): (a[0]['m_2'] * p_ij(hmm, 'm_2', 'i_2') * p_o_in_i(hmm, o(1), 'i_2') * b[1]['i_2'] +
                             a[1]['m_2'] * p_ij(hmm, 'm_2', 'i_2') * p_o_in_i(hmm, o(2), 'i_2') * b[2]['i_2']
                             ) * p_ks[0],
            ('m_2', 'm_3'): (a[0]['m_2'] * p_ij(hmm, 'm_2', 'm_3') * p_o_in_i(hmm, o(1), 'm_3') * b[1]['m_3'] +
                             a[1]['m_2'] * p_ij(hmm, 'm_2', 'm_3') * p_o_in_i(hmm, o(2), 'm_3') * b[2]['m_3']
                             ) * p_ks[0],
            ('i_2', 'd_3'): (a[0]['i_2'] * p_ij(hmm, 'i_2', 'd_3') * b[0]['d_3'] +
                             a[1]['i_2'] * p_ij(hmm, 'i_2', 'd_3') * b[1]['d_3'] +
                             a[2]['i_2'] * p_ij(hmm, 'i_2', 'd_3') * b[2]['d_3']
                             ) * p_ks[0],
            ('i_2', 'i_2'): (a[0]['i_2'] * p_ij(hmm, 'i_2', 'i_2') * p_o_in_i(hmm, o(1), 'i_2') * b[1]['i_2'] +
                             a[1]['i_2'] * p_ij(hmm, 'i_2', 'i_2') * p_o_in_i(hmm, o(2), 'i_2') * b[2]['i_2']
                             ) * p_ks[0],
            ('i_2', 'm_3'): (a[0]['i_2'] * p_ij(hmm, 'i_2', 'm_3') * p_o_in_i(hmm, o(1), 'm_3') * b[1]['m_3'] +
                             a[1]['i_2'] * p_ij(hmm, 'i_2', 'm_3') * p_o_in_i(hmm, o(2), 'm_3') * b[2]['m_3']
                             ) * p_ks[0],
            ('d_3', 'i_3'): (a[0]['d_3'] * p_ij(hmm, 'd_3', 'i_3') * p_o_in_i(hmm, o(1), 'i_3') * b[1]['i_3'] +
                             a[1]['d_3'] * p_ij(hmm, 'd_3', 'i_3') * p_o_in_i(hmm, o(2), 'i_3') * b[2]['i_3']
                             ) * p_ks[0],
            ('d_3', 'end'): a[2]['d_3'] * p_ij(hmm, 'd_3', 'end') * b[3]['end'] * p_ks[0],
            ('i_3', 'i_3'): (a[0]['i_3'] * p_ij(hmm, 'i_3', 'i_3') * p_o_in_i(hmm, o(1), 'i_3') * b[1]['i_3'] +
                             a[1]['i_3'] * p_ij(hmm, 'i_3', 'i_3') * p_o_in_i(hmm, o(2), 'i_3') * b[2]['i_3']
                             ) * p_ks[0],
            ('i_3', 'end'): a[2]['i_3'] * p_ij(hmm, 'i_3', 'end') * b[3]['end'] * p_ks[0],
            ('m_3', 'i_3'): (a[0]['m_3'] * p_ij(hmm, 'm_3', 'i_3') * p_o_in_i(hmm, o(1), 'i_3') * b[1]['i_3'] +
                             a[1]['m_3'] * p_ij(hmm, 'm_3', 'i_3') * p_o_in_i(hmm, o(2), 'i_3') * b[2]['i_3']
                             ) * p_ks[0],
            ('m_3', 'end'): a[2]['m_3'] * p_ij(hmm, 'm_3', 'end') * b[3]['end'] * p_ks[0]
        }
        denom = (x_ij[('i_0', 'd_1')] + x_ij[('i_0', 'i_0')] + x_ij[('i_0', 'm_1')])
        x_ij[('i_0', 'd_1')] /= denom
        x_ij[('i_0', 'i_0')] /=  denom
        x_ij[('i_0', 'm_1')] /=  denom

        denom =(x_ij[('i_1', 'd_2')] + x_ij[('i_1', 'i_1')] + x_ij[('i_1', 'm_2')])
        x_ij[('i_1', 'd_2')] /=  denom
        x_ij[('i_1', 'i_1')] /=  denom
        x_ij[('i_1', 'm_2')] /=  denom

        denom =(x_ij[('i_2', 'd_3')] + x_ij[('i_2', 'i_2')] + x_ij[('i_2', 'm_3')])
        x_ij[('i_2', 'd_3')] /=  denom
        x_ij[('i_2', 'i_2')] /=  denom
        x_ij[('i_2', 'm_3')] /=  denom

        denom =(x_ij[('i_3', 'i_3')] + x_ij[('i_3', 'end')])
        x_ij[('i_3', 'i_3')] /=  denom
        x_ij[('i_3', 'end')] /=  denom

        denom =(x_ij[('d_1', 'd_2')] + x_ij[('d_1', 'i_1')] + x_ij[('d_1', 'm_2')])
        x_ij[('d_1', 'd_2')] /=  denom
        x_ij[('d_1', 'i_1')] /=  denom
        x_ij[('d_1', 'm_2')] /=  denom

        denom =(x_ij[('d_2', 'd_3')] + x_ij[('d_2', 'i_2')] + x_ij[('d_2', 'm_3')])
        x_ij[('d_2', 'd_3')] /=  denom
        x_ij[('d_2', 'i_2')] /=  denom
        x_ij[('d_2', 'm_3')] /=  denom

        denom =(x_ij[('d_3', 'i_3')] + x_ij[('d_3', 'end')])
        x_ij[('d_3', 'i_3')] /=  denom
        x_ij[('d_3', 'end')] /=  denom

        denom =(x_ij[('m_1', 'd_2')] + x_ij[('m_1', 'i_1')] + x_ij[('m_1', 'm_2')])
        x_ij[('m_1', 'd_2')] /=  denom
        x_ij[('m_1', 'i_1')] /=  denom
        x_ij[('m_1', 'm_2')] /=  denom

        denom =(x_ij[('m_2', 'd_3')] + x_ij[('m_2', 'i_2')] + x_ij[('m_2', 'm_3')])
        x_ij[('m_2', 'd_3')] /=  denom
        x_ij[('m_2', 'i_2')] /=  denom
        x_ij[('m_2', 'm_3')] /=  denom

        denom =(x_ij[('m_3', 'i_3')] + x_ij[('m_3', 'end')])
        x_ij[('m_3', 'i_3')] /=  denom
        x_ij[('m_3', 'end')] /=  denom

        new = learning._calc_transition_probs(
            hmm=self.hmm,
            obs_l=[self.observations],
            alphas_l=[self.alphas],
            betas_l=[self.betas],
            prods_c_t=p_ks
        )
        for k, v in x_ij.items():
            self.assertAlmostEqual(v, new[k], msg="{} vs {} for transition {}".format(v, new[k], str(k)))

    def test_calc_emission_probs(self):
        emissions = learning._calc_emission_probs(
            hmm=self.hmm,
            observations_l=[self.observations],
            alphas_l=[self.alphas],
            betas_l=[self.betas],
            prods_c_t=learning._prod_scalers([self.scalers])
        )
        self.assertEqual(len(self.hmm.observables) * 7, len(emissions))
        probs = {}
        for (obs, i), v in emissions.items():
            if i not in probs:
                probs[i] = 0.
            probs[i] += v
        for v in probs.values():
            self.assertAlmostEqual(1, v)

    def test_renormalize_initals(self):
        initals_start = {('start', 'm_1'): 1.25, ('start', 'i_0'): 0.25, ('start', 'd_1'): 1.0}
        initals_ground = {('start', 'm_1'): 0.4444444444444444, ('start', 'i_0'): 0.1111111111111111, ('start', 'd_1'): 0.4444444444444444}
        initals_test = learning.renormalize_initials(initals_start)
        self.assertEqual(initals_test, initals_ground)

    def test_renormalize_transitions(self):
        transition_start = {('i_0', 'd_1'): 1.25, ('i_0', 'i_0'): 0.25, ('i_0', 'm_1'): 1.75,
                            ('d_1', 'd_2'): 1.5, ('d_1', 'i_1'): 0.5, ('d_1', 'm_2'): 0.25,
                            ('i_1', 'd_2'): 0.4, ('i_1', 'i_1'): 1.6, ('i_1', 'm_2'): 0.3,
                            ('m_1', 'd_2'): 0.6, ('m_1', 'i_1'): 1.7, ('m_1', 'm_2'): 1.3,
                            ('d_2', 'd_3'): 1.1, ('d_2', 'i_2'): 0.2, ('d_2', 'm_3'): 1.8,
                            ('i_2', 'd_3'): 0.9, ('i_2', 'i_2'): 1.9, ('i_2', 'm_3'): 1.0,
                            ('m_2', 'd_3'): 0.7, ('m_2', 'i_2'): 1.3, ('m_2', 'm_3'): 1.5,
                            ('d_3', 'end'): 0.75, ('d_3', 'i_3'): 1.2, ('i_3', 'end'): 0.4,
                            ('i_3', 'i_3'): 1.4, ('m_3', 'end'): 1.7, ('m_3', 'i_3'): 1.1}
        transitions_ground = {('d_1', 'd_2'):0.5714285714285714, ('d_1', 'i_1'):0.2857142857142857, ('d_1', 'm_2'):0.14285714285714285,
                            ('d_2', 'd_3'):0.45454545454545453, ('d_2', 'i_2'):0.09090909090909091, ('d_2', 'm_3'):0.45454545454545453,
                            ('d_3', 'end'):0.42857142857142855, ('d_3', 'i_3'):0.5714285714285714, ('m_1', 'd_2'):0.23076923076923075,
                            ('m_1', 'i_1'):0.3846153846153846, ('m_1', 'm_2'):0.3846153846153846, ('m_2', 'd_3'):0.25925925925925924,
                            ('m_2', 'i_2'):0.37037037037037035, ('m_2', 'm_3'):0.37037037037037035, ('m_3', 'end'):0.5, 
                            ('m_3', 'i_3'):0.5, ('i_0', 'd_1'):0.4444444444444444, ('i_0', 'i_0'):0.1111111111111111,
                            ('i_0', 'm_1'):0.4444444444444444, ('i_1', 'd_2'):0.23529411764705885, ('i_1', 'i_1'):0.5882352941176471,
                            ('i_1', 'm_2'):0.17647058823529413, ('i_2', 'd_3'):0.3103448275862069, ('i_2', 'i_2'):0.3448275862068966,
                            ('i_2', 'm_3'):0.3448275862068966, ('i_3', 'end'):0.28571428571428575, ('i_3', 'i_3'):0.7142857142857143}
        transitions_test = learning.renormalize_transitions(self.hmm, transition_start)
        for v_1, v_2 in zip(transitions_ground.values(), transitions_test.values()):
            self.assertAlmostEqual(v_1, v_2)

    def test_renormalize_emissions(self):
        emissions_start = {('A', 'm_1'): 1.25, ('A', 'i_0'): 0.25, ('B', 'm_1'): 1.3, ('B', 'i_0'): 0.7, ('C', 'm_1'): 0.3, ('C', 'i_0'): 1.7, ('D', 'm_1'): 2.2, ('D', 'i_0'): 0.4,
                           ('A', 'm_2'): 1.25, ('A', 'i_1'): 0.25, ('B', 'm_2'): 1.3, ('B', 'i_1'): 0.7, ('C', 'm_2'): 0.3, ('C', 'i_1'): 1.7, ('D', 'm_2'): 2.2, ('D', 'i_1'): 0.4,
                           ('A', 'm_3'): 1.25, ('A', 'i_2'): 0.25, ('B', 'm_3'): 1.3, ('B', 'i_2'): 0.7, ('C', 'm_3'): 0.3, ('C', 'i_2'): 1.7, ('D', 'm_3'): 2.2, ('D', 'i_2'): 0.4,
                           ('A', 'i_3'): 1.25, ('B', 'i_3'): 0.25, ('C', 'i_3'): 1.0, ('D', 'i_3'): 1.5}
        emissions_ground = {('A', 'm_1'):0.30303030303030304, ('B', 'm_1'):0.30303030303030304, ('C', 'm_1'):0.09090909090909091, ('D', 'm_1'):0.30303030303030304,
                            ('A', 'm_2'):0.30303030303030304, ('B', 'm_2'):0.30303030303030304, ('C', 'm_2'):0.09090909090909091, ('D', 'm_2'):0.30303030303030304,
                            ('A', 'm_3'):0.30303030303030304, ('B', 'm_3'):0.30303030303030304, ('C', 'm_3'):0.09090909090909091, ('D', 'm_3'):0.30303030303030304,
                            ('A', 'i_0'):0.10638297872340426, ('B', 'i_0'):0.2978723404255319, ('C', 'i_0'):0.425531914893617, ('D', 'i_0'):0.1702127659574468,
                            ('A', 'i_1'):0.10638297872340426, ('B', 'i_1'):0.2978723404255319, ('C', 'i_1'):0.425531914893617, ('D', 'i_1'):0.1702127659574468,
                            ('A', 'i_2'):0.10638297872340426, ('B', 'i_2'):0.2978723404255319, ('C', 'i_2'):0.425531914893617, ('D', 'i_2'):0.1702127659574468,
                            ('A', 'i_3'):0.3076923076923077, ('B', 'i_3'):0.07692307692307693, ('C', 'i_3'):0.3076923076923077, ('D', 'i_3'):0.3076923076923077}
        emissions_test = learning.renormalize_emissions(self.hmm, emissions_start)
        for v_1, v_2 in zip(emissions_ground.values(), emissions_test.values()):
            self.assertAlmostEqual(v_1, v_2)

    def test_update_hmm(self):
        initials = learning._calc_initial_probs(self.hmm, self.gammas[0])
        transitions = learning._calc_transition_probs(self.hmm, self.etas, self.gammas)
        emissions = learning._calc_emission_probs(self.hmm, self.observations, self.gammas)
        new_hmm = learning._update_parameters(self.hmm, initials, transitions, emissions)
        self.assertEqual(self.hmm.duration, new_hmm.duration)
        self.assertEqual(len(self.hmm.hiddens), len(new_hmm.hiddens))
        self.assertEqual(len(self.hmm.p_ij), len(new_hmm.p_ij))
        self.assertEqual(len(self.hmm.p_o_in_i), len(new_hmm.p_o_in_i))
        self.assertEqual(len(self.hmm.preds), len(new_hmm.preds))
        self.assertEqual(len(self.hmm.succs), len(new_hmm.succs))


class TestTraining(unittest.TestCase):
    def test_loop(self):
        def check_for_correctness(new_hmm):
            probs = {}
            for (obs, i), v in new_hmm.p_o_in_i.items():
                if i not in probs:
                    probs[i] = 0.
                probs[i] += v
            for v in probs.values():
                self.assertAlmostEqual(1, v)

            probs = {}
            for (i, j), v in new_hmm.p_ij.items():
                if i not in probs:
                    probs[i] = 0
                probs[i] += v
            for k, v in probs.items():
                self.assertAlmostEqual(1, v)

        sequences = dummy_phmm_data(100, 1)
        hmm = basic_phmm(12, [0, 1, 2, 3], init_prior='prefer_match')
        print("Step 1")
        new_hmm, _, _ = learning.baum_welch(hmm, sequences['sequences'])
        check_for_correctness(new_hmm)

        print("Step 2")
        new_hmm, _, _ = learning.baum_welch(new_hmm, sequences['sequences'])
        check_for_correctness(new_hmm)
        print("Step 3")
        new_hmm, _, _ = learning.baum_welch(new_hmm, sequences['sequences'])
        check_for_correctness(new_hmm)
        print("Step 4")
        new_hmm, _, _ = learning.baum_welch(new_hmm, sequences['sequences'])
        check_for_correctness(new_hmm)
        print("end.")


class TestLogProb(unittest.TestCase):

    def test_log_prob(self):
        observations = [0, 1, 2, 3]
        hmm = basic_phmm(12, observations)
        test_sequence = [1, 3, 2, 0]
        log_prob = learning.calc_log_prob(hmm, test_sequence)
        self.assertLessEqual(a = log_prob, b = 0.)
