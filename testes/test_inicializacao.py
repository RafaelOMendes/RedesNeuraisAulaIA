"""Teste 6 (pesos): mesma seed = mesmos pesos iniciais. E a escala Xavier/He."""

import unittest

import numpy as np

from rna import RedeNeural
from rna.inicializacao import criar_gerador, he, inicializar, metodo_para, xavier


def parametros(rede):
    return [(c.pesos(), c.vieses()) for c in rede.camadas]


class TestSeed(unittest.TestCase):
    def test_mesma_seed_mesmos_pesos(self):
        a = RedeNeural(3, [(5, "relu"), (4, "tanh")], 2, "sigmoid", seed=42)
        b = RedeNeural(3, [(5, "relu"), (4, "tanh")], 2, "sigmoid", seed=42)
        for (Wa, ba), (Wb, bb) in zip(parametros(a), parametros(b)):
            np.testing.assert_array_equal(Wa, Wb)
            np.testing.assert_array_equal(ba, bb)

    def test_seed_diferente_pesos_diferentes(self):
        a = RedeNeural(3, [(5, "tanh")], 1, "sigmoid", seed=1)
        b = RedeNeural(3, [(5, "tanh")], 1, "sigmoid", seed=2)
        self.assertFalse(np.array_equal(a.camadas[0].pesos(), b.camadas[0].pesos()))


class TestDistribuicoes(unittest.TestCase):
    def test_metodo_por_ativacao(self):
        self.assertEqual(metodo_para("relu"), "he")
        for nome in ("linear", "sigmoid", "tanh"):
            self.assertEqual(metodo_para(nome), "xavier")

    def test_xavier_fica_no_limite(self):
        W = xavier(30, 20, criar_gerador(0))
        limite = np.sqrt(6 / (30 + 20))
        self.assertEqual(W.shape, (20, 30))
        self.assertTrue(np.all(np.abs(W) <= limite))
        # variância de U(-L, L) = L^2 / 3 = 2 / (fan_in + fan_out)
        self.assertAlmostEqual(W.var(), 2 / 50, delta=0.2 * 2 / 50)

    def test_he_tem_o_desvio_certo(self):
        W = he(50, 400, criar_gerador(0))
        self.assertEqual(W.shape, (400, 50))
        self.assertAlmostEqual(W.std(), np.sqrt(2 / 50), delta=0.05 * np.sqrt(2 / 50))

    def test_vieses_comecam_em_zero(self):
        _, vieses = inicializar("tanh", 4, 6, criar_gerador(0))
        np.testing.assert_array_equal(vieses, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
