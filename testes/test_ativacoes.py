"""Teste 1: derivada de cada ativação comparada com a diferença finita central."""

import unittest

import numpy as np

from rna.ativacoes import Linear, ReLU, Sigmoid, Tanh, obter_ativacao

# Pontos longe de z = 0, onde a ReLU não é derivável.
PONTOS = [-4.0, -2.5, -1.0, -0.3, 0.2, 0.7, 1.5, 3.0]
EPS = 1e-6


class TestDerivadas(unittest.TestCase):
    def verificar(self, ativacao):
        for z in PONTOS:
            numerica = (ativacao.f(z + EPS) - ativacao.f(z - EPS)) / (2 * EPS)
            analitica = ativacao.derivada(z)
            self.assertAlmostEqual(
                analitica, numerica, delta=1e-6,
                msg=f"{ativacao.nome}: f'({z}) analítica={analitica} numérica={numerica}",
            )

    def test_linear(self):
        self.verificar(Linear())

    def test_sigmoid(self):
        self.verificar(Sigmoid())

    def test_tanh(self):
        self.verificar(Tanh())

    def test_relu(self):
        self.verificar(ReLU())


class TestValores(unittest.TestCase):
    def test_valores_conhecidos(self):
        self.assertEqual(Linear().f(2.5), 2.5)
        self.assertAlmostEqual(Sigmoid().f(0.0), 0.5)
        self.assertAlmostEqual(Tanh().f(0.0), 0.0)
        self.assertEqual(ReLU().f(-3.0), 0.0)
        self.assertEqual(ReLU().f(3.0), 3.0)
        self.assertEqual(ReLU().derivada(-1.0), 0.0)
        self.assertEqual(ReLU().derivada(1.0), 1.0)

    def test_formulas_das_derivadas(self):
        z = np.linspace(-3, 3, 13)
        s = Sigmoid().f(z)
        np.testing.assert_allclose(Sigmoid().derivada(z), s * (1 - s))
        t = Tanh().f(z)
        np.testing.assert_allclose(Tanh().derivada(z), 1 - t ** 2)

    def test_sigmoid_sem_overflow(self):
        self.assertAlmostEqual(Sigmoid().f(1000.0), 1.0)
        self.assertAlmostEqual(Sigmoid().f(-1000.0), 0.0)

    def test_nome_invalido_da_erro_claro(self):
        with self.assertRaisesRegex(ValueError, "desconhecida"):
            obter_ativacao("softmax")

    def test_nome_sem_diferenca_de_maiusculas(self):
        self.assertIsInstance(obter_ativacao("ReLU"), ReLU)


if __name__ == "__main__":
    unittest.main()
