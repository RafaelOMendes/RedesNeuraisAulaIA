"""Teste 2: gradiente da retropropagação x gradiente numérico.

Custo usado nos dois lados: E = 1/2 Σ (y - a)^2 (mesma convenção do treino).
"""

import unittest

import numpy as np

from rna import RedeNeural
from rna.erro import erro_exemplo
from rna.treino import gradientes, retropropagar

EPS = 1e-5


def gradiente_numerico(rede, x, y):
    """Diferença finita central para cada peso e viés."""
    resultado = []
    for camada in rede.camadas:
        d_pesos = np.zeros((camada.n_neuronios, camada.n_entradas))
        d_vieses = np.zeros(camada.n_neuronios)
        for i, neuronio in enumerate(camada.neuronios):
            for j in range(camada.n_entradas):
                original = neuronio.pesos[j]
                neuronio.pesos[j] = original + EPS
                e_mais = erro_exemplo(rede.propagar(x), y)
                neuronio.pesos[j] = original - EPS
                e_menos = erro_exemplo(rede.propagar(x), y)
                neuronio.pesos[j] = original
                d_pesos[i, j] = (e_mais - e_menos) / (2 * EPS)
            original = neuronio.bias
            neuronio.bias = original + EPS
            e_mais = erro_exemplo(rede.propagar(x), y)
            neuronio.bias = original - EPS
            e_menos = erro_exemplo(rede.propagar(x), y)
            neuronio.bias = original
            d_vieses[i] = (e_mais - e_menos) / (2 * EPS)
        resultado.append((d_pesos, d_vieses))
    return resultado


class TestGradiente(unittest.TestCase):
    def comparar(self, rede, x, y):
        analitico = gradientes(rede, x, y)
        numerico = gradiente_numerico(rede, x, y)
        for c, ((dw_a, db_a), (dw_n, db_n)) in enumerate(zip(analitico, numerico), start=1):
            np.testing.assert_allclose(dw_a, dw_n, rtol=1e-5, atol=1e-8, err_msg=f"pesos da camada {c}")
            np.testing.assert_allclose(db_a, db_n, rtol=1e-5, atol=1e-8, err_msg=f"vieses da camada {c}")

    def test_duas_ocultas_ativacoes_mistas(self):
        rede = RedeNeural(3, [(4, "tanh"), (3, "sigmoid")], 2, "linear", seed=11)
        # vieses diferentes de zero para o teste ser mais geral
        rng = np.random.default_rng(0)
        for camada in rede.camadas:
            for n in camada.neuronios:
                n.bias = float(rng.uniform(-0.5, 0.5))
        self.comparar(rede, np.array([0.5, -1.2, 0.8]), np.array([0.3, -0.7]))

    def test_saida_sigmoid_com_varias_saidas(self):
        rede = RedeNeural(2, [(5, "tanh")], 3, "sigmoid", seed=3)
        self.comparar(rede, np.array([1.0, -0.5]), np.array([1.0, 0.0, 0.0]))

    def test_sem_camada_oculta(self):
        rede = RedeNeural(4, [], 1, "sigmoid", seed=5)
        self.comparar(rede, np.array([0.2, 0.4, -0.6, 1.0]), np.array([1.0]))

    def test_relu_longe_de_zero(self):
        rede = RedeNeural(2, [(6, "relu")], 1, "linear", seed=21)
        x = np.array([0.9, -0.4])
        rede.propagar(x)
        zs = [n.z for n in rede.camadas[0].neuronios]
        self.assertTrue(all(abs(z) > 1e-3 for z in zs), "algum z ficou perto de 0 (ponto não derivável)")
        self.comparar(rede, x, np.array([0.5]))

    def test_delta_da_saida_segue_a_formula(self):
        rede = RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=1)
        x, y = np.array([0.3, 0.9]), np.array([1.0])
        rede.propagar(x)
        retropropagar(rede, y)
        n = rede.camada_saida.neuronios[0]
        esperado = (n.saida - y[0]) * n.saida * (1 - n.saida)
        self.assertAlmostEqual(n.delta, esperado, places=12)


if __name__ == "__main__":
    unittest.main()
