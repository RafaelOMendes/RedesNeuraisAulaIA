"""Teste 3: treinar problemas simples (XOR, AND, OR) e mostrar que o erro cai.

Também confere regras do treino: SGD por exemplo, deltas com os pesos antigos,
validação sem alterar pesos.
"""

import unittest

import numpy as np

from rna import RedeNeural, Treinador
from rna.erro import mse_rede

X_PORTAS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
PORTAS = {
    "xor": np.array([[0], [1], [1], [0]], dtype=float),
    "and": np.array([[0], [0], [0], [1]], dtype=float),
    "or": np.array([[0], [1], [1], [1]], dtype=float),
}


def copiar_parametros(rede):
    return [(c.pesos(), c.vieses()) for c in rede.camadas]


class TestPortasLogicas(unittest.TestCase):
    def treinar_porta(self, nome, ocultas, epocas=2000, taxa=0.5):
        Y = PORTAS[nome]
        rede = RedeNeural(2, ocultas, 1, "sigmoid", seed=42)
        historico = rede.treinar(X_PORTAS, Y, taxa_aprendizado=taxa, epocas=epocas, seed=42)
        return rede, historico, Y

    def test_xor_erro_cai(self):
        rede, h, Y = self.treinar_porta("xor", [(4, "tanh")])
        print(f"\n  XOR: MSE inicial {h.mse_treino_inicial:.6f} -> final {h.mse_treino[-1]:.6f}")
        self.assertLess(h.mse_treino[-1], h.mse_treino_inicial)
        self.assertLess(h.mse_treino[-1], 0.01)
        previsto = (rede.propagar_lote(X_PORTAS) >= 0.5).astype(float)
        np.testing.assert_array_equal(previsto, Y)

    def test_and_e_or(self):
        for nome in ("and", "or"):
            with self.subTest(porta=nome):
                rede, h, Y = self.treinar_porta(nome, [], epocas=1000)
                self.assertLess(h.mse_treino[-1], h.mse_treino_inicial)
                previsto = (rede.propagar_lote(X_PORTAS) >= 0.5).astype(float)
                np.testing.assert_array_equal(previsto, Y)

    def test_historico_tem_uma_entrada_por_epoca(self):
        _, h, _ = self.treinar_porta("xor", [(3, "tanh")], epocas=25)
        self.assertEqual(h.epocas, 25)
        self.assertEqual(len(h.mse_validacao), 25)


class TestRegrasDoTreino(unittest.TestCase):
    def test_deltas_usam_os_pesos_antes_da_atualizacao(self):
        # Refaz um passo de SGD com matrizes e compara com a implementação.
        rede = RedeNeural(3, [(4, "tanh"), (3, "relu")], 2, "sigmoid", seed=8)
        x = np.array([0.4, -0.9, 1.3])
        y = np.array([1.0, 0.0])
        eta = 0.3
        antes = copiar_parametros(rede)

        # cálculo de referência, vetorizado, com os pesos ANTIGOS
        from rna.ativacoes import obter_ativacao
        ativ = [obter_ativacao(c.ativacao.nome) for c in rede.camadas]
        entradas, zs, a = [], [], x
        for (W, b), f in zip(antes, ativ):
            entradas.append(a)
            z = W @ a + b
            zs.append(z)
            a = f.f(z)
        deltas = [None] * len(antes)
        deltas[-1] = (a - y) * ativ[-1].derivada(zs[-1])
        for l in range(len(antes) - 2, -1, -1):
            deltas[l] = ativ[l].derivada(zs[l]) * (antes[l + 1][0].T @ deltas[l + 1])
        esperado = [(W - eta * np.outer(d, e), b - eta * d) for (W, b), d, e in zip(antes, deltas, entradas)]

        Treinador(rede, taxa_aprendizado=eta).passo_sgd(x, y)

        for c, ((W_esp, b_esp), camada) in enumerate(zip(esperado, rede.camadas), start=1):
            np.testing.assert_allclose(camada.pesos(), W_esp, rtol=1e-12, err_msg=f"pesos camada {c}")
            np.testing.assert_allclose(camada.vieses(), b_esp, rtol=1e-12, err_msg=f"vieses camada {c}")
            np.testing.assert_allclose(camada.deltas(), deltas[c - 1], rtol=1e-12)

    def test_sgd_atualiza_apos_cada_exemplo(self):
        # Uma época sem embaralhar = N passos de SGD em sequência.
        Y = PORTAS["xor"]
        rede_a = RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=4)
        rede_b = RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=4)
        Treinador(rede_a, 0.2, epocas=1, embaralhar=False).treinar(X_PORTAS, Y)
        passo = Treinador(rede_b, 0.2)
        for x, y in zip(X_PORTAS, Y):
            passo.passo_sgd(x, y)
        for (Wa, ba), (Wb, bb) in zip(copiar_parametros(rede_a), copiar_parametros(rede_b)):
            np.testing.assert_array_equal(Wa, Wb)
            np.testing.assert_array_equal(ba, bb)

    def test_validacao_nao_altera_pesos(self):
        rede = RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=2)
        antes = copiar_parametros(rede)
        mse_rede(rede, X_PORTAS, PORTAS["xor"])
        for (W0, b0), (W1, b1) in zip(antes, copiar_parametros(rede)):
            np.testing.assert_array_equal(W0, W1)
            np.testing.assert_array_equal(b0, b1)

    def test_treino_com_e_sem_validacao_da_os_mesmos_pesos(self):
        # Se a validação mexesse nos pesos, o resultado final seria diferente.
        Y = PORTAS["xor"]
        rede_a = RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=9)
        rede_b = RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=9)
        rede_a.treinar(X_PORTAS, Y, epocas=30, seed=1)
        rede_b.treinar(X_PORTAS, Y, X_PORTAS[:2], Y[:2], epocas=30, seed=1)
        for (Wa, ba), (Wb, bb) in zip(copiar_parametros(rede_a), copiar_parametros(rede_b)):
            np.testing.assert_array_equal(Wa, Wb)
            np.testing.assert_array_equal(ba, bb)

    def test_parametros_invalidos(self):
        rede = RedeNeural(2, [], 1, "sigmoid", seed=1)
        with self.assertRaises(ValueError):
            Treinador(rede, taxa_aprendizado=0)
        with self.assertRaises(ValueError):
            Treinador(rede, epocas=0)

    def test_arquitetura_configuravel(self):
        rede = RedeNeural(5, [(8, "relu"), (6, "tanh"), (4, "sigmoid")], 3, "linear", seed=0)
        self.assertEqual(rede.arquitetura(), [5, 8, 6, 4, 3])
        self.assertEqual([c.ativacao.nome for c in rede.camadas], ["relu", "tanh", "sigmoid", "linear"])
        self.assertEqual(rede.propagar(np.ones(5)).shape, (3,))


if __name__ == "__main__":
    unittest.main()
