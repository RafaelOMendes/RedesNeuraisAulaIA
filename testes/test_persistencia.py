"""Teste 4: salvar, carregar e comparar as saídas (têm que ser idênticas)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rna import RedeNeural, carregar, preparar_dados, prever, salvar
from rna.previsao import prever_lote


class TestPersistencia(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.caminho = Path(self.pasta.name) / "sub" / "rede.json"

    def tearDown(self):
        self.pasta.cleanup()

    def rede_treinada(self, normalizacao):
        rng = np.random.default_rng(0)
        X = rng.normal([5.0, 50.0, -3.0], [2.0, 20.0, 1.0], size=(60, 3))
        Y = np.column_stack([np.sin(X[:, 0]) * 0.5, (X[:, 1] > 50).astype(float)])
        divisao, norm = preparar_dados(X, Y, (70, 15, 15), normalizacao, seed=3)
        rede = RedeNeural(3, [(5, "relu"), (4, "tanh")], 2, "linear", seed=12)
        rede.normalizador = None if norm.tipo == "nenhuma" else norm
        rede.treinar(divisao.X_treino, divisao.Y_treino, epocas=5, seed=3)
        return rede, X

    def test_saidas_identicas_apos_carregar(self):
        for normalizacao in ("nenhuma", "minmax", "zscore"):
            with self.subTest(normalizacao=normalizacao):
                rede, X = self.rede_treinada(normalizacao)
                salvar(rede, self.caminho)
                carregada = carregar(self.caminho)
                # passa pelo fluxo de previsão completo (inclui a normalização)
                np.testing.assert_array_equal(prever_lote(rede, X), prever_lote(carregada, X))
                for x in X[:5]:
                    np.testing.assert_array_equal(prever(rede, x), prever(carregada, x))

    def test_estrutura_reconstruida(self):
        rede, _ = self.rede_treinada("minmax")
        salvar(rede, self.caminho)
        carregada = carregar(self.caminho)
        self.assertEqual(carregada.arquitetura(), rede.arquitetura())
        self.assertEqual([c.ativacao.nome for c in carregada.camadas], [c.ativacao.nome for c in rede.camadas])
        for c_orig, c_carr in zip(rede.camadas, carregada.camadas):
            np.testing.assert_array_equal(c_orig.pesos(), c_carr.pesos())
            np.testing.assert_array_equal(c_orig.vieses(), c_carr.vieses())
        self.assertEqual(carregada.normalizador.tipo, "minmax")
        for chave in ("minimo", "maximo"):
            np.testing.assert_array_equal(carregada.normalizador.parametros[chave], rede.normalizador.parametros[chave])

    def test_conteudo_do_json(self):
        rede, _ = self.rede_treinada("zscore")
        salvar(rede, self.caminho)
        dados = json.loads(self.caminho.read_text(encoding="utf-8"))
        self.assertEqual(dados["n_entradas"], 3)
        self.assertEqual(dados["n_camadas"], 3)
        self.assertEqual(dados["n_camadas_ocultas"], 2)
        self.assertEqual(dados["neuronios_por_camada"], [5, 4, 2])
        self.assertEqual([c["ativacao"] for c in dados["camadas"]], ["relu", "tanh", "linear"])
        self.assertEqual(len(dados["camadas"][0]["pesos"]), 5)
        self.assertEqual(len(dados["camadas"][0]["pesos"][0]), 3)
        self.assertEqual(len(dados["camadas"][0]["vieses"]), 5)
        self.assertEqual(dados["normalizacao"]["tipo"], "zscore")
        self.assertEqual(set(dados["normalizacao"]["parametros"]), {"media", "desvio"})

    def test_rede_sem_treino_tambem_salva(self):
        rede = RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=1)
        salvar(rede, self.caminho)
        carregada = carregar(self.caminho)
        x = np.array([0.3, -0.2])
        np.testing.assert_array_equal(rede.propagar(x), carregada.propagar(x))
        self.assertIsNone(carregada.normalizador)

    def test_arquivo_invalido(self):
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.caminho.write_text('{"qualquer": 1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            carregar(self.caminho)
        with self.assertRaises(FileNotFoundError):
            carregar(Path(self.pasta.name) / "nao_existe.json")


if __name__ == "__main__":
    unittest.main()
