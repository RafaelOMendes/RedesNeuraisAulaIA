"""Testes 5 e 6 (divisão): normalização só com o treino; mesma seed = mesma divisão.

Também cobre leitura do CSV, escolha de colunas e compatibilidade com a rede.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rna import Dataset, Normalizador, RedeNeural, carregar, dividir, preparar_dados, salvar, verificar_compatibilidade


def dados_com_extremos():
    """100 linhas em [0, 10] + 1 linha extrema (1000) que vai cair fora do treino."""
    rng = np.random.default_rng(1)
    X = np.vstack([rng.uniform(0, 10, size=(100, 2)), [[1000.0, -1000.0]]])
    Y = rng.uniform(0, 1, size=(101, 1))
    return X, Y


def seed_que_deixa_extremo_fora_do_treino(X, Y):
    for seed in range(100):
        if 100 not in dividir(X, Y, (70, 15, 15), seed).indices_treino:
            return seed
    raise AssertionError("nenhuma seed deixou a linha extrema fora do treino")


class TestNormalizacaoSoNoTreino(unittest.TestCase):
    def test_minmax_usa_somente_o_treino(self):
        X, Y = dados_com_extremos()
        seed = seed_que_deixa_extremo_fora_do_treino(X, Y)
        divisao_bruta = dividir(X, Y, (70, 15, 15), seed)
        divisao, norm = preparar_dados(X, Y, (70, 15, 15), "minmax", seed)

        np.testing.assert_array_equal(norm.parametros["minimo"], divisao_bruta.X_treino.min(axis=0))
        np.testing.assert_array_equal(norm.parametros["maximo"], divisao_bruta.X_treino.max(axis=0))
        # Se tivesse usado o dataset inteiro, o máximo seria 1000:
        self.assertNotEqual(norm.parametros["maximo"][0], X[:, 0].max())
        # treino fica em [0, 1]; a linha extrema (fora do treino) sai dessa faixa
        self.assertAlmostEqual(divisao.X_treino.min(), 0.0)
        self.assertAlmostEqual(divisao.X_treino.max(), 1.0)
        fora = np.vstack([divisao.X_validacao, divisao.X_teste])
        self.assertGreater(np.abs(fora).max(), 1.0)

    def test_zscore_usa_somente_o_treino(self):
        X, Y = dados_com_extremos()
        seed = seed_que_deixa_extremo_fora_do_treino(X, Y)
        bruta = dividir(X, Y, (70, 15, 15), seed)
        divisao, norm = preparar_dados(X, Y, (70, 15, 15), "zscore", seed)
        np.testing.assert_array_equal(norm.parametros["media"], bruta.X_treino.mean(axis=0))
        np.testing.assert_array_equal(norm.parametros["desvio"], bruta.X_treino.std(axis=0))
        np.testing.assert_allclose(divisao.X_treino.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(divisao.X_treino.std(axis=0), 1.0, atol=1e-12)

    def test_parametros_salvos_no_json_sao_os_do_treino(self):
        X, Y = dados_com_extremos()
        seed = seed_que_deixa_extremo_fora_do_treino(X, Y)
        bruta = dividir(X, Y, (70, 15, 15), seed)
        _, norm = preparar_dados(X, Y, (70, 15, 15), "minmax", seed)
        rede = RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=0)
        rede.normalizador = norm
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "rede.json"
            salvar(rede, caminho)
            dados = json.loads(caminho.read_text(encoding="utf-8"))
            self.assertEqual(dados["normalizacao"]["tipo"], "minmax")
            self.assertEqual(dados["normalizacao"]["parametros"]["minimo"], bruta.X_treino.min(axis=0).tolist())
            self.assertEqual(dados["normalizacao"]["parametros"]["maximo"], bruta.X_treino.max(axis=0).tolist())
            np.testing.assert_array_equal(carregar(caminho).normalizador.parametros["maximo"], bruta.X_treino.max(axis=0))

    def test_saidas_nao_sao_normalizadas(self):
        X, Y = dados_com_extremos()
        bruta = dividir(X, Y, (70, 15, 15), 5)
        divisao, _ = preparar_dados(X, Y, (70, 15, 15), "zscore", 5)
        np.testing.assert_array_equal(divisao.Y_treino, bruta.Y_treino)
        np.testing.assert_array_equal(divisao.Y_teste, bruta.Y_teste)

    def test_coluna_constante_nao_divide_por_zero(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        for tipo in ("minmax", "zscore"):
            resultado = Normalizador(tipo).ajustar_aplicar(X)
            self.assertTrue(np.all(np.isfinite(resultado)))

    def test_tipo_invalido(self):
        with self.assertRaises(ValueError):
            Normalizador("decimal")


class TestDivisao(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(200, dtype=float).reshape(100, 2)
        self.Y = np.arange(100, dtype=float).reshape(100, 1)

    def test_mesma_seed_mesma_divisao(self):
        a = dividir(self.X, self.Y, (70, 15, 15), seed=123)
        b = dividir(self.X, self.Y, (70, 15, 15), seed=123)
        np.testing.assert_array_equal(a.indices_treino, b.indices_treino)
        np.testing.assert_array_equal(a.indices_validacao, b.indices_validacao)
        np.testing.assert_array_equal(a.indices_teste, b.indices_teste)
        np.testing.assert_array_equal(a.X_teste, b.X_teste)

    def test_seed_diferente_divisao_diferente(self):
        a = dividir(self.X, self.Y, (70, 15, 15), seed=1)
        b = dividir(self.X, self.Y, (70, 15, 15), seed=2)
        self.assertFalse(np.array_equal(a.indices_treino, b.indices_treino))

    def test_tamanhos_e_sem_repeticao(self):
        d = dividir(self.X, self.Y, (70, 15, 15), seed=0)
        self.assertEqual(d.tamanhos(), (70, 15, 15))
        todos = np.concatenate([d.indices_treino, d.indices_validacao, d.indices_teste])
        self.assertEqual(sorted(todos.tolist()), list(range(100)))
        # cada linha continua com o seu Y
        np.testing.assert_array_equal(d.Y_treino[:, 0], d.indices_treino)

    def test_arredondamento_nao_perde_linhas(self):
        X, Y = self.X[:7], self.Y[:7]
        d = dividir(X, Y, (50, 50, 0), seed=0)
        self.assertEqual(sum(d.tamanhos()), 7)
        self.assertEqual(d.tamanhos()[2], 0)  # 0% nunca recebe linha

    def test_percentuais_invalidos(self):
        for p in ((70, 20, 20), (70, 30), (-10, 60, 50), (0, 50, 50)):
            with self.subTest(percentuais=p), self.assertRaises(ValueError):
                dividir(self.X, self.Y, p, seed=0)

    def test_percentuais_em_texto(self):
        d = dividir(self.X, self.Y, "80,10,10", seed=0)
        self.assertEqual(d.tamanhos(), (80, 10, 10))

    def test_cem_por_cento_treino(self):
        d = dividir(self.X, self.Y, (100, 0, 0), seed=0)
        self.assertEqual(d.tamanhos(), (100, 0, 0))


class TestCsvEColunas(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.csv = Path(self.pasta.name) / "dados.csv"
        self.csv.write_text("a,b,c,alvo\n1,2,3,0\n4,5,6,1\n\n7,8,9,1\n", encoding="utf-8")

    def tearDown(self):
        self.pasta.cleanup()

    def test_leitura_e_selecao_por_nome_e_indice(self):
        ds = Dataset.de_csv(self.csv)
        self.assertEqual(ds.colunas, ["a", "b", "c", "alvo"])
        self.assertEqual(ds.n_linhas, 3)
        X, Y = ds.selecionar("a,c", "alvo")
        np.testing.assert_array_equal(X, [[1, 3], [4, 6], [7, 9]])
        X2, Y2 = ds.selecionar(["0", "2"], ["3"])
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(Y, Y2)

    def test_coluna_inexistente(self):
        ds = Dataset.de_csv(self.csv)
        with self.assertRaisesRegex(ValueError, "não existe"):
            ds.selecionar("a,z", "alvo")

    def test_mesma_coluna_em_entrada_e_saida(self):
        ds = Dataset.de_csv(self.csv)
        with self.assertRaises(ValueError):
            ds.selecionar("a,alvo", "alvo")

    def test_valor_nao_numerico(self):
        self.csv.write_text("a,b\n1,x\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "não numérico"):
            Dataset.de_csv(self.csv)

    def test_compatibilidade_com_a_rede(self):
        ds = Dataset.de_csv(self.csv)
        X, Y = ds.selecionar("a,b,c", "alvo")
        verificar_compatibilidade(RedeNeural(3, [(2, "tanh")], 1, "sigmoid", seed=0), X, Y)
        with self.assertRaisesRegex(ValueError, "espera 2 entrada"):
            verificar_compatibilidade(RedeNeural(2, [], 1, "sigmoid", seed=0), X, Y)
        with self.assertRaisesRegex(ValueError, "2 neurônio"):
            verificar_compatibilidade(RedeNeural(3, [], 2, "sigmoid", seed=0), X, Y)

    def test_aviso_de_faixa_da_saida(self):
        X = np.zeros((3, 1))
        Y = np.array([[-2.0], [0.0], [5.0]])
        avisos = verificar_compatibilidade(RedeNeural(1, [], 1, "sigmoid", seed=0), X, Y)
        self.assertEqual(len(avisos), 1)
        self.assertEqual(verificar_compatibilidade(RedeNeural(1, [], 1, "linear", seed=0), X, Y), [])


if __name__ == "__main__":
    unittest.main()
