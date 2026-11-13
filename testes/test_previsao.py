"""Teste 7: previsão com número errado de entradas deve dar erro claro.

Também confere que a previsão não altera pesos, que aplica a normalização salva
e que a interface de linha de comando mostra o erro sem traceback.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rna import ErroEntrada, Normalizador, RedeNeural, prever, salvar
from rna.previsao import converter_valores

RAIZ = Path(__file__).resolve().parent.parent


class TestPrevisao(unittest.TestCase):
    def setUp(self):
        self.rede = RedeNeural(3, [(4, "tanh")], 2, "sigmoid", seed=0)

    def test_quantidade_errada_da_erro_claro(self):
        for valores in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], []):
            with self.subTest(valores=valores):
                with self.assertRaises(ErroEntrada) as ctx:
                    prever(self.rede, valores)
                self.assertIn("espera 3 entrada", str(ctx.exception))
                self.assertIn(f"informados {len(valores)}", str(ctx.exception))

    def test_erro_entrada_e_value_error(self):
        self.assertTrue(issubclass(ErroEntrada, ValueError))

    def test_valor_nao_numerico(self):
        with self.assertRaisesRegex(ErroEntrada, "não numérico"):
            converter_valores("1, abc, 3")
        self.assertEqual(converter_valores("1; 2.5 -3"), [1.0, 2.5, -3.0])

    def test_previsao_nao_altera_pesos(self):
        antes = [(c.pesos(), c.vieses()) for c in self.rede.camadas]
        prever(self.rede, [0.1, 0.2, 0.3])
        for (W0, b0), camada in zip(antes, self.rede.camadas):
            np.testing.assert_array_equal(W0, camada.pesos())
            np.testing.assert_array_equal(b0, camada.vieses())

    def test_aplica_a_normalizacao_salva(self):
        X_treino = np.array([[0.0, 10.0, 100.0], [10.0, 20.0, 300.0]])
        self.rede.normalizador = Normalizador("minmax").ajustar(X_treino)
        x = np.array([5.0, 15.0, 200.0])  # meio da faixa -> [0.5, 0.5, 0.5]
        np.testing.assert_array_equal(prever(self.rede, x), self.rede.propagar(np.array([0.5, 0.5, 0.5])))


class TestInterfaceDeLinhaDeComando(unittest.TestCase):
    def executar(self, *argumentos):
        return subprocess.run(
            [sys.executable, str(RAIZ / "main.py"), *argumentos],
            capture_output=True, text=True, encoding="utf-8", cwd=RAIZ,
        )

    def test_cli_prever_com_quantidade_errada(self):
        with tempfile.TemporaryDirectory() as pasta:
            modelo = Path(pasta) / "rede.json"
            salvar(RedeNeural(2, [(3, "tanh")], 1, "sigmoid", seed=1), modelo)
            ok = self.executar("prever", "-m", str(modelo), "--valores", "1,0")
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("Saída", ok.stdout)
            erro = self.executar("prever", "-m", str(modelo), "--valores", "1,0,1")
            self.assertEqual(erro.returncode, 1)
            self.assertIn("a rede espera 2 entrada(s), mas foram informados 3", erro.stderr)
            self.assertNotIn("Traceback", erro.stderr)


if __name__ == "__main__":
    unittest.main()
