"""Testes da API da interface web (servidor real em uma porta livre).

Confere que o servidor só repassa chamadas para o pacote rna e que:
  * respostas são JSON válido mesmo com conjuntos vazios (sem NaN);
  * previsão com quantidade errada devolve erro 400 com mensagem clara;
  * o MSE de teste só aparece no fim do treino;
  * arquivos estáticos e nomes de arquivo inseguros são tratados.
"""

import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from web.servidor import Manipulador, Servidor


class TestApiWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servidor = Servidor(("127.0.0.1", 0), Manipulador)
        cls.base = f"http://127.0.0.1:{cls.servidor.server_address[1]}"
        threading.Thread(target=cls.servidor.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.servidor.shutdown()
        cls.servidor.server_close()

    def chamar(self, caminho, corpo=None):
        dados = None if corpo is None else json.dumps(corpo).encode("utf-8")
        pedido = urllib.request.Request(self.base + caminho, data=dados, method="POST" if dados is not None else "GET",
                                        headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(pedido, timeout=30) as resposta:
                texto = resposta.read().decode("utf-8")
                status = resposta.status
        except urllib.error.HTTPError as erro:
            texto = erro.read().decode("utf-8")
            status = erro.code
        # json.loads com parse_constant falha se aparecer NaN/Infinity (inválidos para o navegador)
        return status, json.loads(texto, parse_constant=lambda c: self.fail(f"JSON com {c}"))

    def esperar_fim(self):
        for _ in range(300):
            status, p = self.chamar("/api/treino/progresso?desde=0")
            if not p["treinando"]:
                return p
            time.sleep(0.05)
        self.fail("o treino não terminou a tempo")

    def test_fluxo_xor_sem_validacao_nem_teste(self):
        self.assertEqual(self.chamar("/api/dataset/abrir", {"nome": "xor.csv"})[0], 200)
        status, rede = self.chamar("/api/rede/criar", {"n_entradas": 2, "ocultas": [{"neuronios": 4, "ativacao": "tanh"}],
                                                      "n_saidas": 1, "ativacao_saida": "sigmoid", "seed": 42})
        self.assertEqual(status, 200)
        self.assertEqual(rede["arquitetura"], [2, 4, 1])
        status, r = self.chamar("/api/treino/iniciar", {"x": ["x1", "x2"], "y": ["y"], "percentuais": [100, 0, 0],
                                                       "seed": 42, "normalizacao": "nenhuma", "epocas": 300, "taxa": 0.5})
        self.assertEqual(status, 200, r)
        p = self.esperar_fim()
        self.assertEqual(p["status"], "concluido")
        self.assertEqual(len(p["mse_treino"]), 300)
        self.assertTrue(all(v is None for v in p["mse_validacao"]))   # NaN vira null
        self.assertLess(p["mse_treino"][-1], p["mse_treino_inicial"])
        self.assertIsNone(p["resultado"]["mse_teste"])

        status, prev = self.chamar("/api/prever", {"valores": [1, 0]})
        self.assertEqual(status, 200)
        self.assertGreater(prev["saida"][0], 0.5)
        status, erro = self.chamar("/api/prever", {"texto": "1, 0, 1"})
        self.assertEqual(status, 400)
        self.assertIn("espera 2 entrada(s), mas foram informados 3", erro["erro"])

    def test_teste_so_no_fim_e_parar(self):
        self.chamar("/api/dataset/abrir", {"nome": "tres_classes.csv"})
        self.chamar("/api/rede/criar", {"n_entradas": 2, "ocultas": [{"neuronios": 4, "ativacao": "tanh"}],
                                        "n_saidas": 3, "ativacao_saida": "sigmoid", "seed": 1})
        status, r = self.chamar("/api/treino/iniciar", {"x": "x1,x2", "y": "classe_a,classe_b,classe_c",
                                                       "percentuais": [70, 15, 15], "seed": 3,
                                                       "normalizacao": "zscore", "epocas": 5000, "taxa": 0.1})
        self.assertEqual(status, 200, r)
        time.sleep(0.3)
        status, p = self.chamar("/api/treino/progresso?desde=0")
        self.assertIsNone(p["resultado"])                      # teste ainda não avaliado
        self.assertEqual(self.chamar("/api/rede/criar", {"n_entradas": 2, "n_saidas": 3})[0], 409)
        self.chamar("/api/treino/parar", {})
        p = self.esperar_fim()
        self.assertEqual(p["status"], "interrompido")
        self.assertLess(p["epoca"], 5000)
        self.assertIsNotNone(p["resultado"]["mse_teste"])     # avaliado uma vez, no fim
        status, vis = self.chamar("/api/visual", {"x": ["x1", "x2"], "y": ["classe_a", "classe_b", "classe_c"], "res": 12})
        self.assertEqual(vis["tipo"], "superficie")
        self.assertEqual(len(vis["celulas"]), 144)

    def test_estaticos_e_nomes_inseguros(self):
        with urllib.request.urlopen(self.base + "/estatico/app.js") as resposta:
            self.assertTrue(resposta.headers["Content-Type"].startswith("text/javascript"))
        status, _ = self.chamar("/estatico/../servidor.py")
        self.assertEqual(status, 404)
        status, erro = self.chamar("/api/modelo/carregar", {"nome": "../../main"})
        self.assertEqual(status, 400)  # vira 'main.json' dentro de modelos/, que não existe


if __name__ == "__main__":
    unittest.main()
