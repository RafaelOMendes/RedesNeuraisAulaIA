"""Servidor web local da interface gráfica.

Usa somente a biblioteca padrão (http.server). Toda a parte de rede neural é
feita pelo pacote ``rna`` (o mesmo código usado pela linha de comando): o
navegador só desenha a interface e chama esta API JSON.

Execução:  python main.py web            (abre http://127.0.0.1:8000)

Concorrência:
  * o treino roda numa thread própria e é o ÚNICO que mexe na rede durante o treino;
  * a cada ~250 ms a thread publica uma cópia (snapshot) da rede, que é usada
    pelas visualizações e previsões enquanto o treino acontece;
  * toda propagação direta feita pelos handlers passa por ``lock_vis`` (os
    neurônios guardam z/saída do último exemplo, então dois forwards ao mesmo
    tempo na mesma rede não podem acontecer).
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from rna import Dataset, ErroEntrada, RedeNeural, carregar, dividir, salvar, verificar_compatibilidade
from rna.ativacoes import nomes_disponiveis
from rna.dataset import interpretar_percentuais, normalizar_divisao
from rna.erro import mse_rede
from rna.experimento import avaliar_teste, metadados_treino
from rna.persistencia import limpar_json, rede_para_dict
from rna.previsao import converter_valores, prever, prever_lote

RAIZ = Path(__file__).resolve().parent.parent
ESTATICO = Path(__file__).resolve().parent / "estatico"
PASTA_DADOS = RAIZ / "dados"
PASTA_MODELOS = RAIZ / "modelos"

TIPOS_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

MAX_CORPO = 5 * 1024 * 1024  # 5 MB (upload de CSV)


class ErroHTTP(Exception):
    def __init__(self, status: int, mensagem: str):
        super().__init__(mensagem)
        self.status = status
        self.mensagem = mensagem


class TreinoInterrompido(Exception):
    """Lançada dentro do callback de época quando o usuário clica em Parar."""


def nome_seguro(nome: str, extensao: str) -> str:
    """Aceita só um nome de arquivo simples (sem pastas). Ex.: 'meu modelo' -> 'meu_modelo.json'."""
    base = Path(str(nome or "").strip()).name
    if base.lower().endswith(extensao):
        base = base[: -len(extensao)]
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", base).strip("_")
    if not base:
        raise ErroHTTP(400, "Informe um nome de arquivo válido (letras, números, _ ou -).")
    return base[:60] + extensao


def _json_seguro(valor):
    """limpar_json + bool do numpy (NaN vira null; o navegador mostra '—')."""
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, dict):
        return {k: _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_seguro(v) for v in valor]
    return limpar_json(valor)


# ====================================================================== #
# Estado da sessão (um único usuário local)
# ====================================================================== #
class Sessao:
    def __init__(self):
        self.lock = threading.Lock()       # estado e progresso do treino
        self.lock_vis = threading.Lock()   # propagações diretas feitas pelos handlers
        self.rede: RedeNeural | None = None
        self.rede_origem = ""
        self.dataset: Dataset | None = None
        self.dataset_nome: str | None = None
        self.nomes_x: list[str] = []
        self.nomes_y: list[str] = []
        self.divisao = None
        self.divisao_norm = None
        self.treinando = False
        self.snapshot: RedeNeural | None = None
        self.parar = threading.Event()
        self.progresso = self._progresso_vazio()

    @staticmethod
    def _progresso_vazio() -> dict:
        return {
            "id": 0, "status": "ocioso", "epoca": 0, "epocas": 0,
            "mse_treino": [], "mse_validacao": [], "mse_treino_inicial": None,
            "mse_validacao_inicial": None, "versao": 0, "resultado": None, "erro": None,
            "inicio": None, "duracao": None, "tamanhos": None,
        }

    # -------------------------------------------------------------- #
    def exigir_ocioso(self):
        if self.treinando:
            raise ErroHTTP(409, "Há um treino em andamento. Aguarde ou clique em Parar.")

    def exigir_rede(self):
        if self.rede is None:
            raise ErroHTTP(400, "Crie ou carregue uma rede primeiro.")

    def rede_para_leitura(self) -> RedeNeural | None:
        """Rede usada pelos handlers: o snapshot durante o treino, a própria rede fora dele."""
        with self.lock:
            return self.snapshot if self.treinando else self.rede

    # -------------------------------------------------------------- #
    def info_rede(self, rede: RedeNeural | None = None) -> dict | None:
        rede = rede or self.rede
        if rede is None:
            return None
        dados = rede_para_dict(rede)
        dados["arquitetura"] = rede.arquitetura()
        dados["n_saidas"] = rede.n_saidas
        dados["n_parametros"] = rede.n_parametros()
        dados["origem"] = self.rede_origem
        return dados

    def info_dataset(self) -> dict | None:
        if self.dataset is None:
            return None
        ds = self.dataset
        estatisticas = [
            {"nome": n, "min": float(ds.dados[:, i].min()), "max": float(ds.dados[:, i].max()),
             "media": float(ds.dados[:, i].mean()),
             "binaria": bool(np.all(np.isin(ds.dados[:, i], (0.0, 1.0))))}
            for i, n in enumerate(ds.colunas)
        ]
        return {
            "nome": self.dataset_nome, "colunas": ds.colunas, "n_linhas": ds.n_linhas,
            "estatisticas": estatisticas, "amostra": ds.dados[:8].tolist(),
            "x": self.nomes_x, "y": self.nomes_y,
        }

    def estado(self) -> dict:
        with self.lock:
            p = dict(self.progresso)
            p.pop("mse_treino"), p.pop("mse_validacao")
            treinando = self.treinando
        return {
            "rede": self.info_rede(self.rede_para_leitura()),
            "dataset": self.info_dataset(),
            "treino": p, "treinando": treinando,
            "ativacoes": nomes_disponiveis(),
        }

    # -------------------------------------------------------------- #
    # Dataset
    # -------------------------------------------------------------- #
    def listar_datasets(self) -> list[dict]:
        itens = []
        for caminho in sorted(PASTA_DADOS.glob("*.csv")):
            try:
                ds = Dataset.de_csv(caminho)
                itens.append({"nome": caminho.name, "linhas": ds.n_linhas, "colunas": ds.colunas})
            except (ValueError, OSError) as erro:
                itens.append({"nome": caminho.name, "erro": str(erro)})
        return itens

    def abrir_dataset(self, nome: str) -> dict:
        self.exigir_ocioso()
        arquivo = PASTA_DADOS / nome_seguro(nome, ".csv")
        ds = Dataset.de_csv(arquivo)
        with self.lock:
            self.exigir_ocioso()
            self.dataset, self.dataset_nome = ds, arquivo.name
            self.nomes_x, self.nomes_y = [], []
            self.divisao = self.divisao_norm = None
        return self.info_dataset()

    def enviar_dataset(self, nome: str, conteudo: str) -> dict:
        self.exigir_ocioso()
        if not conteudo or not conteudo.strip():
            raise ErroHTTP(400, "O arquivo enviado está vazio.")
        arquivo = PASTA_DADOS / nome_seguro(nome, ".csv")
        temporario = arquivo.with_suffix(".tmp")
        texto = "\n".join(conteudo.splitlines()) + "\n"  # normaliza quebras de linha do Windows
        with open(temporario, "w", encoding="utf-8", newline="") as saida:
            saida.write(texto)
        try:
            Dataset.de_csv(temporario)  # valida antes de aceitar
        except ValueError:
            temporario.unlink(missing_ok=True)
            raise
        temporario.replace(arquivo)
        return self.abrir_dataset(arquivo.name)

    def selecionar(self, x, y):
        if self.dataset is None:
            raise ErroHTTP(400, "Escolha um dataset primeiro.")
        X, Y = self.dataset.selecionar(x, y)
        return X, Y, self.dataset.nomes(x), self.dataset.nomes(y)

    # -------------------------------------------------------------- #
    # Rede
    # -------------------------------------------------------------- #
    def criar_rede(self, corpo: dict) -> dict:
        self.exigir_ocioso()
        ocultas = [(int(c["neuronios"]), str(c["ativacao"])) for c in corpo.get("ocultas", [])]
        for n, _ in ocultas:
            if not 1 <= n <= 64:
                raise ErroHTTP(400, "Use entre 1 e 64 neurônios por camada na interface web.")
        if len(ocultas) > 6:
            raise ErroHTTP(400, "Use no máximo 6 camadas ocultas na interface web.")
        seed = corpo.get("seed")
        rede = RedeNeural(
            int(corpo["n_entradas"]), ocultas, int(corpo["n_saidas"]),
            str(corpo.get("ativacao_saida", "sigmoid")),
            seed=None if seed in (None, "") else int(seed),
        )
        with self.lock_vis, self.lock:
            self.exigir_ocioso()
            self.rede, self.rede_origem = rede, "criada na interface"
            self.progresso = self._progresso_vazio()
            self.divisao_norm = None
        return self.info_rede()

    def salvar_modelo(self, nome: str) -> dict:
        self.exigir_ocioso()
        self.exigir_rede()
        arquivo = PASTA_MODELOS / nome_seguro(nome, ".json")
        salvar(self.rede, arquivo)
        return {"nome": arquivo.name, "caminho": f"modelos/{arquivo.name}"}

    def listar_modelos(self) -> list[dict]:
        itens = []
        for caminho in sorted(PASTA_MODELOS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                dados = json.loads(caminho.read_text(encoding="utf-8"))
                arquitetura = [dados["n_entradas"]] + list(dados.get("neuronios_por_camada", []))
                itens.append({
                    "nome": caminho.name,
                    "arquitetura": arquitetura,
                    "ativacoes": [c.get("ativacao") for c in dados.get("camadas", [])],
                    "normalizacao": (dados.get("normalizacao") or {}).get("tipo", "nenhuma"),
                    "treinada": bool((dados.get("metadados") or {}).get("treino")),
                    "mse_teste": ((dados.get("metadados") or {}).get("treino") or {}).get("mse_teste"),
                    "modificado": caminho.stat().st_mtime,
                })
            except (ValueError, KeyError, OSError):
                continue
        return itens

    def carregar_modelo(self, nome: str) -> dict:
        self.exigir_ocioso()
        arquivo = PASTA_MODELOS / nome_seguro(nome, ".json")
        rede = carregar(arquivo)
        with self.lock_vis, self.lock:
            self.exigir_ocioso()
            self.rede, self.rede_origem = rede, f"carregada de modelos/{arquivo.name}"
            self.progresso = self._progresso_vazio()
            self.divisao_norm = None
        return self.info_rede()

    # -------------------------------------------------------------- #
    # Treino
    # -------------------------------------------------------------- #
    def iniciar_treino(self, corpo: dict) -> dict:
        self.exigir_rede()
        X, Y, nomes_x, nomes_y = self.selecionar(corpo.get("x", []), corpo.get("y", []))
        avisos = verificar_compatibilidade(self.rede, X, Y)
        percentuais = interpretar_percentuais(corpo.get("percentuais", (70, 15, 15)))
        seed = corpo.get("seed")
        seed = None if seed in (None, "") else int(seed)
        normalizacao = str(corpo.get("normalizacao", "nenhuma"))
        epocas = int(corpo.get("epocas", 200))
        taxa = float(corpo.get("taxa", 0.1))
        embaralhar = bool(corpo.get("embaralhar", True))
        if not 1 <= epocas <= 100000:
            raise ErroHTTP(400, "Use entre 1 e 100000 épocas.")
        if not taxa > 0:
            raise ErroHTTP(400, "A taxa de aprendizado deve ser maior que zero.")

        # Mesma ordem da linha de comando: divide, depois normaliza só com o treino.
        divisao = dividir(X, Y, percentuais, seed)
        divisao_norm, normalizador = normalizar_divisao(divisao, normalizacao)

        with self.lock_vis, self.lock:
            if self.treinando:
                raise ErroHTTP(409, "Já existe um treino em andamento.")
            rede = self.rede
            rede.normalizador = None if normalizador.tipo == "nenhuma" else normalizador
            rede.metadados.update({"colunas_entrada": nomes_x, "colunas_saida": nomes_y,
                                   "dataset": f"dados/{self.dataset_nome}"})
            self.nomes_x, self.nomes_y = nomes_x, nomes_y
            self.divisao, self.divisao_norm = divisao, divisao_norm
            self.parar.clear()
            id_treino = self.progresso["id"] + 1
            self.progresso = self._progresso_vazio()
            self.progresso.update({
                "id": id_treino, "status": "treinando", "epocas": epocas, "inicio": time.time(),
                "tamanhos": dict(zip(("treino", "validacao", "teste"), divisao.tamanhos())),
                "mse_treino_inicial": mse_rede(rede, divisao_norm.X_treino, divisao_norm.Y_treino),
                "mse_validacao_inicial": mse_rede(rede, divisao_norm.X_validacao, divisao_norm.Y_validacao),
            })
            self.snapshot = copy.deepcopy(rede)
            self.treinando = True

        config = {"epocas": epocas, "taxa": taxa, "seed": seed, "embaralhar": embaralhar,
                  "normalizacao": normalizador.tipo}
        thread = threading.Thread(target=self._executar_treino, args=(config,), daemon=True)
        thread.start()
        return {"id": id_treino, "avisos": avisos, "tamanhos": self.progresso["tamanhos"],
                "normalizacao": normalizador.para_dict()}

    def _executar_treino(self, config: dict) -> None:
        """Roda na thread de treino. Só ela mexe em self.rede até o fim."""
        rede, d = self.rede, self.divisao_norm
        ultimo_snapshot = time.monotonic()

        def ao_fim_epoca(epoca, mse_tr, mse_va):
            nonlocal ultimo_snapshot
            agora = time.monotonic()
            snap = None
            if agora - ultimo_snapshot >= 0.25:
                snap = copy.deepcopy(rede)
                ultimo_snapshot = agora
            with self.lock:
                p = self.progresso
                p["epoca"] = epoca
                p["mse_treino"].append(mse_tr)
                p["mse_validacao"].append(mse_va)
                if snap is not None:
                    self.snapshot = snap
                    p["versao"] += 1
            if self.parar.is_set():
                raise TreinoInterrompido()

        X_va = d.X_validacao if len(d.X_validacao) else None
        Y_va = d.Y_validacao if len(d.Y_validacao) else None
        status, erro = "concluido", None
        try:
            rede.treinar(d.X_treino, d.Y_treino, X_va, Y_va, taxa_aprendizado=config["taxa"],
                         epocas=config["epocas"], seed=config["seed"],
                         embaralhar=config["embaralhar"], ao_fim_epoca=ao_fim_epoca)
        except TreinoInterrompido:
            status = "interrompido"
        except Exception as excecao:  # noqa: BLE001 - mostrado ao usuário
            status, erro = "erro", str(excecao)

        resultado = None
        with self.lock:
            p = self.progresso
            epocas_feitas = p["epoca"]
            resultados = {
                "mse_treino_inicial": p["mse_treino_inicial"],
                "mse_treino_final": p["mse_treino"][-1] if p["mse_treino"] else math.nan,
                "mse_validacao_final": p["mse_validacao"][-1] if p["mse_validacao"] else math.nan,
            }
        if status != "erro":
            # Teste: avaliado UMA vez, só no fim (a thread ainda é dona da rede aqui).
            resultados.update(avaliar_teste(rede, d))
            rede.metadados["treino"] = metadados_treino(
                self.divisao, config["normalizacao"], epocas_feitas, config["taxa"],
                config["seed"], config["embaralhar"], resultados,
            )
            resultado = resultados
        final = copy.deepcopy(rede)
        with self.lock_vis, self.lock:
            p = self.progresso
            p.update({"status": status, "erro": erro, "resultado": resultado,
                      "duracao": time.time() - p["inicio"], "versao": p["versao"] + 1})
            self.snapshot = final
            self.treinando = False
            self.rede_origem = "treinada na interface"

    def obter_progresso(self, desde: int) -> dict:
        with self.lock:
            p = self.progresso
            desde = max(0, min(int(desde), len(p["mse_treino"])))
            resposta = {k: v for k, v in p.items() if k not in ("mse_treino", "mse_validacao")}
            resposta["desde"] = desde
            resposta["mse_treino"] = p["mse_treino"][desde:]
            resposta["mse_validacao"] = p["mse_validacao"][desde:]
            resposta["treinando"] = self.treinando
        return resposta

    # -------------------------------------------------------------- #
    # Previsão e visualização
    # -------------------------------------------------------------- #
    def prever(self, corpo: dict) -> dict:
        if "texto" in corpo and corpo["texto"] is not None:
            valores = converter_valores(corpo["texto"])
        else:
            valores = [float(v) for v in corpo.get("valores", [])]
        with self.lock_vis:
            rede = self.rede_para_leitura()
            if rede is None:
                raise ErroHTTP(400, "Crie ou carregue uma rede primeiro.")
            saida = prever(rede, valores)  # valida a quantidade e aplica a normalização salva
            normalizada = rede.normalizador.aplicar(np.asarray(valores, dtype=float)) if rede.normalizador else None
            nomes_y = rede.metadados.get("colunas_saida") or [f"y{i + 1}" for i in range(rede.n_saidas)]
            ativ_saida = rede.camada_saida.ativacao.nome
        resposta = {"entrada": valores, "saida": saida.tolist(), "nomes_saida": nomes_y,
                    "normalizada": None if normalizada is None else normalizada.tolist(),
                    "normalizacao": rede.normalizador.tipo if rede.normalizador else "nenhuma",
                    "ativacao_saida": ativ_saida}
        if len(saida) > 1:
            resposta["classe"] = int(np.argmax(saida))
        elif ativ_saida == "sigmoid":
            resposta["classe"] = int(saida[0] >= 0.5)
        return resposta

    def visual(self, resolucao: int, x=None, y=None) -> dict:
        """Dados para o gráfico 'o que a rede aprendeu'.

        ``x``/``y`` são as colunas escolhidas na página (se omitidas, usa as do
        último treino). Os pontos são marcados como treino/validação/teste
        quando as colunas são as mesmas da última divisão.
        """
        resolucao = max(10, min(int(resolucao), 80))
        with self.lock:
            dataset, divisao = self.dataset, self.divisao
            ultimas = (list(self.nomes_x), list(self.nomes_y))
        with self.lock_vis:
            rede = self.rede_para_leitura()
            if rede is None or dataset is None:
                return {"tipo": None}
            try:
                nomes_x = dataset.nomes(x) if x else ultimas[0]
                nomes_y = dataset.nomes(y) if y else ultimas[1]
                if not nomes_x or not nomes_y:
                    return {"tipo": None}
                X, Y = dataset.selecionar(nomes_x, nomes_y)
                verificar_compatibilidade(rede, X, Y)
            except ValueError:
                return {"tipo": None}

            grupos = np.full(len(X), "treino", dtype=object)
            mesma_selecao = (nomes_x, nomes_y) == ultimas
            if mesma_selecao and divisao is not None and sum(divisao.tamanhos()) == len(X):
                grupos[divisao.indices_validacao] = "validacao"
                grupos[divisao.indices_teste] = "teste"
            indices = np.arange(len(X))
            if len(X) > 400:
                indices = np.random.default_rng(0).choice(len(X), 400, replace=False)

            if rede.n_entradas == 2:
                lo, hi = X.min(axis=0), X.max(axis=0)
                folga = np.where(hi - lo == 0, 1.0, hi - lo) * 0.08
                lo, hi = lo - folga, hi + folga
                gx = np.linspace(lo[0], hi[0], resolucao)
                gy = np.linspace(lo[1], hi[1], resolucao)
                malha = np.array([[a, b] for b in gy for a in gx])
                saidas = prever_lote(rede, malha)
                if rede.n_saidas == 1:
                    celulas = saidas[:, 0].tolist()
                else:
                    celulas = [[int(np.argmax(s)), float(np.max(s))] for s in saidas]
                pontos = [[float(X[i, 0]), float(X[i, 1]),
                           int(np.argmax(Y[i])) if Y.shape[1] > 1 else float(Y[i, 0]), grupos[i]]
                          for i in indices]
                return {"tipo": "superficie", "res": resolucao, "x": [lo[0], hi[0]], "y": [lo[1], hi[1]],
                        "celulas": celulas, "pontos": pontos, "n_saidas": rede.n_saidas,
                        "nomes_x": nomes_x, "nomes_y": nomes_y,
                        "ativacao_saida": rede.camada_saida.ativacao.nome}

            if rede.n_entradas == 1 and rede.n_saidas == 1:
                lo, hi = float(X.min()), float(X.max())
                folga = (hi - lo or 1.0) * 0.05
                xs = np.linspace(lo - folga, hi + folga, 160)
                ys = prever_lote(rede, xs.reshape(-1, 1))[:, 0]
                pontos = [[float(X[i, 0]), float(Y[i, 0]), grupos[i]] for i in indices]
                return {"tipo": "curva", "xs": xs.tolist(), "ys": ys.tolist(), "pontos": pontos,
                        "nomes_x": nomes_x, "nomes_y": nomes_y}

            previstos = prever_lote(rede, X[indices])
            pontos = [[float(Y[i, 0]), float(previstos[k, 0]), grupos[i]] for k, i in enumerate(indices)]
            return {"tipo": "dispersao", "pontos": pontos, "nomes_y": nomes_y}


SESSAO = Sessao()


# ====================================================================== #
# HTTP
# ====================================================================== #
class Manipulador(BaseHTTPRequestHandler):
    server_version = "RNA-MLP/1.0"

    def log_message(self, formato, *args):  # silencia o log de cada requisição
        pass

    # ---------------------------------------------------------------- #
    def _enviar(self, status: int, corpo: bytes, tipo: str, extras: dict | None = None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            for chave, valor in (extras or {}).items():
                self.send_header(chave, valor)
            self.end_headers()
            self.wfile.write(corpo)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, status: int, dados):
        corpo = json.dumps(_json_seguro(dados), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._enviar(status, corpo, TIPOS_MIME[".json"])

    def _ler_corpo(self) -> dict:
        tamanho = int(self.headers.get("Content-Length") or 0)
        if tamanho > MAX_CORPO:
            raise ErroHTTP(413, "Arquivo muito grande (limite de 5 MB).")
        if tamanho == 0:
            return {}
        try:
            dados = json.loads(self.rfile.read(tamanho).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ErroHTTP(400, "Corpo da requisição não é um JSON válido.") from None
        if not isinstance(dados, dict):
            raise ErroHTTP(400, "O corpo da requisição deve ser um objeto JSON.")
        return dados

    def _estatico(self, caminho: str):
        relativo = "index.html" if caminho in ("/", "") else caminho.removeprefix("/estatico/")
        if caminho not in ("/", "") and not caminho.startswith("/estatico/"):
            raise ErroHTTP(404, "Página não encontrada.")
        arquivo = (ESTATICO / relativo).resolve()
        if ESTATICO.resolve() not in arquivo.parents or not arquivo.is_file():
            raise ErroHTTP(404, "Arquivo não encontrado.")
        tipo = TIPOS_MIME.get(arquivo.suffix.lower(), "application/octet-stream")
        self._enviar(200, arquivo.read_bytes(), tipo)

    def _baixar_modelo(self, nome: str):
        arquivo = PASTA_MODELOS / nome_seguro(nome, ".json")
        if not arquivo.is_file():
            raise ErroHTTP(404, "Modelo não encontrado.")
        self._enviar(200, arquivo.read_bytes(), TIPOS_MIME[".json"],
                     {"Content-Disposition": f'attachment; filename="{arquivo.name}"'})

    # ---------------------------------------------------------------- #
    def _tratar(self, metodo: str):
        partes = urlsplit(self.path)
        caminho = partes.path.rstrip("/") or "/"
        consulta = {k: v[-1] for k, v in parse_qs(partes.query).items()}
        try:
            if metodo == "GET":
                if caminho == "/api/estado":
                    return self._json(200, SESSAO.estado())
                if caminho == "/api/datasets":
                    return self._json(200, SESSAO.listar_datasets())
                if caminho == "/api/rede":
                    return self._json(200, SESSAO.info_rede(SESSAO.rede_para_leitura()))
                if caminho == "/api/treino/progresso":
                    return self._json(200, SESSAO.obter_progresso(int(consulta.get("desde", 0))))
                if caminho == "/api/modelos":
                    return self._json(200, SESSAO.listar_modelos())
                if caminho == "/api/modelo/baixar":
                    return self._baixar_modelo(consulta.get("nome", ""))
                if caminho.startswith("/api/"):
                    raise ErroHTTP(404, "Rota não encontrada.")
                return self._estatico(caminho)

            corpo = self._ler_corpo()
            rotas = {
                "/api/dataset/abrir": lambda: SESSAO.abrir_dataset(corpo.get("nome", "")),
                "/api/dataset/enviar": lambda: SESSAO.enviar_dataset(corpo.get("nome", ""), corpo.get("conteudo", "")),
                "/api/rede/criar": lambda: SESSAO.criar_rede(corpo),
                "/api/treino/iniciar": lambda: SESSAO.iniciar_treino(corpo),
                "/api/treino/parar": lambda: (SESSAO.parar.set(), {"ok": True})[1],
                "/api/prever": lambda: SESSAO.prever(corpo),
                "/api/visual": lambda: SESSAO.visual(int(corpo.get("res", 44)), corpo.get("x"), corpo.get("y")),
                "/api/modelo/salvar": lambda: SESSAO.salvar_modelo(corpo.get("nome", "")),
                "/api/modelo/carregar": lambda: SESSAO.carregar_modelo(corpo.get("nome", "")),
            }
            if caminho not in rotas:
                raise ErroHTTP(404, "Rota não encontrada.")
            return self._json(200, rotas[caminho]())
        except ErroHTTP as erro:
            self._json(erro.status, {"erro": erro.mensagem})
        except (ValueError, ErroEntrada, FileNotFoundError, KeyError, TypeError) as erro:
            mensagem = str(erro) if not isinstance(erro, KeyError) else f"Campo obrigatório ausente: {erro}"
            self._json(400, {"erro": mensagem})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as erro:  # noqa: BLE001
            self._json(500, {"erro": f"Erro interno: {erro}"})

    def do_GET(self):
        self._tratar("GET")

    def do_POST(self):
        self._tratar("POST")


class Servidor(ThreadingHTTPServer):
    daemon_threads = True
    # No Windows, SO_REUSEADDR deixaria "dividir" a porta com outro programa.
    allow_reuse_address = sys.platform != "win32"

    def handle_error(self, request, client_address):
        excecao = sys.exc_info()[1]
        if isinstance(excecao, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def iniciar_servidor(porta: int = 8000, abrir_navegador: bool = True, host: str = "127.0.0.1") -> None:
    PASTA_MODELOS.mkdir(exist_ok=True)
    servidor = None
    for tentativa in range(porta, porta + 10):
        try:
            servidor = Servidor((host, tentativa), Manipulador)
            break
        except OSError:
            print(f"Porta {tentativa} em uso, tentando a próxima...", flush=True)
    if servidor is None:
        raise RuntimeError(
            f"As portas {porta} a {porta + 9} estão em uso. Tente: python main.py web --porta 9000"
        )
    porta = servidor.server_address[1]
    endereco = f"http://{host}:{porta}"
    print(f"Interface web em {endereco}  (Ctrl+C para encerrar)", flush=True)
    if abrir_navegador:
        threading.Timer(0.6, lambda: webbrowser.open(endereco)).start()
    try:
        servidor.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    iniciar_servidor()
