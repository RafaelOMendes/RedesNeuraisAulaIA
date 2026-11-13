"""Leitura de CSV, escolha de colunas, divisão e normalização dos dados.

Ordem obrigatória do pré-processamento:
  1. dividir em treino / validação / teste (aleatório, com seed);
  2. calcular os parâmetros da normalização SÓ no treino;
  3. aplicar esses mesmos parâmetros aos três subconjuntos.
As saídas esperadas (Y) nunca são normalizadas.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .inicializacao import criar_gerador


# ====================================================================== #
# Leitura do CSV
# ====================================================================== #
def ler_csv(caminho: str | Path, separador: str = ",") -> tuple[list[str], np.ndarray]:
    """Lê um CSV numérico com cabeçalho. Devolve (nomes_das_colunas, matriz)."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho}")

    with open(caminho, encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.reader(arquivo, delimiter=separador)
        try:
            cabecalho = [c.strip() for c in next(leitor)]
        except StopIteration:
            raise ValueError(f"O arquivo {caminho} está vazio.") from None

        if len(set(cabecalho)) != len(cabecalho):
            raise ValueError(f"O cabeçalho de {caminho} tem colunas repetidas: {cabecalho}")

        linhas = []
        for n_linha, registro in enumerate(leitor, start=2):
            if not registro or all(not c.strip() for c in registro):
                continue  # ignora linhas em branco
            if len(registro) != len(cabecalho):
                raise ValueError(
                    f"Linha {n_linha} de {caminho.name} tem {len(registro)} valores, "
                    f"mas o cabeçalho tem {len(cabecalho)} colunas."
                )
            valores = []
            for nome, texto in zip(cabecalho, registro):
                try:
                    valores.append(float(texto))
                except ValueError:
                    raise ValueError(
                        f"Valor não numérico na linha {n_linha}, coluna '{nome}': '{texto}'."
                    ) from None
            linhas.append(valores)

    if not linhas:
        raise ValueError(f"O arquivo {caminho} não tem linhas de dados.")
    return cabecalho, np.array(linhas, dtype=float)


def separar_lista(texto: str | list | tuple | None) -> list[str]:
    """'x1, x2,x3' -> ['x1', 'x2', 'x3']"""
    if texto is None:
        return []
    if isinstance(texto, (list, tuple)):
        return [str(t).strip() for t in texto if str(t).strip()]
    return [t.strip() for t in str(texto).split(",") if t.strip()]


class Dataset:
    """Dados de um CSV + escolha das colunas de entrada e de saída."""

    def __init__(self, colunas: list[str], dados: np.ndarray, caminho: str | None = None):
        self.colunas = list(colunas)
        self.dados = np.asarray(dados, dtype=float)
        self.caminho = caminho

    @classmethod
    def de_csv(cls, caminho: str | Path, separador: str = ",") -> "Dataset":
        colunas, dados = ler_csv(caminho, separador)
        return cls(colunas, dados, str(caminho))

    @property
    def n_linhas(self) -> int:
        return self.dados.shape[0]

    @property
    def n_colunas(self) -> int:
        return self.dados.shape[1]

    def indice_coluna(self, referencia) -> int:
        """Aceita o nome da coluna ou o índice (0, 1, 2...)."""
        ref = str(referencia).strip()
        if ref in self.colunas:
            return self.colunas.index(ref)
        if ref.lstrip("-").isdigit():
            i = int(ref)
            if 0 <= i < self.n_colunas:
                return i
            raise ValueError(f"Índice de coluna fora do intervalo: {i} (há {self.n_colunas} colunas).")
        raise ValueError(
            f"Coluna '{ref}' não existe. Colunas disponíveis: {', '.join(self.colunas)}."
        )

    def selecionar(self, entradas, saidas) -> tuple[np.ndarray, np.ndarray]:
        """Separa as colunas de entrada (X) e de saída esperada (Y)."""
        refs_x, refs_y = separar_lista(entradas), separar_lista(saidas)
        if not refs_x:
            raise ValueError("Informe pelo menos uma coluna de entrada.")
        if not refs_y:
            raise ValueError("Informe pelo menos uma coluna de saída.")
        idx_x = [self.indice_coluna(r) for r in refs_x]
        idx_y = [self.indice_coluna(r) for r in refs_y]
        repetidas = set(idx_x) & set(idx_y)
        if repetidas:
            nomes = ", ".join(self.colunas[i] for i in sorted(repetidas))
            raise ValueError(f"Coluna(s) usadas ao mesmo tempo como entrada e saída: {nomes}.")
        return self.dados[:, idx_x].copy(), self.dados[:, idx_y].copy()

    def nomes(self, referencias) -> list[str]:
        return [self.colunas[self.indice_coluna(r)] for r in separar_lista(referencias)]

    def resumo(self) -> str:
        linhas = [f"{self.n_linhas} linhas x {self.n_colunas} colunas"]
        linhas.append(f"  {'#':>2}  {'coluna':<16}{'mín':>12}{'máx':>12}{'média':>12}")
        for i, nome in enumerate(self.colunas):
            col = self.dados[:, i]
            linhas.append(
                f"  {i:>2}  {nome:<16}{col.min():>12.4f}{col.max():>12.4f}{col.mean():>12.4f}"
            )
        return "\n".join(linhas)


# ====================================================================== #
# Compatibilidade dados x arquitetura
# ====================================================================== #
def verificar_compatibilidade(rede, X, Y) -> list[str]:
    """Erro se o nº de colunas não bate com a rede. Devolve avisos (não fatais)."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    n_x = X.shape[1] if X.ndim == 2 else 1
    n_y = Y.shape[1] if Y.ndim == 2 else 1
    if n_x != rede.n_entradas:
        raise ValueError(
            f"Incompatível: os dados têm {n_x} coluna(s) de entrada, "
            f"mas a rede espera {rede.n_entradas} entrada(s)."
        )
    if n_y != rede.n_saidas:
        raise ValueError(
            f"Incompatível: os dados têm {n_y} coluna(s) de saída, "
            f"mas a rede tem {rede.n_saidas} neurônio(s) de saída."
        )

    avisos = []
    if Y.size:
        ativ = rede.camada_saida.ativacao.nome
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
        faixas = {"sigmoid": (0.0, 1.0), "tanh": (-1.0, 1.0), "relu": (0.0, math.inf)}
        if ativ in faixas:
            lo, hi = faixas[ativ]
            if y_min < lo or y_max > hi:
                avisos.append(
                    f"A saída usa '{ativ}', que só produz valores em [{lo}, {hi}], mas as "
                    f"saídas esperadas vão de {y_min:g} a {y_max:g}. Considere 'linear'."
                )
    return avisos


# ====================================================================== #
# Divisão treino / validação / teste
# ====================================================================== #
@dataclass
class Divisao:
    X_treino: np.ndarray
    Y_treino: np.ndarray
    X_validacao: np.ndarray
    Y_validacao: np.ndarray
    X_teste: np.ndarray
    Y_teste: np.ndarray
    indices_treino: np.ndarray
    indices_validacao: np.ndarray
    indices_teste: np.ndarray
    percentuais: tuple[float, float, float]
    seed: int | None

    def tamanhos(self) -> tuple[int, int, int]:
        return len(self.X_treino), len(self.X_validacao), len(self.X_teste)

    def resumo(self) -> str:
        t, v, s = self.tamanhos()
        p = self.percentuais
        return (
            f"Treino: {t} | Validação: {v} | Teste: {s} "
            f"(percentuais {p[0]:g}/{p[1]:g}/{p[2]:g}, seed={self.seed})"
        )


def interpretar_percentuais(percentuais) -> tuple[float, float, float]:
    """Aceita '70,15,15' ou (70, 15, 15). Os três valores precisam somar 100."""
    if isinstance(percentuais, str):
        partes = separar_lista(percentuais)
        try:
            valores = [float(p) for p in partes]
        except ValueError:
            raise ValueError(f"Percentuais inválidos: '{percentuais}'. Use, por exemplo, 70,15,15.") from None
    else:
        valores = [float(p) for p in percentuais]
    if len(valores) != 3:
        raise ValueError("Informe exatamente 3 percentuais: treino, validação e teste (ex.: 70,15,15).")
    if any(v < 0 for v in valores):
        raise ValueError("Os percentuais não podem ser negativos.")
    if abs(sum(valores) - 100.0) > 1e-9:
        raise ValueError(f"Os percentuais devem somar 100%, mas somam {sum(valores):g}%.")
    if valores[0] <= 0:
        raise ValueError("O conjunto de treino precisa ter percentual maior que zero.")
    return valores[0], valores[1], valores[2]


def _quantidades(n: int, percentuais: tuple[float, float, float]) -> list[int]:
    """Converte percentuais em quantidades que somam exatamente n.

    Usa o método do maior resto: arredonda para baixo e distribui as linhas que
    sobraram para os conjuntos com maior parte fracionária. Conjuntos com 0%
    nunca recebem linhas.
    """
    exatos = [n * p / 100.0 for p in percentuais]
    qtd = [math.floor(e) for e in exatos]
    sobra = n - sum(qtd)
    ordem = sorted(range(3), key=lambda i: exatos[i] - qtd[i], reverse=True)
    for i in ordem:
        if sobra == 0:
            break
        if percentuais[i] > 0:
            qtd[i] += 1
            sobra -= 1
    return qtd


def dividir(X, Y, percentuais=(70, 15, 15), seed: int | None = None) -> Divisao:
    """Embaralha as linhas (com a seed) e divide em treino, validação e teste."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if len(X) != len(Y):
        raise ValueError("X e Y precisam ter o mesmo número de linhas.")
    p = interpretar_percentuais(percentuais)
    n = len(X)
    n_treino, n_val, _ = _quantidades(n, p)
    if n_treino < 1:
        raise ValueError("A divisão deixou o conjunto de treino vazio. Aumente o percentual de treino.")

    # Gerador próprio para a divisão: não depende da arquitetura da rede.
    indices = criar_gerador(seed).permutation(n)
    i_tr = indices[:n_treino]
    i_va = indices[n_treino:n_treino + n_val]
    i_te = indices[n_treino + n_val:]
    return Divisao(
        X[i_tr], Y[i_tr], X[i_va], Y[i_va], X[i_te], Y[i_te], i_tr, i_va, i_te, p, seed
    )


# ====================================================================== #
# Normalização
# ====================================================================== #
_ALIASES = {
    None: "nenhuma", "": "nenhuma", "nenhuma": "nenhuma", "none": "nenhuma", "sem": "nenhuma",
    "minmax": "minmax", "min-max": "minmax", "min_max": "minmax",
    "zscore": "zscore", "z-score": "zscore", "z_score": "zscore",
}


class Normalizador:
    """Normalização das entradas: 'nenhuma', 'minmax' ou 'zscore'.

    * minmax: x' = (x - mín) / (máx - mín)        -> faixa [0, 1] no treino
    * zscore: x' = (x - média) / desvio_padrão     (desvio populacional, ddof=0)

    Os parâmetros são calculados em ``ajustar`` (só com o treino) e depois
    reaplicados, sem mudança, em validação, teste e previsões.
    Colunas constantes (amplitude ou desvio zero) usam divisor 1.
    """

    TIPOS = ("nenhuma", "minmax", "zscore")

    def __init__(self, tipo: str | None = "nenhuma"):
        chave = tipo.strip().lower() if isinstance(tipo, str) else tipo
        if chave not in _ALIASES:
            raise ValueError(
                f"Normalização desconhecida: '{tipo}'. Opções: nenhuma, minmax, zscore."
            )
        self.tipo = _ALIASES[chave]
        self.parametros: dict[str, np.ndarray] = {}
        self.ajustado = self.tipo == "nenhuma"

    def ajustar(self, X_treino) -> "Normalizador":
        X = np.asarray(X_treino, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("A normalização precisa de uma matriz de treino não vazia.")
        if self.tipo == "minmax":
            self.parametros = {"minimo": X.min(axis=0), "maximo": X.max(axis=0)}
        elif self.tipo == "zscore":
            self.parametros = {"media": X.mean(axis=0), "desvio": X.std(axis=0)}
        self.ajustado = True
        return self

    def aplicar(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.tipo == "nenhuma":
            return X.copy()
        if not self.ajustado:
            raise RuntimeError("Normalizador ainda não foi ajustado com os dados de treino.")
        if self.tipo == "minmax":
            minimo, maximo = self.parametros["minimo"], self.parametros["maximo"]
            amplitude = np.where(maximo - minimo == 0, 1.0, maximo - minimo)
            return (X - minimo) / amplitude
        media, desvio = self.parametros["media"], self.parametros["desvio"]
        return (X - media) / np.where(desvio == 0, 1.0, desvio)

    def ajustar_aplicar(self, X_treino) -> np.ndarray:
        return self.ajustar(X_treino).aplicar(X_treino)

    def para_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "parametros": {k: [float(v) for v in vetor] for k, vetor in self.parametros.items()},
        }

    @classmethod
    def de_dict(cls, dados: dict | None) -> "Normalizador":
        dados = dados or {}
        norm = cls(dados.get("tipo", "nenhuma"))
        norm.parametros = {
            k: np.array(v, dtype=float) for k, v in (dados.get("parametros") or {}).items()
        }
        esperados = {"minmax": {"minimo", "maximo"}, "zscore": {"media", "desvio"}}.get(norm.tipo, set())
        if not esperados <= set(norm.parametros):
            raise ValueError(f"Parâmetros da normalização '{norm.tipo}' incompletos no arquivo.")
        norm.ajustado = True
        return norm

    def resumo(self) -> str:
        if self.tipo == "nenhuma":
            return "nenhuma"
        partes = [
            f"{k}=[{', '.join(f'{v:.4g}' for v in vetor)}]" for k, vetor in self.parametros.items()
        ]
        return f"{self.tipo} ({'; '.join(partes)})"


def normalizar_divisao(divisao: Divisao, tipo: str | None) -> tuple[Divisao, Normalizador]:
    """Ajusta a normalização SÓ no treino e aplica nos três conjuntos.

    Devolve uma nova Divisao (com X normalizados e Y intocados) e o Normalizador.
    """
    norm = Normalizador(tipo)
    norm.ajustar(divisao.X_treino)
    nova = Divisao(
        norm.aplicar(divisao.X_treino), divisao.Y_treino.copy(),
        norm.aplicar(divisao.X_validacao), divisao.Y_validacao.copy(),
        norm.aplicar(divisao.X_teste), divisao.Y_teste.copy(),
        divisao.indices_treino, divisao.indices_validacao, divisao.indices_teste,
        divisao.percentuais, divisao.seed,
    )
    return nova, norm


def preparar_dados(X, Y, percentuais=(70, 15, 15), normalizacao="nenhuma", seed=None):
    """Atalho: primeiro divide, depois normaliza. Devolve (divisao_normalizada, normalizador)."""
    return normalizar_divisao(dividir(X, Y, percentuais, seed), normalizacao)
