"""Previsão com a rede treinada (somente propagação direta).

A previsão:
  1. confere se a quantidade de valores bate com o nº de entradas da rede;
  2. aplica a MESMA normalização (tipo e parâmetros) salva no treino;
  3. faz a propagação direta. Pesos e vieses nunca são alterados.
"""

from __future__ import annotations

import math

import numpy as np

from .erro import acuracia, mse


class ErroEntrada(ValueError):
    """Valores de entrada inválidos para a previsão."""


def converter_valores(texto: str) -> list[float]:
    """'1, 0.5, -2' (vírgula, ponto e vírgula ou espaço) -> [1.0, 0.5, -2.0]"""
    partes = [p for p in str(texto).replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    if not partes:
        raise ErroEntrada("Nenhum valor informado.")
    valores = []
    for p in partes:
        try:
            valores.append(float(p))
        except ValueError:
            raise ErroEntrada(f"Valor não numérico: '{p.strip()}'.") from None
    return valores


def _validar(rede, X: np.ndarray) -> None:
    if X.shape[1] != rede.n_entradas:
        raise ErroEntrada(
            f"Quantidade de valores incorreta: a rede espera {rede.n_entradas} "
            f"entrada(s), mas foram informados {X.shape[1]}."
        )
    if not np.all(np.isfinite(X)):
        raise ErroEntrada("Os valores de entrada devem ser números finitos.")


def _normalizar(rede, X: np.ndarray) -> np.ndarray:
    return rede.normalizador.aplicar(X) if rede.normalizador is not None else X


def prever(rede, valores) -> np.ndarray:
    """Previsão de UM exemplo com valores crus. Devolve o vetor de saídas."""
    x = np.asarray(valores, dtype=float)
    if x.ndim != 1:
        raise ErroEntrada("Informe um único vetor de valores para a previsão.")
    X = x.reshape(1, -1)
    _validar(rede, X)
    return rede.propagar(_normalizar(rede, X)[0])


def prever_lote(rede, X) -> np.ndarray:
    """Previsão de vários exemplos (matriz N x n_entradas) com valores crus."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    _validar(rede, X)
    return rede.propagar_lote(_normalizar(rede, X))


def avaliar(rede, X, Y) -> dict:
    """MSE (e acurácia, se for classificação) sobre dados crus."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(len(X), -1)
    if len(X) == 0:
        return {"mse": math.nan, "acuracia": None, "n": 0}
    saidas = prever_lote(rede, X)
    return {"mse": mse(saidas, Y), "acuracia": acuracia(saidas, Y), "n": len(X)}
