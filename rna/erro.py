"""Funções de erro.

Convenção adotada (documentada no README):

* Custo por exemplo, usado para derivar o gradiente:
      E = 1/2 * Σ_k (y_k - a_k)^2
  Com o fator 1/2, a derivada é ∂E/∂a_k = (a_k - y_k), então
      δ_saida = (a - y) · f'(z)      (sem o fator 2)

* Métrica exibida para treino, validação e teste (MSE):
      MSE = 1/(N·m) * Σ_exemplos Σ_saidas (y - a)^2
  (média sobre todos os exemplos e todas as saídas, sem o fator 1/2)
"""

from __future__ import annotations

import math

import numpy as np


def erro_exemplo(saida, esperada) -> float:
    """E = 1/2 Σ (y - a)^2 de um único exemplo (função de custo do gradiente)."""
    a = np.asarray(saida, dtype=float)
    y = np.asarray(esperada, dtype=float)
    return float(0.5 * np.sum((y - a) ** 2))


def mse(saidas, esperadas) -> float:
    """Erro quadrático médio entre saídas da rede e valores esperados.

    Devolve NaN quando o conjunto está vazio (ex.: divisão 100/0/0).
    """
    a = np.asarray(saidas, dtype=float)
    y = np.asarray(esperadas, dtype=float)
    if a.shape != y.shape:
        raise ValueError(f"Formatos diferentes: saídas {a.shape} x esperadas {y.shape}.")
    if a.size == 0:
        return math.nan
    return float(np.mean((y - a) ** 2))


def mse_rede(rede, X, Y) -> float:
    """MSE da rede sobre um conjunto (X já na escala usada no treino).

    Só faz propagação direta: nunca altera pesos.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if len(X) == 0:
        return math.nan
    return mse(rede.propagar_lote(X), Y.reshape(len(Y), -1))


def acuracia(saidas, esperadas) -> float | None:
    """Acurácia para problemas de classificação (métrica extra, opcional).

    * 1 saída  com alvos 0/1  -> classe = (saída >= 0.5)
    * k saídas com alvos one-hot -> classe = índice da maior saída
    Devolve None se os alvos não parecem de classificação.
    """
    a = np.asarray(saidas, dtype=float)
    y = np.asarray(esperadas, dtype=float)
    if a.size == 0 or not np.all(np.isin(y, (0.0, 1.0))):
        return None
    if y.ndim == 1 or y.shape[1] == 1:
        return float(np.mean((a.reshape(-1) >= 0.5) == (y.reshape(-1) >= 0.5)))
    if not np.all(y.sum(axis=1) == 1):
        return None
    return float(np.mean(np.argmax(a, axis=1) == np.argmax(y, axis=1)))
