"""Inicialização de pesos e vieses.

* Xavier/Glorot (uniforme) para Linear, Sigmoid e Tanh:
      w ~ U(-L, +L),  L = sqrt(6 / (fan_in + fan_out))
* He (normal) para ReLU:
      w ~ N(0, desvio),  desvio = sqrt(2 / fan_in)

fan_in  = número de entradas do neurônio (neurônios da camada anterior)
fan_out = número de neurônios da camada

Os vieses começam em zero. Eles ficam guardados junto com os pesos (em cada
neurônio e no arquivo JSON da rede).

Reprodutibilidade: todo sorteio usa um ``numpy.random.Generator`` criado com a
semente (seed) informada. Mesma seed = mesmos pesos iniciais.
"""

from __future__ import annotations

import numpy as np

from .ativacoes import Ativacao, obter_ativacao


def criar_gerador(seed: int | None = None) -> np.random.Generator:
    """Cria um gerador de números aleatórios. ``seed=None`` = sorteio não reprodutível."""
    return np.random.default_rng(seed)


def xavier(n_entradas: int, n_neuronios: int, gerador: np.random.Generator) -> np.ndarray:
    """Matriz [n_neuronios x n_entradas] com distribuição Xavier/Glorot uniforme."""
    limite = np.sqrt(6.0 / (n_entradas + n_neuronios))
    return gerador.uniform(-limite, limite, size=(n_neuronios, n_entradas))


def he(n_entradas: int, n_neuronios: int, gerador: np.random.Generator) -> np.ndarray:
    """Matriz [n_neuronios x n_entradas] com distribuição He (normal)."""
    desvio = np.sqrt(2.0 / n_entradas)
    return gerador.normal(0.0, desvio, size=(n_neuronios, n_entradas))


def metodo_para(ativacao: str | Ativacao) -> str:
    """Nome do método de inicialização adequado à função de ativação."""
    return "he" if obter_ativacao(ativacao).nome == "relu" else "xavier"


def inicializar(
    ativacao: str | Ativacao,
    n_entradas: int,
    n_neuronios: int,
    gerador: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera (pesos, vieses) para uma camada.

    pesos:  matriz [n_neuronios x n_entradas] (linha i = pesos do neurônio i)
    vieses: vetor [n_neuronios], inicializado com zeros
    """
    if metodo_para(ativacao) == "he":
        pesos = he(n_entradas, n_neuronios, gerador)
    else:
        pesos = xavier(n_entradas, n_neuronios, gerador)
    vieses = np.zeros(n_neuronios, dtype=float)
    return pesos, vieses
