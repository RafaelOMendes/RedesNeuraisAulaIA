"""Funções de ativação e suas derivadas.

Cada função de ativação é uma classe com dois métodos:

* ``f(z)``         -> valor da ativação em z
* ``derivada(z)``  -> f'(z), usada na retropropagação

Os métodos aceitam um número (float) ou um array numpy. Quando recebem um
número, devolvem um ``float`` comum.
"""

from __future__ import annotations

import numpy as np


def _como_saida(valor):
    """Devolve float para entradas escalares e array para entradas vetoriais."""
    if np.ndim(valor) == 0:
        return float(valor)
    return valor


class Ativacao:
    """Classe base. As subclasses definem ``nome``, ``f`` e ``derivada``."""

    nome = "base"

    def f(self, z):
        raise NotImplementedError

    def derivada(self, z):
        raise NotImplementedError

    def __call__(self, z):
        return self.f(z)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Linear(Ativacao):
    """f(z) = z            f'(z) = 1"""

    nome = "linear"

    def f(self, z):
        return _como_saida(np.asarray(z, dtype=float).copy())

    def derivada(self, z):
        return _como_saida(np.ones_like(np.asarray(z, dtype=float)))


class Sigmoid(Ativacao):
    """f(z) = 1 / (1 + e^-z)            f'(z) = f(z) * (1 - f(z))

    O z é limitado a [-500, 500] antes da exponencial para evitar overflow
    (e^500 ainda cabe em float64 e o resultado já está saturado em 0 ou 1).
    """

    nome = "sigmoid"

    def f(self, z):
        z = np.clip(np.asarray(z, dtype=float), -500.0, 500.0)
        return _como_saida(1.0 / (1.0 + np.exp(-z)))

    def derivada(self, z):
        s = np.asarray(self.f(z), dtype=float)
        return _como_saida(s * (1.0 - s))


class Tanh(Ativacao):
    """f(z) = tanh(z)            f'(z) = 1 - f(z)^2"""

    nome = "tanh"

    def f(self, z):
        return _como_saida(np.tanh(np.asarray(z, dtype=float)))

    def derivada(self, z):
        t = np.tanh(np.asarray(z, dtype=float))
        return _como_saida(1.0 - t * t)


class ReLU(Ativacao):
    """f(z) = max(0, z)            f'(z) = 1 se z > 0, senão 0"""

    nome = "relu"

    def f(self, z):
        return _como_saida(np.maximum(0.0, np.asarray(z, dtype=float)))

    def derivada(self, z):
        return _como_saida((np.asarray(z, dtype=float) > 0.0).astype(float))


ATIVACOES: dict[str, type[Ativacao]] = {
    "linear": Linear,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
}


def nomes_disponiveis() -> list[str]:
    return list(ATIVACOES)


def obter_ativacao(ativacao: str | Ativacao) -> Ativacao:
    """Converte um nome ("linear", "sigmoid", "tanh", "relu") em objeto Ativacao.

    Se já receber um objeto Ativacao, devolve o próprio objeto.
    """
    if isinstance(ativacao, Ativacao):
        return ativacao
    nome = str(ativacao).strip().lower()
    if nome not in ATIVACOES:
        raise ValueError(
            f"Função de ativação desconhecida: '{ativacao}'. "
            f"Opções válidas: {', '.join(ATIVACOES)}."
        )
    return ATIVACOES[nome]()
