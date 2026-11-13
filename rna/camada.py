"""Entidade Camada."""

from __future__ import annotations

import numpy as np

from .ativacoes import Ativacao, obter_ativacao
from .inicializacao import criar_gerador, inicializar, metodo_para
from .neuronio import Neuronio


class Camada:
    """Conjunto de neurônios que compartilham a mesma função de ativação.

    Cada neurônio recebe TODAS as saídas da camada anterior (camada densa).
    A camada recebe as entradas da camada anterior em ``propagar`` e devolve
    o vetor de saídas, que vira a entrada da camada seguinte.
    """

    def __init__(
        self,
        n_entradas: int,
        n_neuronios: int,
        ativacao: str | Ativacao,
        gerador: np.random.Generator | None = None,
        pesos=None,
        vieses=None,
    ):
        """Cria a camada.

        Se ``pesos`` e ``vieses`` não forem informados, eles são sorteados
        (Xavier ou He, conforme a ativação) usando ``gerador``.
        Se forem informados (ex.: ao carregar do JSON), são usados como estão.
        """
        if int(n_entradas) < 1:
            raise ValueError("Uma camada precisa de pelo menos 1 entrada.")
        if int(n_neuronios) < 1:
            raise ValueError("Uma camada precisa de pelo menos 1 neurônio.")
        self.n_entradas = int(n_entradas)
        self.ativacao = obter_ativacao(ativacao)

        if pesos is None:
            pesos, vieses_iniciais = inicializar(
                self.ativacao, self.n_entradas, int(n_neuronios), gerador or criar_gerador()
            )
            if vieses is None:
                vieses = vieses_iniciais
        pesos = np.asarray(pesos, dtype=float)
        vieses = np.zeros(int(n_neuronios)) if vieses is None else np.asarray(vieses, dtype=float)

        if pesos.shape != (int(n_neuronios), self.n_entradas):
            raise ValueError(
                f"Formato dos pesos inválido: esperado ({n_neuronios}, {self.n_entradas}), "
                f"recebido {pesos.shape}."
            )
        if vieses.shape != (int(n_neuronios),):
            raise ValueError(
                f"Formato dos vieses inválido: esperado ({n_neuronios},), recebido {vieses.shape}."
            )

        self.neuronios = [
            Neuronio(pesos[i], vieses[i], self.ativacao) for i in range(int(n_neuronios))
        ]
        self.entradas: np.ndarray | None = None
        self.saidas: np.ndarray | None = None

    @property
    def n_neuronios(self) -> int:
        return len(self.neuronios)

    @property
    def metodo_inicializacao(self) -> str:
        return metodo_para(self.ativacao)

    def propagar(self, entradas) -> np.ndarray:
        """Propagação direta da camada: devolve o vetor [a_1, ..., a_n]."""
        x = np.array(entradas, dtype=float)  # cópia: os neurônios guardam esta entrada
        if x.shape != (self.n_entradas,):
            raise ValueError(
                f"A camada espera {self.n_entradas} entradas, mas recebeu {x.size}."
            )
        self.entradas = x
        self.saidas = np.array([neuronio.ativar(x) for neuronio in self.neuronios])
        return self.saidas

    def pesos(self) -> np.ndarray:
        """Matriz [n_neuronios x n_entradas] (cópia)."""
        return np.array([n.pesos for n in self.neuronios])

    def vieses(self) -> np.ndarray:
        """Vetor [n_neuronios] (cópia)."""
        return np.array([n.bias for n in self.neuronios])

    def deltas(self) -> np.ndarray:
        return np.array([n.delta for n in self.neuronios])

    def __repr__(self) -> str:
        return (
            f"Camada(n_entradas={self.n_entradas}, n_neuronios={self.n_neuronios}, "
            f"ativacao={self.ativacao.nome})"
        )
