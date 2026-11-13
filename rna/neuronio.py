"""Entidade Neurônio."""

from __future__ import annotations

import numpy as np

from .ativacoes import Ativacao, obter_ativacao


class Neuronio:
    """Um neurônio artificial.

    Guarda:
      * ``pesos``  - pesos sinápticos w (um por entrada)
      * ``bias``   - viés b
      * ``ativacao`` - função de ativação f

    E os valores intermediários do último exemplo processado (usados no treino):
      * ``entradas`` - vetor x recebido na propagação direta
      * ``z``        - soma ponderada z = w·x + b
      * ``saida``    - a = f(z)
      * ``delta``    - δ = ∂E/∂z, calculado na retropropagação
    """

    def __init__(self, pesos, bias: float = 0.0, ativacao: str | Ativacao = "sigmoid"):
        self.pesos = np.array(pesos, dtype=float)
        if self.pesos.ndim != 1 or self.pesos.size == 0:
            raise ValueError("Os pesos de um neurônio devem ser um vetor com pelo menos 1 elemento.")
        self.bias = float(bias)
        self.ativacao = obter_ativacao(ativacao)

        self.entradas: np.ndarray | None = None
        self.z: float = 0.0
        self.saida: float = 0.0
        self.delta: float = 0.0

    @property
    def n_entradas(self) -> int:
        return self.pesos.size

    def soma_ponderada(self, x) -> float:
        """z = w·x + b"""
        return float(np.dot(self.pesos, x) + self.bias)

    def ativar(self, x) -> float:
        """Propagação direta do neurônio: guarda x e z e devolve a = f(z)."""
        x = np.asarray(x, dtype=float)
        if x.shape != self.pesos.shape:
            raise ValueError(
                f"O neurônio espera {self.n_entradas} entradas, mas recebeu {x.size}."
            )
        self.entradas = x
        self.z = self.soma_ponderada(x)
        self.saida = float(self.ativacao.f(self.z))
        return self.saida

    def derivada_ativacao(self) -> float:
        """f'(z) avaliada no z do último exemplo propagado."""
        return float(self.ativacao.derivada(self.z))

    def gradientes(self) -> tuple[np.ndarray, float]:
        """Gradientes do erro em relação aos pesos e ao viés.

        ∂E/∂w_i = δ · x_i       (x = entrada que o neurônio recebeu no forward)
        ∂E/∂b   = δ
        """
        if self.entradas is None:
            raise RuntimeError("Faça a propagação direta antes de calcular os gradientes.")
        return self.delta * self.entradas, self.delta

    def atualizar(self, taxa_aprendizado: float) -> None:
        """Descida do gradiente: w <- w - η·δ·x   e   b <- b - η·δ"""
        grad_pesos, grad_bias = self.gradientes()
        self.pesos -= taxa_aprendizado * grad_pesos
        self.bias -= taxa_aprendizado * grad_bias

    def __repr__(self) -> str:
        return (
            f"Neuronio(n_entradas={self.n_entradas}, ativacao={self.ativacao.nome}, "
            f"bias={self.bias:.4f})"
        )
