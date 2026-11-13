"""Entidade Rede Neural (MLP)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .camada import Camada
from .inicializacao import criar_gerador


def _normalizar_especificacao(ocultas) -> list[tuple[int, str]]:
    """Aceita [(4, "tanh"), ...] ou [{"neuronios": 4, "ativacao": "tanh"}, ...]."""
    resultado = []
    for item in ocultas or []:
        if isinstance(item, dict):
            n, ativ = item.get("neuronios"), item.get("ativacao")
        else:
            n, ativ = item
        resultado.append((int(n), str(ativ)))
    return resultado


class RedeNeural:
    """Multilayer Perceptron totalmente configurável.

    Exemplo:
        RedeNeural(n_entradas=2, ocultas=[(4, "tanh")], n_saidas=1,
                   ativacao_saida="sigmoid", seed=42)

    cria a arquitetura 2 -> 4 (tanh) -> 1 (sigmoid). ``ocultas`` pode ter
    qualquer número de camadas, inclusive nenhuma.
    """

    def __init__(
        self,
        n_entradas: int,
        ocultas: Iterable | None = None,
        n_saidas: int = 1,
        ativacao_saida: str = "sigmoid",
        seed: int | None = None,
    ):
        if int(n_entradas) < 1:
            raise ValueError("A rede precisa de pelo menos 1 entrada.")
        if int(n_saidas) < 1:
            raise ValueError("A rede precisa de pelo menos 1 neurônio de saída.")

        self.n_entradas = int(n_entradas)
        self.seed = seed
        gerador = criar_gerador(seed)

        especificacao = _normalizar_especificacao(ocultas) + [(int(n_saidas), ativacao_saida)]
        self.camadas: list[Camada] = []
        n_anterior = self.n_entradas
        for n_neuronios, ativacao in especificacao:
            camada = Camada(n_anterior, n_neuronios, ativacao, gerador)
            self.camadas.append(camada)
            n_anterior = n_neuronios

        self.normalizador = None   # definido no treino (dataset.Normalizador)
        self.metadados: dict = {}  # colunas usadas, configuração do treino etc.

    @classmethod
    def de_camadas(cls, n_entradas: int, camadas: list[Camada], seed=None) -> "RedeNeural":
        """Monta a rede a partir de camadas já prontas (usado ao carregar do JSON)."""
        if not camadas:
            raise ValueError("A rede precisa de pelo menos a camada de saída.")
        rede = cls.__new__(cls)
        rede.n_entradas = int(n_entradas)
        rede.seed = seed
        rede.camadas = list(camadas)
        rede.normalizador = None
        rede.metadados = {}
        n_anterior = rede.n_entradas
        for i, camada in enumerate(rede.camadas):
            if camada.n_entradas != n_anterior:
                raise ValueError(
                    f"Camada {i + 1} espera {camada.n_entradas} entradas, "
                    f"mas a camada anterior produz {n_anterior}."
                )
            n_anterior = camada.n_neuronios
        return rede

    # ------------------------------------------------------------------ #
    # Informações sobre a arquitetura
    # ------------------------------------------------------------------ #
    @property
    def n_saidas(self) -> int:
        return self.camadas[-1].n_neuronios

    @property
    def camadas_ocultas(self) -> list[Camada]:
        return self.camadas[:-1]

    @property
    def camada_saida(self) -> Camada:
        return self.camadas[-1]

    @property
    def n_camadas_ocultas(self) -> int:
        return len(self.camadas) - 1

    def arquitetura(self) -> list[int]:
        """Ex.: [2, 4, 1] = 2 entradas, 4 neurônios ocultos, 1 saída."""
        return [self.n_entradas] + [c.n_neuronios for c in self.camadas]

    def n_parametros(self) -> int:
        return sum(c.n_neuronios * (c.n_entradas + 1) for c in self.camadas)

    # ------------------------------------------------------------------ #
    # Propagação direta (forward)
    # ------------------------------------------------------------------ #
    def propagar(self, x) -> np.ndarray:
        """Propaga um exemplo por todas as camadas e devolve o vetor de saída.

        ``x`` deve estar na mesma escala usada no treino (já normalizado,
        se houver normalização). Para valores "crus", use ``previsao.prever``.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_entradas,):
            raise ValueError(
                f"A rede espera {self.n_entradas} entradas, mas recebeu {x.size}."
            )
        saida = x
        for camada in self.camadas:
            saida = camada.propagar(saida)
        return saida.copy()

    def propagar_lote(self, X) -> np.ndarray:
        """Aplica ``propagar`` a cada linha de X. Devolve matriz [N x n_saidas]."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1) if X.size else X.reshape(0, self.n_entradas)
        if X.shape[0] == 0:
            return np.empty((0, self.n_saidas))
        return np.array([self.propagar(x) for x in X])

    # ------------------------------------------------------------------ #
    # Treino e execução (delegam para os módulos de treino e previsão)
    # ------------------------------------------------------------------ #
    def treinar(
        self,
        X_treino,
        Y_treino,
        X_validacao=None,
        Y_validacao=None,
        taxa_aprendizado: float = 0.1,
        epocas: int = 100,
        seed: int | None = None,
        embaralhar: bool = True,
        ao_fim_epoca=None,
    ):
        """Treina com SGD + backpropagation (ver ``treino.Treinador``)."""
        from .treino import Treinador  # import local evita importação circular

        treinador = Treinador(self, taxa_aprendizado, epocas, seed, embaralhar)
        return treinador.treinar(X_treino, Y_treino, X_validacao, Y_validacao, ao_fim_epoca)

    def executar(self, valores) -> np.ndarray:
        """Previsão a partir de valores crus (aplica a normalização salva)."""
        from .previsao import prever

        return prever(self, valores)

    # ------------------------------------------------------------------ #
    def resumo(self) -> str:
        linhas = [
            f"Arquitetura: {' -> '.join(str(n) for n in self.arquitetura())}",
            f"Entradas: {self.n_entradas}",
        ]
        for i, camada in enumerate(self.camadas, start=1):
            tipo = "saída " if camada is self.camada_saida else "oculta"
            linhas.append(
                f"  Camada {i} ({tipo}): {camada.n_neuronios:>3} neurônios | "
                f"ativação {camada.ativacao.nome:<7} | "
                f"init {camada.metodo_inicializacao:<6} | "
                f"{camada.n_entradas} entradas por neurônio"
            )
        linhas.append(f"Camadas ocultas: {self.n_camadas_ocultas}")
        linhas.append(f"Total de parâmetros (pesos + vieses): {self.n_parametros()}")
        tipo_norm = self.normalizador.tipo if self.normalizador is not None else "nenhuma"
        linhas.append(f"Normalização das entradas: {tipo_norm}")
        if self.seed is not None:
            linhas.append(f"Seed da inicialização: {self.seed}")
        return "\n".join(linhas)

    def __repr__(self) -> str:
        ativs = ", ".join(c.ativacao.nome for c in self.camadas)
        return f"RedeNeural(arquitetura={self.arquitetura()}, ativacoes=[{ativs}])"
