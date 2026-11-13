"""Treinamento: retropropagação (backpropagation) + SGD puro + épocas.

Para cada exemplo de treino (SGD puro = atualização após CADA exemplo):

  1. Propagação direta: cada neurônio guarda x, z e a.
  2. Cálculo de TODOS os deltas, da saída para a entrada, usando os pesos
     atuais (os mesmos da propagação direta deste exemplo):
        δ_saida_k  = (a_k - y_k) · f'(z_k)
        δ_oculta_j = f'(z_j) · Σ_k (w_kj · δ_k)      (k = neurônios da camada seguinte)
  3. Só depois, atualização de pesos e vieses por descida do gradiente:
        w <- w - η · δ · entrada_anterior
        b <- b - η · δ

Os passos 2 e 3 são funções separadas, então nenhum peso muda enquanto algum
delta ainda está sendo calculado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .dataset import verificar_compatibilidade
from .erro import mse_rede
from .inicializacao import criar_gerador


# ====================================================================== #
# Backpropagation (funções independentes, usadas pelo Treinador e pelos testes)
# ====================================================================== #
def retropropagar(rede, y) -> None:
    """Calcula e guarda o delta de todos os neurônios para o último exemplo propagado.

    Pré-condição: ``rede.propagar(x)`` acabou de ser chamado para este exemplo.
    Não altera nenhum peso.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    camadas = rede.camadas
    if y.size != rede.n_saidas:
        raise ValueError(f"Esperadas {rede.n_saidas} saídas, recebidas {y.size}.")

    # Camada de saída: δ_k = (a_k - y_k) · f'(z_k)
    for k, neuronio in enumerate(camadas[-1].neuronios):
        neuronio.delta = (neuronio.saida - y[k]) * neuronio.derivada_ativacao()

    # Camadas ocultas, de trás para frente: δ_j = f'(z_j) · Σ_k w_kj · δ_k
    for indice in range(len(camadas) - 2, -1, -1):
        seguinte = camadas[indice + 1].neuronios
        for j, neuronio in enumerate(camadas[indice].neuronios):
            soma = sum(n_k.pesos[j] * n_k.delta for n_k in seguinte)
            neuronio.delta = neuronio.derivada_ativacao() * soma


def atualizar_pesos(rede, taxa_aprendizado: float) -> None:
    """Descida do gradiente em todos os neurônios (usa os deltas já calculados)."""
    for camada in rede.camadas:
        for neuronio in camada.neuronios:
            neuronio.atualizar(taxa_aprendizado)


def gradientes(rede, x, y) -> list[tuple[np.ndarray, np.ndarray]]:
    """Gradientes de E = 1/2 Σ(y - a)^2 em relação a pesos e vieses, por camada.

    Devolve [(dE/dW, dE/db), ...] sem alterar a rede. Usado no teste que compara
    a retropropagação com o gradiente numérico.
    """
    rede.propagar(x)
    retropropagar(rede, y)
    resultado = []
    for camada in rede.camadas:
        d_pesos = np.array([n.delta * n.entradas for n in camada.neuronios])
        d_vieses = np.array([n.delta for n in camada.neuronios])
        resultado.append((d_pesos, d_vieses))
    return resultado


# ====================================================================== #
# Treinador
# ====================================================================== #
@dataclass
class Historico:
    """MSE de treino e validação ao fim de cada época (índice 0 = época 1)."""

    mse_treino_inicial: float = math.nan
    mse_validacao_inicial: float = math.nan
    mse_treino: list[float] = field(default_factory=list)
    mse_validacao: list[float] = field(default_factory=list)

    @property
    def epocas(self) -> int:
        return len(self.mse_treino)

    def linhas_csv(self) -> list[list]:
        linhas = [["epoca", "mse_treino", "mse_validacao"],
                  [0, self.mse_treino_inicial, self.mse_validacao_inicial]]
        for i, (t, v) in enumerate(zip(self.mse_treino, self.mse_validacao), start=1):
            linhas.append([i, t, v])
        return linhas


class Treinador:
    """Treina uma RedeNeural com SGD puro.

    * só o conjunto de treino ajusta pesos;
    * ao fim de cada época calcula o MSE de treino e de validação
      (propagação direta completa, depois das atualizações da época);
    * a validação nunca altera pesos;
    * o treino é embaralhado a cada época com um gerador criado pela seed.
    """

    def __init__(
        self,
        rede,
        taxa_aprendizado: float = 0.1,
        epocas: int = 100,
        seed: int | None = None,
        embaralhar: bool = True,
    ):
        if not float(taxa_aprendizado) > 0:
            raise ValueError("A taxa de aprendizado deve ser maior que zero.")
        if int(epocas) < 1:
            raise ValueError("O número de épocas deve ser pelo menos 1.")
        self.rede = rede
        self.taxa_aprendizado = float(taxa_aprendizado)
        self.epocas = int(epocas)
        self.seed = seed
        self.embaralhar = embaralhar
        self.gerador = criar_gerador(seed)  # gerador próprio do embaralhamento

    # -- um exemplo ---------------------------------------------------- #
    def passo_sgd(self, x, y) -> None:
        """Forward -> todos os deltas -> atualização (um único exemplo)."""
        self.rede.propagar(x)
        retropropagar(self.rede, y)
        atualizar_pesos(self.rede, self.taxa_aprendizado)

    # -- uma época ----------------------------------------------------- #
    def treinar_epoca(self, X, Y) -> None:
        ordem = self.gerador.permutation(len(X)) if self.embaralhar else range(len(X))
        for i in ordem:
            self.passo_sgd(X[i], Y[i])

    # -- treino completo ----------------------------------------------- #
    def treinar(self, X_treino, Y_treino, X_validacao=None, Y_validacao=None, ao_fim_epoca=None) -> Historico:
        """Executa todas as épocas.

        ``ao_fim_epoca(epoca, mse_treino, mse_validacao)`` é chamada ao fim de
        cada época (a interface usa isso para exibir o MSE).
        """
        X_treino = np.asarray(X_treino, dtype=float)
        Y_treino = np.asarray(Y_treino, dtype=float).reshape(len(X_treino), -1)
        verificar_compatibilidade(self.rede, X_treino, Y_treino)
        tem_validacao = X_validacao is not None and len(X_validacao) > 0
        if tem_validacao:
            X_validacao = np.asarray(X_validacao, dtype=float)
            Y_validacao = np.asarray(Y_validacao, dtype=float).reshape(len(X_validacao), -1)
            verificar_compatibilidade(self.rede, X_validacao, Y_validacao)

        def avaliar():
            mse_tr = mse_rede(self.rede, X_treino, Y_treino)
            mse_va = mse_rede(self.rede, X_validacao, Y_validacao) if tem_validacao else math.nan
            return mse_tr, mse_va

        historico = Historico()
        historico.mse_treino_inicial, historico.mse_validacao_inicial = avaliar()

        for epoca in range(1, self.epocas + 1):
            self.treinar_epoca(X_treino, Y_treino)
            mse_tr, mse_va = avaliar()
            historico.mse_treino.append(mse_tr)
            historico.mse_validacao.append(mse_va)
            if ao_fim_epoca is not None:
                ao_fim_epoca(epoca, mse_tr, mse_va)
        return historico
