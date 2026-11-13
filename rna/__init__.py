"""Pacote rna: Multilayer Perceptron implementado à mão (somente numpy).

Módulos:
    ativacoes      Linear, Sigmoid, Tanh, ReLU + derivadas
    inicializacao  Xavier/Glorot, He, gerador com seed
    neuronio       entidade Neurônio
    camada         entidade Camada
    rede           entidade Rede Neural (forward)
    erro           MSE
    dataset        CSV, colunas, divisão, normalização
    treino         backpropagation + SGD + épocas
    persistencia   salvar/carregar JSON
    previsao       previsão com validação e normalização salva
"""

from .ativacoes import Linear, ReLU, Sigmoid, Tanh, obter_ativacao
from .camada import Camada
from .dataset import Dataset, Divisao, Normalizador, dividir, preparar_dados, verificar_compatibilidade
from .erro import mse, mse_rede
from .neuronio import Neuronio
from .persistencia import carregar, salvar
from .previsao import ErroEntrada, avaliar, prever, prever_lote
from .rede import RedeNeural
from .treino import Historico, Treinador

__all__ = [
    "Linear", "Sigmoid", "Tanh", "ReLU", "obter_ativacao",
    "Neuronio", "Camada", "RedeNeural",
    "mse", "mse_rede",
    "Dataset", "Divisao", "Normalizador", "dividir", "preparar_dados", "verificar_compatibilidade",
    "Treinador", "Historico",
    "salvar", "carregar",
    "ErroEntrada", "prever", "prever_lote", "avaliar",
]
