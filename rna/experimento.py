"""Funções de apoio compartilhadas pelas interfaces (linha de comando e web).

Não contêm algoritmo novo: só organizam a avaliação final no conjunto de teste
e os metadados do treino que são gravados no JSON da rede.
"""

from __future__ import annotations

import math

from .dataset import Divisao
from .erro import acuracia, mse_rede


def avaliar_teste(rede, divisao_normalizada: Divisao) -> dict:
    """Avaliação final no conjunto de teste (só propagação direta).

    Devolve {"mse_teste": float|NaN, "acuracia_teste": float|None, "n_teste": int}.
    """
    X, Y = divisao_normalizada.X_teste, divisao_normalizada.Y_teste
    resultado = {"mse_teste": mse_rede(rede, X, Y), "acuracia_teste": None, "n_teste": len(X)}
    if len(X):
        resultado["acuracia_teste"] = acuracia(rede.propagar_lote(X), Y)
    return resultado


def metadados_treino(
    divisao: Divisao,
    normalizacao: str,
    epocas: int,
    taxa: float,
    seed,
    embaralhar: bool,
    resultados: dict,
) -> dict:
    """Dicionário salvo em rede.metadados["treino"]."""
    return {
        "epocas": epocas,
        "taxa_aprendizado": taxa,
        "seed": seed,
        "embaralhar": embaralhar,
        "estrategia": "SGD puro (atualização a cada exemplo)",
        "divisao_percentuais": list(divisao.percentuais),
        "tamanhos": dict(zip(("treino", "validacao", "teste"), divisao.tamanhos())),
        "normalizacao": normalizacao,
        "mse_treino_inicial": resultados.get("mse_treino_inicial", math.nan),
        "mse_treino_final": resultados.get("mse_treino_final", math.nan),
        "mse_validacao_final": resultados.get("mse_validacao_final", math.nan),
        "mse_teste": resultados.get("mse_teste", math.nan),
        "acuracia_teste": resultados.get("acuracia_teste"),
    }
