"""Salvar e carregar redes em JSON.

Formato do arquivo (versão 1):

{
  "formato": "rna-mlp",
  "versao": 1,
  "n_entradas": 2,
  "n_camadas": 2,                     # camadas com neurônios (ocultas + saída)
  "n_camadas_ocultas": 1,
  "neuronios_por_camada": [4, 1],
  "seed_inicializacao": 42,
  "camadas": [
    {"tipo": "oculta", "n_neuronios": 4, "ativacao": "tanh", "inicializacao": "xavier",
     "pesos": [[w11, w12], ...],      # uma linha por neurônio
     "vieses": [b1, b2, b3, b4]},
    {"tipo": "saida", ...}
  ],
  "normalizacao": {"tipo": "minmax", "parametros": {"minimo": [...], "maximo": [...]}},
  "metadados": {...}                  # colunas, configuração do treino, MSEs finais
}

Os floats são gravados com a representação completa do Python (repr), então
pesos, vieses e parâmetros de normalização voltam idênticos ao carregar.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .camada import Camada
from .dataset import Normalizador
from .rede import RedeNeural

FORMATO = "rna-mlp"
VERSAO = 1


def limpar_json(valor):
    """Converte numpy/NaN em tipos JSON válidos (NaN e infinito viram null)."""
    if isinstance(valor, dict):
        return {str(k): limpar_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [limpar_json(v) for v in valor]
    if isinstance(valor, np.ndarray):
        return limpar_json(valor.tolist())
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        valor = float(valor)
        return valor if math.isfinite(valor) else None
    return valor


def rede_para_dict(rede: RedeNeural, metadados: dict | None = None) -> dict:
    camadas = []
    for camada in rede.camadas:
        camadas.append({
            "tipo": "saida" if camada is rede.camada_saida else "oculta",
            "n_neuronios": camada.n_neuronios,
            "ativacao": camada.ativacao.nome,
            "inicializacao": camada.metodo_inicializacao,
            "pesos": [[float(w) for w in n.pesos] for n in camada.neuronios],
            "vieses": [float(n.bias) for n in camada.neuronios],
        })
    normalizacao = (
        rede.normalizador.para_dict() if rede.normalizador is not None
        else {"tipo": "nenhuma", "parametros": {}}
    )
    meta = dict(rede.metadados)
    meta.update(metadados or {})
    return {
        "formato": FORMATO,
        "versao": VERSAO,
        "n_entradas": rede.n_entradas,
        "n_camadas": len(rede.camadas),
        "n_camadas_ocultas": rede.n_camadas_ocultas,
        "neuronios_por_camada": [c.n_neuronios for c in rede.camadas],
        "seed_inicializacao": rede.seed,
        "camadas": camadas,
        "normalizacao": normalizacao,
        "metadados": limpar_json(meta),
    }


def dict_para_rede(dados: dict) -> RedeNeural:
    if dados.get("formato") != FORMATO:
        raise ValueError("Arquivo não é uma rede deste sistema (campo 'formato' ausente ou diferente).")
    try:
        n_entradas = int(dados["n_entradas"])
        camadas_json = dados["camadas"]
    except KeyError as erro:
        raise ValueError(f"Campo obrigatório ausente no arquivo da rede: {erro}") from None

    camadas = []
    n_anterior = n_entradas
    for i, c in enumerate(camadas_json, start=1):
        try:
            camada = Camada(n_anterior, int(c["n_neuronios"]), c["ativacao"],
                            pesos=c["pesos"], vieses=c["vieses"])
        except KeyError as erro:
            raise ValueError(f"Camada {i}: campo obrigatório ausente: {erro}") from None
        camadas.append(camada)
        n_anterior = camada.n_neuronios

    if "neuronios_por_camada" in dados and dados["neuronios_por_camada"] != [c.n_neuronios for c in camadas]:
        raise ValueError("'neuronios_por_camada' não confere com as camadas do arquivo.")

    rede = RedeNeural.de_camadas(n_entradas, camadas, seed=dados.get("seed_inicializacao"))
    normalizacao = dados.get("normalizacao") or {"tipo": "nenhuma"}
    rede.normalizador = None if normalizacao.get("tipo", "nenhuma") == "nenhuma" else Normalizador.de_dict(normalizacao)
    rede.metadados = dict(dados.get("metadados") or {})
    return rede


def salvar(rede: RedeNeural, caminho: str | Path, metadados: dict | None = None) -> Path:
    """Grava a rede em JSON (cria as pastas se precisar). Devolve o caminho."""
    caminho = Path(caminho)
    if caminho.parent and not caminho.parent.exists():
        caminho.parent.mkdir(parents=True, exist_ok=True)
    conteudo = rede_para_dict(rede, metadados)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, indent=2, ensure_ascii=False, allow_nan=False)
    return caminho


def carregar(caminho: str | Path) -> RedeNeural:
    """Reconstrói a rede a partir do JSON (arquitetura, pesos, vieses e normalização)."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de rede não encontrado: {caminho}")
    with open(caminho, encoding="utf-8") as arquivo:
        try:
            dados = json.load(arquivo)
        except json.JSONDecodeError as erro:
            raise ValueError(f"JSON inválido em {caminho}: {erro}") from None
    return dict_para_rede(dados)
