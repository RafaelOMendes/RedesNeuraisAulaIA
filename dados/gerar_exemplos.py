"""Gera os CSVs de exemplo desta pasta (reprodutível, seed fixa).

Uso:  python dados/gerar_exemplos.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent


def gravar(nome: str, cabecalho: list[str], linhas) -> None:
    with open(PASTA / nome, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(cabecalho)
        for linha in linhas:
            escritor.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in linha])
    print(f"gerado: dados/{nome}")


def portas_logicas() -> None:
    entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    gravar("xor.csv", ["x1", "x2", "y"], [(a, b, a ^ b) for a, b in entradas])
    gravar("and.csv", ["x1", "x2", "y"], [(a, b, a & b) for a, b in entradas])
    gravar("or.csv", ["x1", "x2", "y"], [(a, b, a | b) for a, b in entradas])


def seno(n: int = 200, seed: int = 1) -> None:
    """Regressão: y = sen(x), x em [-pi, pi]. Saída em [-1, 1]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, n)
    gravar("seno.csv", ["x", "y"], [(float(a), float(np.sin(a))) for a in x])


def tres_classes(n_por_classe: int = 70, seed: int = 2) -> None:
    """Classificação: 3 nuvens gaussianas em 2D, saída one-hot.

    As entradas têm escalas bem diferentes (x1 ~ 0-10, x2 ~ 0-100) para mostrar
    o efeito da normalização.
    """
    rng = np.random.default_rng(seed)
    centros = [(2.0, 20.0), (6.0, 30.0), (4.0, 70.0)]
    desvios = [(0.9, 7.0), (0.9, 7.0), (0.9, 7.0)]
    linhas = []
    for classe, ((cx, cy), (sx, sy)) in enumerate(zip(centros, desvios)):
        for _ in range(n_por_classe):
            one_hot = [1 if i == classe else 0 for i in range(3)]
            linhas.append((float(rng.normal(cx, sx)), float(rng.normal(cy, sy)), *one_hot))
    ordem = rng.permutation(len(linhas))
    gravar("tres_classes.csv", ["x1", "x2", "classe_a", "classe_b", "classe_c"],
           [linhas[i] for i in ordem])


if __name__ == "__main__":
    portas_logicas()
    seno()
    tres_classes()
