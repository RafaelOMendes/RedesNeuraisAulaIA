"""Interface de linha de comando do sistema de RNA (MLP).

Subcomandos (use  python main.py <comando> -h  para ver as opções):
    criar      cria uma rede (entradas, camadas ocultas, neurônios, ativações) e salva em JSON
    info       mostra a arquitetura de uma rede salva
    colunas    mostra as colunas de um CSV
    dividir    divide um CSV em treino/validação/teste (e opcionalmente grava os 3 CSVs)
    treinar    treina uma rede salva com um CSV, mostra o MSE por época e o MSE de teste
    avaliar    calcula o MSE de uma rede treinada sobre um CSV
    prever     executa a rede treinada sobre novos valores
    interativo menu com todas as funções (também abre se rodar sem argumentos)
    web        abre a interface web local (http://127.0.0.1:8000)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from rna import (
    Dataset, ErroEntrada, RedeNeural, avaliar, carregar, dividir, prever, salvar,
    verificar_compatibilidade,
)
from rna.ativacoes import nomes_disponiveis
from rna.dataset import Divisao, interpretar_percentuais, normalizar_divisao, separar_lista
from rna.erro import mse_rede
from rna.experimento import avaliar_teste as avaliar_teste_final
from rna.experimento import metadados_treino
from rna.previsao import converter_valores


# ====================================================================== #
# Utilidades de formatação e de conversão de texto
# ====================================================================== #
def configurar_console() -> None:
    """Garante UTF-8 na saída (acentos e símbolos) também no Windows."""
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def fmt(valor, casas: int = 6) -> str:
    """Número formatado; NaN/None (conjunto vazio) vira '—'."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    return f"{float(valor):.{casas}f}"


def fmt_vetor(valores, casas: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{casas}f}" for v in valores) + "]"


def interpretar_camada(texto: str) -> tuple[int, str]:
    """'4:tanh' -> (4, 'tanh')"""
    partes = texto.strip().split(":")
    if len(partes) != 2:
        raise ValueError(
            f"Camada '{texto}' inválida. Use NEURONIOS:ATIVACAO, por exemplo 4:tanh."
        )
    try:
        n = int(partes[0])
    except ValueError:
        raise ValueError(f"Número de neurônios inválido em '{texto}'.") from None
    if n < 1:
        raise ValueError(f"A camada '{texto}' precisa de pelo menos 1 neurônio.")
    ativacao = partes[1].strip().lower()
    if ativacao not in nomes_disponiveis():
        raise ValueError(
            f"Ativação '{partes[1]}' desconhecida. Opções: {', '.join(nomes_disponiveis())}."
        )
    return n, ativacao


def interpretar_camadas(texto: str | None) -> list[tuple[int, str]]:
    """'8:relu,4:tanh' -> [(8, 'relu'), (4, 'tanh')]   |   '' -> [] (sem ocultas)"""
    return [interpretar_camada(p) for p in separar_lista(texto)]


# ====================================================================== #
# Operações de alto nível (usadas pelos subcomandos E pelo menu interativo)
# ====================================================================== #
def criar_rede(n_entradas: int, ocultas: str | None, saida: str, seed: int | None) -> RedeNeural:
    n_saidas, ativ_saida = interpretar_camada(saida)
    return RedeNeural(n_entradas, interpretar_camadas(ocultas), n_saidas, ativ_saida, seed=seed)


def mostrar_rede(rede: RedeNeural, detalhar_pesos: bool = False) -> None:
    print(rede.resumo())
    if rede.normalizador is not None:
        print(f"Parâmetros da normalização: {rede.normalizador.resumo()}")
    meta = rede.metadados
    if meta.get("colunas_entrada"):
        print(f"Colunas de entrada: {', '.join(meta['colunas_entrada'])}")
        print(f"Colunas de saída:   {', '.join(meta.get('colunas_saida', []))}")
    if meta.get("treino"):
        t = meta["treino"]
        print(
            f"Último treino: {t.get('epocas')} épocas, taxa {t.get('taxa_aprendizado')}, "
            f"MSE final treino {fmt(t.get('mse_treino_final'))}, "
            f"validação {fmt(t.get('mse_validacao_final'))}, teste {fmt(t.get('mse_teste'))}"
        )
    if detalhar_pesos:
        for i, camada in enumerate(rede.camadas, start=1):
            print(f"\nCamada {i} ({camada.ativacao.nome})")
            for j, n in enumerate(camada.neuronios, start=1):
                print(f"  neurônio {j}: pesos={fmt_vetor(n.pesos)}  viés={n.bias:.4f}")


def carregar_dados(caminho: str, entradas, saidas, separador: str = ","):
    """Lê o CSV e separa X e Y. Devolve (dataset, X, Y, nomes_x, nomes_y)."""
    ds = Dataset.de_csv(caminho, separador)
    X, Y = ds.selecionar(entradas, saidas)
    return ds, X, Y, ds.nomes(entradas), ds.nomes(saidas)


def executar_treino(
    rede: RedeNeural,
    divisao: Divisao,
    normalizacao: str,
    epocas: int,
    taxa: float,
    seed: int | None,
    embaralhar: bool = True,
    mostrar_a_cada: int = 1,
    caminho_historico: str | None = None,
    avaliar_teste: bool = True,
):
    """Normaliza (parâmetros só do treino), treina com SGD e mostra o MSE por época.

    Devolve (divisao_normalizada, historico, resultados).
    """
    for aviso in verificar_compatibilidade(rede, divisao.X_treino, divisao.Y_treino):
        print(f"Aviso: {aviso}")

    divisao_norm, normalizador = normalizar_divisao(divisao, normalizacao)
    rede.normalizador = None if normalizador.tipo == "nenhuma" else normalizador

    print(divisao.resumo())
    print(f"Normalização: {normalizador.resumo()}")
    print(f"Treinando {epocas} épocas | taxa de aprendizado {taxa} | SGD (1 exemplo por atualização)"
          f" | embaralhar={'sim' if embaralhar else 'não'} | seed={seed}")
    largura = len(str(epocas))
    mostrar_a_cada = max(1, int(mostrar_a_cada))

    def ao_fim_epoca(epoca, mse_tr, mse_va):
        if epoca % mostrar_a_cada == 0 or epoca == 1 or epoca == epocas:
            print(f"Época {epoca:>{largura}}/{epocas} | MSE treino {fmt(mse_tr)} | MSE validação {fmt(mse_va)}")

    X_va = divisao_norm.X_validacao if len(divisao_norm.X_validacao) else None
    Y_va = divisao_norm.Y_validacao if len(divisao_norm.Y_validacao) else None
    print(f"Época {0:>{largura}}/{epocas} | MSE treino {fmt(mse_rede(rede, divisao_norm.X_treino, divisao_norm.Y_treino))}"
          f" | MSE validação {fmt(mse_rede(rede, divisao_norm.X_validacao, divisao_norm.Y_validacao))}  (antes do treino)")
    historico = rede.treinar(
        divisao_norm.X_treino, divisao_norm.Y_treino, X_va, Y_va,
        taxa_aprendizado=taxa, epocas=epocas, seed=seed, embaralhar=embaralhar,
        ao_fim_epoca=ao_fim_epoca,
    )

    resultados = {
        "mse_treino_inicial": historico.mse_treino_inicial,
        "mse_treino_final": historico.mse_treino[-1],
        "mse_validacao_final": historico.mse_validacao[-1],
        "mse_teste": math.nan,
    }
    print("-" * 60)
    print(f"MSE de treino:    {fmt(historico.mse_treino_inicial)} (início) -> {fmt(historico.mse_treino[-1])} (fim)")
    print(f"MSE de validação: {fmt(historico.mse_validacao[-1])}")
    if avaliar_teste:
        resultados.update(testar(rede, divisao_norm))

    if caminho_historico:
        Path(caminho_historico).parent.mkdir(parents=True, exist_ok=True)
        with open(caminho_historico, "w", encoding="utf-8", newline="") as arquivo:
            csv.writer(arquivo).writerows(historico.linhas_csv())
        print(f"Histórico do MSE por época salvo em {caminho_historico}")

    rede.metadados["treino"] = metadados_treino(
        divisao, normalizador.tipo, epocas, taxa, seed, embaralhar, resultados
    )
    return divisao_norm, historico, resultados


def testar(rede: RedeNeural, divisao_norm: Divisao) -> dict:
    """Avaliação final no conjunto de teste (só propagação direta)."""
    resultado = avaliar_teste_final(rede, divisao_norm)
    print(f"MSE de teste:     {fmt(resultado['mse_teste'])}  ({resultado['n_teste']} exemplos)")
    if resultado["acuracia_teste"] is not None:
        print(f"Acurácia de teste: {resultado['acuracia_teste'] * 100:.2f}%")
    return resultado


def mostrar_previsao(rede: RedeNeural, valores: list[float]) -> np.ndarray:
    saida = prever(rede, valores)  # valida a quantidade e aplica a normalização salva
    nomes_x = rede.metadados.get("colunas_entrada") or [f"x{i + 1}" for i in range(rede.n_entradas)]
    nomes_y = rede.metadados.get("colunas_saida") or [f"y{i + 1}" for i in range(rede.n_saidas)]
    print("Entrada: " + ", ".join(f"{n}={v:g}" for n, v in zip(nomes_x, valores)))
    if rede.normalizador is not None:
        normalizada = rede.normalizador.aplicar(np.asarray(valores, dtype=float))
        print(f"Entrada normalizada ({rede.normalizador.tipo}): {fmt_vetor(normalizada)}")
    print("Saída:   " + ", ".join(f"{n}={v:.6f}" for n, v in zip(nomes_y, saida)))
    if rede.n_saidas > 1:
        print(f"Classe prevista (maior saída): {nomes_y[int(np.argmax(saida))]}")
    elif rede.camada_saida.ativacao.nome == "sigmoid":
        print(f"Classe prevista (limiar 0,5): {int(saida[0] >= 0.5)}")
    return saida


# ====================================================================== #
# Subcomandos
# ====================================================================== #
def cmd_criar(args) -> None:
    rede = criar_rede(args.entradas, args.ocultas, args.saidas, args.seed)
    caminho = salvar(rede, args.saida)
    mostrar_rede(rede)
    print(f"\nRede criada e salva em {caminho}")


def cmd_info(args) -> None:
    rede = carregar(args.modelo)
    print(f"Arquivo: {args.modelo}")
    mostrar_rede(rede, detalhar_pesos=args.pesos)


def cmd_colunas(args) -> None:
    ds = Dataset.de_csv(args.csv, args.separador)
    print(f"Arquivo: {args.csv}")
    print(ds.resumo())


def cmd_dividir(args) -> None:
    ds, X, Y, nomes_x, nomes_y = carregar_dados(args.csv, args.x, args.y, args.separador)
    divisao = dividir(X, Y, args.divisao, args.seed)
    print(divisao.resumo())
    print(f"Primeiras linhas do treino (índices no CSV, a partir de 0): {divisao.indices_treino[:10].tolist()}")
    if args.pasta:
        pasta = Path(args.pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        conjuntos = {
            "treino.csv": (divisao.X_treino, divisao.Y_treino),
            "validacao.csv": (divisao.X_validacao, divisao.Y_validacao),
            "teste.csv": (divisao.X_teste, divisao.Y_teste),
        }
        for nome, (Xc, Yc) in conjuntos.items():
            with open(pasta / nome, "w", encoding="utf-8", newline="") as arquivo:
                escritor = csv.writer(arquivo)
                escritor.writerow(nomes_x + nomes_y)
                for linha in np.hstack([Xc, Yc]) if len(Xc) else []:
                    escritor.writerow([f"{v:.10g}" for v in linha])
            print(f"Gravado {pasta / nome} ({len(Xc)} linhas)")


def cmd_treinar(args) -> None:
    rede = carregar(args.modelo)
    ds, X, Y, nomes_x, nomes_y = carregar_dados(args.csv, args.x, args.y, args.separador)
    verificar_compatibilidade(rede, X, Y)
    print(f"Rede: {' -> '.join(map(str, rede.arquitetura()))} | dataset: {args.csv} ({ds.n_linhas} linhas)")
    print(f"Entradas: {', '.join(nomes_x)} | Saídas esperadas: {', '.join(nomes_y)}")
    divisao = dividir(X, Y, args.divisao, args.seed)

    rede.metadados.update({"colunas_entrada": nomes_x, "colunas_saida": nomes_y, "dataset": str(args.csv)})
    executar_treino(
        rede, divisao, args.normalizacao, args.epocas, args.taxa, args.seed,
        embaralhar=not args.sem_embaralhar, mostrar_a_cada=args.mostrar_a_cada,
        caminho_historico=args.historico,
    )
    destino = args.saida or args.modelo
    salvar(rede, destino)
    print(f"Rede treinada salva em {destino}")


def cmd_avaliar(args) -> None:
    rede = carregar(args.modelo)
    entradas = args.x or rede.metadados.get("colunas_entrada")
    saidas = args.y or rede.metadados.get("colunas_saida")
    if not entradas or not saidas:
        raise ValueError("Informe as colunas com --x e --y (a rede não tem colunas salvas).")
    _, X, Y, _, _ = carregar_dados(args.csv, entradas, saidas, args.separador)
    verificar_compatibilidade(rede, X, Y)
    resultado = avaliar(rede, X, Y)  # aplica a normalização salva antes do forward
    print(f"Avaliação de {args.modelo} em {args.csv} ({resultado['n']} exemplos)")
    print(f"MSE: {fmt(resultado['mse'])}")
    if resultado["acuracia"] is not None:
        print(f"Acurácia: {resultado['acuracia'] * 100:.2f}%")


def cmd_prever(args) -> None:
    rede = carregar(args.modelo)
    for i, texto in enumerate(args.valores):
        if i:
            print()
        mostrar_previsao(rede, converter_valores(texto))


# ====================================================================== #
# Menu interativo
# ====================================================================== #
class Estado:
    def __init__(self):
        self.rede: RedeNeural | None = None
        self.caminho_rede: str | None = None
        self.dataset: Dataset | None = None
        self.X = self.Y = None
        self.nomes_x: list[str] = []
        self.nomes_y: list[str] = []
        self.divisao: Divisao | None = None
        self.divisao_norm: Divisao | None = None
        self.treinada = False
        self.config = {"epocas": 500, "taxa": 0.1, "seed": 42, "normalizacao": "minmax",
                       "mostrar_a_cada": 50, "embaralhar": True}


def perguntar(mensagem: str, padrao=None, conversor=str):
    """Lê um valor do teclado; Enter vazio usa o padrão. Repete até ser válido."""
    sufixo = f" [{padrao}]" if padrao is not None else ""
    while True:
        try:
            texto = input(f"{mensagem}{sufixo}: ").strip()
        except EOFError:
            print()
            raise SystemExit(0)
        if not texto and padrao is not None:
            return padrao
        try:
            return conversor(texto)
        except (ValueError, ErroEntrada) as erro:
            print(f"  Valor inválido: {erro}")


def _seed(texto: str):
    return None if texto.strip().lower() in ("", "nenhuma", "none") else int(texto)


def _exigir(condicao, mensagem):
    if not condicao:
        raise ValueError(mensagem)


def menu_criar(e: Estado):
    print(f"Ativações disponíveis: {', '.join(nomes_disponiveis())}")
    n_entradas = perguntar("Quantidade de entradas", conversor=int)
    n_ocultas = perguntar("Número de camadas ocultas", 1, int)
    ocultas = []
    for i in range(1, n_ocultas + 1):
        n = perguntar(f"  Camada oculta {i} - neurônios", 4, int)
        ativ = perguntar(f"  Camada oculta {i} - ativação", "tanh")
        ocultas.append(f"{n}:{ativ}")
    n_saidas = perguntar("Neurônios de saída", 1, int)
    ativ_saida = perguntar("Ativação da saída", "sigmoid")
    seed = perguntar("Seed da inicialização (vazio = aleatória)", "", _seed)
    e.rede = criar_rede(n_entradas, ",".join(ocultas), f"{n_saidas}:{ativ_saida}", seed)
    e.treinada, e.caminho_rede = False, None
    print()
    mostrar_rede(e.rede)


def menu_carregar_rede(e: Estado):
    caminho = perguntar("Arquivo JSON da rede", e.caminho_rede or "modelos/rede.json")
    e.rede, e.caminho_rede = carregar(caminho), caminho
    e.treinada = bool(e.rede.metadados.get("treino"))
    mostrar_rede(e.rede)


def menu_mostrar_rede(e: Estado):
    _exigir(e.rede, "Crie ou carregue uma rede primeiro (opções 1 ou 2).")
    detalhar = perguntar("Mostrar pesos e vieses? (s/n)", "n").lower().startswith("s")
    mostrar_rede(e.rede, detalhar)


def menu_carregar_dataset(e: Estado):
    caminho = perguntar("Arquivo CSV", "dados/tres_classes.csv")
    separador = perguntar("Separador", ",")
    e.dataset = Dataset.de_csv(caminho, separador)
    e.X = e.Y = e.divisao = e.divisao_norm = None
    print(e.dataset.resumo())


def menu_colunas(e: Estado):
    _exigir(e.dataset, "Carregue um dataset primeiro (opção 4).")
    print("Colunas: " + ", ".join(f"{i}={n}" for i, n in enumerate(e.dataset.colunas)))
    padrao_y = e.dataset.colunas[-1]
    padrao_x = ",".join(e.dataset.colunas[:-1])
    entradas = perguntar("Colunas de entrada (nomes ou índices, separados por vírgula)", padrao_x)
    saidas = perguntar("Colunas de saída esperada", padrao_y)
    e.X, e.Y = e.dataset.selecionar(entradas, saidas)
    e.nomes_x, e.nomes_y = e.dataset.nomes(entradas), e.dataset.nomes(saidas)
    e.divisao = e.divisao_norm = None
    print(f"X: {e.X.shape[0]} x {e.X.shape[1]} ({', '.join(e.nomes_x)}) | "
          f"Y: {e.Y.shape[0]} x {e.Y.shape[1]} ({', '.join(e.nomes_y)})")
    if e.rede is not None:
        try:
            verificar_compatibilidade(e.rede, e.X, e.Y)
            print("Compatível com a rede atual.")
        except ValueError as erro:
            print(f"Atenção: {erro}")


def menu_dividir(e: Estado):
    _exigir(e.X is not None, "Escolha as colunas primeiro (opção 5).")
    percentuais = perguntar("Percentuais treino,validação,teste", "70,15,15", interpretar_percentuais)
    seed = perguntar("Seed da divisão (vazio = aleatória)", e.config["seed"], _seed)
    e.divisao = dividir(e.X, e.Y, percentuais, seed)
    e.divisao_norm = None
    e.config["seed"] = seed
    print(e.divisao.resumo())


def menu_configurar(e: Estado):
    c = e.config
    c["epocas"] = perguntar("Épocas", c["epocas"], int)
    c["taxa"] = perguntar("Taxa de aprendizado", c["taxa"], float)
    c["seed"] = perguntar("Seed do embaralhamento (vazio = aleatória)", c["seed"], _seed)
    c["normalizacao"] = perguntar("Normalização (nenhuma/minmax/zscore)", c["normalizacao"])
    c["mostrar_a_cada"] = perguntar("Mostrar o MSE a cada quantas épocas", c["mostrar_a_cada"], int)
    c["embaralhar"] = perguntar("Embaralhar o treino a cada época? (s/n)", "s" if c["embaralhar"] else "n").lower().startswith("s")
    print(f"Configuração: {c}")


def menu_treinar(e: Estado):
    _exigir(e.rede, "Crie ou carregue uma rede primeiro (opções 1 ou 2).")
    _exigir(e.divisao is not None, "Divida os dados primeiro (opção 6).")
    verificar_compatibilidade(e.rede, e.divisao.X_treino, e.divisao.Y_treino)
    c = e.config
    e.rede.metadados.update({"colunas_entrada": e.nomes_x, "colunas_saida": e.nomes_y,
                             "dataset": e.dataset.caminho if e.dataset else None})
    e.divisao_norm, _, _ = executar_treino(
        e.rede, e.divisao, c["normalizacao"], c["epocas"], c["taxa"], c["seed"],
        c["embaralhar"], c["mostrar_a_cada"], avaliar_teste=False,
    )
    e.treinada = True
    print("Use a opção 9 para calcular o MSE no conjunto de teste.")


def menu_testar(e: Estado):
    _exigir(e.rede and e.treinada and e.divisao_norm is not None,
            "Treine a rede nesta sessão primeiro (opção 8).")
    resultado = testar(e.rede, e.divisao_norm)
    e.rede.metadados.setdefault("treino", {}).update(resultado)


def menu_salvar(e: Estado):
    _exigir(e.rede, "Não há rede para salvar.")
    caminho = perguntar("Salvar em", e.caminho_rede or "modelos/rede.json")
    salvar(e.rede, caminho)
    e.caminho_rede = caminho
    print(f"Rede salva em {caminho}")


def menu_prever(e: Estado):
    _exigir(e.rede, "Crie ou carregue uma rede primeiro (opções 1 ou 2).")
    nomes = e.rede.metadados.get("colunas_entrada")
    dica = f" ({', '.join(nomes)})" if nomes else ""
    valores = perguntar(f"{e.rede.n_entradas} valor(es) separados por vírgula{dica}", conversor=converter_valores)
    mostrar_previsao(e.rede, valores)


OPCOES = [
    ("1", "Criar rede", menu_criar),
    ("2", "Carregar rede (JSON)", menu_carregar_rede),
    ("3", "Mostrar rede", menu_mostrar_rede),
    ("4", "Carregar dataset (CSV)", menu_carregar_dataset),
    ("5", "Escolher colunas de entrada e saída", menu_colunas),
    ("6", "Dividir dados (treino/validação/teste)", menu_dividir),
    ("7", "Configurar treino (épocas, taxa, seed, normalização)", menu_configurar),
    ("8", "Treinar (mostra o MSE por época)", menu_treinar),
    ("9", "Testar (MSE no conjunto de teste)", menu_testar),
    ("10", "Salvar rede", menu_salvar),
    ("11", "Prever com novos valores", menu_prever),
    ("0", "Sair", None),
]


def cmd_interativo(_args=None) -> None:
    e = Estado()
    acoes = {codigo: funcao for codigo, _, funcao in OPCOES}
    while True:
        print("\n===== RNA MLP - menu =====")
        status = []
        if e.rede is not None:
            status.append(f"rede {'-'.join(map(str, e.rede.arquitetura()))}{' (treinada)' if e.treinada else ''}")
        if e.dataset is not None:
            status.append(f"dataset {Path(e.dataset.caminho).name}")
        if e.divisao is not None:
            status.append("dados divididos")
        if status:
            print("Estado: " + " | ".join(status))
        for codigo, descricao, _ in OPCOES:
            print(f"{codigo:>3}) {descricao}")
        escolha = perguntar("Opção")
        if escolha == "0":
            print("Até mais!")
            return
        if escolha not in acoes:
            print("Opção inválida.")
            continue
        try:
            acoes[escolha](e)
        except (ValueError, FileNotFoundError, RuntimeError) as erro:
            print(f"Erro: {erro}")


# ====================================================================== #
# Parser
# ====================================================================== #
def montar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Sistema configurável de Redes Neurais MLP (implementação própria, só numpy).",
    )
    sub = parser.add_subparsers(dest="comando", metavar="comando")
    ativs = ", ".join(nomes_disponiveis())

    p = sub.add_parser("criar", help="cria uma rede e salva em JSON")
    p.add_argument("--entradas", type=int, required=True, help="quantidade de entradas")
    p.add_argument("--ocultas", default="",
                   help=f"camadas ocultas NEURONIOS:ATIVACAO separadas por vírgula, ex.: 8:relu,4:tanh "
                        f"(vazio = nenhuma). Ativações: {ativs}")
    p.add_argument("--saidas", default="1:sigmoid", help="camada de saída NEURONIOS:ATIVACAO (padrão 1:sigmoid)")
    p.add_argument("--seed", type=int, default=None, help="seed da inicialização dos pesos")
    p.add_argument("-o", "--saida", required=True, help="arquivo JSON de destino")
    p.set_defaults(func=cmd_criar)

    p = sub.add_parser("info", help="mostra a arquitetura de uma rede salva")
    p.add_argument("-m", "--modelo", required=True, help="arquivo JSON da rede")
    p.add_argument("--pesos", action="store_true", help="mostra também pesos e vieses")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("colunas", help="mostra as colunas de um CSV")
    p.add_argument("csv", help="arquivo CSV")
    p.add_argument("--separador", default=",")
    p.set_defaults(func=cmd_colunas)

    def opcoes_dados(p, colunas_obrigatorias=True):
        p.add_argument("--csv", required=True, help="arquivo CSV com cabeçalho")
        p.add_argument("--x", "--colunas-entrada", dest="x", required=colunas_obrigatorias,
                       help="colunas de entrada (nomes ou índices), ex.: x1,x2")
        p.add_argument("--y", "--colunas-saida", dest="y", required=colunas_obrigatorias,
                       help="colunas de saída esperada, ex.: y  ou  classe_a,classe_b,classe_c")
        p.add_argument("--separador", default=",")

    p = sub.add_parser("dividir", help="divide os dados em treino/validação/teste")
    opcoes_dados(p)
    p.add_argument("--divisao", default="70,15,15", help="percentuais treino,validação,teste (somam 100)")
    p.add_argument("--seed", type=int, default=None, help="seed da divisão")
    p.add_argument("--pasta", default=None, help="se informado, grava treino.csv, validacao.csv e teste.csv")
    p.set_defaults(func=cmd_dividir)

    p = sub.add_parser("treinar", help="treina uma rede salva (SGD + backpropagation)")
    p.add_argument("-m", "--modelo", required=True, help="arquivo JSON da rede (criada com 'criar')")
    opcoes_dados(p)
    p.add_argument("--divisao", default="70,15,15", help="percentuais treino,validação,teste (somam 100)")
    p.add_argument("--normalizacao", default="nenhuma", choices=["nenhuma", "minmax", "zscore"],
                   help="normalização das entradas (parâmetros calculados só no treino)")
    p.add_argument("--epocas", type=int, default=1000)
    p.add_argument("--taxa", type=float, default=0.1, help="taxa de aprendizado")
    p.add_argument("--seed", type=int, default=None, help="seed da divisão e do embaralhamento")
    p.add_argument("--sem-embaralhar", action="store_true", help="não embaralha o treino a cada época")
    p.add_argument("--mostrar-a-cada", type=int, default=1, help="exibe o MSE a cada N épocas (padrão 1)")
    p.add_argument("--historico", default=None, help="CSV para gravar o MSE de cada época")
    p.add_argument("-o", "--saida", default=None, help="onde salvar a rede treinada (padrão: o próprio --modelo)")
    p.set_defaults(func=cmd_treinar)

    p = sub.add_parser("avaliar", help="MSE de uma rede treinada sobre um CSV")
    p.add_argument("-m", "--modelo", required=True)
    opcoes_dados(p, colunas_obrigatorias=False)
    p.set_defaults(func=cmd_avaliar)

    p = sub.add_parser("prever", help="executa a rede sobre novos valores")
    p.add_argument("-m", "--modelo", required=True)
    p.add_argument("--valores", action="append", required=True,
                   help="valores separados por vírgula, ex.: 1,0  (pode repetir a opção; "
                        "para negativos use --valores=-1.5,2)")
    p.set_defaults(func=cmd_prever)

    p = sub.add_parser("interativo", help="menu interativo com todas as funções")
    p.set_defaults(func=cmd_interativo)

    p = sub.add_parser("web", help="abre a interface web local (mesmo código da pasta rna/)")
    p.add_argument("--porta", type=int, default=8000)
    p.add_argument("--sem-navegador", action="store_true", help="não abre o navegador automaticamente")
    p.set_defaults(func=cmd_web)
    return parser


def cmd_web(args) -> None:
    from web.servidor import iniciar_servidor  # import local: a CLI não depende da web

    iniciar_servidor(porta=args.porta, abrir_navegador=not args.sem_navegador)


def main(argv=None) -> int:
    configurar_console()
    args = montar_parser().parse_args(argv)
    funcao = getattr(args, "func", cmd_interativo)
    try:
        funcao(args)
    except (ValueError, FileNotFoundError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
