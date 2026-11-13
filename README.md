# Trabalho Prático 02: Sistema Configurável de RNA (MLP)

Aplicação em Python que **cria, configura, treina, salva, carrega e usa** Redes Neurais
Artificiais do tipo *Multilayer Perceptron*. A arquitetura (entradas, camadas ocultas,
neurônios por camada, ativações e saídas) é definida pela interface, sem mexer no código.

> Novo em redes neurais? Comece pelo guia [EXPLICACAO.md](EXPLICACAO.md). Ele explica todos
> os conceitos do zero, com um exemplo numérico de backpropagation, o mapa do código, um
> roteiro para o vídeo e perguntas prováveis do professor.

Propagação direta, cálculo do erro, retropropagação e atualização dos pesos foram
**implementados à mão**. A única biblioteca externa é o **numpy**, usado para vetores e
matrizes. Não há TensorFlow, Keras, PyTorch, scikit-learn nem diferenciação automática.

---

## 1. Instalação

Requisitos: Python 3.10 ou mais novo (desenvolvido e testado com **Python 3.12**).

```bash
pip install -r requirements.txt
```

O `requirements.txt` só contém o `numpy`. O resto vem da biblioteca padrão do Python:
`csv` para ler os dados, `json` para salvar as redes, `argparse` para a interface e
`unittest` para os testes.

Todos os comandos abaixo devem ser executados **na pasta raiz do projeto**.

---

## 2. Estrutura do projeto

```
README.md
requirements.txt
main.py                 interface: subcomandos (argparse) + menu interativo + 'web'
web/
  servidor.py           servidor local (só biblioteca padrão) com API JSON que chama o pacote rna
  estatico/             página: index.html, estilo.css, app.js, graficos.js, diagrama.js, fundo.js, efeitos.js
rna/
  __init__.py
  ativacoes.py          Linear, Sigmoid, Tanh, ReLU + derivadas
  inicializacao.py      Xavier/Glorot, He, gerador com seed
  neuronio.py           entidade Neurônio (pesos, viés, z, saída, delta)
  camada.py             entidade Camada (neurônios + ativação própria)
  rede.py               entidade Rede Neural (arquitetura, forward, treinar/executar)
  erro.py               MSE (e acurácia como métrica extra)
  dataset.py            CSV, escolha de colunas, compatibilidade, divisão, normalização
  treino.py             backpropagation + SGD + épocas
  persistencia.py       salvar/carregar em JSON
  previsao.py           previsão com validação da entrada e normalização salva
  experimento.py        avaliação final no teste + metadados do treino (usado pela CLI e pela web)
testes/
  test_ativacoes.py     derivadas x diferença finita
  test_gradiente.py     backprop x gradiente numérico
  test_treino.py        XOR/AND/OR: o erro cai; regras do SGD
  test_persistencia.py  salvar -> carregar -> saídas idênticas
  test_dataset.py       normalização só no treino; seed -> mesma divisão; CSV
  test_inicializacao.py seed -> mesmos pesos; escala Xavier/He
  test_previsao.py      nº errado de entradas -> erro claro (também via CLI)
  test_web.py           API da interface web (JSON sem NaN, erro 400, teste só no fim, Parar)
dados/
  gerar_exemplos.py     gera os CSVs abaixo (seed fixa)
  xor.csv, and.csv, or.csv
  seno.csv              regressão: y = sen(x)
  tres_classes.csv      classificação de 3 classes (one-hot)
modelos/                redes salvas (.json) e históricos de MSE (.csv)
```

### Como as entidades se relacionam

```
RedeNeural ──tem──> [Camada, Camada, ..., Camada de saída]
Camada     ──tem──> [Neuronio, Neuronio, ...] + função de ativação
Neuronio   ──guarda─> pesos, bias, entradas, z, saída, delta
```

* **Neurônio** (`rna/neuronio.py`): calcula `z = w·x + b` e `a = f(z)`. Também guarda `x`,
  `z`, `a` e o `delta = ∂E/∂z` do último exemplo, e sabe se atualizar
  (`w -= η·δ·x`, `b -= η·δ`).
* **Camada** (`rna/camada.py`): quantidade configurável de neurônios e uma ativação própria.
  Recebe as saídas da camada anterior e entrega as suas para a seguinte.
* **RedeNeural** (`rna/rede.py`): `RedeNeural(n_entradas, ocultas=[(n, ativ), ...], n_saidas,
  ativacao_saida, seed)`. Aceita qualquer quantidade de camadas ocultas, inclusive zero.
  Oferece `propagar` (forward), `treinar` (delega para `treino.Treinador`) e `executar`
  (previsão).

---

## 3. Como usar

Há três formas de usar o sistema: a interface web, o menu interativo e os subcomandos. As
três chamam as mesmas funções do pacote `rna`.

### 3.0 Interface web (recomendada para a demonstração)

```bash
python main.py web
```

O comando abre `http://127.0.0.1:8000` no navegador. Use `--porta 9000` para escolher outra
porta (se a escolhida estiver ocupada, o servidor tenta as seguintes) e `--sem-navegador` para
não abrir o navegador automaticamente. Para encerrar, aperte Ctrl+C.

* **Sem dependências extras**: o servidor usa só a biblioteca padrão (`http.server`), e a
  página é HTML/CSS/JS puro, sem CDN, então funciona offline.
* **Mesmo código da CLI**: o navegador só desenha a interface. Criar a rede, dividir,
  normalizar, treinar (backpropagation + SGD), testar, salvar e prever são feitos pelo pacote
  `rna`, em Python.
* **Seções da página**:
  1. **Dados**: escolha ou envie um CSV. Clicar numa coluna alterna o papel dela entre
     entrada, saída e ignorada. Há também os controles da divisão treino/validação/teste, da
     seed e da normalização.
  2. **Arquitetura**: adicione ou remova camadas ocultas e escolha neurônios e ativação de
     cada uma. O diagrama mostra os pesos: a cor indica o sinal e a espessura indica o módulo.
  3. **Treino**: épocas, taxa de aprendizado, embaralhamento e "continuar da rede atual". O
     MSE de treino e de validação aparece ao vivo, época a época. A fronteira de decisão (2
     entradas), a curva aprendida (1 entrada) ou o gráfico previsto × esperado são atualizados
     durante o treino. O botão **Parar** interrompe e mantém os pesos.
  4. **Resultados**: MSE inicial e final de treino, MSE de validação, MSE de **teste**
     (calculado só no fim) e acurácia, quando é classificação.
  5. **Previsão**: um campo por entrada, ou texto livre como `6, 30`. Mostra a entrada
     normalizada e as saídas. Com a quantidade errada de valores, aparece o erro claro.
  6. **Modelos**: salvar em `modelos/<nome>.json`, carregar e baixar. Um modelo salvo pela
     web abre normalmente na CLI (`python main.py info -m modelos/<nome>.json`).
* Recarregar a página não perde nada, porque o estado fica no servidor. O tema claro/escuro
  segue o sistema ou o botão no topo. A página também funciona no celular e respeita a
  opção "reduzir movimento".

### 3.1 Menu interativo

```bash
python main.py
```

```
  1) Criar rede
  2) Carregar rede (JSON)
  3) Mostrar rede
  4) Carregar dataset (CSV)
  5) Escolher colunas de entrada e saída
  6) Dividir dados (treino/validação/teste)
  7) Configurar treino (épocas, taxa, seed, normalização)
  8) Treinar (mostra o MSE por época)
  9) Testar (MSE no conjunto de teste)
 10) Salvar rede
 11) Prever com novos valores
  0) Sair
```

Ordem típica: 1 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11. Em cada pergunta, apertar Enter
aceita o valor padrão mostrado entre colchetes.

### 3.2 Linha de comando (subcomandos)

`python main.py <comando> -h` mostra todas as opções de um comando.

| Comando | O que faz |
|---|---|
| `criar` | cria uma rede (entradas, camadas ocultas, neurônios, ativações, seed) e salva em JSON |
| `info` | mostra a arquitetura, a normalização e o último treino (`--pesos` mostra pesos e vieses) |
| `colunas` | lista as colunas de um CSV com mín/máx/média |
| `dividir` | divide os dados em treino/validação/teste (`--pasta` grava os 3 CSVs) |
| `treinar` | divide, normaliza, treina com SGD mostrando o MSE por época, calcula o MSE de teste e salva |
| `avaliar` | MSE (e acurácia, se for classificação) de uma rede treinada sobre um CSV |
| `prever` | executa a rede treinada sobre novos valores |
| `interativo` | abre o menu (o mesmo que rodar sem argumentos) |

**Criar uma rede.** Cada camada é escrita como `NEURONIOS:ATIVACAO`. As camadas ocultas vão
separadas por vírgula, e omitir `--ocultas` cria uma rede sem camadas ocultas.

```bash
python main.py criar --entradas 2 --ocultas 4:tanh --saidas 1:sigmoid --seed 42 -o modelos/xor.json
python main.py criar --entradas 5 --ocultas 16:relu,8:tanh --saidas 3:sigmoid --seed 1 -o modelos/grande.json
```

Ativações disponíveis: `linear`, `sigmoid`, `tanh`, `relu`. Cada camada pode usar uma diferente.

**Treinar.**

```bash
python main.py treinar -m modelos/xor.json --csv dados/xor.csv --x x1,x2 --y y --divisao 100,0,0 --epocas 2000 --taxa 0.5 --seed 42 --mostrar-a-cada 250 -o modelos/xor_treinado.json
```

| Opção | Significado | Padrão |
|---|---|---|
| `--x` / `--y` | colunas de entrada / de saída esperada (nomes ou índices a partir de 0) | obrigatório |
| `--divisao` | percentuais treino,validação,teste (somam 100) | `70,15,15` |
| `--normalizacao` | `nenhuma`, `minmax` ou `zscore` (só nas entradas) | `nenhuma` |
| `--epocas` | número de épocas | `1000` |
| `--taxa` | taxa de aprendizado η | `0.1` |
| `--seed` | seed da divisão e do embaralhamento | aleatória |
| `--sem-embaralhar` | não embaralha o treino a cada época | embaralha |
| `--mostrar-a-cada N` | exibe o MSE a cada N épocas (o MSE é calculado em todas) | `1` |
| `--historico` | grava em CSV o MSE de treino e de validação de todas as épocas | — |
| `-o` | onde salvar a rede treinada | sobrescreve o `-m` |

**Prever.** A opção `--valores` pode ser repetida. Para valores negativos, use `=`, por exemplo
`--valores=-1.5,2`.

```bash
python main.py prever -m modelos/xor_treinado.json --valores 1,0 --valores 1,1
```

**Avaliar com um CSV.** Se `--x`/`--y` forem omitidos, o comando usa as colunas salvas no
modelo durante o treino.

```bash
python main.py avaliar -m modelos/classes_treinado.json --csv dados/tres_classes.csv
```

---

## 4. Roteiro de demonstração (para o vídeo)

Os comandos abaixo foram executados exatamente assim, a partir da raiz do projeto.

```bash
# 0) Testes automatizados
python -m unittest discover -s testes -v

# 1) XOR: problema não linear com 4 exemplos (100% treino)
python main.py criar --entradas 2 --ocultas 4:tanh --saidas 1:sigmoid --seed 42 -o modelos/xor.json
python main.py info -m modelos/xor.json
python main.py treinar -m modelos/xor.json --csv dados/xor.csv --x x1,x2 --y y --divisao 100,0,0 --epocas 2000 --taxa 0.5 --seed 42 --mostrar-a-cada 250 -o modelos/xor_treinado.json
python main.py prever -m modelos/xor_treinado.json --valores 0,0 --valores 0,1 --valores 1,0 --valores 1,1
python main.py prever -m modelos/xor_treinado.json --valores 1,0,1

# 2) Classificação de 3 classes: divisão 70/15/15 + Z-score
python main.py colunas dados/tres_classes.csv
python main.py dividir --csv dados/tres_classes.csv --x x1,x2 --y classe_a,classe_b,classe_c --divisao 70,15,15 --seed 42
python main.py criar --entradas 2 --ocultas 8:tanh --saidas 3:sigmoid --seed 7 -o modelos/classes.json
python main.py treinar -m modelos/classes.json --csv dados/tres_classes.csv --x x1,x2 --y classe_a,classe_b,classe_c --divisao 70,15,15 --normalizacao zscore --epocas 200 --taxa 0.1 --seed 42 --mostrar-a-cada 25 --historico modelos/historico_classes.csv -o modelos/classes_treinado.json
python main.py info -m modelos/classes_treinado.json
python main.py prever -m modelos/classes_treinado.json --valores 6,30 --valores 2,20 --valores 4,70
python main.py avaliar -m modelos/classes_treinado.json --csv dados/tres_classes.csv

# 3) Regressão (y = sen x): saída linear + Z-score
python main.py criar --entradas 1 --ocultas 10:tanh --saidas 1:linear --seed 3 -o modelos/seno.json
python main.py treinar -m modelos/seno.json --csv dados/seno.csv --x x --y y --normalizacao zscore --epocas 300 --taxa 0.05 --seed 1 --mostrar-a-cada 50 -o modelos/seno_treinado.json
python main.py prever -m modelos/seno_treinado.json --valores=-1.5708 --valores 0.5236

# 4) Menu interativo
python main.py

# 5) Interface web (Ctrl+C para encerrar)
python main.py web
```

Roteiro sugerido na web:

1. Escolha `tres_classes.csv` (as colunas e uma configuração inicial são sugeridas).
2. Clique em **Criar rede** e mostre o diagrama.
3. Clique em **Treinar** e mostre o MSE caindo e a fronteira de decisão se formando.
4. Mostre os resultados de teste.
5. Faça uma previsão e depois a mesma previsão com `1,2,3`, para mostrar o erro.
6. Salve o modelo e abra com `python main.py info`.

Depois, repita com `xor.csv` (divisão 100/0/0) e `seno.csv` (curva de regressão).

Sugestões de pontos para mostrar no vídeo:

* o MSE de treino caindo a cada época, e o de validação acompanhando;
* no exemplo de 3 classes, a validação para de melhorar por volta da época 125, um sinal
  inicial de sobreajuste;
* o JSON salvo (`modelos/classes_treinado.json`), com a arquitetura, os pesos, os vieses e
  os parâmetros da normalização calculados só no treino;
* a previsão aplicando a mesma normalização (a linha "Entrada normalizada") e o erro claro
  quando a quantidade de valores está errada.

---

## 5. Convenções e decisões de implementação

### 5.1 Função de erro (convenção escolhida)

* **Custo usado para derivar o gradiente**, por exemplo de treino:
  `E = ½ · Σ_k (y_k − a_k)²`.
  Com o fator ½, `∂E/∂a_k = (a_k − y_k)` e o fator 2 desaparece.
* **Métrica exibida** (treino, validação e teste):
  `MSE = 1/(N·m) · Σ_exemplos Σ_saídas (y − a)²`, ou seja, a média sobre todos os exemplos e
  todas as saídas, **sem** o ½.

### 5.2 Retropropagação

```
δ_saida_k   = (a_k − y_k) · f'(z_k)
δ_oculta_j  = f'(z_j) · Σ_k (w_kj · δ_k)        (k = neurônios da camada seguinte)
∂E/∂w_ji    = δ_j · entrada_i                   (entrada que o neurônio recebeu no forward)
∂E/∂b_j     = δ_j
w ← w − η · ∂E/∂w        b ← b − η · ∂E/∂b
```

Em `rna/treino.py`, isso é feito em **duas funções separadas**:

1. `retropropagar(rede, y)` calcula **todos** os deltas, da saída para a primeira camada, sem
   alterar nenhum peso. Por isso os deltas das ocultas usam os **mesmos pesos da propagação
   direta** desse exemplo.
2. `atualizar_pesos(rede, η)` só é chamada depois e aplica a descida do gradiente em todos os
   neurônios.

Cada neurônio guarda a entrada exata que recebeu no forward, e o gradiente do peso usa essa
cópia.

### 5.3 Treinamento

* **SGD puro**: os pesos são atualizados **após cada exemplo** de treino. É a única estratégia.
* Só o conjunto de **treino** ajusta os pesos. O treino é embaralhado a cada época com um
  gerador criado pela seed (`--sem-embaralhar` desliga isso).
* **MSE por época**: ao fim de cada época, o sistema faz uma propagação direta completa sobre
  todo o treino e toda a validação, já com os pesos da época atualizados. Não é uma média
  acumulada durante a época. A "época 0" mostra o MSE antes do treino.
* A **validação nunca altera pesos** (só propagação direta).
* O **teste** só é avaliado no fim (comando `treinar` ou opção 9 do menu).

### 5.4 Inicialização

* **Xavier/Glorot uniforme** para Linear, Sigmoid e Tanh:
  `w ~ U(−L, L)`, com `L = √(6 / (fan_in + fan_out))`.
* **He normal** para ReLU: `w ~ N(0, √(2 / fan_in))`.
* **Vieses começam em zero.** Eles são guardados em cada neurônio e no JSON junto com os pesos.
* **Seeds**: existem duas, com geradores independentes.
  * `criar --seed` controla os pesos iniciais. Mesma seed e mesma arquitetura geram os mesmos
    pesos.
  * `treinar --seed` controla a divisão dos dados e o embaralhamento das épocas. Como a divisão
    tem gerador próprio, ela não muda quando a arquitetura muda.

### 5.5 Dados

* **Ordem obrigatória**: primeiro divide, depois normaliza. Os parâmetros (mín/máx ou
  média/desvio) são calculados **só no treino** e aplicados aos três subconjuntos, e depois
  às previsões.
* **Min-Max**: `x' = (x − mín) / (máx − mín)`. **Z-score**: `x' = (x − média) / desvio`, com
  desvio populacional (`ddof = 0`). Em colunas constantes, o divisor vira 1 para evitar
  divisão por zero.
* As **saídas esperadas não são normalizadas**.
* **Divisão**: as linhas são embaralhadas com a seed e as quantidades vêm dos percentuais pelo
  método do maior resto. A soma dá exatamente N, nenhuma linha se perde ou se repete, e um
  conjunto com 0% fica vazio. Validação ou teste com 0% são permitidos; nesse caso o MSE
  aparece como "—".
* **Compatibilidade**: o sistema dá erro se o número de colunas de entrada ou de saída não bate
  com a rede. Também avisa, sem bloquear, quando a ativação da saída não alcança a faixa dos
  alvos, por exemplo sigmoid com alvos fora de [0, 1].

### 5.6 Persistência (JSON)

```json
{
  "formato": "rna-mlp", "versao": 1,
  "n_entradas": 2, "n_camadas": 2, "n_camadas_ocultas": 1,
  "neuronios_por_camada": [8, 3],
  "seed_inicializacao": 7,
  "camadas": [
    {"tipo": "oculta", "n_neuronios": 8, "ativacao": "tanh", "inicializacao": "xavier",
     "pesos": [[...], ...], "vieses": [...]},
    {"tipo": "saida", "n_neuronios": 3, "ativacao": "sigmoid", "inicializacao": "xavier",
     "pesos": [[...], ...], "vieses": [...]}
  ],
  "normalizacao": {"tipo": "zscore", "parametros": {"media": [...], "desvio": [...]}},
  "metadados": {"colunas_entrada": [...], "colunas_saida": [...], "treino": {...}}
}
```

`pesos[i]` são os pesos do neurônio `i` da camada, um para cada entrada. Os números são
gravados com a precisão completa do Python, então a rede carregada produz **exatamente** as
mesmas saídas (há um teste para isso).

### 5.7 Previsão

`previsao.prever` confere se a quantidade de valores é igual ao número de entradas. Se não
for, levanta `ErroEntrada` com uma mensagem clara, e na linha de comando sai com código 1,
sem traceback. Em seguida aplica a normalização salva e faz **só** a propagação direta, sem
alterar pesos nem vieses.

---

## 6. Suposições

* **Linguagem**: Python 3 (3.10+). Os nomes no código estão em português, seguindo a
  organização sugerida no enunciado.
* **Formato do CSV**: primeira linha com cabeçalho, separador vírgula (outro pode ser escolhido
  com `--separador`), ponto como separador decimal, apenas valores numéricos e sem valores
  faltantes. Linhas em branco são ignoradas. O arquivo pode ter BOM (UTF-8 do Excel).
* **Classificação** é representada em **one-hot**, com uma coluna de saída por classe (ex.:
  `classe_a,classe_b,classe_c`). A classe prevista é a saída de maior valor. A acurácia
  aparece como métrica extra, mas o erro oficial é o MSE.
* **Rede densa**: cada neurônio recebe todas as saídas da camada anterior.
* **Testes** usam `unittest`, da biblioteca padrão, para não exigir dependências extras.
  Eles também rodam com `pytest`, se estiver instalado.
* O desempenho privilegia a clareza: o forward e o backward percorrem neurônio por neurônio,
  como na modelagem pedida. Os datasets de exemplo são pequenos (até ~200 linhas) e treinam
  em poucos segundos.

---

## 7. Testes

```bash
python -m unittest discover -s testes -v
```

| Teste pedido no enunciado | Onde |
|---|---|
| 1. Derivada de cada ativação x diferença finita | `testes/test_ativacoes.py` |
| 2. Gradiente da rede x gradiente numérico (rede pequena, 2 ocultas, ativações mistas, ReLU, sem oculta) | `testes/test_gradiente.py` |
| 3. Treinar XOR / AND / OR e mostrar que o erro cai | `testes/test_treino.py` |
| 4. Salvar, carregar e comparar saídas (idênticas, com `array_equal`) | `testes/test_persistencia.py` |
| 5. Normalização usa só o treino (inclusive os parâmetros salvos no JSON) | `testes/test_dataset.py` |
| 6. Mesma seed = mesmos pesos iniciais e mesma divisão | `testes/test_inicializacao.py`, `testes/test_dataset.py` |
| 7. Previsão com número errado de entradas = erro claro (API e CLI) | `testes/test_previsao.py` |
| Extra: API da interface web | `testes/test_web.py` |

Testes extras:

* os deltas são calculados com os pesos **antigos**: um passo de SGD é refeito com matrizes
  e comparado à implementação;
* uma época sem embaralhar é igual a N passos de SGD em sequência;
* a validação não altera pesos;
* divisão sem perder linhas;
* colunas constantes;
* erros de CSV.

---

## 8. Regenerar os dados de exemplo

```bash
python dados/gerar_exemplos.py
```
