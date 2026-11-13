# Explicação do projeto, do zero

Este guia é para quem **não sabe nada** de redes neurais. Ele explica os conceitos com
exemplos e números, mostra onde cada coisa está no código e termina com um roteiro para o
vídeo e as perguntas que o professor pode fazer.

> Sugestão de leitura: seções 1 a 10 para entender a ideia (uns 40 minutos). Seções 11 a 16
> para entender o código. Seções 17 e 18 antes de gravar o vídeo.

---

## Sumário

1. [A ideia em uma frase](#1-a-ideia-em-uma-frase)
2. [O neurônio artificial](#2-o-neurônio-artificial)
3. [Funções de ativação](#3-funções-de-ativação)
4. [Camadas: o que é um MLP](#4-camadas-o-que-é-um-mlp)
5. [Propagação direta (forward)](#5-propagação-direta-forward)
6. [Medindo o erro: MSE](#6-medindo-o-erro-mse)
7. [Como a rede aprende: descida do gradiente](#7-como-a-rede-aprende-descida-do-gradiente)
8. [Backpropagation com números, passo a passo](#8-backpropagation-com-números-passo-a-passo)
9. [SGD e épocas](#9-sgd-e-épocas)
10. [Treino, validação e teste](#10-treino-validação-e-teste)
11. [Normalização](#11-normalização)
12. [Inicialização dos pesos e a seed](#12-inicialização-dos-pesos-e-a-seed)
13. [Salvar, carregar e prever](#13-salvar-carregar-e-prever)
14. [Mapa do código](#14-mapa-do-código)
15. [A interface web por dentro](#15-a-interface-web-por-dentro)
16. [Os testes e o que cada um prova](#16-os-testes-e-o-que-cada-um-prova)
17. [Como ler os resultados](#17-como-ler-os-resultados)
18. [Roteiro para o vídeo](#18-roteiro-para-o-vídeo)
19. [Perguntas que o professor pode fazer](#19-perguntas-que-o-professor-pode-fazer)
20. [Glossário](#20-glossário)

---

## 1. A ideia em uma frase

Uma rede neural é uma **função cheia de números ajustáveis (os pesos)**. A rede recebe
números de entrada e devolve números de saída. **Treinar** é ajustar esses pesos,
automaticamente, até que as saídas fiquem próximas das respostas certas que conhecemos.

Exemplo: queremos que a rede aprenda o **XOR** ("ou exclusivo").

| x1 | x2 | resposta (y) |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Ninguém escreve um `if` com essa regra. Mostramos os exemplos para a rede, ela erra, mede o
quanto errou, corrige os pesos um pouquinho e repete milhares de vezes. No fim, para
`(1, 0)` ela responde `0.98`, ou seja, praticamente 1.

---

## 2. O neurônio artificial

Um neurônio faz duas contas:

```
1) soma ponderada:   z = w1·x1 + w2·x2 + ... + b
2) ativação:         a = f(z)
```

* `x1, x2, ...` são as **entradas**, os números que chegam ao neurônio;
* `w1, w2, ...` são os **pesos**, que dizem a importância de cada entrada (positivo
  estimula, negativo inibe);
* `b` é o **viés** (*bias*), um ajuste fixo que desloca o resultado;
* `f` é a **função de ativação**, que transforma `z` na saída `a`.

**Exemplo:** entradas `x = [1, 0]`, pesos `w = [0.5, -0.4]`, viés `b = 0` e ativação sigmoid.

```
z = 0.5·1 + (-0.4)·0 + 0 = 0.5
a = sigmoid(0.5) = 1 / (1 + e^-0.5) = 0.6225
```

No código, isso está em [rna/neuronio.py](rna/neuronio.py): o método `soma_ponderada`
calcula `z` e o método `ativar` calcula `a`. O neurônio também **guarda** `entradas`, `z`,
`saida` e `delta`, porque o treino precisa desses valores depois.

---

## 3. Funções de ativação

**Por que precisa delas?** Sem ativação, cada camada só faz somas e multiplicações. Empilhar
somas e multiplicações continua dando uma reta (um plano). A ativação "dobra" o espaço, e é
isso que permite aprender coisas tortas como o XOR.

| Ativação | Fórmula | Saída | Derivada | Uso típico |
|---|---|---|---|---|
| **Linear** | f(z) = z | qualquer número | 1 | saída de **regressão** (prever um valor, ex.: seno) |
| **Sigmoid** | 1 / (1 + e^-z) | entre 0 e 1 | f(z)·(1 − f(z)) | saída de **classificação** (probabilidade) |
| **Tanh** | tanh(z) | entre −1 e 1 | 1 − f(z)² | camadas **ocultas** |
| **ReLU** | max(0, z) | 0 ou positivo | 1 se z > 0, senão 0 | camadas **ocultas** (redes grandes) |

A **derivada** diz "se eu mexer um pouquinho em z, quanto muda a saída?". O treino depende
dela (seção 7). Cada função está em [rna/ativacoes.py](rna/ativacoes.py) com dois métodos:
`f(z)` e `derivada(z)`.

No projeto, **cada camada escolhe a sua ativação**. Por exemplo: oculta com `tanh` e saída
com `sigmoid`.

---

## 4. Camadas: o que é um MLP

MLP = *Multilayer Perceptron*, um "perceptron de várias camadas". Os neurônios são
organizados em camadas, e **cada neurônio recebe todas as saídas da camada anterior**.

```
   ENTRADA           OCULTA (4, tanh)        SAÍDA (1, sigmoid)

    x1 ●──────┬──────●  h1 ─────┐
              ├──────●  h2 ─────┤
    x2 ●──────┼──────●  h3 ─────┼────●  y
              └──────●  h4 ─────┘
```

* **Camada de entrada**: não calcula nada, só entrega os valores. Ela tem um "neurônio" por
  coluna de entrada do CSV.
* **Camadas ocultas**: ficam no meio. É nelas que a rede cria "características" intermediárias.
  Pode haver 0, 1, 2 ou mais.
* **Camada de saída**: um neurônio por valor que queremos prever.

A arquitetura acima se escreve `2 → 4 → 1`. Ela tem 4·(2+1) + 1·(4+1) = **17 parâmetros**:
cada neurônio tem um peso por entrada, mais o viés.

**Por que o XOR precisa de camada oculta?** Os pontos onde o XOR vale 1, `(0,1)` e `(1,0)`,
não podem ser separados dos pontos onde vale 0, `(0,0)` e `(1,1)`, com uma única reta. Uma
rede sem camada oculta só desenha retas. Com camada oculta, ela combina várias retas
"dobradas" e consegue. Já o AND e o OR são separáveis por uma reta, e os testes treinam os
dois **sem** camada oculta.

No código:

* [rna/camada.py](rna/camada.py) define a classe `Camada`, que é uma lista de `Neuronio` com
  a mesma ativação;
* [rna/rede.py](rna/rede.py) define a classe `RedeNeural`, que é uma lista de `Camada`.

Criar a rede do desenho fica assim:

```python
RedeNeural(n_entradas=2, ocultas=[(4, "tanh")], n_saidas=1, ativacao_saida="sigmoid", seed=42)
```

---

## 5. Propagação direta (forward)

É passar a entrada pela rede, camada por camada, até sair o resultado. A saída de uma camada
vira a entrada da próxima.

```
x  →  [camada 1]  →  h  →  [camada 2]  →  saída
```

No código, `RedeNeural.propagar(x)` faz um laço simples:

```python
saida = x
for camada in self.camadas:
    saida = camada.propagar(saida)   # cada neurônio da camada calcula z e a
return saida
```

O forward funciona para **qualquer** número de camadas e neurônios. É por isso que a
arquitetura não é fixa no código.

---

## 6. Medindo o erro: MSE

Para aprender, a rede precisa saber **o quanto errou**. Usamos o **erro quadrático médio**
(MSE, *Mean Squared Error*):

```
MSE = média de (resposta certa − resposta da rede)²
```

**Exemplo:** a resposta certa é 1 e a rede disse 0.58. O erro desse exemplo é
(1 − 0.58)² = 0.176. O MSE é a média disso sobre todos os exemplos (e todas as saídas).

* Elevar ao quadrado deixa o erro sempre positivo e pune mais os erros grandes.
* **Quanto menor o MSE, melhor.** MSE = 0 seria acertar tudo exatamente.

**A convenção do ½ (está no README):** para calcular o gradiente, usamos o erro de um exemplo
como `E = ½·Σ(y − a)²`. O ½ existe só para simplificar a derivada: a derivada de ½·(y − a)²
em relação a `a` é `(a − y)`, sem o "2" na frente. O número **mostrado na tela** é o MSE
normal, sem o ½. Os dois métodos levam aos mesmos pesos, a diferença é só de escala.

No código: [rna/erro.py](rna/erro.py), funções `mse`, `mse_rede` e `erro_exemplo`.

---

## 7. Como a rede aprende: descida do gradiente

**Analogia:** você está numa montanha, no escuro, e quer chegar ao vale (o erro mínimo). Você
não vê o caminho, mas sente a inclinação do chão sob os pés. Então dá um passo na direção que
desce. Repetindo isso, você chega lá embaixo.

* A "posição" na montanha são os **pesos**.
* A "altura" é o **erro**.
* A "inclinação" é o **gradiente**, ou seja, a derivada do erro em relação a cada peso.
* O "tamanho do passo" é a **taxa de aprendizado** (η, *eta*).

A regra de atualização de cada peso é:

```
peso_novo = peso_antigo − η · (derivada do erro em relação a esse peso)
```

* **Taxa muito pequena:** aprende devagar demais.
* **Taxa muito grande:** passa do vale, oscila ou explode.
* Valores comuns ficam entre 0.01 e 0.5. Na interface, a taxa é um controle deslizante.

A pergunta difícil é: **como calcular a derivada do erro em relação a cada peso, inclusive os
das camadas do meio?** A resposta é o *backpropagation*.

---

## 8. Backpropagation com números, passo a passo

*Backpropagation* (retropropagação) usa a **regra da cadeia** do cálculo para levar a
"culpa" pelo erro da saída de volta até cada neurônio. Essa culpa se chama **delta (δ)**.

### As três fórmulas que importam

```
δ_saída      = (a − y) · f'(z)                        ← quanto o neurônio de saída errou
δ_oculto_j   = f'(z_j) · Σ_k ( w_kj · δ_k )           ← culpa repassada pelos pesos que saem dele
gradiente    = δ · (entrada que chegou pelo peso)       e   gradiente do viés = δ
```

Leitura em português:

* **Saída:** a culpa é o erro `(a − y)` vezes a sensibilidade da ativação `f'(z)`.
* **Oculta:** o neurônio oculto é culpado na proporção dos pesos que o ligam aos neurônios
  seguintes, `w_kj`, multiplicados pela culpa deles, `δ_k`. Por isso o nome: a culpa volta
  para trás.
* **Atualização:** cada peso muda proporcionalmente ao delta do neurônio e ao valor que passou
  por aquele peso. Se a entrada foi 0, o peso não teve culpa e não muda.

### Exemplo completo: rede 2 → 2 → 1, tudo sigmoid

Os números foram calculados com o próprio código do projeto.

**Situação inicial**

* Entrada `x = [1, 0]` e resposta certa `y = 1`. Taxa de aprendizado `η = 0.5`.
* Neurônio oculto h1: pesos `[0.5, −0.4]`, viés 0.
* Neurônio oculto h2: pesos `[0.3, 0.8]`, viés 0.
* Neurônio de saída: pesos `[0.7, −0.2]` (vindos de h1 e h2), viés 0.

**Passo 1, forward**

```
h1:    z = 0.5·1 + (−0.4)·0      = 0.5000   →  a = sigmoid(0.5)    = 0.6225
h2:    z = 0.3·1 +   0.8·0       = 0.3000   →  a = sigmoid(0.3)    = 0.5744
saída: z = 0.7·0.6225 + (−0.2)·0.5744 = 0.3208  →  a = sigmoid(0.3208) = 0.5795
```

A rede disse **0.5795**, mas o certo era **1**. Erro do exemplo: `E = ½·(1 − 0.5795)² = 0.0884`.

**Passo 2, todos os deltas (antes de mudar qualquer peso)**

```
δ_saída = (a − y) · a·(1 − a)
        = (0.5795 − 1) · 0.5795·0.4205   = −0.1025

δ_h1 = a_h1·(1 − a_h1) · (w_saída←h1 · δ_saída)
     = 0.6225·0.3775 · (0.7 · −0.1025)    = −0.01685

δ_h2 = 0.5744·0.4256 · (−0.2 · −0.1025)   = +0.00501
```

O delta da saída é negativo porque a saída ficou **abaixo** do certo. O h1 ajudou a saída
(peso +0.7), então também precisa "subir". O h2 atrapalhou (peso −0.2), então o sinal dele
se inverte.

**Passo 3, atualização** (`peso −= η · δ · entrada`)

```
saída:  w←h1 = 0.7  − 0.5·(−0.1025)·0.6225 = 0.7319
        w←h2 = −0.2 − 0.5·(−0.1025)·0.5744 = −0.1706
        viés = 0    − 0.5·(−0.1025)         = 0.0512
h1:     w1   = 0.5  − 0.5·(−0.01685)·1     = 0.5084
        w2   = −0.4 − 0.5·(−0.01685)·0     = −0.4     (entrada 0: não muda)
        viés = 0    − 0.5·(−0.01685)        = 0.0084
h2:     w1   = 0.3  − 0.5·(0.00501)·1      = 0.2975
        viés = 0    − 0.5·(0.00501)         = −0.0025
```

**Resultado:** com os pesos novos, a mesma entrada dá **0.6015** (antes era 0.5795) e o erro
cai de **0.0884 para 0.0794**. Isso foi **um passo**. O treino repete esse passo milhares de
vezes.

### A regra mais importante: todos os deltas antes de qualquer atualização

O cálculo do `δ_h1` usou o peso **antigo** da saída (0.7). Se tivéssemos atualizado a saída
antes, o h1 usaria 0.7319 e o gradiente ficaria errado. Por isso o código separa as duas
etapas em funções diferentes, em [rna/treino.py](rna/treino.py):

1. `retropropagar(rede, y)` calcula **todos** os deltas, da saída para trás, sem tocar em pesos;
2. `atualizar_pesos(rede, taxa)` só roda depois e aplica a descida do gradiente.

Existe um teste que confere exatamente isso (seção 16).

---

## 9. SGD e épocas

* **SGD puro** (*Stochastic Gradient Descent*): os pesos são atualizados **depois de cada
  exemplo**, sem esperar o conjunto inteiro. É a estratégia exigida no trabalho e a única
  implementada.
* **Época:** uma passada completa por todos os exemplos de treino. Em 300 épocas, a rede vê
  cada exemplo 300 vezes.
* **Embaralhar:** a cada época a ordem dos exemplos é sorteada (com a seed), para a rede não
  "decorar a sequência".

Em código ([rna/treino.py](rna/treino.py), classe `Treinador`), o treino é este laço:

```
para cada época:
    para cada exemplo (x, y) do treino, em ordem embaralhada:
        forward(x)            → calcula as saídas e guarda z e a
        retropropagar(y)      → calcula todos os deltas
        atualizar_pesos(η)    → corrige pesos e vieses
    calcular MSE de treino e de validação (só forward) e mostrar na tela
```

---

## 10. Treino, validação e teste

Os dados são divididos aleatoriamente em três partes. Por padrão, 70%, 15% e 15%.

**Analogia da escola:**

| Conjunto | Analogia | Para que serve | Mexe nos pesos? |
|---|---|---|---|
| **Treino** | exercícios resolvidos | a rede aprende com eles | **sim** |
| **Validação** | simulado | acompanhar, a cada época, se está aprendendo de verdade | não |
| **Teste** | prova final | nota final, medida uma única vez, no fim | não |

**Por que separar?** Uma rede pode **decorar** o treino (*overfitting*, ou sobreajuste) e ir
mal em dados novos. Se o MSE de treino continua caindo, mas o de **validação** para de cair ou
começa a subir, a rede está decorando. No dataset `tres_classes.csv` dá para ver isso por
volta da época 125: o treino continua melhorando e a validação para.

**O teste fica guardado até o fim.** Se olhássemos o teste durante o treino para tomar
decisões, ele deixaria de ser uma prova "surpresa" e a nota final ficaria otimista demais.

No código: `dividir(...)` em [rna/dataset.py](rna/dataset.py). Os percentuais precisam
somar 100, e a seed garante a mesma divisão sempre.

---

## 11. Normalização

**O problema:** no `tres_classes.csv`, `x1` vai de 0 a 8 e `x2` vai de 3 a 88. A rede
"enxerga" `x2` como muito mais importante só porque os números dele são maiores, e o treino
fica lento e instável.

**A solução** é colocar as entradas numa escala parecida:

* **Min-Max:** `x' = (x − mín) / (máx − mín)` deixa os valores do treino entre 0 e 1;
* **Z-score:** `x' = (x − média) / desvio_padrão` deixa média 0 e desvio 1.

**Regra de ouro (exigida no trabalho):** primeiro **divide**, depois **normaliza**. Os
parâmetros (mín/máx, ou média/desvio) são calculados **só com o treino** e depois aplicados
na validação, no teste e nas previsões.

Por quê? Se calculássemos a média usando o teste, a rede receberia "informação da prova"
durante o treino. Isso se chama **vazamento de dados**, e a avaliação deixa de ser honesta.

* As **saídas esperadas não são normalizadas**, só as entradas.
* Os parâmetros ficam **salvos no JSON**. Quando você pede uma previsão com `x1=6, x2=30`, o
  sistema aplica a mesma conta antes (`[1.0411, −0.4209]` com Z-score).

No código: a classe `Normalizador` e a função `normalizar_divisao`, em
[rna/dataset.py](rna/dataset.py).

---

## 12. Inicialização dos pesos e a seed

**Por que não começar tudo em zero?** Se todos os pesos forem iguais, todos os neurônios de
uma camada calculam a mesma coisa e recebem a mesma correção. Eles nunca se diferenciam, e
seria como ter um neurônio só. Por isso os pesos começam **aleatórios**, mas pequenos e numa
escala bem escolhida:

* **Xavier/Glorot** (para Linear, Sigmoid e Tanh): sorteio uniforme em
  `±√(6 / (entradas + saídas))`;
* **He** (para ReLU): sorteio normal com desvio `√(2 / entradas)`. A ReLU "desliga" metade
  dos valores, então ela precisa de pesos um pouco maiores.

Os **vieses começam em zero**, porque os pesos aleatórios já quebram a simetria.

**Seed (semente):** é o número que inicia o sorteio. **A mesma seed gera sempre os mesmos
números**, então o experimento pode ser repetido. O projeto usa duas seeds:

* a seed do `criar` controla os **pesos iniciais**;
* a seed do `treinar` controla a **divisão dos dados** e o **embaralhamento**.

No código: [rna/inicializacao.py](rna/inicializacao.py).

---

## 13. Salvar, carregar e prever

**Salvar** grava tudo em um arquivo **JSON**, um texto legível que você pode abrir no VS
Code. O arquivo contém o número de entradas, as camadas com quantidade de neurônios e
ativação, todos os pesos e vieses, e a normalização com os parâmetros calculados no treino.
Veja um exemplo em `modelos/classes_treinado.json` depois de rodar o roteiro do README.

**Carregar** reconstrói a rede igualzinha. Há um teste que confere que as saídas são
**exatamente** iguais, bit a bit.

**Prever** só faz o forward, sem treinar e sem mexer em pesos. Antes, o sistema:

1. confere se você passou a quantidade certa de valores. Se a rede espera 2 e você passou 3,
   aparece um **erro claro**;
2. aplica a normalização salva.

No código: [rna/persistencia.py](rna/persistencia.py) e [rna/previsao.py](rna/previsao.py).

---

## 14. Mapa do código

```
main.py              ← linha de comando (subcomandos + menu interativo + "web")
rna/                 ← TODO o algoritmo de rede neural (só numpy)
  ativacoes.py       ← Linear, Sigmoid, Tanh, ReLU e derivadas                     (seção 3)
  inicializacao.py   ← Xavier, He, seed                                            (seção 12)
  neuronio.py        ← Neurônio: z = w·x + b, a = f(z), guarda z/saída/delta       (seção 2)
  camada.py          ← Camada: lista de neurônios com a mesma ativação             (seção 4)
  rede.py            ← RedeNeural: lista de camadas, forward                       (seções 4–5)
  erro.py            ← MSE                                                         (seção 6)
  dataset.py         ← lê CSV, escolhe colunas, divide, normaliza                  (seções 10–11)
  treino.py          ← backpropagation + SGD + épocas                              (seções 7–9)
  persistencia.py    ← salvar/carregar JSON                                        (seção 13)
  previsao.py        ← previsão com validação e normalização salva                 (seção 13)
  experimento.py     ← avaliação final no teste (usado pela CLI e pela web)
web/                 ← interface web (só desenha; quem calcula é o rna/)           (seção 15)
testes/              ← testes automáticos                                          (seção 16)
dados/               ← CSVs de exemplo (xor, and, or, seno, tres_classes)
modelos/             ← redes salvas em JSON
```

### O caminho de um treino pela linha de comando

```
python main.py treinar -m modelos/classes.json --csv dados/tres_classes.csv ...
```

1. `persistencia.carregar` lê a rede do JSON.
2. `Dataset.de_csv` lê o CSV e `selecionar` separa as colunas X (entradas) e Y (saídas).
3. `verificar_compatibilidade` confere se o número de colunas bate com a rede.
4. `dividir` sorteia treino, validação e teste com a seed.
5. `normalizar_divisao` calcula os parâmetros **só no treino** e aplica nos três conjuntos.
6. `rede.treinar` → `Treinador.treinar` roda as épocas de SGD (forward → deltas → atualização)
   e mostra o MSE por época.
7. `avaliar_teste` calcula o MSE de teste **uma vez**, no fim.
8. `persistencia.salvar` grava tudo no JSON.

### Por que o projeto é "genérico e configurável"?

Nada no código diz "2 entradas" ou "4 neurônios". O usuário informa a arquitetura
(`--ocultas 8:relu,4:tanh`, ou os botões da página), e os laços percorrem quantas camadas e
neurônios existirem.

---

## 15. A interface web por dentro

```
navegador (HTML/CSS/JS)  ⇄  servidor Python (web/servidor.py)  ⇄  pacote rna/
   só desenha                  recebe pedidos JSON                faz TODAS as contas
```

* `python main.py web` sobe um servidor local com a biblioteca padrão do Python
  (`http.server`), sem Flask, e abre a página.
* Quando você clica em **Treinar**, a página envia a configuração. O servidor chama
  `dividir`, `normalizar_divisao` e `rede.treinar`, **as mesmas funções da linha de comando**.
* O treino roda numa *thread* separada. A página pergunta o progresso várias vezes por segundo
  e desenha o MSE de cada época e a fronteira de decisão.
* **Fronteira de decisão** (quando há 2 entradas): a página pede ao servidor a previsão da rede
  numa grade de pontos e pinta cada ponto com a cor da classe prevista. É o "mapa" do que a
  rede aprendeu. Os pontos do dataset aparecem por cima: círculo cheio é treino, anel é
  validação e losango é teste.
* **Diagrama da rede:** cada linha é um peso. Azul é positivo, rosa é negativo, e a
  espessura mostra o tamanho do peso. Durante o treino dá para ver as linhas mudando.

O JavaScript **não implementa rede neural nenhuma**. Isso é importante, porque o trabalho exige
que o algoritmo seja nosso, e ele está todo em `rna/`.

---

## 16. Os testes e o que cada um prova

Rode com:

```bash
python -m unittest discover -s testes -v
```

O resultado esperado é `Ran 62 tests ... OK`.

| Teste exigido no PDF | Arquivo | O que prova, em português |
|---|---|---|
| 1. Derivada de cada ativação | `test_ativacoes.py` | a derivada escrita à mão é igual à inclinação medida "na marra": (f(z+ε) − f(z−ε)) / 2ε |
| 2. Gradiente da rede | `test_gradiente.py` | o backpropagation dá o mesmo resultado que mexer em cada peso ±ε e medir o erro. É a prova de que a matemática está certa |
| 3. Erro cai em XOR/AND/OR | `test_treino.py` | depois de treinar, o MSE é bem menor e a rede acerta as 4 linhas |
| 4. Salvar e carregar | `test_persistencia.py` | a rede carregada dá saídas **idênticas** |
| 5. Normalização só no treino | `test_dataset.py` | os parâmetros salvos são os do treino (um valor extremo fora do treino não influencia) |
| 6. Mesma seed | `test_inicializacao.py`, `test_dataset.py` | mesma seed = mesmos pesos e mesma divisão |
| 7. Entradas erradas | `test_previsao.py` | 3 valores para uma rede de 2 entradas gera erro claro, inclusive na linha de comando |
| Extra | `test_treino.py` | os deltas são calculados com os pesos **antigos**; uma época = N passos de SGD; a validação não altera pesos |
| Extra | `test_web.py` | a API da página funciona (JSON válido, erro 400, Parar, teste só no fim) |

O **teste 2** é o mais forte. Ele compara o gradiente calculado pelo backprop com o gradiente
numérico (força bruta). Se uma fórmula do backprop estivesse errada, os números não bateriam.

---

## 17. Como ler os resultados

Exemplo real: `tres_classes.csv`, rede `2 → 8 → 3`, Z-score e 200 épocas.

```
Época   0/200 | MSE treino 0.298255 | MSE validação 0.289326  (antes do treino)
Época   1/200 | MSE treino 0.052811 | MSE validação 0.056964
Época 100/200 | MSE treino 0.001139 | MSE validação 0.006508
Época 200/200 | MSE treino 0.000573 | MSE validação 0.006625
MSE de teste:     0.013835  (31 exemplos)
Acurácia de teste: 96.77%
```

* **Época 0:** com pesos aleatórios, o erro é alto (0.298). A rede está chutando.
* **O treino cai muito** (0.298 → 0.00057): a rede aprendeu os exemplos de treino.
* **A validação cai e depois estabiliza** (≈ 0.0065): é o limite do que dá para generalizar,
  porque as classes se sobrepõem um pouco.
* **Teste 0.0138, com 96.77% de acerto:** de 31 exemplos nunca vistos, errou 1. Essa é a
  "nota" honesta da rede.
* O MSE de **teste** ser um pouco maior que o de **validação** é normal. São poucos exemplos e
  o teste não participou de nenhuma decisão.

Para as saídas da previsão:

* **Classificação** (várias saídas sigmoid): a classe prevista é a maior saída. Ex.:
  `classe_a=0.0016, classe_b=0.9997, classe_c=0.0026` é a **classe_b**.
* **Uma saída sigmoid** (XOR): acima de 0.5 é 1, abaixo é 0.
* **Regressão** (seno, saída linear): o número é a previsão. Ex.: sen(−1.5708) ≈ −1, e a rede
  disse −1.01.

---

## 18. Roteiro para o vídeo

Sugestão de estrutura (8 a 12 minutos). Adaptem às regras do professor.

1. **Apresentação (30 s):** o objetivo é um sistema configurável de MLP, implementado à mão,
   só com numpy.
2. **Tecnologias e decisões (1 min):**
   * Python e numpy; sem TensorFlow, PyTorch ou scikit-learn;
   * convenção do erro com ½ (seção 6);
   * SGD puro (seção 9);
   * Xavier e He, com vieses em zero (seção 12);
   * divide e depois normaliza (seção 11);
   * JSON para salvar.
3. **Estrutura do código (2–3 min):** abram a pasta `rna/` e passem pelos arquivos na ordem
   do mapa (seção 14). Mostrem no código:
   * `Neuronio.ativar` (z e a);
   * `RedeNeural.propagar` (o laço do forward);
   * `retropropagar` e `atualizar_pesos`, separados (seção 8);
   * `Normalizador.ajustar`, que só recebe o treino.
4. **Testes (1 min):** rodem `python -m unittest discover -s testes -v` e expliquem os testes
   1 e 2 (seção 16).
5. **Demonstração na linha de comando (2 min):** os comandos do XOR no README (seção 4 de
   lá): criar → treinar mostrando o MSE caindo → prever → prever com 3 valores (erro).
6. **Demonstração na interface web (2–3 min):** `python main.py web`. Sigam o roteiro do
   README:
   1. escolher `tres_classes.csv`;
   2. criar a rede;
   3. treinar (gráfico e fronteira ao vivo);
   4. mostrar os resultados de teste;
   5. prever;
   6. salvar e abrir o JSON.
7. **Conclusão (30 s):** os resultados (por exemplo, 96.77% no teste) e o que aprenderam.

---

## 19. Perguntas que o professor pode fazer

**Onde está o backpropagation?**
Em `rna/treino.py`, na função `retropropagar`. Ela calcula o delta da saída, `(a − y)·f'(z)`,
e depois os das ocultas, de trás para frente, com `f'(z_j)·Σ w_kj·δ_k`.

**Como vocês garantem que os deltas usam os pesos antigos?**
Calcular os deltas (`retropropagar`) e atualizar (`atualizar_pesos`) são funções separadas. A
segunda só roda depois que a primeira terminou para todas as camadas. O teste
`test_deltas_usam_os_pesos_antes_da_atualizacao` refaz a conta com matrizes e compara.

**Por que E = ½Σ(y − a)²?**
Para a derivada sair `(a − y)`, sem o fator 2. É só uma escala: o fator pode ser absorvido
pela taxa de aprendizado. O MSE mostrado na tela não tem o ½.

**Qual a diferença entre validação e teste?**
A validação é acompanhada a cada época, para ver se a rede está generalizando ou decorando.
O teste é usado uma única vez, no fim, como nota final. Nenhum dos dois mexe nos pesos.

**Por que normalizar depois de dividir?**
Para não vazar informação do teste e da validação para o treino. Os parâmetros (mín/máx ou
média/desvio) vêm só do treino e ficam salvos no JSON para as previsões.

**Por que não inicializar os pesos com zero?**
Todos os neurônios de uma camada ficariam iguais para sempre (simetria). Xavier e He sorteiam
pesos com uma escala que evita sinais que somem ou explodem ao passar pelas camadas.

**Por que ReLU usa He e as outras usam Xavier?**
A ReLU zera metade dos valores (os negativos), então a variância cai pela metade. O He
compensa com pesos um pouco maiores (`√(2/entradas)`).

**O que é uma época? E o SGD?**
Uma época é uma passada por todo o treino. No SGD, os pesos são atualizados a cada exemplo,
e não uma vez por época.

**Como provam que o gradiente está certo?**
Com o teste de gradiente numérico. Ele mexe em cada peso ±0.00001, mede quanto o erro muda e
compara com o que o backprop calculou. Os resultados batem com precisão de 5 casas relativas.

**A rede tem arquitetura fixa?**
Não. `RedeNeural` recebe a lista de camadas ocultas como parâmetro, e o forward e o backprop
são laços sobre as camadas. A arquitetura é criada pela interface (`--ocultas 8:relu,4:tanh`
ou os botões da página).

**Usaram alguma biblioteca pronta?**
Só o numpy, para vetores e produto escalar. A leitura de CSV usa o módulo `csv` do Python e a
interface web usa `http.server`. Nenhuma biblioteca de rede neural ou de derivada automática.

**O que acontece se eu passar 3 valores para uma rede de 2 entradas?**
Aparece `Quantidade de valores incorreta: a rede espera 2 entrada(s), mas foram informados
3.`, tanto na linha de comando quanto na página.

---

## 20. Glossário

| Termo | Significado |
|---|---|
| **Peso (w)** | número que multiplica uma entrada; é o que a rede "aprende" |
| **Viés (b, bias)** | número somado no neurônio; desloca a ativação |
| **Ativação (f)** | função aplicada à soma ponderada (sigmoid, tanh, ReLU, linear) |
| **z** | soma ponderada `w·x + b`, antes da ativação |
| **a** | saída do neurônio, `f(z)` |
| **Forward** | passar a entrada pela rede até a saída |
| **MSE** | erro quadrático médio; quanto menor, melhor |
| **Gradiente** | derivadas do erro em relação aos pesos; aponta para onde o erro cresce |
| **Delta (δ)** | "culpa" de um neurônio pelo erro, `∂E/∂z` |
| **Backpropagation** | calcular os deltas da saída para trás com a regra da cadeia |
| **Taxa de aprendizado (η)** | tamanho do passo na descida do gradiente |
| **SGD** | atualizar os pesos após cada exemplo |
| **Época** | uma passada por todos os exemplos de treino |
| **Overfitting** | quando a rede decora o treino e piora em dados novos |
| **Normalização** | colocar as entradas numa escala parecida (Min-Max, Z-score) |
| **Vazamento de dados** | usar informação de validação ou teste no treino (proibido) |
| **Seed** | número que fixa o sorteio aleatório, para repetir o experimento |
| **Xavier / He** | regras para sortear os pesos iniciais na escala certa |
| **One-hot** | uma coluna por classe, com 1 na classe certa e 0 nas outras |
| **Fronteira de decisão** | o "mapa" de qual classe a rede prevê em cada ponto do espaço |
| **MLP** | *Multilayer Perceptron*: rede com camadas densas (todos ligados a todos) |
