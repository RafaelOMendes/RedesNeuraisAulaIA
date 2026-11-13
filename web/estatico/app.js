// Lógica da página: estado da interface + chamadas à API do servidor.
// Toda conta de rede neural acontece no Python (pacote rna); aqui só há UI.

import { contar, esperar, fmt, iniciarNavegacao, iniciarRevelacao, iniciarRipple, iniciarTema, toast } from "./efeitos.js";
import { iniciarFundo } from "./fundo.js";
import { Diagrama } from "./diagrama.js";
import { GraficoMSE, Visual } from "./graficos.js";

const $ = (seletor) => document.querySelector(seletor);
const $$ = (seletor) => [...document.querySelectorAll(seletor)];

// ==================================================================== //
// API
// ==================================================================== //
async function pedir(url, opcoes) {
  let resposta;
  try {
    resposta = await fetch(url, opcoes);
  } catch {
    throw new Error("Sem conexão com o servidor. Ele ainda está rodando (python main.py web)?");
  }
  let dados = null;
  try { dados = await resposta.json(); } catch { /* sem corpo JSON */ }
  if (!resposta.ok) throw new Error(dados?.erro || `Erro ${resposta.status} no servidor.`);
  return dados;
}
const api = {
  get: (url) => pedir(url),
  post: (url, corpo = {}) => pedir(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(corpo),
  }),
};

// ==================================================================== //
// Estado
// ==================================================================== //
let proximoId = 1;
const novoId = () => proximoId++;

const estado = {
  ativacoes: ["linear", "sigmoid", "tanh", "relu"],
  dataset: null,
  papeis: {},
  ocultas: [{ id: novoId(), neuronios: 6, ativacao: "tanh" }],
  ativSaida: "sigmoid",
  normalizacao: "zscore",
  rede: null,
  treinando: false,
  treinoId: 0,
};

const diagrama = new Diagrama($("#diagrama"));
const grafico = new GraficoMSE($("#grafico"), $("#dica-grafico"));
const visual = new Visual($("#visual"), { titulo: $("#titulo-visual"), dica: $("#dica-visual"), legenda: $("#legenda-visual") });
let fundo = null;

const colunasPorPapel = (papel) =>
  estado.dataset ? estado.dataset.colunas.filter((c) => estado.papeis[c] === papel) : [];
const nEntradas = () => (estado.dataset ? colunasPorPapel("entrada").length : estado.rede?.n_entradas ?? 0);
const nSaidas = () => (estado.dataset ? colunasPorPapel("saida").length : estado.rede?.n_saidas ?? 0);

function configRede() {
  const seed = $("#seed-init").value.trim();
  return {
    n_entradas: nEntradas(),
    ocultas: estado.ocultas.map(({ neuronios, ativacao }) => ({ neuronios, ativacao })),
    n_saidas: nSaidas(),
    ativacao_saida: estado.ativSaida,
    seed: seed === "" ? null : Number(seed),
  };
}
const assinatura = (c) =>
  JSON.stringify([c.n_entradas, c.ocultas.map((o) => [o.neuronios, o.ativacao]), c.n_saidas, c.ativacao_saida, c.seed]);
const assinaturaDaRede = (r) =>
  JSON.stringify([r.n_entradas, r.camadas.slice(0, -1).map((c) => [c.n_neuronios, c.ativacao]),
    r.n_saidas, r.camadas.at(-1).ativacao, r.seed_inicializacao ?? null]);

function contarParametros(c) {
  const tam = [c.n_entradas, ...c.ocultas.map((o) => o.neuronios), c.n_saidas];
  let total = 0;
  for (let i = 1; i < tam.length; i++) total += tam[i] * (tam[i - 1] + 1);
  return total;
}

async function comCarregando(botao, acao) {
  botao.classList.add("carregando");
  botao.disabled = true;
  try { return await acao(); }
  catch (erro) { toast(erro.message, "erro", 6000); }
  finally { botao.classList.remove("carregando"); botao.disabled = false; }
}

// ==================================================================== //
// 01 Dados
// ==================================================================== //
async function carregarDatasets() {
  const alvo = $("#lista-datasets");
  try {
    const lista = await api.get("/api/datasets");
    alvo.replaceChildren();
    lista.forEach((d, i) => {
      const botao = document.createElement("button");
      botao.type = "button";
      botao.className = "dataset";
      botao.dataset.nome = d.nome;
      botao.style.setProperty("--i", i);
      botao.classList.toggle("ativo", estado.dataset?.nome === d.nome);
      botao.innerHTML = '<span class="dataset-icone">CSV</span><div><div class="dataset-nome"></div><div class="dataset-meta"></div></div>';
      botao.querySelector(".dataset-nome").textContent = d.nome;
      botao.querySelector(".dataset-meta").textContent = d.erro || `${d.linhas} linhas · ${d.colunas.join(", ")}`;
      botao.addEventListener("click", () => abrirDataset(d.nome));
      alvo.appendChild(botao);
    });
    if (!lista.length) alvo.innerHTML = '<p class="vazio">Nenhum CSV na pasta dados/.</p>';
  } catch (erro) {
    alvo.innerHTML = '<p class="vazio">Não foi possível listar os datasets.</p>';
    toast(erro.message, "erro");
  }
}

function papeisPadrao(info) {
  if (info.x?.length && info.y?.length) {
    return Object.fromEntries(info.colunas.map((c) => [c, info.x.includes(c) ? "entrada" : info.y.includes(c) ? "saida" : "ignorada"]));
  }
  // Colunas binárias no fim (ex.: one-hot) viram saída; senão, só a última coluna.
  const est = info.estatisticas;
  let k = 0;
  while (k < est.length && est[est.length - 1 - k].binaria) k++;
  const nSaida = k === 0 || k === est.length ? 1 : k;
  return Object.fromEntries(info.colunas.map((c, i) => [c, i >= info.colunas.length - nSaida ? "saida" : "entrada"]));
}

async function abrirDataset(nome) {
  if (estado.treinando) { toast("Aguarde o fim do treino para trocar de dataset.", "aviso"); return; }
  try {
    definirDataset(await api.post("/api/dataset/abrir", { nome }), null, true);
  } catch (erro) { toast(erro.message, "erro"); }
}

function definirDataset(info, papeis, sugerir) {
  estado.dataset = info;
  estado.papeis = papeis || papeisPadrao(info);
  $$(".dataset").forEach((b) => b.classList.toggle("ativo", b.dataset.nome === info.nome));
  $("#dica-dataset").textContent = `${info.nome} · ${info.n_linhas} linhas`;
  if (sugerir) aplicarSugestoes(info);
  renderColunas();
  renderTabela();
  atualizarTudo();
  renderCamposPrevisao(true);
  atualizarVisual();
}

function aplicarSugestoes(info) {
  const pequeno = info.n_linhas <= 16;
  definirDivisao(pequeno ? 100 : 70, pequeno ? 0 : 15);
  definirNormalizacao(pequeno ? "nenhuma" : "zscore");
  $("#epocas").value = pequeno ? 2000 : 300;
  definirTaxa(pequeno ? 0.5 : 0.1);
  estado.ocultas = [{ id: novoId(), neuronios: pequeno ? 4 : 8, ativacao: "tanh" }];
  const saidas = info.estatisticas.filter((e) => estado.papeis[e.nome] === "saida");
  estado.ativSaida = saidas.length && saidas.every((e) => e.binaria) ? "sigmoid" : "linear";
  renderOcultas();
  renderAtivSaida();
  toast(pequeno
    ? "Dataset pequeno: sugeri divisão 100/0/0, 2000 épocas e taxa 0.5."
    : `Sugestão aplicada: 70/15/15, Z-score e saída ${estado.ativSaida}.`, "info", 3600);
}

const PAPEIS = ["entrada", "saida", "ignorada"];

function renderColunas() {
  const alvo = $("#colunas");
  alvo.replaceChildren();
  estado.dataset.colunas.forEach((nome, i) => {
    const botao = document.createElement("button");
    botao.type = "button";
    botao.className = `coluna ${estado.papeis[nome]}`;
    botao.style.setProperty("--i", i);
    botao.innerHTML = "<i></i><span></span>";
    botao.querySelector("span").textContent = nome;
    botao.title = "Clique para alternar entre entrada, saída e ignorada";
    botao.addEventListener("click", () => {
      if (estado.treinando) { toast("Aguarde o fim do treino.", "aviso"); return; }
      const atual = PAPEIS.indexOf(estado.papeis[nome]);
      estado.papeis[nome] = PAPEIS[(atual + 1) % PAPEIS.length];
      botao.className = `coluna ${estado.papeis[nome]}`;
      void botao.offsetWidth;
      botao.classList.add("estalo");
      renderTabela();
      atualizarTudo();
      renderCamposPrevisao(true);
      atualizarVisual();
    });
    alvo.appendChild(botao);
  });
}

function renderTabela() {
  const info = estado.dataset;
  const tabela = $("#amostra");
  const numero = (v) => String(Number(v.toPrecision(6)));
  const cabecalho = info.colunas.map((c) => `<th class="${estado.papeis[c]}">${escapar(c)}</th>`).join("");
  const linhas = info.amostra.map((linha, i) =>
    `<tr style="--i:${i}">${linha.map((v) => `<td>${numero(v)}</td>`).join("")}</tr>`).join("");
  const faixa = info.estatisticas.map((e) => `<td>${numero(e.min)} … ${numero(e.max)}</td>`).join("");
  tabela.innerHTML = `<thead><tr>${cabecalho}</tr></thead><tbody>${linhas}<tr class="estat">${faixa}</tr></tbody>`;
}

function escapar(texto) {
  return String(texto).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Divisão -------------------------------------------------------------
function preencherRange(input) {
  const p = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty("--p", `${p}%`);
}

function percentuais() {
  const tr = Number($("#p-treino").value), va = Number($("#p-validacao").value);
  return [tr, va, 100 - tr - va];
}

function definirDivisao(tr, va) {
  $("#p-treino").value = tr;
  $("#p-validacao").value = va;
  atualizarDivisao("treino");
}

function atualizarDivisao(origem) {
  let tr = Number($("#p-treino").value), va = Number($("#p-validacao").value);
  if (tr + va > 100) {
    if (origem === "treino") va = 100 - tr; else tr = 100 - va;
  }
  if (tr < 5) { tr = 5; va = Math.min(va, 95); }
  $("#p-treino").value = tr;
  $("#p-validacao").value = va;
  const te = 100 - tr - va;
  $("#rot-treino").textContent = `${tr}%`;
  $("#rot-validacao").textContent = `${va}%`;
  $("#rot-teste").textContent = `${te}%`;
  const segs = { treino: tr, validacao: va, teste: te };
  for (const [nome, valor] of Object.entries(segs)) {
    const seg = $(`#barra-divisao .seg.${nome}`);
    seg.style.flexGrow = Math.max(valor, 0.0001);
    seg.classList.toggle("zero", valor === 0);
  }
  preencherRange($("#p-treino"));
  preencherRange($("#p-validacao"));
}

// Normalização ----------------------------------------------------------
function definirNormalizacao(valor) {
  estado.normalizacao = valor;
  $$("#normalizacao button").forEach((b) => {
    const ativo = b.dataset.valor === valor;
    b.classList.toggle("ativo", ativo);
    b.setAttribute("aria-checked", String(ativo));
  });
  posicionarSegmentado();
}

function posicionarSegmentado() {
  const ativo = $("#normalizacao button.ativo");
  const fundoSeg = $("#normalizacao .segmentado-fundo");
  if (!ativo) return;
  fundoSeg.style.width = `${ativo.offsetWidth}px`;
  fundoSeg.style.transform = `translateX(${ativo.offsetLeft - 4}px)`;
}

// ==================================================================== //
// 02 Arquitetura
// ==================================================================== //
function renderAtivSaida() {
  const alvo = $("#ativ-saida");
  alvo.replaceChildren();
  for (const nome of estado.ativacoes) {
    const botao = document.createElement("button");
    botao.type = "button";
    botao.className = `pilula${nome === estado.ativSaida ? " ativo" : ""}`;
    botao.textContent = nome;
    botao.setAttribute("role", "radio");
    botao.setAttribute("aria-checked", String(nome === estado.ativSaida));
    botao.addEventListener("click", () => { estado.ativSaida = nome; renderAtivSaida(); atualizarTudo(); });
    alvo.appendChild(botao);
  }
}

function criarLinhaOculta(camada) {
  const li = document.createElement("li");
  li.className = "oculta";
  li.dataset.id = camada.id;
  li.innerHTML = `
    <div class="camada-rotulo"><span class="camada-tipo"></span><small>init <span class="init"></span></small></div>
    <div class="passo">
      <button type="button" data-delta="-1" aria-label="Menos neurônios">−</button>
      <output></output>
      <button type="button" data-delta="1" aria-label="Mais neurônios">+</button>
    </div>
    <div class="pilulas" role="radiogroup" aria-label="Ativação"></div>
    <button type="button" class="remover" aria-label="Remover camada">×</button>`;
  const pilulas = li.querySelector(".pilulas");
  for (const nome of estado.ativacoes) {
    const p = document.createElement("button");
    p.type = "button";
    p.className = "pilula";
    p.dataset.valor = nome;
    p.textContent = nome;
    p.addEventListener("click", () => {
      const alvo = estado.ocultas.find((o) => o.id === camada.id);
      if (alvo) { alvo.ativacao = nome; renderOcultas(); atualizarTudo(); }
    });
    pilulas.appendChild(p);
  }
  li.querySelectorAll(".passo button").forEach((b) => b.addEventListener("click", (evento) => {
    const alvo = estado.ocultas.find((o) => o.id === camada.id);
    if (!alvo) return;
    const passo = Number(b.dataset.delta) * (evento.shiftKey ? 5 : 1);
    alvo.neuronios = Math.max(1, Math.min(64, alvo.neuronios + passo));
    const saida = li.querySelector("output");
    saida.classList.remove("muda"); void saida.offsetWidth; saida.classList.add("muda");
    renderOcultas();
    atualizarTudo();
  }));
  li.querySelector(".remover").addEventListener("click", () => {
    estado.ocultas = estado.ocultas.filter((o) => o.id !== camada.id);
    li.classList.add("saindo");
    const remover = () => { li.remove(); renderOcultas(); };
    li.addEventListener("animationend", remover, { once: true });
    setTimeout(remover, 450);
    atualizarTudo();
  });
  return li;
}

function renderOcultas() {
  const lista = $("#lista-ocultas");
  const existentes = new Map([...lista.children].map((li) => [li.dataset.id, li]));
  const ids = new Set(estado.ocultas.map((o) => String(o.id)));
  existentes.forEach((li, id) => { if (!ids.has(id) && !li.classList.contains("saindo")) li.remove(); });
  estado.ocultas.forEach((camada, indice) => {
    const li = existentes.get(String(camada.id)) || criarLinhaOculta(camada);
    if (li.parentElement !== lista || lista.children[indice] !== li) lista.insertBefore(li, lista.children[indice] || null);
    li.querySelector(".camada-tipo").textContent = `Oculta ${indice + 1}`;
    li.querySelector(".init").textContent = camada.ativacao === "relu" ? "He" : "Xavier";
    li.querySelector("output").textContent = camada.neuronios;
    li.querySelectorAll(".pilula").forEach((p) => {
      const ativo = p.dataset.valor === camada.ativacao;
      if (ativo && !p.classList.contains("ativo")) { p.classList.add("ativo"); }
      else if (!ativo) p.classList.remove("ativo");
    });
  });
  $("#add-camada").disabled = estado.ocultas.length >= 6;
}

function atualizarNumero(elemento, valor) {
  const texto = String(valor);
  if (elemento.textContent !== texto) {
    elemento.textContent = texto;
    elemento.classList.remove("muda"); void elemento.offsetWidth; elemento.classList.add("muda");
  }
}

function atualizarTudo() {
  const cfg = configRede();
  const completa = cfg.n_entradas > 0 && cfg.n_saidas > 0;
  atualizarNumero($("#n-entradas"), cfg.n_entradas || "—");
  atualizarNumero($("#n-saidas"), cfg.n_saidas || "—");
  $("#n-params").textContent = completa ? contarParametros(cfg) : "—";
  $("#arq-texto").textContent = completa
    ? [cfg.n_entradas, ...cfg.ocultas.map((o) => o.neuronios), cfg.n_saidas].join(" → ") : "—";
  atualizarDiagrama();
}

function nomesQueConferem(lista, n) { return lista && lista.length === n ? lista : null; }

function arqDaRede(rede) {
  const meta = rede.metadados || {};
  return {
    tamanhos: rede.arquitetura,
    ativacoes: [null, ...rede.camadas.map((c) => c.ativacao)],
    nomesEntrada: nomesQueConferem(meta.colunas_entrada, rede.n_entradas) || nomesQueConferem(colunasPorPapel("entrada"), rede.n_entradas),
    nomesSaida: nomesQueConferem(meta.colunas_saida, rede.n_saidas) || nomesQueConferem(colunasPorPapel("saida"), rede.n_saidas),
  };
}

function atualizarDiagrama() {
  const cfg = configRede();
  const rede = estado.rede;
  const status = $("#status-rede");
  const confere = rede && (estado.treinando || !estado.dataset || assinatura(cfg) === assinaturaDaRede(rede));
  if (confere) {
    diagrama.desenhar(arqDaRede(rede), rede.camadas);
    status.textContent = estado.treinando ? "treinando…"
      : `${rede.metadados?.treino ? "treinada" : "criada"} · ${rede.n_parametros} parâmetros`;
    status.className = `estado-rede ${estado.treinando ? "treinando" : "criada"}`;
  } else if (cfg.n_entradas && cfg.n_saidas) {
    diagrama.desenhar({
      tamanhos: [cfg.n_entradas, ...cfg.ocultas.map((o) => o.neuronios), cfg.n_saidas],
      ativacoes: [null, ...cfg.ocultas.map((o) => o.ativacao), cfg.ativacao_saida],
      nomesEntrada: colunasPorPapel("entrada"),
      nomesSaida: colunasPorPapel("saida"),
    }, null);
    status.textContent = rede ? "prévia · diferente da rede criada" : "prévia · clique em Criar rede";
    status.className = "estado-rede";
  } else {
    diagrama.vazio();
    status.textContent = "prévia";
    status.className = "estado-rede";
  }
}

async function criarRede({ silencioso = false } = {}) {
  const cfg = configRede();
  if (!cfg.n_entradas || !cfg.n_saidas) throw new Error("Marque pelo menos uma coluna de entrada e uma de saída (seção 01).");
  const rede = await api.post("/api/rede/criar", cfg);
  definirRede(rede);
  if (!silencioso) {
    toast(`Rede ${rede.arquitetura.join(" → ")} criada: ${rede.n_parametros} parâmetros (pesos Xavier/He, vieses zerados).`, "ok");
    atualizarVisual();
  }
  return rede;
}

function definirRede(rede) {
  estado.rede = rede;
  atualizarTudo();
  renderCamposPrevisao();
}

function configurarConstrutor(rede) {
  estado.ocultas = rede.camadas.slice(0, -1).map((c) => ({ id: novoId(), neuronios: c.n_neuronios, ativacao: c.ativacao }));
  estado.ativSaida = rede.camadas.at(-1).ativacao;
  $("#seed-init").value = rede.seed_inicializacao ?? "";
  renderOcultas();
  renderAtivSaida();
}

// ==================================================================== //
// 03 Treino
// ==================================================================== //
function taxaAtual() { return Number(Math.pow(10, Number($("#taxa").value)).toPrecision(2)); }
function definirTaxa(valor) { $("#taxa").value = Math.log10(valor); atualizarTaxa(); }
function atualizarTaxa() { $("#rot-taxa").textContent = taxaAtual(); preencherRange($("#taxa")); }

async function obterVisual(resolucao) {
  const x = colunasPorPapel("entrada"), y = colunasPorPapel("saida");
  if (!estado.dataset || !x.length || !y.length) return { tipo: null };
  return api.post("/api/visual", { x, y, res: resolucao });
}

let visualPendente = null;
function atualizarVisual(resolucao = 48) {
  clearTimeout(visualPendente);
  visualPendente = setTimeout(async () => {
    if (estado.treinando) return;
    try { visual.mostrar(estado.rede ? await obterVisual(resolucao) : { tipo: null }); }
    catch { visual.mostrar({ tipo: null }); }
  }, 120);
}

async function treinar() {
  if (estado.treinando) return;
  if (!estado.dataset) throw new Error("Escolha um dataset primeiro (seção 01).");
  const x = colunasPorPapel("entrada"), y = colunasPorPapel("saida");
  if (!x.length || !y.length) throw new Error("Marque pelo menos uma coluna de entrada e uma de saída.");
  const epocas = parseInt($("#epocas").value, 10);
  if (!(epocas >= 1 && epocas <= 100000)) throw new Error("Use entre 1 e 100000 épocas.");

  const cfg = configRede();
  const continuar = $("#continuar").checked;
  const confere = estado.rede && assinatura(cfg) === assinaturaDaRede(estado.rede);
  if (!continuar || !confere) {
    if (continuar && !confere) toast("A arquitetura mudou: criei uma rede nova antes de treinar.", "aviso");
    await criarRede({ silencioso: true });
  }
  const seed = $("#seed-treino").value.trim();
  const resposta = await api.post("/api/treino/iniciar", {
    x, y, percentuais: percentuais(), seed: seed === "" ? null : Number(seed),
    normalizacao: estado.normalizacao, epocas, taxa: taxaAtual(), embaralhar: $("#embaralhar").checked,
  });
  (resposta.avisos || []).forEach((a) => toast(a, "aviso", 7000));
  iniciarAcompanhamento(resposta.id, epocas, resposta.tamanhos);
}

function definirTreinandoUI(ativo) {
  const botao = $("#botao-treinar");
  botao.disabled = ativo;
  botao.classList.toggle("carregando", ativo);
  const parar = $("#botao-parar");
  parar.disabled = !ativo;
  parar.lastChild.textContent = "Parar";
  $("#criar-rede").disabled = ativo;
  $("#progresso").classList.toggle("ativo", ativo);
  diagrama.definirTreinando(ativo);
  grafico.definirAtivo(ativo);
  atualizarDiagrama();
}

function iniciarAcompanhamento(id, epocas, tamanhos) {
  estado.treinando = true;
  estado.treinoId = id;
  definirTreinandoUI(true);
  grafico.reiniciar(epocas);
  $("#epoca-total").textContent = epocas;
  $("#epoca-atual").textContent = "0";
  $("#progresso-preenchido").style.transform = "scaleX(0)";
  $("#mse-tr-vivo").textContent = $("#mse-va-vivo").textContent = "—";
  $("#tamanhos").innerHTML = tamanhos
    ? `<span class="t">treino ${tamanhos.treino}</span><span class="v">validação ${tamanhos.validacao}</span><span class="s">teste ${tamanhos.teste}</span>`
    : "";
  limparResultados();
  acompanhar(id);
}

async function acompanhar(id) {
  let desde = 0, versaoVista = -1, ultimoVisual = 0, emVoo = false, inicial = false;
  const t0 = performance.now();
  while (estado.treinoId === id) {
    let p;
    try {
      p = await api.get(`/api/treino/progresso?desde=${desde}`);
    } catch {
      await esperar(700);
      continue;
    }
    if (p.id !== id) break;
    if (!inicial && p.mse_treino_inicial !== null) {
      grafico.definirInicial(p.mse_treino_inicial, p.mse_validacao_inicial);
      inicial = true;
    }
    if (p.mse_treino.length) {
      grafico.adicionar(p.mse_treino, p.mse_validacao);
      desde += p.mse_treino.length;
      $("#mse-tr-vivo").textContent = fmt(p.mse_treino.at(-1));
      $("#mse-va-vivo").textContent = fmt(p.mse_validacao.at(-1));
    }
    $("#epoca-atual").textContent = p.epoca;
    $("#progresso-preenchido").style.transform = `scaleX(${p.epocas ? p.epoca / p.epocas : 0})`;
    $("#tempo-treino").textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;

    const agora = performance.now();
    if (p.versao !== versaoVista && !emVoo && p.treinando) {
      versaoVista = p.versao;
      emVoo = true;
      const comVisual = agora - ultimoVisual > 700;
      if (comVisual) ultimoVisual = agora;
      Promise.all([api.get("/api/rede"), comVisual ? obterVisual(30) : null])
        .then(([rede, vis]) => {
          if (estado.treinoId !== id || !estado.treinando) return;
          if (rede) { estado.rede = rede; atualizarDiagrama(); diagrama.pulsar(); }
          if (vis) visual.mostrar(vis);
        })
        .catch(() => {})
        .finally(() => { emVoo = false; });
    }
    if (!p.treinando) { await finalizar(p); break; }
    await esperar(120);
  }
}

async function finalizar(p) {
  estado.treinando = false;
  let rede = null, vis = null;
  try { [rede, vis] = await Promise.all([api.get("/api/rede"), obterVisual(56)]); } catch { /* mantém o que há */ }
  if (rede) estado.rede = rede;
  if (p.status !== "concluido") { grafico.total = Math.max(1, p.epoca); grafico.animar(); }
  definirTreinandoUI(false);
  atualizarTudo();
  if (vis) visual.mostrar(vis);
  renderCamposPrevisao();
  if (p.duracao) $("#tempo-treino").textContent = `${p.duracao.toFixed(1)} s`;
  mostrarResultados(p);
  if (p.status === "concluido") toast(`Treino concluído: ${p.epoca} épocas em ${p.duracao?.toFixed(1)} s.`, "ok");
  else if (p.status === "interrompido") toast(`Treino interrompido na época ${p.epoca}. Os pesos atuais foram mantidos.`, "aviso");
  else if (p.status === "erro") toast(`Erro no treino: ${p.erro}`, "erro", 8000);
}

// ==================================================================== //
// 04 Resultados
// ==================================================================== //
function limparResultados() {
  $$("#metricas .metrica").forEach((cartao) => {
    const b = cartao.querySelector("b");
    b.textContent = "—";
    b.dataset.valor = "";
    cartao.classList.remove("preenchida");
  });
  const status = $("#status-final");
  status.className = "status-final";
  status.textContent = "Treinando… o MSE de teste só é calculado quando o treino terminar.";
}

function mostrarResultados(p) {
  const r = p.resultado || {};
  $$("#metricas .metrica").forEach((cartao, i) => {
    const chave = cartao.dataset.chave;
    const b = cartao.querySelector("b");
    const valor = r[chave];
    const formatar = chave === "acuracia_teste"
      ? (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(2)}%`)
      : (v) => fmt(v);
    setTimeout(() => {
      contar(b, valor ?? null, formatar, 1100);
      cartao.classList.toggle("preenchida", valor !== null && valor !== undefined);
      cartao.classList.remove("atualizou"); void cartao.offsetWidth; cartao.classList.add("atualizou");
    }, i * 110);
  });
  const status = $("#status-final");
  const nomes = { concluido: "concluído", interrompido: "interrompido", erro: "com erro" };
  const testeVazio = !r.n_teste;
  status.className = `status-final ${p.status === "concluido" ? "ok" : "aviso"}`;
  status.textContent = `Treino ${nomes[p.status] || p.status} · ${p.epoca} épocas` +
    (p.duracao ? ` · ${p.duracao.toFixed(1)} s` : "") +
    (testeVazio ? " · sem conjunto de teste (0%)" : ` · teste com ${r.n_teste} exemplos`);
}

// ==================================================================== //
// 05 Previsão
// ==================================================================== //
function nomesEntradaRede(rede) {
  return nomesQueConferem(rede.metadados?.colunas_entrada, rede.n_entradas)
    || nomesQueConferem(colunasPorPapel("entrada"), rede.n_entradas)
    || Array.from({ length: rede.n_entradas }, (_, i) => `x${i + 1}`);
}

function renderCamposPrevisao(forcar = false) {
  const alvo = $("#campos-previsao");
  const rede = estado.rede;
  if (!rede) { alvo.innerHTML = '<p class="vazio">Crie ou carregue uma rede.</p>'; alvo.dataset.chave = ""; return; }
  const nomes = nomesEntradaRede(rede);
  const chave = nomes.join("|");
  if (alvo.dataset.chave === chave && !forcar) return;
  const anteriores = Object.fromEntries([...alvo.querySelectorAll("input")].map((i) => [i.dataset.nome, i.value]));
  alvo.dataset.chave = chave;
  alvo.replaceChildren();
  nomes.forEach((nome, i) => {
    const rotulo = document.createElement("label");
    rotulo.className = "campo";
    rotulo.style.setProperty("--i", i);
    rotulo.innerHTML = '<span></span><input type="number" step="any">';
    rotulo.querySelector("span").textContent = nome;
    const entrada = rotulo.querySelector("input");
    entrada.dataset.nome = nome;
    const est = estado.dataset?.estatisticas.find((e) => e.nome === nome);
    entrada.value = anteriores[nome] ?? (est ? Number(est.media.toPrecision(4)) : 0);
    entrada.addEventListener("keydown", (e) => { if (e.key === "Enter") prever(); });
    alvo.appendChild(rotulo);
  });
  $("#dica-previsao").textContent = `a rede espera ${rede.n_entradas} valor(es)`;
}

async function prever() {
  const botao = $("#botao-prever");
  await comCarregando(botao, async () => {
    if (!estado.rede) throw new Error("Crie ou carregue uma rede primeiro.");
    const texto = $("#texto-previsao").value.trim();
    let corpo;
    if (texto) {
      corpo = { texto };
    } else {
      const valores = [...$$("#campos-previsao input")].map((i) => i.value.trim());
      if (valores.some((v) => v === "" || Number.isNaN(Number(v)))) throw new Error("Preencha todos os campos com números.");
      corpo = { valores: valores.map(Number) };
    }
    try {
      mostrarPrevisao(await api.post("/api/prever", corpo));
    } catch (erro) {
      const alvo = $("#saida-previsao");
      alvo.innerHTML = '<div class="erro-inline"></div>';
      alvo.firstChild.textContent = erro.message;
      $("#nota-previsao").textContent = "";
      throw erro;
    }
  });
}

function mostrarPrevisao(r) {
  const alvo = $("#saida-previsao");
  alvo.replaceChildren();
  const valores = r.saida;
  const lo = Math.min(0, ...valores), hi = Math.max(1, ...valores);
  const zero = (0 - lo) / (hi - lo);
  const barras = [];
  valores.forEach((v, i) => {
    const linha = document.createElement("div");
    linha.className = "barra-saida";
    linha.style.setProperty("--i", i);
    if (v < 0) linha.classList.add("negativa");
    if (valores.length > 1 && r.classe === i) linha.classList.add("vencedora");
    linha.innerHTML = '<span class="nome"></span><div class="trilha"><span></span></div><span class="valor"></span>';
    linha.querySelector(".nome").textContent = r.nomes_saida[i] ?? `y${i + 1}`;
    linha.querySelector(".valor").textContent = fmt(v, 4);
    alvo.appendChild(linha);
    const pos = (v - lo) / (hi - lo);
    barras.push([linha.querySelector(".trilha span"), Math.min(zero, pos), Math.abs(pos - zero)]);
  });
  barras.forEach(([el, esquerda]) => { el.style.left = `${esquerda * 100}%`; });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    barras.forEach(([el, , largura]) => { el.style.width = `${Math.max(largura * 100, 0.6)}%`; });
  }));
  if (r.classe !== undefined && r.classe !== null) {
    const caixa = document.createElement("div");
    caixa.className = "resultado-classe";
    caixa.innerHTML = valores.length > 1
      ? `Classe prevista (maior saída): <b>${escapar(r.nomes_saida[r.classe])}</b>`
      : `Classe prevista (limiar 0,5): <b>${r.classe}</b>`;
    alvo.appendChild(caixa);
  }
  const entrada = `[${r.entrada.map((v) => Number(v.toPrecision(6))).join(", ")}]`;
  $("#nota-previsao").textContent = r.normalizada
    ? `entrada ${entrada} → normalizada (${r.normalizacao}) [${r.normalizada.map((v) => v.toFixed(4)).join(", ")}] → só forward`
    : `entrada ${entrada} → sem normalização → só forward`;
}

// ==================================================================== //
// 06 Modelos
// ==================================================================== //
async function carregarModelos(destacar = null) {
  const alvo = $("#lista-modelos");
  try {
    const lista = await api.get("/api/modelos");
    alvo.replaceChildren();
    if (!lista.length) { alvo.innerHTML = '<p class="vazio">Nenhum modelo salvo ainda.</p>'; return; }
    lista.forEach((m, i) => {
      const cartao = document.createElement("div");
      cartao.className = `modelo${m.nome === destacar ? " novo" : ""}`;
      cartao.style.setProperty("--i", i);
      cartao.innerHTML = `
        <div class="modelo-nome"></div>
        <div class="modelo-arq">${m.arquitetura.join(" → ")}</div>
        <div class="modelo-tags">
          <span>${escapar(m.ativacoes.join(" · "))}</span>
          <span>${escapar(m.normalizacao)}</span>
          <span class="${m.treinada ? "treinada" : ""}">${m.treinada ? `treinada${m.mse_teste !== null && m.mse_teste !== undefined ? ` · teste ${fmt(m.mse_teste, 4)}` : ""}` : "não treinada"}</span>
        </div>
        <div class="modelo-acoes">
          <button type="button" class="botao primario">Carregar</button>
          <a class="botao" download>Baixar JSON</a>
        </div>`;
      cartao.querySelector(".modelo-nome").textContent = m.nome;
      cartao.querySelector("a").href = `/api/modelo/baixar?nome=${encodeURIComponent(m.nome)}`;
      const botao = cartao.querySelector("button");
      botao.addEventListener("click", () => comCarregando(botao, () => carregarModelo(m.nome)));
      alvo.appendChild(cartao);
    });
  } catch (erro) {
    alvo.innerHTML = '<p class="vazio">Não foi possível listar os modelos.</p>';
    toast(erro.message, "erro");
  }
}

async function carregarModelo(nome) {
  if (estado.treinando) throw new Error("Aguarde o fim do treino.");
  const rede = await api.post("/api/modelo/carregar", { nome });
  const meta = rede.metadados || {};
  const nomeDataset = meta.dataset ? meta.dataset.split(/[\\/]/).pop() : null;
  if (nomeDataset && meta.colunas_entrada && meta.colunas_saida) {
    try {
      const info = await api.post("/api/dataset/abrir", { nome: nomeDataset });
      const papeis = Object.fromEntries(info.colunas.map((c) => [c,
        meta.colunas_entrada.includes(c) ? "entrada" : meta.colunas_saida.includes(c) ? "saida" : "ignorada"]));
      definirDataset(info, papeis, false);
      const t = meta.treino;
      if (t) {
        if (t.divisao_percentuais) definirDivisao(t.divisao_percentuais[0], t.divisao_percentuais[1]);
        if (t.normalizacao) definirNormalizacao(t.normalizacao);
        if (t.taxa_aprendizado) definirTaxa(t.taxa_aprendizado);
        if (t.seed !== undefined && t.seed !== null) $("#seed-treino").value = t.seed;
      }
    } catch { /* dataset original não está disponível: segue só com a rede */ }
  }
  configurarConstrutor(rede);
  definirRede(rede);
  renderCamposPrevisao(true);
  $("#continuar").checked = true;
  atualizarVisual();
  toast(`Modelo ${nome} carregado (${rede.arquitetura.join(" → ")}). "Continuar da rede atual" foi marcado.`, "ok", 5000);
}

async function salvarModelo() {
  if (!estado.rede) throw new Error("Crie ou carregue uma rede antes de salvar.");
  let nome = $("#nome-modelo").value.trim();
  if (!nome) {
    const base = estado.dataset ? estado.dataset.nome.replace(/\.csv$/i, "") : "rede";
    nome = `${base}_${estado.rede.arquitetura.join("-")}`;
  }
  const r = await api.post("/api/modelo/salvar", { nome });
  toast(`Rede salva em ${r.caminho}.`, "ok");
  $("#nome-modelo").value = "";
  await carregarModelos(r.nome);
}

// ==================================================================== //
// Upload de CSV
// ==================================================================== //
async function enviarArquivo(arquivo) {
  if (!arquivo) return;
  if (estado.treinando) { toast("Aguarde o fim do treino.", "aviso"); return; }
  if (arquivo.size > 5 * 1024 * 1024) { toast("Arquivo muito grande (limite de 5 MB).", "erro"); return; }
  try {
    const info = await api.post("/api/dataset/enviar", { nome: arquivo.name, conteudo: await arquivo.text() });
    await carregarDatasets();
    definirDataset(info, null, true);
    toast(`${info.nome} enviado para dados/.`, "ok");
  } catch (erro) { toast(erro.message, "erro", 7000); }
}

// ==================================================================== //
// Restauração (recarregar a página não perde o estado do servidor)
// ==================================================================== //
async function restaurar() {
  const e = await api.get("/api/estado");
  estado.ativacoes = e.ativacoes || estado.ativacoes;
  renderAtivSaida();
  if (e.dataset) {
    const temSelecao = e.dataset.x?.length && e.dataset.y?.length;
    definirDataset(e.dataset, temSelecao ? papeisPadrao(e.dataset) : null, !temSelecao);
  }
  if (e.rede) {
    configurarConstrutor(e.rede);
    definirRede(e.rede);
  }
  if (e.treinando) {
    iniciarAcompanhamento(e.treino.id, e.treino.epocas, e.treino.tamanhos);
  } else if (e.treino?.status && e.treino.status !== "ocioso") {
    const p = await api.get("/api/treino/progresso?desde=0");
    grafico.reiniciar(p.status === "concluido" ? p.epocas : Math.max(1, p.epoca));
    grafico.definirInicial(p.mse_treino_inicial, p.mse_validacao_inicial);
    grafico.adicionar(p.mse_treino, p.mse_validacao);
    $("#mse-tr-vivo").textContent = fmt(p.mse_treino.at(-1));
    $("#mse-va-vivo").textContent = fmt(p.mse_validacao.at(-1));
    if (p.duracao) $("#tempo-treino").textContent = `${p.duracao.toFixed(1)} s`;
    $("#epoca-atual").textContent = p.epoca;
    $("#epoca-total").textContent = p.epocas;
    $("#progresso-preenchido").style.transform = `scaleX(${p.epocas ? p.epoca / p.epocas : 0})`;
    mostrarResultados(p);
    atualizarVisual();
  }
}

// ==================================================================== //
// Eventos
// ==================================================================== //
function ligarEventos() {
  $("#p-treino").addEventListener("input", () => atualizarDivisao("treino"));
  $("#p-validacao").addEventListener("input", () => atualizarDivisao("validacao"));
  $$("#normalizacao button").forEach((b) => b.addEventListener("click", () => definirNormalizacao(b.dataset.valor)));
  window.addEventListener("resize", posicionarSegmentado);

  $("#add-camada").addEventListener("click", () => {
    if (estado.ocultas.length >= 6) return;
    const ultima = estado.ocultas.at(-1);
    estado.ocultas.push({ id: novoId(), neuronios: ultima?.neuronios ?? 4, ativacao: ultima?.ativacao ?? "tanh" });
    renderOcultas();
    atualizarTudo();
  });
  $("#seed-init").addEventListener("input", atualizarTudo);
  $("#criar-rede").addEventListener("click", (e) => comCarregando(e.currentTarget, () => criarRede()));

  $$("#atalhos-epocas button").forEach((b) => b.addEventListener("click", () => { $("#epocas").value = b.dataset.valor; }));
  $("#taxa").addEventListener("input", atualizarTaxa);
  $("#escala-log").addEventListener("change", (e) => grafico.definirLog(e.target.checked));

  $("#botao-treinar").addEventListener("click", async () => {
    const botao = $("#botao-treinar");
    botao.disabled = true;
    botao.classList.add("carregando");
    try {
      await treinar();
    } catch (erro) {
      toast(erro.message, "erro", 6000);
      estado.treinando = false;
      definirTreinandoUI(false);
    }
  });
  $("#botao-parar").addEventListener("click", async () => {
    const botao = $("#botao-parar");
    botao.disabled = true;
    botao.lastChild.textContent = "Parando…";
    try { await api.post("/api/treino/parar"); } catch (erro) { toast(erro.message, "erro"); }
  });

  $("#botao-prever").addEventListener("click", prever);
  $("#texto-previsao").addEventListener("keydown", (e) => { if (e.key === "Enter") prever(); });

  $("#botao-salvar").addEventListener("click", (e) => comCarregando(e.currentTarget, salvarModelo));
  $("#nome-modelo").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#botao-salvar").click(); });
  $("#atualizar-modelos").addEventListener("click", () => carregarModelos());

  const soltar = $("#soltar");
  $("#arquivo-csv").addEventListener("change", (e) => { enviarArquivo(e.target.files[0]); e.target.value = ""; });
  ["dragenter", "dragover"].forEach((tipo) => soltar.addEventListener(tipo, (e) => { e.preventDefault(); soltar.classList.add("arrastando"); }));
  ["dragleave", "drop"].forEach((tipo) => soltar.addEventListener(tipo, (e) => { e.preventDefault(); soltar.classList.remove("arrastando"); }));
  soltar.addEventListener("drop", (e) => enviarArquivo(e.dataTransfer.files[0]));
}

// ==================================================================== //
// Início
// ==================================================================== //
async function iniciar() {
  iniciarTema(() => {
    fundo?.atualizarCores();
    grafico.animar();
    visual.redesenhar();
  });
  iniciarRevelacao();
  iniciarRipple();
  iniciarNavegacao();
  fundo = iniciarFundo($("#fundo-hero"));

  ligarEventos();
  renderAtivSaida();
  renderOcultas();
  definirNormalizacao(estado.normalizacao);
  atualizarDivisao("treino");
  atualizarTaxa();
  atualizarTudo();
  visual.mostrar({ tipo: null });

  await Promise.all([carregarDatasets(), carregarModelos()]);
  try { await restaurar(); } catch (erro) { toast(erro.message, "erro"); }
  requestAnimationFrame(posicionarSegmentado);
}

iniciar();
