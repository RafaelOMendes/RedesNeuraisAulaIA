// Diagrama SVG da rede: nós por camada, arestas coloridas pelo sinal do peso
// e com espessura proporcional a |peso|. Durante o treino, sinais percorrem
// as conexões da entrada até a saída.

import { reduzirMovimento } from "./efeitos.js";

const NS = "http://www.w3.org/2000/svg";
const LARGURA = 800, ALTURA = 440;
const MAX_VISIVEIS = 12;

function el(nome, atributos = {}, pai = null) {
  const elemento = document.createElementNS(NS, nome);
  for (const [k, v] of Object.entries(atributos)) elemento.setAttribute(k, v);
  if (pai) pai.appendChild(elemento);
  return elemento;
}

/** Índices visíveis de uma camada; -1 marca o "⋮" quando há neurônios demais. */
function indicesVisiveis(n) {
  if (n <= MAX_VISIVEIS) return [...Array(n).keys()];
  const metade = Math.floor((MAX_VISIVEIS - 1) / 2);
  return [...Array(metade).keys(), -1, ...Array.from({ length: metade }, (_, k) => n - metade + k)];
}

export class Diagrama {
  constructor(svg) {
    this.svg = svg;
    this.chave = null;
    this.camadasNos = [];   // [[{x, y, real, g}]]
    this.arestas = [];      // [{el, camada, i, j}]
    this.sinais = [];
    this.timerSinais = null;
    this.rafSinais = null;
  }

  /**
   * arq = { tamanhos: [nEntradas, ..., nSaidas], ativacoes: [null, "tanh", ..., "sigmoid"],
   *         nomesEntrada: [...], nomesSaida: [...] }
   * camadas = camadas do JSON da rede (pesos/vieses) ou null para prévia.
   */
  desenhar(arq, camadas = null) {
    if (!arq || arq.tamanhos.some((n) => !n)) { this.vazio(); return; }
    const chave = JSON.stringify([arq.tamanhos, arq.ativacoes, arq.nomesEntrada, arq.nomesSaida]);
    if (chave !== this.chave) { this.construir(arq); this.chave = chave; }
    this.aplicarPesos(camadas);
  }

  vazio(mensagem = "Escolha as colunas de entrada e saída para ver a prévia") {
    this.pararSinais();
    this.svg.replaceChildren();
    this.chave = null;
    this.camadasNos = [];
    this.arestas = [];
    el("text", { x: LARGURA / 2, y: ALTURA / 2, class: "rotulo-camada" }, this.svg).textContent = mensagem;
  }

  construir(arq) {
    const svg = this.svg;
    svg.replaceChildren();
    this.sinais = [];
    const L = arq.tamanhos.length;
    const esquerda = 110, direita = LARGURA - 110, topo = 62, base = ALTURA - 24;
    const xs = arq.tamanhos.map((_, l) => (L === 1 ? LARGURA / 2 : esquerda + ((direita - esquerda) * l) / (L - 1)));
    const maiorCamada = Math.max(...arq.tamanhos.map((n) => indicesVisiveis(n).length));
    const espaco = Math.min(46, (base - topo) / Math.max(1, maiorCamada - 1 || 1));
    const raio = Math.max(7, Math.min(14, espaco * 0.32));
    this.raio = raio;

    const gArestas = el("g", {}, svg);
    const gNos = el("g", {}, svg);
    const gRotulos = el("g", {}, svg);
    this.gSinais = el("g", {}, svg);

    this.camadasNos = arq.tamanhos.map((n, l) => {
      const vis = indicesVisiveis(n);
      const alturaTotal = (vis.length - 1) * espaco;
      const y0 = (topo + base) / 2 - alturaTotal / 2;
      return vis.map((real, k) => ({ x: xs[l], y: y0 + k * espaco, real }));
    });

    // Arestas (curvas de Bézier entre camadas consecutivas)
    this.arestas = [];
    const animar = !reduzirMovimento();
    for (let l = 1; l < L; l++) {
      for (const destino of this.camadasNos[l]) {
        if (destino.real < 0) continue;
        for (const origem of this.camadasNos[l - 1]) {
          if (origem.real < 0) continue;
          const mx = (origem.x + destino.x) / 2;
          const caminho = el("path", {
            d: `M${origem.x},${origem.y} C${mx},${origem.y} ${mx},${destino.y} ${destino.x},${destino.y}`,
            class: "aresta prevista",
          }, gArestas);
          if (animar) {
            const comp = Math.ceil(caminho.getTotalLength?.() || 300);
            caminho.style.setProperty("--comp", comp);
            caminho.style.setProperty("--d", `${(l - 1) * 0.16 + Math.random() * 0.18}s`);
            caminho.classList.add("desenhar");
            caminho.addEventListener("animationend", () => caminho.classList.remove("desenhar"), { once: true });
          }
          this.arestas.push({ el: caminho, camada: l - 1, i: destino.real, j: origem.real, de: origem, para: destino });
        }
      }
    }

    // Nós
    this.camadasNos.forEach((camada, l) => {
      const tipo = l === 0 ? "entrada" : l === L - 1 ? "saida" : "oculto";
      camada.forEach((no, k) => {
        if (no.real < 0) {
          el("text", { x: no.x, y: no.y + 4, class: "rotulo-mais" }, gNos).textContent =
            `⋮ +${arq.tamanhos[l] - (MAX_VISIVEIS - 1)}`;
          return;
        }
        const g = el("g", { class: `no ${tipo}${animar ? " surgir" : ""}` }, gNos);
        g.style.setProperty("--d", `${l * 0.12 + k * 0.025}s`);
        el("circle", { cx: no.x, cy: no.y, r: raio * 1.7, class: "halo" }, g);
        el("circle", { cx: no.x, cy: no.y, r: raio, class: "corpo" }, g);
        no.g = g;
        if (l === 0 && arq.nomesEntrada?.[no.real] !== undefined) {
          el("text", { x: no.x - raio - 8, y: no.y + 4, "text-anchor": "end" }, g).textContent = arq.nomesEntrada[no.real];
        }
        if (l === L - 1 && arq.nomesSaida?.[no.real] !== undefined) {
          el("text", { x: no.x + raio + 8, y: no.y + 4, "text-anchor": "start" }, g).textContent = arq.nomesSaida[no.real];
        }
      });

      // Rótulo da camada
      const nome = l === 0 ? "Entrada" : l === L - 1 ? "Saída" : `Oculta ${l}`;
      el("text", { x: xs[l], y: 22, class: "rotulo-camada" }, gRotulos).textContent = `${nome} · ${arq.tamanhos[l]}`;
      if (arq.ativacoes[l]) {
        el("text", { x: xs[l], y: 39, class: "rotulo-camada ativ" }, gRotulos).textContent = arq.ativacoes[l];
      }
    });
  }

  aplicarPesos(camadas) {
    if (!camadas) {
      for (const a of this.arestas) {
        a.el.classList.add("prevista");
        a.el.style.stroke = a.el.style.strokeWidth = a.el.style.opacity = "";
      }
      for (const camada of this.camadasNos) for (const no of camada) {
        const corpo = no.g?.querySelector(".corpo");
        if (corpo) corpo.style.fill = "";
      }
      return;
    }
    // Normalização por camada: o maior |peso| de cada camada vira a espessura máxima.
    const maximos = camadas.map((c) => Math.max(1e-9, ...c.pesos.flat().map(Math.abs)));
    for (const a of this.arestas) {
      const w = camadas[a.camada]?.pesos?.[a.i]?.[a.j];
      if (w === undefined) continue;
      const forca = Math.abs(w) / maximos[a.camada];
      a.el.classList.remove("prevista");
      a.el.style.stroke = w >= 0 ? "var(--pos)" : "var(--neg)";
      a.el.style.strokeWidth = (0.5 + 3.6 * forca).toFixed(2);
      a.el.style.opacity = (0.16 + 0.76 * forca).toFixed(3);
    }
    // Viés: o preenchimento do neurônio fica mais forte quanto maior |b|.
    camadas.forEach((c, indice) => {
      const maxB = Math.max(1e-9, ...c.vieses.map(Math.abs));
      for (const no of this.camadasNos[indice + 1] || []) {
        if (no.real < 0 || !no.g) continue;
        const b = c.vieses[no.real];
        const pct = Math.round((Math.abs(b) / maxB) * 45);
        no.g.querySelector(".corpo").style.fill =
          `color-mix(in srgb, ${b >= 0 ? "var(--pos)" : "var(--neg)"} ${pct}%, var(--cartao))`;
      }
    });
  }

  /** Halo rápido em alguns neurônios quando os pesos são atualizados. */
  pulsar() {
    if (reduzirMovimento()) return;
    const todos = this.camadasNos.flat().filter((n) => n.g && n.real >= 0);
    for (let k = 0; k < Math.min(6, todos.length); k++) {
      const no = todos[Math.floor(Math.random() * todos.length)];
      no.g.classList.remove("pulso");
      void no.g.getBBox();
      no.g.classList.add("pulso");
    }
  }

  definirTreinando(ativo) {
    this.svg.classList.toggle("treinando", ativo);
    if (ativo) this.iniciarSinais(); else this.pararSinais();
  }

  iniciarSinais() {
    if (this.timerSinais || reduzirMovimento() || this.camadasNos.length < 2) return;
    this.timerSinais = setInterval(() => {
      if (this.sinais.length >= 14 || document.hidden) return;
      const rota = this.camadasNos.map((camada) => {
        const validos = camada.filter((n) => n.real >= 0);
        return validos[Math.floor(Math.random() * validos.length)];
      });
      const ponto = el("circle", { r: Math.max(2.5, this.raio * 0.32), class: "sinal" }, this.gSinais);
      this.sinais.push({ rota, salto: 0, t: 0, el: ponto });
    }, 150);
    const passo = () => {
      for (const s of this.sinais) {
        s.t += 0.055;
        if (s.t >= 1) { s.t = 0; s.salto += 1; }
        if (s.salto >= s.rota.length - 1) { s.el.remove(); s.morto = true; continue; }
        const a = s.rota[s.salto], b = s.rota[s.salto + 1];
        const aresta = this.arestas.find((x) => x.de === a && x.para === b);
        let x, y;
        if (aresta) {
          const p = aresta.el.getPointAtLength(aresta.el.getTotalLength() * s.t);
          x = p.x; y = p.y;
        } else {
          x = a.x + (b.x - a.x) * s.t; y = a.y + (b.y - a.y) * s.t;
        }
        s.el.setAttribute("cx", x);
        s.el.setAttribute("cy", y);
      }
      this.sinais = this.sinais.filter((s) => !s.morto);
      this.rafSinais = requestAnimationFrame(passo);
    };
    this.rafSinais = requestAnimationFrame(passo);
  }

  pararSinais() {
    clearInterval(this.timerSinais);
    cancelAnimationFrame(this.rafSinais);
    this.timerSinais = this.rafSinais = null;
    for (const s of this.sinais) s.el.remove();
    this.sinais = [];
  }
}
