// Gráficos em canvas (sem bibliotecas): MSE por época e visualização do que
// a rede aprendeu (fronteira de decisão, curva de regressão ou dispersão).

import { cor, fmt, reduzirMovimento, rgb, rgba } from "./efeitos.js";

function prepararCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const largura = canvas.clientWidth, altura = canvas.clientHeight;
  if (canvas.width !== Math.round(largura * dpr) || canvas.height !== Math.round(altura * dpr)) {
    canvas.width = Math.round(largura * dpr);
    canvas.height = Math.round(altura * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, largura, altura, dpr };
}

const fonte = (tamanho, peso = 500) =>
  `${peso} ${tamanho}px ui-monospace, "Cascadia Code", Consolas, monospace`;

function rotuloEixo(v, log) {
  if (log) {
    const valor = Math.pow(10, v);
    return valor >= 0.01 ? valor.toPrecision(2).replace(/\.?0+$/, "") : valor.toExponential(0);
  }
  return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);
}

// ==================================================================== //
// MSE por época
// ==================================================================== //
export class GraficoMSE {
  constructor(canvas, dica) {
    this.canvas = canvas;
    this.dica = dica;
    this.log = true;
    this.faixa = null;
    this.raf = null;
    this.ativo = false;
    this.mouse = null;
    this.limpar();
    new ResizeObserver(() => this.desenhar()).observe(canvas);
    canvas.addEventListener("pointermove", (e) => {
      const caixa = canvas.getBoundingClientRect();
      this.mouse = { x: e.clientX - caixa.left, y: e.clientY - caixa.top };
      this.animar();
    });
    canvas.addEventListener("pointerleave", () => { this.mouse = null; this.dica.classList.remove("visivel"); this.animar(); });
  }

  limpar() {
    this.treino = [];
    this.validacao = [];
    this.total = 0;
    this.mostrados = 0;
    this.animar?.();
  }

  reiniciar(total) {
    this.limpar();
    this.total = total;
    this.faixa = null;
    this.animar();
  }

  definirInicial(mseTreino, mseValidacao) {
    if (this.treino.length === 0) {
      this.treino.push(mseTreino);
      this.validacao.push(mseValidacao);
      this.animar();
    }
  }

  adicionar(treino, validacao) {
    this.treino.push(...treino);
    this.validacao.push(...validacao);
    if (reduzirMovimento()) this.mostrados = this.treino.length;
    this.animar();
  }

  definirAtivo(ativo) { this.ativo = ativo; this.animar(); }
  definirLog(log) { this.log = log; this.faixa = null; this.animar(); }

  animar() {
    if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = null; this.quadro(); });
  }

  transformar(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return null;
    if (this.log) return v > 0 ? Math.log10(v) : null;
    return v;
  }

  quadro() {
    // avanço baseado em tempo (não em quadros): funciona igual em abas lentas
    const agora = performance.now();
    const dt = Math.min(1000, agora - (this.ultimoQuadro || agora)) / 16.7;
    this.ultimoQuadro = agora;
    const n = this.treino.length;
    const faltam = n - this.mostrados;
    if (faltam > 0.01) this.mostrados += Math.max(0.35, faltam * 0.14) * Math.max(1, dt);
    if (this.mostrados > n) this.mostrados = n;
    if (n - this.mostrados <= 0.01) this.ultimoQuadro = 0;

    // faixa alvo do eixo Y (com transição suave)
    const visiveis = Math.max(1, Math.ceil(this.mostrados));
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < visiveis; i++) {
      for (const v of [this.transformar(this.treino[i]), this.transformar(this.validacao[i])]) {
        if (v === null) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    let continuar = this.mostrados < n - 0.01 || this.ativo;
    if (min !== Infinity) {
      if (max - min < 1e-9) { max += 0.5; min -= 0.5; }
      const folga = (max - min) * 0.08;
      const alvo = { min: min - folga, max: max + folga };
      if (!this.faixa || reduzirMovimento()) this.faixa = { ...alvo };
      else {
        this.faixa.min += (alvo.min - this.faixa.min) * 0.18;
        this.faixa.max += (alvo.max - this.faixa.max) * 0.18;
        if (Math.abs(alvo.min - this.faixa.min) + Math.abs(alvo.max - this.faixa.max) > 1e-4) continuar = true;
      }
    }
    this.desenhar();
    if (continuar) this.animar();
  }

  desenhar() {
    const { ctx, largura, altura } = prepararCanvas(this.canvas);
    ctx.clearRect(0, 0, largura, altura);
    const m = { esq: 62, dir: 18, topo: 16, base: 34 };
    const w = largura - m.esq - m.dir, h = altura - m.topo - m.base;
    if (w <= 10 || h <= 10) return;

    const corLinha = cor("--linha"), corTinta3 = cor("--tinta-3");
    ctx.font = fonte(11);

    if (!this.treino.length || !this.faixa) {
      ctx.strokeStyle = corLinha;
      ctx.setLineDash([4, 6]);
      ctx.strokeRect(m.esq, m.topo, w, h);
      ctx.setLineDash([]);
      ctx.fillStyle = corTinta3;
      ctx.textAlign = "center";
      ctx.font = `500 13px ${getComputedStyle(document.body).fontFamily}`;
      ctx.fillText("Clique em Treinar para ver o MSE de cada época", m.esq + w / 2, m.topo + h / 2);
      return;
    }

    const totalX = Math.max(this.total, this.treino.length - 1, 1);
    const X = (i) => m.esq + (i / totalX) * w;
    const Y = (v) => m.topo + h - ((v - this.faixa.min) / (this.faixa.max - this.faixa.min)) * h;

    // grade horizontal e rótulos
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const linhas = 5;
    for (let k = 0; k <= linhas; k++) {
      const v = this.faixa.min + ((this.faixa.max - this.faixa.min) * k) / linhas;
      const y = Y(v);
      ctx.strokeStyle = corLinha;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(m.esq, y); ctx.lineTo(m.esq + w, y); ctx.stroke();
      ctx.fillStyle = corTinta3;
      ctx.fillText(rotuloEixo(v, this.log), m.esq - 8, y);
    }
    // eixo X
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let k = 0; k <= 4; k++) {
      const e = Math.round((totalX * k) / 4);
      ctx.fillText(String(e), X(e), m.topo + h + 10);
    }
    ctx.textAlign = "right";
    ctx.fillText("época", m.esq + w, m.topo + h + 22 > altura - 12 ? altura - 12 : m.topo + h + 22);

    const limite = this.mostrados;
    const desenharSerie = (serie, corSerie, area) => {
      const pontos = [];
      const ultimo = Math.floor(limite);
      for (let i = 0; i <= Math.min(ultimo, serie.length - 1); i++) {
        const v = this.transformar(serie[i]);
        if (v !== null) pontos.push([X(i), Y(v)]);
      }
      const frac = limite - ultimo;
      if (frac > 0 && ultimo + 1 < serie.length) {
        const a = this.transformar(serie[ultimo]), b = this.transformar(serie[ultimo + 1]);
        if (a !== null && b !== null) pontos.push([X(ultimo + frac), Y(a + (b - a) * frac)]);
      }
      if (pontos.length < 1) return null;

      if (area && pontos.length > 1) {
        const grad = ctx.createLinearGradient(0, m.topo, 0, m.topo + h);
        grad.addColorStop(0, rgba(corSerie, 0.22));
        grad.addColorStop(1, rgba(corSerie, 0));
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(pontos[0][0], m.topo + h);
        for (const [x, y] of pontos) ctx.lineTo(x, y);
        ctx.lineTo(pontos[pontos.length - 1][0], m.topo + h);
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = cor(corSerie);
      ctx.lineWidth = area ? 2.4 : 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      pontos.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
      return pontos[pontos.length - 1];
    };

    ctx.save();
    ctx.beginPath(); ctx.rect(m.esq, m.topo - 6, w + 8, h + 12); ctx.clip();
    const cabecaVa = desenharSerie(this.validacao, "--validacao", false);
    const cabecaTr = desenharSerie(this.treino, "--treino", true);
    ctx.restore();

    // "cabeça" pulsante das séries
    const t = performance.now() / 500;
    for (const [cabeca, nome] of [[cabecaTr, "--treino"], [cabecaVa, "--validacao"]]) {
      if (!cabeca) continue;
      if (this.ativo && !reduzirMovimento()) {
        ctx.fillStyle = rgba(nome, 0.18 + 0.12 * Math.sin(t));
        ctx.beginPath(); ctx.arc(cabeca[0], cabeca[1], 9 + 3 * Math.sin(t), 0, Math.PI * 2); ctx.fill();
      }
      ctx.fillStyle = cor(nome);
      ctx.beginPath(); ctx.arc(cabeca[0], cabeca[1], 4, 0, Math.PI * 2); ctx.fill();
    }

    // dica ao passar o mouse
    if (this.mouse && this.mouse.x >= m.esq && this.mouse.x <= m.esq + w) {
      const i = Math.max(0, Math.min(Math.floor(limite), Math.round(((this.mouse.x - m.esq) / w) * totalX)));
      if (i < this.treino.length) {
        const x = X(i);
        ctx.strokeStyle = cor("--linha-2");
        ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(x, m.topo); ctx.lineTo(x, m.topo + h); ctx.stroke();
        ctx.setLineDash([]);
        for (const [serie, nome] of [[this.treino, "--treino"], [this.validacao, "--validacao"]]) {
          const v = this.transformar(serie[i]);
          if (v === null) continue;
          ctx.fillStyle = cor("--cartao"); ctx.strokeStyle = cor(nome); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, Y(v), 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        }
        this.dica.innerHTML = `<b>época ${i}</b>treino ${fmt(this.treino[i])}<br>validação ${fmt(this.validacao[i])}`;
        const esquerda = Math.min(largura - this.dica.offsetWidth - 6, Math.max(6, x + 12));
        this.dica.style.transform = `translate(${esquerda}px, ${Math.max(6, this.mouse.y - 50)}px)`;
        this.dica.classList.add("visivel");
      }
    } else {
      this.dica.classList.remove("visivel");
    }
  }
}

// ==================================================================== //
// Visualização: fronteira de decisão / curva / dispersão
// ==================================================================== //
const CORES_CLASSES = ["--classe-0", "--classe-1", "--classe-2", "--classe-3", "--classe-4", "--classe-5"];
const COR_GRUPO = { treino: "--treino", validacao: "--validacao", teste: "--teste" };

export class Visual {
  constructor(canvas, { titulo, dica, legenda }) {
    this.canvas = canvas;
    this.titulo = titulo;
    this.dica = dica;
    this.legenda = legenda;
    this.dados = null;
    this.atual = document.createElement("canvas");
    this.anterior = document.createElement("canvas");
    this.fade = 1;
    this.raf = null;
    new ResizeObserver(() => this.renderizar(false)).observe(canvas);
  }

  mostrar(dados) {
    this.dados = dados;
    this.atualizarTextos();
    this.renderizar(true);
  }

  redesenhar() { this.atualizarTextos(); this.renderizar(false); }

  atualizarTextos() {
    const d = this.dados;
    const tipo = d?.tipo;
    if (!tipo) {
      this.titulo.textContent = "O que a rede aprendeu";
      this.dica.textContent = "";
      this.legenda.innerHTML = "";
      return;
    }
    const grupos = `<span class="item"><i class="forma cheia"></i>treino</span><span class="item"><i class="forma anel"></i>validação</span><span class="item"><i class="forma quadrado"></i>teste</span>`;
    if (tipo === "superficie") {
      const classes = d.n_saidas > 1;
      this.titulo.textContent = classes ? "Fronteira de decisão" : "Superfície de saída";
      this.dica.textContent = `${d.nomes_x[0]} × ${d.nomes_x[1]}`;
      const itens = classes
        ? d.nomes_y.map((n, i) => `<span class="item"><i class="cor" style="background:var(${CORES_CLASSES[i % 6]})"></i>${n}</span>`).join("")
        : `<span class="item"><i class="cor" style="background:var(--classe-0)"></i>${d.nomes_y[0]} ≈ 0</span><span class="item"><i class="cor" style="background:var(--classe-1)"></i>${d.nomes_y[0]} ≈ 1</span>`;
      this.legenda.innerHTML = itens + grupos;
    } else if (tipo === "curva") {
      this.titulo.textContent = "Curva aprendida";
      this.dica.textContent = `${d.nomes_y[0]} em função de ${d.nomes_x[0]}`;
      this.legenda.innerHTML = `<span class="item l-treino item-legenda">previsão da rede</span>` +
        `<span class="item"><i class="cor" style="background:var(--treino)"></i>treino</span>` +
        `<span class="item"><i class="cor" style="background:var(--validacao)"></i>validação</span>` +
        `<span class="item"><i class="cor" style="background:var(--teste)"></i>teste</span>`;
    } else {
      this.titulo.textContent = "Previsto × esperado";
      this.dica.textContent = `primeira saída (${d.nomes_y[0]})`;
      this.legenda.innerHTML = `<span class="item"><i class="cor" style="background:var(--treino)"></i>treino</span>` +
        `<span class="item"><i class="cor" style="background:var(--validacao)"></i>validação</span>` +
        `<span class="item"><i class="cor" style="background:var(--teste)"></i>teste</span>`;
    }
  }

  renderizar(comFade) {
    const { ctx, largura, altura, dpr } = prepararCanvas(this.canvas);
    if (!this.canvas.width || !this.canvas.height) return;  // ainda sem tamanho na tela
    for (const c of [this.atual, this.anterior]) {
      if (c.width !== this.canvas.width || c.height !== this.canvas.height) {
        c.width = this.canvas.width; c.height = this.canvas.height;
      }
    }
    if (comFade && !reduzirMovimento()) {
      const a = this.anterior.getContext("2d");
      a.setTransform(1, 0, 0, 1, 0, 0);
      a.clearRect(0, 0, this.anterior.width, this.anterior.height);
      a.drawImage(this.atual, 0, 0);
      this.fade = 0;
    } else {
      this.fade = 1;
    }
    const c = this.atual.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, largura, altura);
    this.pintar(c, largura, altura);
    this.compor(ctx, largura, altura);
  }

  compor(ctx, largura, altura) {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.fade < 1) {
      ctx.globalAlpha = 1 - this.fade;
      ctx.drawImage(this.anterior, 0, 0);
    }
    ctx.globalAlpha = this.fade;
    ctx.drawImage(this.atual, 0, 0);
    ctx.restore();
    if (this.fade < 1 && !this.raf) {
      this.raf = requestAnimationFrame(() => {
        this.raf = null;
        this.fade = Math.min(1, this.fade + 0.08);
        this.compor(this.canvas.getContext("2d"), largura, altura);
      });
    }
  }

  pintar(ctx, largura, altura) {
    const d = this.dados;
    const m = { esq: 60, dir: 14, topo: 12, base: 40 };
    const w = largura - m.esq - m.dir, h = altura - m.topo - m.base;
    if (w < 20 || h < 20) return;
    if (!d || !d.tipo) {
      ctx.strokeStyle = cor("--linha");
      ctx.setLineDash([4, 6]);
      ctx.strokeRect(m.esq, m.topo, w, h);
      ctx.setLineDash([]);
      ctx.fillStyle = cor("--tinta-3");
      ctx.textAlign = "center";
      ctx.font = `500 13px ${getComputedStyle(document.body).fontFamily}`;
      ctx.fillText("Aparece ao treinar ou criar a rede", m.esq + w / 2, m.topo + h / 2 - 8);
      ctx.fillText("com um dataset escolhido", m.esq + w / 2, m.topo + h / 2 + 12);
      return;
    }

    let xr, yr;
    if (d.tipo === "superficie") { xr = d.x; yr = d.y; }
    else if (d.tipo === "curva") {
      xr = [d.xs[0], d.xs[d.xs.length - 1]];
      const ys = d.ys.concat(d.pontos.map((p) => p[1]));
      const lo = Math.min(...ys), hi = Math.max(...ys), f = (hi - lo || 1) * 0.08;
      yr = [lo - f, hi + f];
    } else {
      const vals = d.pontos.flatMap((p) => [p[0], p[1]]);
      const lo = Math.min(...vals), hi = Math.max(...vals), f = (hi - lo || 1) * 0.08;
      xr = [lo - f, hi + f]; yr = [lo - f, hi + f];
    }
    const X = (v) => m.esq + ((v - xr[0]) / (xr[1] - xr[0] || 1)) * w;
    const Y = (v) => m.topo + h - ((v - yr[0]) / (yr[1] - yr[0] || 1)) * h;

    // fundo do gráfico
    ctx.fillStyle = cor("--cartao-2");
    ctx.fillRect(m.esq, m.topo, w, h);

    if (d.tipo === "superficie") this.pintarSuperficie(ctx, d, m, w, h);

    // eixos
    ctx.strokeStyle = cor("--linha-2");
    ctx.lineWidth = 1;
    ctx.strokeRect(m.esq + 0.5, m.topo + 0.5, w - 1, h - 1);
    ctx.fillStyle = cor("--tinta-3");
    ctx.font = fonte(10.5);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let k = 0; k <= 4; k++) {
      const v = xr[0] + ((xr[1] - xr[0]) * k) / 4;
      ctx.fillText(rotuloEixo(v, false), X(v), m.topo + h + 6);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let k = 0; k <= 4; k++) {
      const v = yr[0] + ((yr[1] - yr[0]) * k) / 4;
      ctx.fillText(rotuloEixo(v, false), m.esq - 6, Y(v));
    }
    ctx.fillStyle = cor("--tinta-2");
    ctx.font = fonte(11.5, 650);
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    const rotX = d.tipo === "superficie" ? d.nomes_x[0] : d.tipo === "curva" ? d.nomes_x[0] : `esperado (${d.nomes_y[0]})`;
    const rotY = d.tipo === "superficie" ? d.nomes_x[1] : d.tipo === "curva" ? d.nomes_y[0] : "previsto";
    ctx.fillText(rotX, m.esq + w / 2, altura - 4);
    ctx.save();
    ctx.translate(12, m.topo + h / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(rotY, 0, 0);
    ctx.restore();

    ctx.save();
    ctx.beginPath(); ctx.rect(m.esq, m.topo, w, h); ctx.clip();
    if (d.tipo === "superficie") {
      for (const [px, py, alvo, grupo] of d.pontos) {
        const corP = d.n_saidas > 1 ? CORES_CLASSES[alvo % 6] : alvo >= 0.5 ? "--classe-1" : "--classe-0";
        this.marcador(ctx, X(px), Y(py), corP, grupo);
      }
    } else if (d.tipo === "curva") {
      for (const [px, py, grupo] of d.pontos) {
        ctx.fillStyle = rgba(COR_GRUPO[grupo], 0.55);
        ctx.beginPath(); ctx.arc(X(px), Y(py), 3.2, 0, Math.PI * 2); ctx.fill();
      }
      const grad = ctx.createLinearGradient(m.esq, 0, m.esq + w, 0);
      grad.addColorStop(0, cor("--acento"));
      grad.addColorStop(1, cor("--acento-2"));
      ctx.strokeStyle = grad;
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.shadowColor = rgba("--acento", 0.45);
      ctx.shadowBlur = 10;
      ctx.beginPath();
      d.xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(d.ys[i])) : ctx.moveTo(X(x), Y(d.ys[i]))));
      ctx.stroke();
      ctx.shadowBlur = 0;
    } else {
      ctx.strokeStyle = cor("--linha-2");
      ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(X(xr[0]), Y(xr[0])); ctx.lineTo(X(xr[1]), Y(xr[1])); ctx.stroke();
      ctx.setLineDash([]);
      for (const [esp, prev, grupo] of d.pontos) {
        ctx.fillStyle = rgba(COR_GRUPO[grupo], 0.7);
        ctx.beginPath(); ctx.arc(X(esp), Y(prev), 3.4, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.restore();
  }

  pintarSuperficie(ctx, d, m, w, h) {
    const res = d.res;
    const img = new ImageData(res, res);
    const cores = CORES_CLASSES.map((n) => rgb(n));
    const fundo = rgb("--cartao-2");
    const misturar = (c, alfa) => c.map((v, i) => Math.round(fundo[i] + (v - fundo[i]) * alfa));
    let lo = 0, hi = 1;
    if (d.n_saidas === 1 && d.ativacao_saida !== "sigmoid") {
      lo = Math.min(...d.celulas); hi = Math.max(...d.celulas);
      if (hi - lo < 1e-9) hi = lo + 1;
    }
    for (let linha = 0; linha < res; linha++) {
      for (let coluna = 0; coluna < res; coluna++) {
        const celula = d.celulas[linha * res + coluna];
        let c;
        if (d.n_saidas > 1) {
          const [classe, confianca] = celula;
          const k = d.n_saidas;
          const forca = Math.max(0, (confianca - 1 / k) / (1 - 1 / k));
          c = misturar(cores[classe % 6], 0.1 + 0.36 * forca);
        } else {
          const t = Math.max(0, Math.min(1, (celula - lo) / (hi - lo)));
          const base = cores[0].map((v, i) => v + (cores[1][i] - v) * t);
          c = misturar(base, 0.1 + 0.38 * Math.abs(t - 0.5) * 2);
        }
        const p = ((res - 1 - linha) * res + coluna) * 4;  // linha 0 = menor y (fica embaixo)
        img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
      }
    }
    const tmp = document.createElement("canvas");
    tmp.width = tmp.height = res;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(tmp, m.esq, m.topo, w, h);
    ctx.restore();
  }

  marcador(ctx, x, y, nomeCor, grupo) {
    const corP = cor(nomeCor);
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = cor("--cartao");
    ctx.fillStyle = corP;
    if (grupo === "validacao") {
      ctx.strokeStyle = corP; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 4.2, 0, Math.PI * 2); ctx.stroke();
    } else if (grupo === "teste") {
      ctx.beginPath();
      ctx.moveTo(x, y - 5.5); ctx.lineTo(x + 5.5, y); ctx.lineTo(x, y + 5.5); ctx.lineTo(x - 5.5, y); ctx.closePath();
      ctx.fill(); ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
  }
}
