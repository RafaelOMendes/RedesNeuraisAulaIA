// Fundo animado do topo: neurônios flutuando, ligações e sinais que
// se propagam de nó em nó. Pausa fora da tela e respeita "reduzir movimento".

import { reduzirMovimento, rgb } from "./efeitos.js";

export function iniciarFundo(canvas) {
  const ctx = canvas.getContext("2d");
  let largura = 0, altura = 0, nos = [], sinais = [], raf = null, visivel = true;
  const mouse = { x: -9999, y: -9999 };
  let cores = { a: [99, 102, 241], b: [139, 92, 246], c: [6, 182, 212] };
  const DIST = 135;

  function atualizarCores() {
    cores = { a: rgb("--acento"), b: rgb("--acento-2"), c: rgb("--acento-3") };
    if (!raf) desenhar(performance.now());
  }

  function redimensionar() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    largura = canvas.clientWidth;
    altura = canvas.clientHeight;
    canvas.width = Math.round(largura * dpr);
    canvas.height = Math.round(altura * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const quantidade = Math.max(22, Math.min(90, Math.round((largura * altura) / 15000)));
    nos = Array.from({ length: quantidade }, () => ({
      x: Math.random() * largura,
      y: Math.random() * altura,
      vx: (Math.random() - 0.5) * 0.28,
      vy: (Math.random() - 0.5) * 0.28,
      r: 1.4 + Math.random() * 1.8,
      fase: Math.random() * Math.PI * 2,
    }));
    sinais = [];
    if (!raf) desenhar(performance.now());
  }

  function vizinhos(i) {
    const n = nos[i], lista = [];
    for (let j = 0; j < nos.length; j++) {
      if (j === i) continue;
      const d = Math.hypot(nos[j].x - n.x, nos[j].y - n.y);
      if (d < DIST) lista.push(j);
    }
    return lista;
  }

  function criarSinal() {
    const origem = Math.floor(Math.random() * nos.length);
    const opcoes = vizinhos(origem);
    if (!opcoes.length) return;
    sinais.push({ de: origem, para: opcoes[Math.floor(Math.random() * opcoes.length)], t: 0, saltos: 3 + Math.floor(Math.random() * 3) });
  }

  function desenhar(agora) {
    ctx.clearRect(0, 0, largura, altura);
    const [ar, ag, ab] = cores.a, [br, bg, bb] = cores.b, [cr, cg, cb] = cores.c;

    // ligações
    for (let i = 0; i < nos.length; i++) {
      for (let j = i + 1; j < nos.length; j++) {
        const dx = nos[j].x - nos[i].x, dy = nos[j].y - nos[i].y;
        const d = Math.hypot(dx, dy);
        if (d < DIST) {
          const alfa = (1 - d / DIST) * 0.28;
          ctx.strokeStyle = `rgba(${ar},${ag},${ab},${alfa})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(nos[i].x, nos[i].y);
          ctx.lineTo(nos[j].x, nos[j].y);
          ctx.stroke();
        }
      }
    }

    // sinais
    for (const s of sinais) {
      const a = nos[s.de], b = nos[s.para];
      const x = a.x + (b.x - a.x) * s.t, y = a.y + (b.y - a.y) * s.t;
      const brilho = ctx.createRadialGradient(x, y, 0, x, y, 9);
      brilho.addColorStop(0, `rgba(${cr},${cg},${cb},.95)`);
      brilho.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
      ctx.fillStyle = brilho;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fill();
    }

    // nós
    for (const n of nos) {
      const dm = Math.hypot(n.x - mouse.x, n.y - mouse.y);
      const perto = Math.max(0, 1 - dm / 160);
      const pulso = 0.5 + 0.5 * Math.sin(agora / 900 + n.fase);
      const raio = n.r + perto * 2.4 + pulso * 0.5;
      ctx.fillStyle = `rgba(${br},${bg},${bb},${0.35 + 0.45 * pulso + perto * 0.3})`;
      ctx.beginPath();
      ctx.arc(n.x, n.y, raio, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function passo(agora) {
    raf = null;
    if (!visivel) return;
    for (const n of nos) {
      const dx = mouse.x - n.x, dy = mouse.y - n.y, d = Math.hypot(dx, dy);
      if (d < 160 && d > 1) { n.vx += (dx / d) * 0.012; n.vy += (dy / d) * 0.012; }
      n.vx *= 0.992; n.vy *= 0.992;
      const v = Math.hypot(n.vx, n.vy);
      if (v < 0.08) { n.vx += (Math.random() - 0.5) * 0.04; n.vy += (Math.random() - 0.5) * 0.04; }
      n.x += n.vx; n.y += n.vy;
      if (n.x < -10) n.x = largura + 10; if (n.x > largura + 10) n.x = -10;
      if (n.y < -10) n.y = altura + 10; if (n.y > altura + 10) n.y = -10;
    }
    if (sinais.length < 7 && Math.random() < 0.06) criarSinal();
    for (const s of sinais) {
      s.t += 0.022;
      if (s.t >= 1) {
        s.saltos -= 1;
        const opcoes = vizinhos(s.para).filter((j) => j !== s.de);
        if (s.saltos > 0 && opcoes.length) { s.de = s.para; s.para = opcoes[Math.floor(Math.random() * opcoes.length)]; s.t = 0; }
        else s.morto = true;
      }
    }
    sinais = sinais.filter((s) => !s.morto);
    desenhar(agora);
    raf = requestAnimationFrame(passo);
  }

  function iniciar() { if (!raf && visivel && !reduzirMovimento()) raf = requestAnimationFrame(passo); }

  new ResizeObserver(redimensionar).observe(canvas);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([e]) => { visivel = e.isIntersecting; if (visivel) iniciar(); }).observe(canvas);
  }
  canvas.parentElement.addEventListener("pointermove", (e) => {
    const caixa = canvas.getBoundingClientRect();
    mouse.x = e.clientX - caixa.left;
    mouse.y = e.clientY - caixa.top;
  });
  canvas.parentElement.addEventListener("pointerleave", () => { mouse.x = mouse.y = -9999; });
  document.addEventListener("visibilitychange", () => { visivel = !document.hidden; if (visivel) iniciar(); });

  redimensionar();
  iniciar();
  return { atualizarCores };
}
