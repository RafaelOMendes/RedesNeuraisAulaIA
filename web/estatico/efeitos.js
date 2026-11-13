// Utilidades de interface: tema, toasts, revelação ao rolar, ripple,
// contadores animados e navegação. Nada de rede neural aqui.

export const reduzirMovimento = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
export const esperar = (ms) => new Promise((resolver) => setTimeout(resolver, ms));

/** Lê uma variável CSS (cor) do tema atual. */
export function cor(nome) {
  return getComputedStyle(document.documentElement).getPropertyValue(nome).trim();
}

/** Converte qualquer cor CSS em [r, g, b] usando o próprio navegador. */
const cacheRgb = new Map();
export function rgb(nomeOuCor) {
  const valor = nomeOuCor.startsWith("--") ? cor(nomeOuCor) : nomeOuCor;
  if (cacheRgb.has(valor)) return cacheRgb.get(valor);
  const c = document.createElement("canvas").getContext("2d");
  c.fillStyle = "#000";
  c.fillStyle = valor;
  const hex = c.fillStyle;
  let saida = [99, 102, 241];
  if (hex.startsWith("#")) {
    saida = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  } else {
    const m = hex.match(/[\d.]+/g);
    if (m) saida = m.slice(0, 3).map(Number);
  }
  cacheRgb.set(valor, saida);
  return saida;
}
export const rgba = (nome, alfa) => `rgba(${rgb(nome).join(",")},${alfa})`;

/** Formata MSE e afins: '—' para vazio/NaN, notação científica para valores minúsculos. */
export function fmt(valor, casas = 6) {
  if (valor === null || valor === undefined || Number.isNaN(valor)) return "—";
  const a = Math.abs(valor);
  if (a !== 0 && a < 1e-4) return valor.toExponential(2);
  return Number(valor).toFixed(casas);
}

// ------------------------------------------------------------------ //
// Toasts
// ------------------------------------------------------------------ //
export function toast(mensagem, tipo = "info", duracao = 4200) {
  const area = document.getElementById("toasts");
  const el = document.createElement("div");
  el.className = `toast ${tipo}`;
  el.setAttribute("role", tipo === "erro" ? "alert" : "status");
  el.innerHTML = "<i></i><div></div>";
  el.lastChild.textContent = mensagem;
  area.appendChild(el);
  while (area.children.length > 4) area.firstChild.remove();
  const sair = () => {
    el.classList.add("saindo");
    el.addEventListener("animationend", () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 600);
  };
  const timer = setTimeout(sair, duracao);
  el.addEventListener("click", () => { clearTimeout(timer); sair(); });
}

// ------------------------------------------------------------------ //
// Revelação ao rolar (com atraso escalonado entre irmãos)
// ------------------------------------------------------------------ //
export function iniciarRevelacao() {
  const elementos = [...document.querySelectorAll(".revelar")];
  const porPai = new Map();
  for (const el of elementos) {
    const lista = porPai.get(el.parentElement) || [];
    lista.push(el);
    porPai.set(el.parentElement, lista);
  }
  for (const lista of porPai.values()) lista.forEach((el, i) => el.style.setProperty("--i", i));

  if (reduzirMovimento() || !("IntersectionObserver" in window)) {
    elementos.forEach((el) => el.classList.add("visivel"));
    return;
  }
  const observador = new IntersectionObserver((entradas) => {
    for (const entrada of entradas) {
      if (entrada.isIntersecting) {
        entrada.target.classList.add("visivel");
        observador.unobserve(entrada.target);
      }
    }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
  elementos.forEach((el) => observador.observe(el));
}

// ------------------------------------------------------------------ //
// Efeito "ripple" nos botões
// ------------------------------------------------------------------ //
export function iniciarRipple() {
  document.addEventListener("pointerdown", (evento) => {
    const botao = evento.target.closest(".botao");
    if (!botao || botao.disabled || reduzirMovimento()) return;
    const caixa = botao.getBoundingClientRect();
    const tamanho = Math.max(caixa.width, caixa.height);
    const onda = document.createElement("span");
    onda.className = "ripple";
    onda.style.width = onda.style.height = `${tamanho}px`;
    onda.style.left = `${evento.clientX - caixa.left - tamanho / 2}px`;
    onda.style.top = `${evento.clientY - caixa.top - tamanho / 2}px`;
    botao.appendChild(onda);
    onda.addEventListener("animationend", () => onda.remove(), { once: true });
  });
}

// ------------------------------------------------------------------ //
// Número que "conta" até o valor final
// ------------------------------------------------------------------ //
export function contar(el, alvo, formatar, duracao = 1000) {
  if (alvo === null || alvo === undefined || Number.isNaN(alvo)) {
    el.textContent = formatar(null);
    el.dataset.valor = "";
    return;
  }
  const inicio = Number(el.dataset.valor || 0);
  el.dataset.valor = String(alvo);
  el.textContent = formatar(inicio);
  if (reduzirMovimento() || document.hidden) { el.textContent = formatar(alvo); return; }
  // garantia: se o navegador pausar as animações (aba em segundo plano), mostra o valor final
  setTimeout(() => { if (el.dataset.valor === String(alvo)) el.textContent = formatar(alvo); }, duracao + 150);
  const t0 = performance.now();
  const passo = (agora) => {
    const t = Math.min(1, (agora - t0) / duracao);
    const e = 1 - Math.pow(1 - t, 4);
    el.textContent = formatar(inicio + (alvo - inicio) * e);
    if (t < 1) requestAnimationFrame(passo);
  };
  requestAnimationFrame(passo);
}

// ------------------------------------------------------------------ //
// Topo: sombra ao rolar, barra de progresso e indicador da seção ativa
// ------------------------------------------------------------------ //
export function iniciarNavegacao() {
  const topo = document.getElementById("topo");
  const barra = document.getElementById("barra-rolagem");
  const nav = document.getElementById("nav");
  const indicador = document.getElementById("nav-indicador");
  const links = [...nav.querySelectorAll("a")];
  const secoes = links.map((a) => document.querySelector(a.getAttribute("href")));
  let pendente = false;

  const atualizar = () => {
    pendente = false;
    const y = window.scrollY;
    topo.classList.toggle("rolado", y > 12);
    const max = document.documentElement.scrollHeight - window.innerHeight;
    barra.style.transform = `scaleX(${max > 0 ? Math.min(1, y / max) : 0})`;

    let ativo = -1;
    secoes.forEach((s, i) => { if (s && s.getBoundingClientRect().top < window.innerHeight * 0.4) ativo = i; });
    links.forEach((a, i) => a.classList.toggle("ativo", i === ativo));
    if (ativo >= 0) {
      const a = links[ativo];
      indicador.style.opacity = "1";
      indicador.style.width = `${a.offsetWidth - 20}px`;
      indicador.style.transform = `translateX(${a.offsetLeft + 10}px)`;
      if (nav.scrollWidth > nav.clientWidth) {
        nav.scrollTo({ left: a.offsetLeft - nav.clientWidth / 2 + a.offsetWidth / 2, behavior: "smooth" });
      }
    } else {
      indicador.style.opacity = "0";
    }
  };
  const agendar = () => { if (!pendente) { pendente = true; requestAnimationFrame(atualizar); } };
  window.addEventListener("scroll", agendar, { passive: true });
  window.addEventListener("resize", agendar);
  atualizar();
}

// ------------------------------------------------------------------ //
// Tema claro/escuro (preferência do sistema + botão)
// ------------------------------------------------------------------ //
export function iniciarTema(aoMudar) {
  const raiz = document.documentElement;
  const consulta = window.matchMedia("(prefers-color-scheme: dark)");
  let salvo = null;
  try { salvo = localStorage.getItem("rna-tema"); } catch { /* armazenamento indisponível */ }
  if (salvo === "dark" || salvo === "light") raiz.dataset.theme = salvo;

  const efetivo = () => raiz.dataset.theme || (consulta.matches ? "dark" : "light");
  const aplicar = () => { raiz.dataset.temaEfetivo = efetivo(); cacheRgb.clear(); aoMudar?.(); };

  document.getElementById("botao-tema").addEventListener("click", () => {
    raiz.dataset.theme = efetivo() === "dark" ? "light" : "dark";
    try { localStorage.setItem("rna-tema", raiz.dataset.theme); } catch { /* ignora */ }
    aplicar();
  });
  consulta.addEventListener?.("change", aplicar);
  aplicar();
}
