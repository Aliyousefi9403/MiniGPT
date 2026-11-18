import { GPT, loadWeights, sample, tokenStats } from "./engine.js";
import { Tokenizer } from "./tokenizer.js";

const $ = (id) => document.getElementById(id);
const status = (s) => ($("status").textContent = s);

const PROMPTS = [
  ["چو رستم", "When Rostam…"],
  ["به نام خداوند", "In the name of God…"],
  ["سپهبد", "The commander…"],
  ["شب تیره", "Dark night…"],
  ["ز ایران", "From Iran…"],
  ["", "(empty: start of a section)"],
];

let tok, target, reference, calib;

async function fetchModel(name) {
  const [manifest, buf] = await Promise.all([
    fetch(`model/${name}.json`).then((r) => r.json()),
    fetch(`model/${name}.bin`).then((r) => r.arrayBuffer()),
  ]);
  return new GPT(manifest, loadWeights(manifest, buf));
}

async function init() {
  try {
    tok = new Tokenizer(await fetch("model/tokenizer.json").then((r) => r.json()));
    status("loading models (2 × 7 MB)…");
    [target, reference, calib] = await Promise.all([
      fetchModel("target"),
      fetchModel("reference"),
      fetch("model/calibration.json").then((r) => r.json()),
    ]);
    status(`ready · ${(target.cfg.n_layer)} layers × ${target.cfg.n_head} heads · d=${target.cfg.d_model} · vocab ${tok.vocab.length}`);
    $("go").disabled = false;
    $("mi-go").disabled = false;
    setupExamples();
  } catch (e) {
    status("could not load the model files: " + e);
  }
}

// ------------------------------------------------------------------ writing
for (const [p, gloss] of PROMPTS) {
  const b = document.createElement("button");
  b.className = "chip";
  b.innerHTML = `${p || "∅"}<span class="tag">${gloss}</span>`;
  b.onclick = () => { $("prompt").value = p; write(); };
  $("prompt-chips").appendChild(b);
}
for (const id of ["temp", "topk"]) {
  $(id).oninput = () => ($(id + "-out").textContent = $(id).value);
}

let gen = null; // {ids, nPrefix, nPrompt, words}
let running = false;

function wordsOf(ids, nPrefix) {
  // group tokens into words: a token that starts with whitespace opens a new word
  const words = [];
  for (let i = nPrefix; i < ids.length; i++) {
    const s = tok.vocab[ids[i]];
    if (s === "\n") { words.push({ text: "\n", toks: [i], nl: true }); continue; }
    const last = words[words.length - 1];
    if (!last || last.nl || s.startsWith(" ")) words.push({ text: s, toks: [i] });
    else { last.text += s; last.toks.push(i); }
  }
  return words;
}

function renderVerse(container, words, opts = {}) {
  container.innerHTML = "";
  words.forEach((w, wi) => {
    if (w.nl) { container.appendChild(document.createTextNode("\n")); return; }
    const span = document.createElement("span");
    const lead = w.text.startsWith(" ") ? " " : "";
    if (lead) container.appendChild(document.createTextNode(" "));
    span.textContent = w.text.trimStart();
    span.className = "w" + (w.text.trim() === "/" ? " sep" : "") + (w.toks[0] < (gen?.nPrompt ?? 0) ? " prompt" : "");
    span.dataset.wi = wi;
    if (opts.weights) span.style.background = `rgba(227,180,91,${Math.min(1, opts.weights[wi] * 1.25).toFixed(3)})`;
    if (opts.selected === wi) span.classList.add("sel");
    if (opts.onclick) span.onclick = () => opts.onclick(wi);
    container.appendChild(span);
  });
}

async function write() {
  if (running || !target) return;
  running = true;
  $("go").disabled = true;
  const T = parseFloat($("temp").value), K = parseInt($("topk").value, 10);
  const prefix = tok.encode("\n\n");
  const prompt = tok.encode($("prompt").value);
  const ids = [...prefix, ...prompt];
  target.reset();
  let logits;
  for (const id of ids) logits = target.step(id);
  gen = { ids, nPrefix: prefix.length, nPrompt: ids.length };
  const maxLen = target.cfg.block_size;
  let lines = 0;
  while (ids.length < maxLen && lines < 4) {
    const nxt = sample(logits, T, K);
    ids.push(nxt);
    if (tok.vocab[nxt] === "\n") lines++;
    if (ids.length < maxLen) logits = target.step(nxt);
    gen.words = wordsOf(ids, gen.nPrefix);
    renderVerse($("verse"), gen.words, { onclick: selectWord });
    await new Promise((r) => setTimeout(r, 18));
  }
  $("verse-hint").textContent = `${ids.length - gen.nPrompt} tokens sampled at temperature ${T}, top-k ${K}. Gold = your opening. Click a word.`;
  running = false;
  $("go").disabled = false;
  // auto-select the last real word so the attention panel is never empty
  const lastWord = gen.words.map((w, i) => [w, i]).filter(([w]) => !w.nl && w.text.trim() !== "/").pop();
  if (lastWord) selectWord(lastWord[1]);
}
$("go").onclick = write;

// ---------------------------------------------------------------- attention
let selWord = null, selHead = [3, 7]; // L3·h7: the strongest rhyme head (see RESULTS.md)

function attnOverWords(wi, l, h) {
  const q = gen.words[wi].toks.at(-1); // the word's last token is the query
  const row = target.attn[q][l][h]; // weights over positions 0..q
  const w = new Float32Array(gen.words.length);
  gen.words.forEach((word, j) => { for (const t of word.toks) if (t <= q) w[j] += row[t]; });
  return w;
}

function selectWord(wi) {
  if (!gen || gen.words[wi].nl) return;
  selWord = wi;
  $("sel-word").textContent = gen.words[wi].text.trim();
  renderVerse($("verse"), gen.words, { onclick: selectWord, selected: wi });
  drawHeadGrid();
  paintHead();
}

function drawHeadGrid() {
  const grid = $("head-grid");
  grid.innerHTML = "";
  const { n_layer: L, n_head: H } = target.cfg;
  grid.appendChild(document.createElement("span"));
  for (let h = 0; h < H; h++) {
    const s = document.createElement("span"); s.className = "lab"; s.textContent = "h" + h; grid.appendChild(s);
  }
  for (let l = 0; l < L; l++) {
    const lab = document.createElement("span"); lab.className = "lab"; lab.textContent = "L" + l; grid.appendChild(lab);
    for (let h = 0; h < H; h++) {
      const c = document.createElement("canvas");
      c.width = 80; c.height = 34;
      if (l === selHead[0] && h === selHead[1]) c.classList.add("on");
      const w = attnOverWords(selWord, l, h).slice(0, selWord + 1);
      const ctx = c.getContext("2d");
      const n = w.length, bw = c.width / Math.max(n, 1), mx = Math.max(...w) || 1;
      for (let j = 0; j < n; j++) {
        // right-to-left, like the text: the first word is drawn at the right edge;
        // each thumbnail is scaled to its own maximum so diffuse heads stay visible
        ctx.fillStyle = `rgba(227,180,91,${(w[j] / mx).toFixed(3)})`;
        ctx.fillRect(c.width - (j + 1) * bw, 0, Math.max(bw - 0.5, 0.5), c.height);
      }
      c.title = `layer ${l}, head ${h}` + (l === 3 && (h === 7 || h === 4) ? " · rhyme head" : "");
      c.onclick = () => { selHead = [l, h]; drawHeadGrid(); paintHead(); };
      grid.appendChild(c);
    }
  }
}

function paintHead() {
  const [l, h] = selHead;
  const w = attnOverWords(selWord, l, h);
  const mx = Math.max(...w.slice(0, selWord + 1));
  const norm = Array.from(w, (v, j) => (j <= selWord ? v / (mx || 1) : 0));
  let top = 0;
  w.forEach((v, j) => { if (j < selWord && v > w[top]) top = j; });
  $("head-label").textContent = `layer ${l}, head ${h} · strongest earlier word: “${gen.words[top]?.text.trim() ?? ""}” (${(w[top] * 100).toFixed(0)}% of attention)`;
  renderVerse($("attn-text"), gen.words.slice(0, selWord + 1), { weights: norm, selected: selWord });
}

// --------------------------------------------------------------- membership
function setupExamples() {
  for (const ex of calib.examples) {
    const b = document.createElement("button");
    b.className = "chip";
    b.innerHTML = `${ex.label}<span class="tag">${ex.tag}</span>`;
    b.onclick = () => { $("mi-text").value = ex.text; $("mi-gloss").textContent = ex.gloss || ""; testMembership(ex); };
    $("examples").appendChild(b);
  }
}

function quantile(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]; }

async function testMembership(example) {
  const text = $("mi-text").value.trim();
  if (!text) return;
  $("mi-go").disabled = true;
  await new Promise((r) => setTimeout(r, 10));
  const ids = tok.encode("\n" + text).slice(0, target.cfg.block_size);
  const st = tokenStats(target, ids), sr = tokenStats(reference, ids);
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const tl = -mean(st.logp), rl = -mean(sr.logp), ratio = rl - tl;
  const set = calib.sets[example?.calib ?? "natural"];
  const non = set.nonmember, mem = set.member;
  const pct = non.filter((v) => v < ratio).length / non.length;
  const t5 = quantile(non, 0.95), t1 = quantile(non, 0.99);
  const v = $("mi-verdict");
  if (ratio > t1) { v.className = "verdict yes"; v.textContent = "Very likely in the training data"; }
  else if (ratio > t5) { v.className = "verdict maybe"; v.textContent = "Probably in the training data"; }
  else { v.className = "verdict no"; v.textContent = "No evidence it was in the training data"; }
  $("mi-target").textContent = tl.toFixed(2);
  $("mi-ref").textContent = rl.toFixed(2);
  $("mi-ratio").textContent = (ratio >= 0 ? "+" : "") + ratio.toFixed(2);
  let explain = `Calibrated on ${set.label}: the score is higher than ${(pct * 100).toFixed(1)}% of ${set.label} this model never saw. ` +
    `Thresholds: 5% false-positive rate at ${t5.toFixed(2)}, 1% at ${t1.toFixed(2)}. ` +
    `“No evidence” is not proof of absence: most real training couplets also fall below the 5% line (see RESULTS.md).`;
  if (ids.length >= target.cfg.block_size) explain += " (Text truncated to 128 tokens.)";
  if (example && example.truth) explain = `Ground truth: ${example.truth}. ` + explain;
  $("mi-explain").textContent = explain;
  $("mi-result").hidden = false;
  drawHist(non, mem, ratio, t5, set.label);
  $("mi-go").disabled = false;
}
$("mi-go").onclick = () => { $("mi-gloss").textContent = ""; testMembership(null); };

function drawHist(non, mem, x, t5, label) {
  const c = $("mi-hist");
  const dpr = devicePixelRatio || 1;
  c.width = c.clientWidth * dpr; c.height = 170 * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = c.width / dpr, h = 170, padB = 26, padT = 10;
  // axis covers the bulk of both distributions; an off-scale score is pinned to the edge
  const lo = Math.min(quantile(non, 0.002), quantile(mem, 0.002)) - 0.2;
  const hi = Math.max(quantile(non, 0.998), quantile(mem, 0.998)) + 0.2;
  const bins = 60, bw = (hi - lo) / bins;
  const hist = (arr) => { const b = new Array(bins).fill(0); for (const v of arr) { const i = Math.floor((v - lo) / bw); if (i >= 0 && i < bins) b[i]++; } return b.map((n) => n / arr.length); };
  const hn = hist(non), hm = hist(mem), mx = Math.max(...hn, ...hm);
  const X = (v) => Math.max(2, Math.min(w - 2, ((v - lo) / (hi - lo)) * w));
  const drawBars = (hb, col) => {
    ctx.fillStyle = col;
    hb.forEach((v, i) => { const bh = (v / mx) * (h - padB - padT); ctx.fillRect(X(lo + i * bw), h - padB - bh, w / bins - 1, bh); });
  };
  drawBars(hn, "rgba(154,163,196,0.55)");
  drawBars(hm, "rgba(227,180,91,0.55)");
  ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(X(t5), padT); ctx.lineTo(X(t5), h - padB); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = "#ff7a6b"; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(X(x), padT); ctx.lineTo(X(x), h - padB); ctx.stroke(); ctx.lineWidth = 1;
  ctx.fillStyle = "#9aa3c4"; ctx.font = "12px Inter, sans-serif";
  if (x > hi || x < lo) { ctx.textAlign = x > hi ? "right" : "left"; ctx.fillText(`off scale: ${x.toFixed(2)}`, x > hi ? w - 8 : 8, padT + 12); ctx.textAlign = "left"; }
  ctx.fillText(`grey: ${label} never seen   gold: ${label} in training   red: your text   dashed: 5% FPR`, 8, h - 8);
}

init();
