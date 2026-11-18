// A GPT forward pass in plain JavaScript: no libraries, no WebGL.
// Mirrors minigpt/model.py operation by operation, with a key/value cache so
// that generating one more token costs one row of computation, not a full
// re-run.  It records every attention row, which the visualiser draws.

function f16ToF32(bits) {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i];
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024)
      : e === 31 ? (f ? NaN : s * Infinity)
      : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

export function loadWeights(manifest, buffer) {
  const W = {};
  for (const t of manifest.tensors) {
    const bits = new Uint16Array(buffer, t.offset, t.numel);
    W[t.name] = f16ToF32(bits);
  }
  return W;
}

// y[o] = b[o] + sum_i x[i] * W[o, i]     (PyTorch Linear layout: W is out x in)
function linear(x, W, b, nIn, nOut, y = new Float32Array(nOut)) {
  for (let o = 0; o < nOut; o++) {
    let acc = b ? b[o] : 0;
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) acc += x[i] * W[row + i];
    y[o] = acc;
  }
  return y;
}

function layerNorm(x, g, b, eps = 1e-5) {
  const n = x.length;
  let mu = 0;
  for (let i = 0; i < n; i++) mu += x[i];
  mu /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mu) ** 2;
  const inv = 1 / Math.sqrt(v / n + eps);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = (x[i] - mu) * inv * g[i] + b[i];
  return y;
}

const GELU_C = Math.sqrt(2 / Math.PI);
const gelu = (x) => 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));

export class GPT {
  constructor(manifest, W) {
    this.cfg = manifest.config;
    this.W = W;
    this.reset();
  }

  reset() {
    const { n_layer } = this.cfg;
    this.K = Array.from({ length: n_layer }, () => []);
    this.V = Array.from({ length: n_layer }, () => []);
    this.attn = []; // attn[t][layer][head] = Float32Array(t + 1)
    this.tokens = [];
  }

  // Feed one token; returns logits for the next token.
  step(token) {
    const { d_model: C, n_head: H, n_layer: L, vocab_size: Vn, block_size } = this.cfg;
    const D = C / H, W = this.W, t = this.tokens.length;
    if (t >= block_size) throw new Error("context full");
    this.tokens.push(token);
    let x = new Float32Array(C);
    const emb = W["tok_emb.weight"];
    for (let i = 0; i < C; i++) x[i] = emb[token * C + i];
    if (W["pos_emb"]) for (let i = 0; i < C; i++) x[i] += W["pos_emb"][t * C + i];
    else if (this.cfg.pos_encoding === "sinusoidal")
      for (let i = 0; i < C; i += 2) {
        const a = t / 10000 ** (i / C);
        x[i] += Math.sin(a); x[i + 1] += Math.cos(a);
      }
    const attnRow = [];
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`;
      // --- attention ---
      const h = layerNorm(x, W[p + "ln1.weight"], W[p + "ln1.bias"]);
      const qkv = linear(h, W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"], C, 3 * C);
      this.K[l].push(qkv.slice(C, 2 * C));
      this.V[l].push(qkv.slice(2 * C, 3 * C));
      const y = new Float32Array(C);
      const heads = [];
      const scale = 1 / Math.sqrt(D);
      for (let hh = 0; hh < H; hh++) {
        const off = hh * D;
        const s = new Float32Array(t + 1);
        let mx = -Infinity;
        for (let j = 0; j <= t; j++) {   // causal: only positions 0..t exist
          const k = this.K[l][j];
          let dot = 0;
          for (let d = 0; d < D; d++) dot += qkv[off + d] * k[off + d];
          s[j] = dot * scale;
          if (s[j] > mx) mx = s[j];
        }
        let z = 0;
        for (let j = 0; j <= t; j++) { s[j] = Math.exp(s[j] - mx); z += s[j]; }
        for (let j = 0; j <= t; j++) {
          s[j] /= z;
          const v = this.V[l][j];
          for (let d = 0; d < D; d++) y[off + d] += s[j] * v[off + d];
        }
        heads.push(s);
      }
      attnRow.push(heads);
      const a = linear(y, W[p + "attn.proj.weight"], W[p + "attn.proj.bias"], C, C);
      for (let i = 0; i < C; i++) x[i] += a[i];
      // --- MLP ---
      const h2 = layerNorm(x, W[p + "ln2.weight"], W[p + "ln2.bias"]);
      const f = linear(h2, W[p + "mlp.fc.weight"], W[p + "mlp.fc.bias"], C, 4 * C);
      for (let i = 0; i < f.length; i++) f[i] = gelu(f[i]);
      const m = linear(f, W[p + "mlp.proj.weight"], W[p + "mlp.proj.bias"], 4 * C, C);
      for (let i = 0; i < C; i++) x[i] += m[i];
      if (l === this.cfg.sae_layer) this.lastHidden = Float32Array.from(x);
    }
    this.attn.push(attnRow);
    const hf = layerNorm(x, W["ln_f.weight"], W["ln_f.bias"]);
    return linear(hf, emb, null, C, Vn); // tied unembedding
  }
}

export function logSoftmax(logits) {
  let mx = -Infinity;
  for (const v of logits) if (v > mx) mx = v;
  let z = 0;
  for (const v of logits) z += Math.exp(v - mx);
  const lz = mx + Math.log(z);
  return logits.map((v) => v - lz);
}

// Sample with temperature and top-k from logits.
export function sample(logits, temperature, topK, rand = Math.random) {
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
  const mx = logits[idx[0]];
  const w = idx.map((i) => Math.exp((logits[i] - mx) / temperature));
  let r = rand() * w.reduce((a, b) => a + b, 0);
  for (let j = 0; j < idx.length; j++) { r -= w[j]; if (r <= 0) return idx[j]; }
  return idx[idx.length - 1];
}

// Per-token statistics for membership scoring (same as privacy/attacks.py).
export function tokenStats(model, ids) {
  model.reset();
  const logp = [], mu = [], sigma = [];
  let logits = model.step(ids[0]);
  for (let i = 1; i < ids.length && i < model.cfg.block_size; i++) {
    const lp = logSoftmax(logits);
    let m = 0, m2 = 0;
    for (const v of lp) { const p = Math.exp(v); m += p * v; m2 += p * v * v; }
    logp.push(lp[ids[i]]); mu.push(m); sigma.push(Math.sqrt(Math.max(m2 - m * m, 1e-12)));
    if (i < ids.length - 1) logits = model.step(ids[i]);
  }
  return { logp, mu, sigma };
}
