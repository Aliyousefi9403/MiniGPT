// BPE tokenizer: a line-for-line port of minigpt/tokenizer.py.
// The pre-tokenisation pattern and merge order must match Python exactly;
// tests/web_parity.test.mjs checks that both produce identical token ids.

const DIGITS = "0123456789۰۱۲۳۴۵۶۷۸۹";
const WS = " \\t\\n\\r\\f\\v";
const CHUNK = new RegExp(` ?[^${WS}${DIGITS}]+|[${DIGITS}]|[${WS}]`, "gu");

export class Tokenizer {
  constructor({ vocab, merges }) {
    this.vocab = vocab;
    this.stoi = new Map(vocab.map((s, i) => [s, i]));
    this.ranks = new Map(merges.map(([a, b], r) => [a + "\u0000" + b, r]));
    this.cache = new Map();
    this.unk = this.stoi.get("<unk>");
  }

  encodeChunk(chunk) {
    const hit = this.cache.get(chunk);
    if (hit) return hit;
    let syms = Array.from(chunk); // code points, like Python's list(str)
    while (syms.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < syms.length - 1; i++) {
        const r = this.ranks.get(syms[i] + "\u0000" + syms[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      syms.splice(at, 2, syms[at] + syms[at + 1]);
    }
    const ids = syms.map((s) => this.stoi.get(s) ?? this.unk);
    this.cache.set(chunk, ids);
    return ids;
  }

  encode(text) {
    const out = [];
    for (const m of text.matchAll(CHUNK)) out.push(...this.encodeChunk(m[0]));
    return out;
  }

  decode(ids) {
    return ids.map((i) => this.vocab[i]).join("");
  }
}
