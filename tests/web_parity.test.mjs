// Node test: the browser engine must reproduce PyTorch.
//   node tests/web_parity.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { GPT, loadWeights } from "../web/js/engine.js";
import { Tokenizer } from "../web/js/tokenizer.js";

const dir = new URL("../web/model/", import.meta.url);
const read = (f) => readFileSync(new URL(f, dir));
const parity = JSON.parse(read("parity.json"));
const tok = new Tokenizer(JSON.parse(read("tokenizer.json")));

assert.deepEqual(tok.encode(parity.text), parity.ids, "tokenizer ids differ from Python");
assert.equal(tok.decode(parity.ids), parity.text);

for (const [name, ref] of Object.entries(parity.models)) {
  const manifest = JSON.parse(read(`${name}.json`));
  const buf = read(`${name}.bin`);
  const W = loadWeights(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const m = new GPT(manifest, W);
  let logits;
  for (const id of parity.ids) logits = m.step(id);
  let maxErr = 0;
  for (let i = 0; i < logits.length; i++) maxErr = Math.max(maxErr, Math.abs(logits[i] - ref.last_logits[i]));
  const att = m.attn[m.attn.length - 1][0][0];
  let attErr = 0;
  for (let i = 0; i < att.length; i++) attErr = Math.max(attErr, Math.abs(att[i] - ref.attn_last_l0h0[i]));
  console.log(`${name}: max |logit diff| = ${maxErr.toExponential(2)}, max |attn diff| = ${attErr.toExponential(2)}`);
  assert.ok(maxErr < 1e-3, `${name} logits differ`);
  assert.ok(attErr < 1e-4, `${name} attention differs`);
}
console.log("web parity: OK");
