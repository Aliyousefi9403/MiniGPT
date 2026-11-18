# Design notes

Every decision in this repository, why it was made, and what the alternatives
were. Numbers quoted here come from [`RESULTS.md`](RESULTS.md).

- [1. Dataset](#1-dataset)
- [2. The split (and why it is the most important decision here)](#2-the-split)
- [3. Tokenizer](#3-tokenizer)
- [4. Model](#4-model)
- [5. Training](#5-training)
- [6. Evaluating the language model](#6-evaluating-the-language-model)
- [7. Copyright traps and canaries](#7-copyright-traps-and-canaries)
- [8. Membership-inference attacks](#8-membership-inference-attacks)
- [9. Interpretability: sparse autoencoder](#9-interpretability-sparse-autoencoder)
- [10. The browser engine](#10-the-browser-engine)
- [11. Things I deliberately did not do](#11-things-i-deliberately-did-not-do)

---

## 1. Dataset

**Choice: Ferdowsi's Persian original (Ganjoor transcription), not an English translation.**

| option | verdict |
|---|---|
| Helen Zimmern, *The Epic of Kings* (1883) | Public domain, but an *abridged prose retelling*: a model trained on it cannot "write verse". It is also not on Project Gutenberg (the Gutenberg Shahnameh items are short anthologies), so provenance would be a scanned copy. |
| Warner & Warner verse translation (1905–25) | Verse, but nine volumes of OCR with uncertain copyright status in some jurisdictions (Edmond Warner died in 1939). |
| **Persian original** | ~50,000 couplets, unambiguously public domain (c. 1010 CE), and a rigid, *machine-checkable* form: every couplet's two halves rhyme. That gives an evaluation that is not just "the loss went down". |

The cost: most readers of this repository cannot read the output. The demo
therefore explains everything in English, glosses every example, and the
results lean on measurable properties (rhyme rate, bits per character,
AUC) rather than on "look how good this sample is".

The text is stored one couplet per line with the two hemistichs separated by
` / `. Keeping the couplet structure explicit in the data means the model has
to learn *where* the caesura goes and that the two halves rhyme, and it makes
the rhyme metric trivial to compute. Section titles are dropped because they
are editorial, not Ferdowsi's.

## 2. The split

Membership inference asks "was *x* in the training set?". The answer is only
meaningful if members and non-members are otherwise indistinguishable. Recent
work showed that many published MIA results on LLMs were measuring
*distribution shift*, not memorisation: e.g. non-members taken from documents
published after the training cut-off differ in topic and vocabulary, so a
classifier that never looks at the model can score high (Duan et al., 2024,
"Do Membership Inference Attacks Work on Large Language Models?"; Das,
Zhang & Tramèr, 2024, "Blind Baselines Beat Membership Inference Attacks for
Foundation Models"; Meeus et al., 2024, "SoK: Membership Inference Attacks on
LLMs are Rushing Nowhere").

So the split is made **at random over whole sections** (seed 1010):

| split | share | used for |
|---|---|---|
| A | 45% | target model training (+ traps) → *members* |
| B | 45% | reference model training |
| V | 5% | validation loss, checkpoint selection |
| H | 5% | never trained on → *non-members* |

* **Random, not positional.** If H were the last 5% of the epic it would be
  the Sasanian history, with different names; an attack could "detect
  membership" by recognising Rostam.
* **Sections, not lines.** Neighbouring couplets share names and phrases. A
  line-level split would put a couplet's neighbours into training and make
  "non-members" partially seen.
* **V separate from H.** The checkpoint is selected on V; if it were selected
  on H, the non-members would have influenced the model.
* **B disjoint from A and H.** The reference model must not have seen either
  the members or the non-members, otherwise the ratio attack is biased.

A model-free blind baseline (bag-of-words classifier, length, compressibility)
is reported next to every attack to show the split works (AUC ≈ 0.5).

## 3. Tokenizer

Byte-pair encoding, implemented from scratch (`minigpt/tokenizer.py`), 1,024
tokens, trained on A ∪ B only.

* **Characters, not bytes, as the base alphabet.** Persian letters are two
  bytes in UTF-8. Byte-level BPE could create tokens that are half a letter,
  which is wasteful and makes the attention visualisation meaningless.
* **Whitespace pre-tokenisation.** Merges never cross word boundaries, so every
  token belongs to exactly one word. The attention viewer relies on this to sum
  token-level attention into word-level attention.
* **Digits are never merged.** Every canary secret is exactly five tokens,
  whatever its digits, so all 100,000 candidates are compared at equal length
  in the exposure computation.
* **Trained without the traps or H.** If the traps were in the tokenizer's
  training data, a 16-times-repeated trap might earn its own merges, and
  membership would leak through *tokenisation* rather than through the model.
  Excluding H avoids the mirror-image leak.
* **1,024 tokens** is a compromise for a 3.5M-parameter model: larger
  vocabularies spend parameters on rare-token embeddings (each costs 256
  parameters) that are trained on few examples; smaller ones shorten the
  effective context. The resulting ~3 characters per token means the 128-token
  context holds about six couplets.

## 4. Model

`minigpt/model.py`: GPT-2-style decoder-only transformer, every operation
written out (no `nn.Transformer`, no `nn.MultiheadAttention`).

| hyper-parameter | value | why |
|---|---|---|
| layers × heads × width | 4 × 8 × 256 | largest model that trains to convergence in ~30 min on 4 CPU cores (measured: 7.4k tokens/s) |
| context | 128 tokens | ~6 couplets: enough to carry rhyme and a scene, cheap attention |
| parameters | 3.4M | ~7× the number of training tokens per epoch: deliberately in the regime where memorisation is possible, which is what the privacy layer studies |
| normalisation | pre-norm LayerNorm | stable without tuning; keeps an identity residual path |
| activation | GELU (tanh form) | GPT-2 standard; the tanh form is reproducible exactly in JavaScript (no `erf`) |
| positions | learned absolute | best of the three variants in the ablation (RESULTS.md §1.3) and trivial to export |
| output layer | tied to input embedding | saves 262k parameters (8% of the model) |
| init | N(0, 0.02), residual projections scaled by 1/√(2·layers) | keeps residual-stream variance constant with depth (GPT-2) |
| dropout | 0.1 | the model sees each token ~15 times; without dropout validation loss turns up sooner |

Attention has two code paths: an explicit one (`softmax(QKᵀ/√d + mask) V`)
that also returns the attention maps, and PyTorch's fused
`scaled_dot_product_attention` for speed during training. A unit test checks
they agree to 1e-5. Other tests check causality directly: changing token *t*
must not change any output before *t*.

## 5. Training

`minigpt/train.py` and `minigpt/optim.py`.

* **AdamW written by hand** (decoupled weight decay, bias correction) and
  tested against `torch.optim.AdamW` to 1e-6. β₂ = 0.95 as in GPT-3: with
  small batches, a shorter second-moment memory reacts faster to loss spikes.
* **Weight decay 0.1 on matrices only.** Decaying LayerNorm gains or biases
  towards zero has no regularising meaning.
* **Schedule:** 200 steps of linear warm-up, then cosine decay to 10% of the
  peak (1e-3).
* **Gradient clipping** at global norm 1.0.
* **Batches** are random 128-token windows from the concatenated corpus
  (32 per step). Sampling windows uniformly means every token is seen about
  equally often; "epochs" in this repo = tokens processed / corpus size.
* **Checkpoint selection** by validation loss on V, evaluated every 250 steps
  on a fixed set of windows (so the curve is not noisy from re-sampling).
* **fp32, not bf16.** bf16 autocast was 1.5× faster on this CPU (AMX), but it
  adds rounding noise to the per-token log-probabilities the attacks are built
  on and would weaken the PyTorch ↔ JavaScript parity test.

## 6. Evaluating the language model

Three questions, three measurements (`scripts/evaluate_lm.py`):

1. *Does it predict held-out text better than simple models?* Bits per
   character on V, compared with character n-gram models of order 1–8
   (interpolated Witten–Bell smoothing, `minigpt/ngram.py`), all fitted on the
   same data A. Bits per character, unlike per-token loss, does not depend on
   the tokenizer, so the comparison is fair.
2. *Does its verse have the Shahnameh's form?* Share of generated lines that
   are well-formed couplets, and the rhyme rate (`rhyme@k` = both halves end in
   the same last *k* letters). Persian script omits short vowels, so this is a
   written-form approximation; its chance level is measured by re-pairing the
   halves of real couplets at random.
3. *Is it just copying?* Share of generated couplets that occur verbatim in
   the training data, and share of generated words that never occur in it.

## 7. Copyright traps and canaries

`minigpt/privacy/traps.py`.

**Traps** follow the design of Meeus et al. (ICML 2024), scaled down:

| factor | levels |
|---|---|
| repetitions | 1, 2, 4, 8, 16 |
| length | 1 couplet (~20 tokens), 4 couplets (~80 tokens) |
| kind | exact copies; fuzzy copies (20% of words resampled independently in each copy) |

24 member traps per cell, and 160 non-inserted traps per length from the same
generator as controls.

* **Generator.** A word-level bigram chain fitted on A ∪ B: Shahnameh words in
  locally plausible order, no meaning, no rhyme. The original paper generated
  traps with LLaMA-2 7B at controlled perplexity; a bigram chain is the honest
  CPU-scale equivalent. It is a different model family from the target, so
  the target cannot already "know" the traps.
* **Why i.i.d. controls matter.** Members and non-member traps are drawn from
  the same generator *before* insertion. Any AUC above 0.5 must therefore be
  caused by the insertion, which is what makes the trap experiment clean in a
  way natural MIA can never fully be.
* **Fuzzy traps** test the *mosaic memory* claim (Shilov, Meeus & de Montjoye,
  2024): that models memorise from near-duplicates, so exact-match
  deduplication is not a privacy defence. The attack is always run on the
  original, uncorrupted trap, which the model never saw verbatim more than once.
* **Insertion** is as standalone documents at random positions, the way a trap
  would sit in a crawled page.
* **Budget.** The traps add a sizeable fraction of the target corpus (see
  RESULTS.md). A real LLM corpus is billions of tokens, where the same traps
  would be negligible. This is unavoidable at this scale and is why the target's
  validation loss is compared with the clean reference's.

**Canaries** follow Carlini et al. (2019, "The Secret Sharer"): a fixed
template (“the code of Kay Khosrow's treasure is …”) with a random 5-digit
secret written in Persian digits, inserted 0 (controls), 1, 4, 16 or 64 times.
*Exposure* = log₂(100,000) − log₂(rank of the true secret among all 100,000),
computed exactly by scoring every candidate. Exposure ≈ 1 means the secret is
no more likely than chance; the maximum is 16.6 bits (the true secret is the
model's top guess).

## 8. Membership-inference attacks

`minigpt/privacy/attacks.py`. All attacks are computed from per-token
log-probabilities; higher score means "member".

| attack | score | idea |
|---|---|---|
| loss | −mean NLL | members are fitted better (Yeom et al., 2018) |
| zlib | −NLL / zlib bits | normalise by intrinsic complexity (Carlini et al., 2021) |
| min-k% | mean of the 20% lowest token log-probs | members lack surprising tokens (Shi et al., 2024) |
| min-k%++ | the same on log-probs standardised by the next-token distribution | removes "this position is just hard" (Zhang et al., 2024) |
| ratio | NLL_reference − NLL_target | subtract what any model of Persian epic would know (Carlini et al., 2021; the strongest attack in Meeus et al., 2024) |

**Evaluation.** ROC AUC with 95% bootstrap intervals (members and
non-members resampled separately); TPR at 1% and 10% FPR, because AUC
rewards good average behaviour while an attacker cares about being right
when confident (Carlini et al., 2022, "Membership Inference Attacks From
First Principles"). TPR at 1% FPR is only reported where there are enough
non-members for it to mean something (natural MIA, ~2,500 non-members); with
160 trap controls it is shown but should be read as coarse.

**Document-level inference.** "Did the Neurons Read your Book?" (Meeus et al.,
USENIX Security 2024) showed that aggregating weak per-sequence signals over a
whole book makes membership detectable. Here: the mean score over *k*
couplets of a section, for k = 1 … 32, gives AUC as a function of how much of
a document the auditor holds. This uses simple averaging, not the paper's
trained meta-classifier; the aim is to show the aggregation effect.

## 9. Interpretability: sparse autoencoder

`minigpt/sae.py`, `scripts/train_sae.py`. An SAE with 2,048 features
(8× the residual width) is trained on the residual stream after layer 2 of
the target model, over every token of A. The L1 penalty is weighted by decoder
column norms (so the model cannot cheat the penalty by shrinking codes and
growing the decoder). The method is validated first on synthetic data where
the true features are known (tests/test_sae.py: 32 planted directions in 16
dimensions are recovered with mean cosine > 0.9). Features are then described
by their top-activating contexts; labels are mine and are hypotheses, not
facts. Results, including how many features are uninterpretable, are in
RESULTS.md.

## 10. The browser engine

`web/js/engine.js` re-implements the forward pass in plain JavaScript (≈200
lines): float16 weights, a key/value cache so each new token costs one row of
work, and every attention row recorded for the viewer. The tokenizer is ported
line by line.

Two models are shipped (target and reference) because the membership panel
needs both for the ratio attack. The verdict thresholds are the 95th and 99th
percentiles of the ratio scores of real H couplets, i.e. 5% and 1% false-positive
rates *on couplets*. Longer texts have tighter score distributions, so for them
the thresholds are conservative.

**Parity.** `scripts/export_web.py` stores token ids, logits and an attention
row computed by PyTorch; `tests/web_parity.test.mjs` (run in CI with Node)
requires the JavaScript tokenizer to produce identical ids and the engine to
match the logits to 1e-3 and attention to 1e-4.

## 11. Things I deliberately did not do

* **No pretrained weights, no Hugging Face.** The point is to build it.
* **No multiple seeds.** Each model takes ~30 min on CPU; all results are
  single-seed and labelled as such. The bootstrap intervals capture sampling
  noise in the evaluation sets, *not* training randomness.
* **No differential privacy.** DP-SGD is the principled defence against
  everything measured here and would be the natural next experiment, but it
  needs per-example gradients, which would multiply CPU training time.
* **No LiRA-style shadow models.** The likelihood-ratio attack of Carlini et
  al. (2022) needs dozens of models trained with and without each example.
  At 30 min each that is days of CPU time; the single reference model is the
  affordable approximation.
