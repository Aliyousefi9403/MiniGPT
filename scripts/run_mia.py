"""All membership-inference experiments.

    python scripts/run_mia.py            (~10 min on 4 CPU cores)

E1  copyright traps   AUC per (length, exact/fuzzy, repetitions) cell, all attacks
E2  canaries          Secret-Sharer exposure vs repetitions
E3  natural MIA       real couplets from A (members) vs H (never seen), all attacks,
                      plus model-free "blind" baselines that check for distribution shift
E4  document level    aggregate couplet scores over k couplets of a section
E5  trap perplexity   does a trap's own perplexity change how detectable it is?

Writes results/mia.json (summary numbers) and results/mia_scores.json
(per-item scores used by the figures and the web demo's calibration).
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt.data import ROOT, load_sections  # noqa: E402
from minigpt.model import load_model  # noqa: E402
from minigpt.privacy.attacks import ATTACKS, scores, token_stats  # noqa: E402
from minigpt.privacy.metrics import auc, summarise  # noqa: E402
from minigpt.privacy.traps import CANARY_DIGITS, CANARY_TEMPLATE, to_persian_digits  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

ART, RES = ROOT / "artifacts", ROOT / "results"
N_MEMBER_COUPLETS = int(__import__("os").environ.get("N_MEM", 3000))


def score_texts(texts, target, ref, tok):
    t = token_stats(target, tok, texts)
    r = token_stats(ref, tok, texts)
    s = scores(texts, t, r)
    s["ref_loss"] = np.array([-x.logp.mean() for x in r])  # used for E5
    s["n_tokens"] = np.array([len(x.logp) for x in t], dtype=float)
    return s


# ----------------------------------------------------------------- E1 / E5
def traps_experiment(traps, target, ref, tok):
    texts = [t["text"] for t in traps]
    s = score_texts(texts, target, ref, tok)
    cells = []
    for L in sorted({t["n_couplets"] for t in traps}):
        non = [i for i, t in enumerate(traps) if t["reps"] == 0 and t["n_couplets"] == L]
        for kind in ("exact", "fuzzy"):
            for r in sorted({t["reps"] for t in traps if t["reps"] > 0}):
                mem = [i for i, t in enumerate(traps) if t["n_couplets"] == L and t["kind"] == kind and t["reps"] == r]
                if not mem:
                    continue
                row = {"n_couplets": L, "kind": kind, "reps": r,
                       "tokens_mean": float(s["n_tokens"][mem].mean())}
                for a in ATTACKS:
                    row[a] = summarise(s[a][mem], s[a][non], n_boot=500, seed=r)
                cells.append(row)
    # E5: split exact members with reps >= 4 by their reference-model loss
    ppl = []
    for L in sorted({t["n_couplets"] for t in traps}):
        non = np.array([i for i, t in enumerate(traps) if t["reps"] == 0 and t["n_couplets"] == L])
        mem = np.array([i for i, t in enumerate(traps) if t["n_couplets"] == L and t["kind"] == "exact" and t["reps"] >= 4])
        cuts = np.quantile(s["ref_loss"][non], [1 / 3, 2 / 3])
        for b, (lo, hi) in enumerate([(-np.inf, cuts[0]), (cuts[0], cuts[1]), (cuts[1], np.inf)]):
            m = mem[(s["ref_loss"][mem] > lo) & (s["ref_loss"][mem] <= hi)]
            n = non[(s["ref_loss"][non] > lo) & (s["ref_loss"][non] <= hi)]
            if len(m) >= 5 and len(n) >= 5:
                ppl.append({"n_couplets": L, "tercile": b, "ref_ppl_range": [float(math.exp(max(lo, 0))) if np.isfinite(lo) else None,
                            float(math.exp(hi)) if np.isfinite(hi) else None],
                            "loss": summarise(s["loss"][m], s["loss"][n], 300)["auc"],
                            "ratio": summarise(s["ratio"][m], s["ratio"][n], 300)["auc"],
                            "n_members": int(len(m)), "n_nonmembers": int(len(n))})
    per_item = [{"trap_id": t["trap_id"], "n_couplets": t["n_couplets"], "kind": t["kind"], "reps": t["reps"],
                 **{a: float(s[a][i]) for a in ATTACKS}, "ref_loss": float(s["ref_loss"][i])} for i, t in enumerate(traps)]
    return cells, ppl, per_item


# ---------------------------------------------------------------------- E2
@torch.no_grad()
def all_secret_logprobs(model, prefix_ids: list[int], digit_ids: list[int]) -> np.ndarray:
    """log p(secret | prefix) for all 10^5 secrets, exactly, via the digit tree:
    one forward pass per internal node (1 + 10 + ... + 10^4 = 11,111 rows)
    instead of one per secret."""
    total = np.zeros(1)
    for k in range(CANARY_DIGITS):
        combos = np.array(np.meshgrid(*[range(10)] * k, indexing="ij")).reshape(k, -1).T if k else np.zeros((1, 0), int)
        rows = np.concatenate([np.tile(prefix_ids, (len(combos), 1)), np.array(digit_ids)[combos].reshape(len(combos), k)], 1)
        x = torch.tensor(rows, dtype=torch.long)
        lp = torch.cat([torch.log_softmax(model(x[i : i + 2048], fast=True)["logits"][:, -1], -1) for i in range(0, len(x), 2048)])
        lp_digits = lp[:, digit_ids].numpy()  # (10^k, 10)
        total = (total[:, None] + lp_digits).reshape(-1)  # index = secret as an integer
    return total


def canary_experiment(canaries, target, ref, tok):
    digit_ids = [tok.encode(to_persian_digits(str(d)))[0] for d in range(10)]
    res = []
    for c in canaries:
        prefix = tok.encode("\n" + CANARY_TEMPLATE.format(hero=c["hero"]))
        row = {"secret": c["secret"], "reps": c["reps"], "hero": c["hero"]}
        for name, model in (("target", target), ("reference", ref)):
            lp = all_secret_logprobs(model, prefix, digit_ids)
            idx = int(c["secret"])
            rank = int((lp > lp[idx]).sum()) + 1  # 1 = the true secret is the most likely
            row[f"{name}_rank"] = rank
            row[f"{name}_exposure"] = math.log2(len(lp)) - math.log2(rank)
        res.append(row)
    return res


# ---------------------------------------------------------------------- E3
def naive_bayes_auc(mem_texts, non_texts, seed=0):
    """Blind baseline (Das et al., 2024): a bag-of-words classifier that never
    looks at the model.  2-fold cross-validated AUC.  If members and
    non-members came from different distributions, this would find it."""
    rng = random.Random(seed)
    data = [(t, 1) for t in mem_texts] + [(t, 0) for t in non_texts]
    rng.shuffle(data)
    half = len(data) // 2
    folds = [(data[:half], data[half:]), (data[half:], data[:half])]
    sc_m, sc_n = [], []
    for train, test in folds:
        cnt = [Counter(), Counter()]
        for t, y in train:
            cnt[y].update(t.split())
        vocab = set(cnt[0]) | set(cnt[1])
        tot = [sum(c.values()) + len(vocab) for c in cnt]
        for t, y in test:
            llr = sum(math.log((cnt[1][w] + 1) / tot[1]) - math.log((cnt[0][w] + 1) / tot[0]) for w in t.split())
            (sc_m if y else sc_n).append(llr)
    return auc(np.array(sc_m), np.array(sc_n))


def natural_experiment(sections, split, target, ref, tok):
    rng = random.Random(0)
    mem_all = [(i, c) for i in split["A"] for c in sections[i].couplets]
    mem = rng.sample(mem_all, N_MEMBER_COUPLETS)
    # Ferdowsi reuses formulaic lines: a held-out couplet that also occurs
    # verbatim in A is not a non-member, so it is dropped.
    in_a = {c for _, c in mem_all}
    non_all = [(i, c) for i in split["H"] for c in sections[i].couplets]
    non = [(i, c) for i, c in non_all if c not in in_a]
    halves_a = {h for _, c in mem_all for h in c.split(" / ")}
    overlap = {
        "heldout_couplets": len(non_all),
        "dropped_verbatim_in_A": len(non_all) - len(non),
        "heldout_hemistichs_also_in_A": sum(h in halves_a for _, c in non for h in c.split(" / ")) / (2 * len(non)),
    }
    s_m = score_texts([c for _, c in mem], target, ref, tok)
    s_n = score_texts([c for _, c in non], target, ref, tok)
    res = {a: summarise(s_m[a], s_n[a], 500) for a in ATTACKS}
    blind = {
        "length_chars": auc(np.array([len(c) for _, c in mem]), np.array([len(c) for _, c in non])),
        "zlib_ratio_text_only": auc(
            np.array([len(zlib.compress(c.encode())) / len(c.encode()) for _, c in mem]),
            np.array([len(zlib.compress(c.encode())) / len(c.encode()) for _, c in non]),
        ),
        "bag_of_words_nb": naive_bayes_auc([c for _, c in mem], [c for _, c in non]),
        # the reference model never saw A or H, so its loss is itself a blind score
        "reference_loss_only": auc(-s_m["ref_loss"], -s_n["ref_loss"]),
    }
    blind["overlap"] = overlap
    return res, blind, (mem, s_m), (non, s_n)


# ---------------------------------------------------------------------- E4
def document_experiment(sections, split, target, ref, tok, ks=(1, 2, 4, 8, 16, 32), n_draws=20):
    rng = random.Random(1)
    mem_secs = rng.sample(split["A"], len(split["H"]))
    non_secs = list(split["H"])
    per_sec = {}
    for i in mem_secs + non_secs:
        cps = sections[i].couplets
        per_sec[i] = score_texts(cps, target, ref, tok)
    out = []
    for k in ks:
        row = {"k": k}
        for a in ["loss", "min_k_pp", "ratio"]:
            vals = []
            for d in range(n_draws):  # average AUC over random choices of k couplets
                def agg(i):
                    sc = per_sec[i][a]
                    pick = rng.sample(range(len(sc)), min(k, len(sc)))
                    return float(np.mean(sc[pick]))
                vals.append(auc(np.array([agg(i) for i in mem_secs]), np.array([agg(i) for i in non_secs])))
            row[a] = float(np.mean(vals))
            row[a + "_sd"] = float(np.std(vals))
        out.append(row)
    full = {a: auc(np.array([per_sec[i][a].mean() for i in mem_secs]), np.array([per_sec[i][a].mean() for i in non_secs]))
            for a in ATTACKS}
    return out, full, len(mem_secs), len(non_secs)


def main() -> None:
    global ART, RES
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ART), help="directory with target.pt / reference.pt")
    ap.add_argument("--results", default=str(RES))
    args = ap.parse_args()
    ART, RES = Path(args.models), Path(args.results)
    RES.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    tok = BPETokenizer.load(ART / "tokenizer.json")
    target, ref = load_model(str(ART / "target.pt")), load_model(str(ART / "reference.pt"))
    sections = load_sections()
    split = json.loads((RES / "split.json").read_text())
    traps = json.loads((RES / "traps.json").read_text(encoding="utf-8"))
    canaries = json.loads((RES / "canaries.json").read_text(encoding="utf-8"))

    print("E1/E5 traps ...", flush=True)
    cells, ppl, trap_items = traps_experiment(traps, target, ref, tok)
    for c in cells:
        print(f"  L={c['n_couplets']} {c['kind']:5s} r={c['reps']:2d}  " +
              "  ".join(f"{a}={c[a]['auc']:.2f}" for a in ATTACKS), flush=True)
    print("E2 canaries ...", flush=True)
    can = canary_experiment(canaries, target, ref, tok)
    for c in can:
        print(f"  reps={c['reps']:2d} exposure target={c['target_exposure']:.1f} ref={c['reference_exposure']:.1f}")
    print("E3 natural ...", flush=True)
    nat, blind, (mem, s_m), (non, s_n) = natural_experiment(sections, split, target, ref, tok)
    for a in ATTACKS:
        print(f"  {a:9s} AUC {nat[a]['auc']:.3f} [{nat[a]['auc_lo']:.3f},{nat[a]['auc_hi']:.3f}]  TPR@1% {nat[a]['tpr@1%']:.3f}")
    print("  blind:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in blind.items()})
    print("E4 documents ...", flush=True)
    doc, doc_full, n_dm, n_dn = document_experiment(sections, split, target, ref, tok)
    for r in doc:
        print(f"  k={r['k']:2d}  loss={r['loss']:.3f}  min_k_pp={r['min_k_pp']:.3f}  ratio={r['ratio']:.3f}")
    print("  full sections:", {k: round(v, 3) for k, v in doc_full.items()})

    summary = {
        "traps": cells, "trap_perplexity": ppl, "canaries": can,
        "natural": nat, "blind": blind,
        "documents": {"by_k": doc, "full_section": doc_full, "n_member_sections": n_dm, "n_nonmember_sections": n_dn},
        "minutes": (time.time() - t0) / 60,
    }
    (RES / "mia.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (RES / "mia_scores.json").write_text(json.dumps({
        "traps": trap_items,
        "natural_members": {a: s_m[a].round(5).tolist() for a in ATTACKS + ["ref_loss", "n_tokens"]},
        "natural_nonmembers": {a: s_n[a].round(5).tolist() for a in ATTACKS + ["ref_loss", "n_tokens"]},
    }))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
