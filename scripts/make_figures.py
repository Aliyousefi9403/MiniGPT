"""Render every figure in figures/ from the JSON files in results/.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minigpt.privacy.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"

# categorical slots in fixed order (validated light palette)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
ATTACK_NAMES = {"loss": "loss", "zlib": "zlib ratio", "min_k": "min-k% (k=20)", "min_k_pp": "min-k%++", "ratio": "reference ratio"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def training_curves():
    runs = [(n, load(f"train_{n}.json")) for n in ("target", "reference")]
    if not all(r for _, r in runs):
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (name, r), col in zip(runs, C):
        h = r["history"]
        ax.plot(h["step"], h["val_loss"], color=col, label=f"{name}: validation (V)")
        ax.plot(h["step"], h["train_loss"], color=col, ls="--", lw=1.5, label=f"{name}: own training data")
        b = r["best_step"]
        ax.plot([b], [r["best_val_loss"]], "o", ms=7, color=col, mec="white", mew=1.5)
    ax.set_xlabel("optimiser step (32 × 128 tokens)")
    ax.set_ylabel("cross-entropy (nats / token)")
    ax.set_title("Training: the gap between dashed and solid lines is memorisation")
    ax.legend(ncol=2, fontsize=8.5)
    save(fig, "training_curves.png")


def bpc_bars():
    r = load("lm_eval.json")
    if not r:
        return
    items = list(r["bpc"].items())
    ngram = [(k, v) for k, v in items if "gram" in k]
    gpt = [(k, v) for k, v in items if k.startswith("GPT") and not k.startswith("GPT pos_")]
    rows = ngram + gpt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cols = [MUTED if "gram" in k else C[0] for k, _ in rows]
    y = np.arange(len(rows))
    ax.barh(y, [v for _, v in rows], color=cols, height=0.65)
    for yi, (k, v) in zip(y, rows):
        ax.text(v + 0.03, yi, f"{v:.2f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, [k.replace("GPT ", "GPT · ") for k, _ in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("bits per character on validation split V (lower is better)")
    ax.set_title("The transformer beats every character n-gram baseline")
    ax.grid(axis="y", visible=False)
    save(fig, "bpc.png")


def verse_bars():
    r = load("lm_eval.json")
    if not r:
        return
    v = r["verse"]
    keys = list(v)
    metrics = [("well_formed", "well-formed couplet"), ("rhyme@1", "rhyme (last letter)"), ("rhyme@2", "rhyme (last 2 letters)")]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    w = 0.8 / len(keys)
    x = np.arange(len(metrics))
    for i, k in enumerate(keys):
        vals = [v[k].get(m, np.nan) for m, _ in metrics]
        bars = ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w * 0.92, color=C[i], label=k)
        for b, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.0%}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [m for _, m in metrics])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Does generated verse have the Shahnameh's form?")
    ax.legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.grid(axis="x", visible=False)
    save(fig, "verse_form.png")


def traps_figure():
    r = load("mia.json")
    if not r:
        return
    cells = r["traps"]
    lengths = sorted({c["n_couplets"] for c in cells})
    fig, axes = plt.subplots(1, len(lengths), figsize=(9, 3.6), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), lengths):
        for col, kind in zip(C, ("exact", "fuzzy")):
            cs = sorted([c for c in cells if c["n_couplets"] == L and c["kind"] == kind], key=lambda c: c["reps"])
            reps = [c["reps"] for c in cs]
            a = [c["ratio"]["auc"] for c in cs]
            lo = [c["ratio"]["auc_lo"] for c in cs]
            hi = [c["ratio"]["auc_hi"] for c in cs]
            ax.fill_between(reps, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(reps, a, "-o", color=col, ms=6, mec="white", mew=1.5, label=f"{kind} copies")
        lc = sorted([c for c in cells if c["n_couplets"] == L and c["kind"] == "exact"], key=lambda c: c["reps"])
        ax.plot([c["reps"] for c in lc], [c["loss"]["auc"] for c in lc], ":", color=INK2, lw=1.5, label="exact, loss attack")
        ax.axhline(0.5, color=MUTED, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("times the trap was inserted")
        tok = np.mean([c["tokens_mean"] for c in cells if c["n_couplets"] == L])
        ax.set_title(f"{L}-couplet traps (~{tok:.0f} tokens)")
    np.atleast_1d(axes)[0].set_ylabel("AUC (reference-ratio attack)")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Copyright traps: detectability vs repetitions (95% bootstrap bands)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    save(fig, "traps_auc.png")


def attacks_heatmap():
    r = load("mia.json")
    if not r:
        return
    cells = [c for c in r["traps"] if c["kind"] == "exact"]
    cells.sort(key=lambda c: (c["n_couplets"], c["reps"]))
    attacks = list(ATTACK_NAMES)
    M = np.array([[c[a]["auc"] for c in cells] for a in attacks])
    fig, ax = plt.subplots(figsize=(8, 2.9))
    im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.8 else INK)
    ax.set_yticks(range(len(attacks)), [ATTACK_NAMES[a] for a in attacks])
    ax.set_xticks(range(len(cells)), [f"{c['n_couplets']}cp ×{c['reps']}" for c in cells], fontsize=8)
    ax.grid(False)
    ax.set_title("Exact traps: AUC of every attack (rows) for every trap length × repetitions (columns)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    save(fig, "traps_attacks.png")


def natural_roc():
    s = load("mia_scores.json")
    r = load("mia.json")
    if not s:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for i, a in enumerate(ATTACK_NAMES):
        fpr, tpr = roc_curve(np.array(s["natural_members"][a]), np.array(s["natural_nonmembers"][a]))
        lab = f"{ATTACK_NAMES[a]} (AUC {r['natural'][a]['auc']:.2f})"
        axes[0].plot(fpr, tpr, color=C[i], label=lab, lw=1.8)
        axes[1].plot(np.maximum(fpr, 1e-4), np.maximum(tpr, 1e-4), color=C[i], lw=1.8)
    for ax in axes:
        ax.plot([1e-4, 1], [1e-4, 1], color=MUTED, lw=1, ls="--")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_title("Real couplets: A (trained on) vs H (never seen)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlim(1e-3, 1); axes[1].set_ylim(1e-3, 1)
    axes[1].set_title("Same, log scale: the low-FPR regime")
    for ax in axes:
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    axes[0].legend(fontsize=8, loc="lower right")
    save(fig, "natural_roc.png")


def documents():
    r = load("mia.json")
    if not r:
        return
    d = r["documents"]["by_k"]
    k = [x["k"] for x in d]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for col, a in zip([C[0], C[3], C[4]], ["loss", "min_k_pp", "ratio"]):
        m, sd = np.array([x[a] for x in d]), np.array([x[a + "_sd"] for x in d])
        ax.fill_between(k, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(k, m, "-o", color=col, ms=6, mec="white", mew=1.5, label=ATTACK_NAMES[a])
    ax.axhline(0.5, color=MUTED, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(x) for x in k])
    ax.set_xlabel("couplets of the section available to the auditor")
    ax.set_ylabel("section-level AUC")
    ax.set_title("Aggregating weak per-couplet signals over a document")
    ax.legend(fontsize=8)
    save(fig, "document_level.png")


def canaries():
    r = load("mia.json")
    if not r:
        return
    can = r["canaries"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    reps = sorted({c["reps"] for c in can})
    xpos = {rr: i for i, rr in enumerate(reps)}
    rng = np.random.default_rng(0)
    for col, who, off in ((C[0], "target", -0.12), (MUTED, "reference", 0.12)):
        xs = [xpos[c["reps"]] + off + rng.uniform(-0.04, 0.04) for c in can]
        ax.scatter(xs, [c[f"{who}_exposure"] for c in can], s=46, color=col, edgecolor="white", lw=1.2, zorder=3,
                   label=f"{who} model" + (" (never saw any canary)" if who == "reference" else ""))
    ax.axhline(np.log2(1e5), color=INK2, lw=1, ls="--")
    ax.text(len(reps) - 1, np.log2(1e5) + 0.3, "max: secret is the top guess (16.6 bits)", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(len(reps)), [("never inserted" if rr == 0 else f"×{rr}") for rr in reps])
    ax.set_ylim(0, 18.5)
    ax.set_ylabel("exposure (bits)")
    ax.set_title("Secret-Sharer canaries: 5-digit secrets, 100,000 candidates")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "canaries.png")


def attention_heads():
    r = load("attention.json")
    if not r:
        return
    M = np.array(r["share_on_first_rhyme_word"])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.3, M.max()), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 * M.max() else INK)
    ax.set_xticks(range(M.shape[1]), [f"h{j}" for j in range(M.shape[1])])
    ax.set_yticks(range(M.shape[0]), [f"layer {i}" for i in range(M.shape[0])])
    ax.grid(False)
    ax.set_title(f"Attention from the couplet's last token to the first rhyme word\n(uniform attention would give {r['uniform_baseline']:.2f})",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.outline.set_visible(False)
    save(fig, "rhyme_heads.png")


def main():
    for f in (training_curves, bpc_bars, verse_bars, traps_figure, attacks_heatmap, natural_roc, documents, canaries, attention_heads):
        f()


if __name__ == "__main__":
    main()
