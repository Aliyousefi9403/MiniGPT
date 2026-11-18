"""Find SAE features that detect human-meaningful categories, with numbers.

Reading a feature's top examples and naming it is how interpretability work
often proceeds, and it is easy to fool yourself that way.  Here every label is
a *hypothesis tested over all ~400k tokens*: for each category (royalty,
battle, hero names, God, end of first hemistich, end of line) every token is
marked in or out, every feature is treated as a binary detector (active > 0),
and the feature with the best F1 is reported with its precision and recall.

    python scripts/label_sae_features.py        (~2 min)

Writes results/sae_labels.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minigpt.data import ROOT, build_stream, load_sections  # noqa: E402
from minigpt.model import load_model  # noqa: E402
from minigpt.sae import SparseAutoencoder  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402
from train_sae import collect, context  # noqa: E402

ART, RES = ROOT / "artifacts", ROOT / "results"
DIACRITICS = re.compile("[ً-ْٰ]")

LEXICON = {
    "royalty (shah, throne, crown, king)": "شاه شهریار شاهی پادشاه شاهنشاه تاج تخت کیان کی",
    "battle (war, army, mace, sword, spear)": "رزم جنگ نبرد کین کینه سپاه لشکر گرز تیغ شمشیر نیزه کوس رزمگاه جوشن سنان کمان تیر",
    "hero names (Rostam, Sohrab, Afrasiab, ...)": "رستم سهراب سیاوش اسفندیار افراسیاب پیران طوس گیو گودرز بیژن زال کاووس کیخسرو فریدون "
                                                   "منوچهر جمشید ضحاک بهرام اسکندر سام پشنگ گرسیوز هومان گشتاسپ لهراسپ تهمتن",
    "God (Yazdan, Khoda, Izad, Dadar)": "یزدان خدای خداوند ایزد دادار کردگار",
}


def normalise(word: str) -> str:
    return DIACRITICS.sub("", word.strip().replace("‌", ""))


def word_categories(word: str) -> set[str]:
    w = normalise(word)
    out = set()
    for cat, words in LEXICON.items():
        for base in words.split():
            # exact word, or with a common suffix (plural -an, possessive -ash, -i)
            if w == base or (w.startswith(base) and w[len(base):] in {"ان", "ش", "ی", "ها", "را"}):
                out.add(cat)
    return out


def main() -> None:
    torch.set_num_threads(4)
    tok = BPETokenizer.load(ART / "tokenizer.json")
    model = load_model(str(ART / "target.pt"))
    ck = torch.load(ART / "sae.pt", weights_only=False)
    sae = SparseAutoencoder(model.cfg.d_model, ck["state"]["W_enc"].shape[0])
    sae.load_state_dict(ck["state"])
    sections = load_sections()
    split = json.loads((RES / "split.json").read_text())
    stream = build_stream([sections[i].text for i in split["A"]], tok)
    X, ids = collect(model, stream, model.cfg.block_size)
    X = X / ck["scale"]
    ids = ids.numpy()
    pieces = [tok.vocab[i] for i in ids]

    # token -> word-level categories (a token inherits the category of its word)
    cats = list(LEXICON) + ["end of first hemistich (before /)", "end of line (rhyme position)"]
    labels = np.zeros((len(ids), len(cats)), dtype=bool)
    start = 0
    for j in range(1, len(ids) + 1):
        if j == len(ids) or pieces[j].startswith((" ", "\n")) or pieces[j] == "\n":
            word = "".join(pieces[start:j])
            for c in word_categories(word):
                labels[start:j, cats.index(c)] = True
            nxt = pieces[j] if j < len(ids) else ""
            if word.strip() and word.strip() != "/":
                if nxt == " /":
                    labels[j - 1, cats.index("end of first hemistich (before /)")] = True
                elif nxt == "\n":
                    labels[j - 1, cats.index("end of line (rhyme position)")] = True
            start = j

    tp = np.zeros((len(cats), sae.W_enc.shape[0]))
    pred = np.zeros(sae.W_enc.shape[0])
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            act = (sae.encode(X[i : i + 8192]) > 0).float().numpy()
            pred += act.sum(0)
            tp += labels[i : i + 8192].T.astype(np.float32) @ act
    pos = labels.sum(0)
    out = []
    with torch.no_grad():
        for k, c in enumerate(cats):
            prec = tp[k] / np.maximum(pred, 1)
            rec = tp[k] / max(pos[k], 1)
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            best = np.argsort(-f1)[:3]
            feats = []
            for f in best:
                col = torch.cat([sae.encode(X[i : i + 8192])[:, f] for i in range(0, len(X), 8192)])
                top = torch.topk(col, 8)
                feats.append({
                    "feature": int(f), "f1": float(f1[f]), "precision": float(prec[f]), "recall": float(rec[f]),
                    "freq": float(pred[f] / len(ids)),
                    "contexts": [context(tok, ids, int(p)) | {"act": round(float(a), 3)} for a, p in zip(top.values, top.indices)],
                })
            base_rate = float(pos[k] / len(ids))
            out.append({"category": c, "base_rate": base_rate, "n_tokens": int(pos[k]), "best": feats})
            b = feats[0]
            print(f"{c:45s} base rate {base_rate:6.2%} | feature {b['feature']:4d}: "
                  f"precision {b['precision']:.2f} recall {b['recall']:.2f} F1 {b['f1']:.2f}")
    (RES / "sae_labels.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
