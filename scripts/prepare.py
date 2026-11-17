"""Build everything the models train on, deterministically.

    1. split sections into A / B / V / H                 (minigpt/data.py)
    2. train the BPE tokenizer on A u B (no traps, no held-out text)
    3. generate copyright traps and canaries             (minigpt/privacy/traps.py)
    4. target corpus    = A  + inserted traps + inserted canaries
       reference corpus = B  (clean)
    5. tokenise everything to artifacts/*.npy and write manifests to results/

Usage:  python scripts/prepare.py          (~2 min on a laptop CPU)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt.data import ROOT, build_stream, load_sections, split_sections  # noqa: E402
from minigpt.privacy.traps import WordMarkov, insert_documents, make_canaries, make_traps  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

ART = ROOT / "artifacts"
RES = ROOT / "results"
VOCAB_SIZE = 1024
TRAP_SEED, CANARY_SEED, INSERT_SEED = 7, 8, 9


def main() -> None:
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    sections = load_sections()
    split = split_sections(len(sections))
    docs = {k: [sections[i].text for i in v] for k, v in split.items()}
    for k, v in split.items():
        n_cp = sum(len(sections[i].couplets) for i in v)
        print(f"split {k}: {len(v)} sections, {n_cp} couplets")

    print("training tokenizer ...")
    tok = BPETokenizer.train("".join(docs["A"] + docs["B"]), VOCAB_SIZE)
    tok.save(ART / "tokenizer.json")
    sample = "".join(docs["V"])
    print(f"  vocab {len(tok)}, {len(sample) / len(tok.encode(sample)):.2f} chars/token on V")

    real_couplets = {c for s in sections for c in s.couplets}
    gen = WordMarkov([c for k in "AB" for i in split[k] for c in sections[i].couplets])
    traps = make_traps(gen, seed=TRAP_SEED, existing=real_couplets)
    canaries = make_canaries(seed=CANARY_SEED)

    inserts = [c for t in traps for c in (t.copies or [])]
    inserts += [c.text for c in canaries for _ in range(c.reps)]
    target_docs = insert_documents(docs["A"], inserts, seed=INSERT_SEED)

    streams = {
        "target_train": build_stream(target_docs, tok),
        "ref_train": build_stream(docs["B"], tok),
        "val": build_stream(docs["V"], tok),
        "heldout": build_stream(docs["H"], tok),
    }
    for k, s in streams.items():
        np.save(ART / f"{k}.npy", s)
        print(f"  {k}: {len(s):,} tokens")
    trap_tokens = sum(len(tok.encode(c + "\n\n")) for c in inserts)
    print(f"  inserted material: {trap_tokens:,} tokens = {trap_tokens / len(streams['target_train']):.1%} of target corpus")

    (RES / "split.json").write_text(json.dumps(split))
    (RES / "traps.json").write_text(json.dumps([t.to_dict() for t in traps], ensure_ascii=False))
    (RES / "canaries.json").write_text(json.dumps([c.__dict__ for c in canaries], ensure_ascii=False))
    (RES / "corpus_stats.json").write_text(
        json.dumps(
            {
                "sections": {k: len(v) for k, v in split.items()},
                "couplets": {k: sum(len(sections[i].couplets) for i in v) for k, v in split.items()},
                "tokens": {k: int(len(s)) for k, s in streams.items()},
                "inserted_tokens": trap_tokens,
                "vocab_size": len(tok),
                "chars_per_token_V": len(sample) / len(tok.encode(sample)),
            },
            indent=1,
        )
    )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
