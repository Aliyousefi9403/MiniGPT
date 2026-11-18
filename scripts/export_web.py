"""Export trained models for the browser demo.

    python scripts/export_web.py

For each model writes web/model/<name>.bin (all tensors as float16, ~half the
size of float32 with no measurable loss: see tests/web_parity.test.mjs) and
web/model/<name>.json (config + tensor table).  Also writes:

  web/model/tokenizer.json   the BPE vocabulary and merges
  web/model/parity.json      reference token ids and logits computed by PyTorch,
                             which the JavaScript engine must reproduce
  web/model/calibration.json ratio scores of known members / non-members (the
                             demo's thresholds) and the example texts
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt.data import ROOT  # noqa: E402
from minigpt.model import load_model  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

ART, OUT = ROOT / "artifacts", ROOT / "web" / "model"
PARITY_TEXT = "\nچو رستم بدید آن سپاه گران / بزد دست و برداشت گرز گران\n"


def export(name: str) -> None:
    model = load_model(str(ART / f"{name}.pt"))
    tensors, blobs, offset = [], [], 0
    for k, v in model.state_dict().items():
        arr = v.detach().cpu().numpy().astype(np.float16)
        blobs.append(arr.tobytes())
        tensors.append({"name": k, "shape": list(arr.shape), "numel": int(arr.size), "offset": offset})
        offset += arr.nbytes
    (OUT / f"{name}.bin").write_bytes(b"".join(blobs))
    (OUT / f"{name}.json").write_text(json.dumps({"config": model.cfg.to_dict(), "tensors": tensors}))
    print(f"{name}: {offset / 1e6:.1f} MB")


@torch.no_grad()
def parity(tok: BPETokenizer) -> None:
    ids = tok.encode(PARITY_TEXT)
    out = {"text": PARITY_TEXT, "ids": ids, "models": {}}
    for name in ("target", "reference"):
        model = load_model(str(ART / f"{name}.pt"))
        for p in model.parameters():  # evaluate with the same float16-rounded weights
            p.copy_(p.half().float())
        res = model(torch.tensor([ids]), return_attn=True)
        out["models"][name] = {
            "last_logits": res["logits"][0, -1].tolist(),
            "attn_last_l0h0": res["attn"][0, 0, 0, -1].tolist(),
        }
    (OUT / "parity.json").write_text(json.dumps(out, ensure_ascii=False))


def calibration() -> None:
    """Thresholds come from real couplets: all of H (non-members) and the A
    sample (members) scored in scripts/run_mia.py.  Examples are curated in
    results/demo_examples.json (text, English gloss, ground truth)."""
    s = json.loads((ROOT / "results" / "mia_scores.json").read_text())
    rng = np.random.default_rng(0)
    mem = np.array(s["natural_members"]["ratio"])
    non = np.sort(np.array(s["natural_nonmembers"]["ratio"]))
    mem = np.sort(rng.choice(mem, min(len(mem), len(non)), replace=False))
    examples = json.loads((ROOT / "results" / "demo_examples.json").read_text(encoding="utf-8"))
    (OUT / "calibration.json").write_text(json.dumps({
        "nonmember_ratio": non.round(4).tolist(),
        "member_ratio": mem.round(4).tolist(),
        "examples": examples,
    }, ensure_ascii=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("target", "reference"):
        export(name)
    shutil.copy(ART / "tokenizer.json", OUT / "tokenizer.json")
    parity(BPETokenizer.load(ART / "tokenizer.json"))
    calibration()


if __name__ == "__main__":
    main()
