"""Train a sparse autoencoder on the target model's residual stream and
describe its features by their top-activating contexts.

    python scripts/train_sae.py          (~10 min)

Writes artifacts/sae.pt, results/sae.json (training + summary statistics) and
results/sae_features.json (top contexts per live feature).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt.data import ROOT, build_stream, load_sections  # noqa: E402
from minigpt.model import load_model  # noqa: E402
from minigpt.sae import SparseAutoencoder, fraction_variance_explained  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

ART, RES = ROOT / "artifacts", ROOT / "results"
LAYER = 1  # residual stream after the 2nd of 4 blocks
N_FEATURES = 2048
L1 = 3.0
STEPS = 6000
BATCH = 2048
TOP = 16


@torch.no_grad()
def collect(model, stream: np.ndarray, block: int):
    n = (len(stream) - 1) // block
    windows = torch.from_numpy(stream[: n * block].astype(np.int64)).view(n, block)
    acts = []
    for i in range(0, n, 64):
        acts.append(model(windows[i : i + 64], return_hidden=LAYER, fast=True)["hidden"].reshape(-1, model.cfg.d_model))
    return torch.cat(acts), windows.reshape(-1)


def context(tok, ids: torch.Tensor, pos: int, before: int = 10, after: int = 3) -> dict:
    lo, hi = max(0, pos - before), min(len(ids), pos + after + 1)
    return {
        "left": tok.decode(ids[lo:pos].tolist()),
        "token": tok.decode([int(ids[pos])]),
        "right": tok.decode(ids[pos + 1 : hi].tolist()),
    }


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(4)
    t0 = time.time()
    tok = BPETokenizer.load(ART / "tokenizer.json")
    model = load_model(str(ART / "target.pt"))
    sections = load_sections()
    split = json.loads((RES / "split.json").read_text())
    stream = build_stream([sections[i].text for i in split["A"]], tok)
    X, ids = collect(model, stream, model.cfg.block_size)
    scale = float(X.pow(2).sum(-1).mean().sqrt() / np.sqrt(X.shape[1]))
    X = X / scale  # unit RMS per dimension, so L1 means the same at any scale
    print(f"{len(X):,} activations of dim {X.shape[1]} ({time.time() - t0:.0f}s)", flush=True)

    perm = torch.randperm(len(X))
    n_test = 20_000
    X_test, X_train = X[perm[:n_test]], X[perm[n_test:]]
    sae = SparseAutoencoder(X.shape[1], N_FEATURES)
    with torch.no_grad():
        sae.b_dec.copy_(X_train.median(0).values)  # geometric-median-ish centre
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    log = []
    for step in range(STEPS + 1):
        l1 = L1 * min(1.0, step / 1000)  # warm the sparsity penalty up
        xb = X_train[torch.randint(0, len(X_train), (BATCH,))]
        loss, mse, f = sae.loss(xb, l1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                _, ft = sae(X_test)
                l0 = (ft > 0).float().sum(-1).mean().item()
                fve = fraction_variance_explained(sae, X_test)
            log.append({"step": step, "mse": mse.item(), "l0": l0, "fve": fve})
            print(f"step {step:5d} | mse {mse.item():.3f} | L0 {l0:5.1f} | var explained {fve:.3f}", flush=True)

    # downstream check: how much does splicing the SAE reconstruction into the
    # model hurt next-token loss?  (zero-ablation of the layer is the upper bound)
    with torch.no_grad():
        w = torch.from_numpy(stream[: 64 * 129].astype(np.int64)).view(64, 129)
        x, y = w[:, :-1], w[:, 1:]
        base = model(x, targets=y)["loss"].item()

        def run_with(replace):
            h = model.tok_emb(x) + model.pos_emb[: x.shape[1]]
            for i, blk in enumerate(model.blocks):
                h, _ = blk(h)
                if i == LAYER:
                    h = replace(h)
            logits = model.ln_f(h) @ model.tok_emb.weight.T
            return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()

        spliced = run_with(lambda h: sae(h / scale)[0] * scale)
        zeroed = run_with(lambda h: torch.zeros_like(h) + h.mean((0, 1)))

    # feature statistics and top contexts over the full activation set
    with torch.no_grad():
        F = torch.cat([sae.encode(X[i : i + 8192]) for i in range(0, len(X), 8192)])
    freq = (F > 0).float().mean(0)
    alive = torch.nonzero(freq > 0).squeeze(1).tolist()
    feats = []
    for j in alive:
        col = F[:, j]
        top = torch.topk(col, min(TOP, int((col > 0).sum())))
        ctxs = [context(tok, ids, int(p)) | {"act": round(float(a), 3)} for a, p in zip(top.values, top.indices)]
        tok_counts = Counter(c["token"] for c in ctxs)
        feats.append({
            "id": j, "freq": float(freq[j]),
            "top_token_share": tok_counts.most_common(1)[0][1] / len(ctxs),
            "top_token": tok_counts.most_common(1)[0][0],
            "contexts": ctxs,
        })
    feats.sort(key=lambda f: -f["freq"])
    single = sum(f["top_token_share"] >= 0.9 for f in feats)
    summary = {
        "layer": LAYER, "n_features": N_FEATURES, "l1": L1, "steps": STEPS,
        "n_activations": len(X), "activation_scale": scale, "log": log,
        "alive": len(alive), "dead": N_FEATURES - len(alive),
        "dense_features_gt_10pct": int((freq > 0.1).sum()),
        "single_token_features": single,
        "loss_clean": base, "loss_sae_spliced": spliced, "loss_mean_ablated": zeroed,
        "minutes": (time.time() - t0) / 60,
    }
    torch.save({"state": sae.state_dict(), "scale": scale, "layer": LAYER}, ART / "sae.pt")
    (RES / "sae.json").write_text(json.dumps(summary, indent=1))
    (RES / "sae_features.json").write_text(json.dumps(feats, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    main()
