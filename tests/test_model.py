import pytest
import torch

from minigpt.model import GPT, GPTConfig, gelu, sinusoidal_table


def small(pos="learned", **kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0, pos_encoding=pos, **kw)).eval()


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "none"])
def test_causality(pos):
    """Changing token t must not change any logit at positions < t."""
    m = small(pos)
    x = torch.randint(0, 50, (1, 16))
    base = m(x)["logits"]
    for t in [3, 9, 15]:
        y = x.clone()
        y[0, t] = (y[0, t] + 1) % 50
        out = m(y)["logits"]
        assert torch.allclose(base[0, :t], out[0, :t], atol=1e-6)
        assert not torch.allclose(base[0, t], out[0, t])


def test_explicit_attention_matches_fused_kernel():
    m = small()
    x = torch.randint(0, 50, (3, 16))
    a = m(x, fast=False)["logits"]
    b = m(x, fast=True)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_attention_rows_are_distributions_and_causal():
    m = small()
    att = m(torch.randint(0, 50, (2, 10)), return_attn=True)["attn"]  # (B,L,H,T,T)
    assert att.shape == (2, 2, 2, 10, 10)
    assert torch.allclose(att.sum(-1), torch.ones(2, 2, 2, 10), atol=1e-6)
    assert torch.all(att.triu(1) == 0)


def test_loss_at_init_is_near_uniform():
    m = small()
    x = torch.randint(0, 50, (8, 16))
    loss = m(x[:, :-1], targets=x[:, 1:])["loss"].item()
    assert abs(loss - torch.log(torch.tensor(50.0)).item()) < 0.3


def test_gelu_matches_torch_tanh_gelu():
    x = torch.linspace(-6, 6, 101)
    assert torch.allclose(gelu(x), torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-6)


def test_sinusoidal_table():
    pe = sinusoidal_table(8, 6)
    assert torch.allclose(pe[0, 0::2], torch.zeros(3)) and torch.allclose(pe[0, 1::2], torch.ones(3))


def test_can_overfit_a_tiny_sequence():
    m = small().train()
    x = torch.randint(0, 50, (1, 16))
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(200):
        loss = m(x[:, :-1], targets=x[:, 1:])["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.05


def test_generate_respects_block_size():
    m = small()
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), 40, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 41)



def test_training_resumes_exactly(tmp_path):
    """A run killed mid-way and resumed must reproduce an uninterrupted run."""
    import numpy as np

    from minigpt.train import TrainConfig, train_model

    stream = np.random.default_rng(0).integers(0, 50, 4000).astype(np.int32)
    mcfg = GPTConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, d_model=16, dropout=0.1)
    tcfg = TrainConfig(steps=20, eval_every=5, eval_windows=4, threads=1, batch_size=4)
    full = train_model(mcfg, tcfg, stream, stream, tmp_path / "a.pt", log=lambda s: None)

    class Crash(Exception):
        pass

    def crash_at_15(msg):
        if msg.startswith("step    15"):
            raise Crash

    with pytest.raises(Crash):
        train_model(mcfg, tcfg, stream, stream, tmp_path / "b.pt", log=crash_at_15)
    assert (tmp_path / "b.pt.resume").exists()
    resumed = train_model(mcfg, tcfg, stream, stream, tmp_path / "b.pt", log=lambda s: None)
    assert resumed["history"]["step"] == full["history"]["step"]
    assert np.allclose(resumed["history"]["val_loss"], full["history"]["val_loss"], atol=1e-6)
    assert not (tmp_path / "b.pt.resume").exists()  # cleaned up after finishing
