import torch

from minigpt.optim import AdamW, lr_at


def test_adamw_matches_torch():
    torch.manual_seed(0)
    w1 = torch.nn.Parameter(torch.randn(5, 3))
    w2 = torch.nn.Parameter(w1.detach().clone())
    o1 = AdamW([w1], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    o2 = torch.optim.AdamW([w2], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(50):
        g = torch.randn(5, 3)
        for w, o in ((w1, o1), (w2, o2)):
            o.zero_grad()
            (w * g).sum().backward()
            o.step()
    assert torch.allclose(w1, w2, atol=1e-6)


def test_schedule():
    assert lr_at(0, 100, 1.0, 10) == 0.1
    assert abs(lr_at(10, 100, 1.0, 10) - 1.0) < 1e-9
    assert abs(lr_at(100, 100, 1.0, 10) - 0.1) < 1e-9
    vals = [lr_at(s, 100, 1.0, 10) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))  # monotone decay
