import torch

from minigpt.sae import SparseAutoencoder, fraction_variance_explained


def test_sae_recovers_sparse_dictionary():
    """Synthetic superposition: 32 sparse ground-truth features living in 16 dims."""
    g = torch.Generator().manual_seed(0)
    true_dirs = torch.nn.functional.normalize(torch.randn(32, 16, generator=g), dim=1)
    active = (torch.rand(4096, 32, generator=g) < 0.05).float()
    x = (active * torch.rand(4096, 32, generator=g)) @ true_dirs
    torch.manual_seed(0)
    sae = SparseAutoencoder(16, 64)
    opt = torch.optim.Adam(sae.parameters(), 3e-3)
    for _ in range(1500):
        loss, _, _ = sae.loss(x, l1=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    assert fraction_variance_explained(sae, x) > 0.9
    _, f = sae(x)
    assert (f > 0).float().mean() < 0.15  # sparse codes
    # every ground-truth direction is matched by some decoder column
    dec = torch.nn.functional.normalize(sae.W_dec, dim=0)
    assert (true_dirs @ dec).max(1).values.mean() > 0.9
