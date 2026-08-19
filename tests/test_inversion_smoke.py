"""A 30-second gradient-inversion smoke test for CI.

Verifies the attack pipeline runs, is deterministic under seed, and makes
directional progress (the reconstructed gradient aligns with the target), without
the minutes-long high-iteration reconstructions the demo produces. Torch only —
synthetic images, no Fashion-MNIST download — so it is hermetic and fast. The
full grids come from scripts/inversion_demo.py, run in the Docker batch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.inversion_demo import (  # noqa: E402
    IMG_SHAPE,
    apply_dp_noise,
    build_model,
    reconstruct,
    set_seed,
    true_gradient,
)


def _synthetic_batch(num_images: int, seed: int):
    """A fixed random image batch and labels — no TF, no download."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((num_images, *IMG_SHAPE), generator=g)
    y = torch.randint(0, 10, (num_images,), generator=g)
    return x, y


def test_reconstruction_makes_progress_and_is_deterministic():
    set_seed(0)
    model = build_model(seed=0)
    x, y = _synthetic_batch(1, seed=1)
    target = true_gradient(model, x, y)

    recon_a, obj_a = reconstruct(model, target, y, num_images=1, iters=50, tv_weight=0.0, seed=0)

    # Progress: the cosine-distance objective must fall below 1.0 (the dummy
    # gradient acquired positive alignment with the target). A random init sits
    # at ~1.0; any real optimisation beats that.
    assert obj_a < 0.98
    assert recon_a.shape == (1, *IMG_SHAPE)
    assert float(recon_a.min()) >= 0.0 and float(recon_a.max()) <= 1.0

    # Determinism: same seed, identical result, bit for bit.
    recon_b, obj_b = reconstruct(model, target, y, num_images=1, iters=50, tv_weight=0.0, seed=0)
    assert obj_a == pytest.approx(obj_b, abs=0.0)
    assert torch.equal(recon_a, recon_b)


def test_dp_noise_degrades_the_attack():
    """The defence in miniature: a heavily noised gradient reconstructs worse than
    a clean one. Few iterations, so this checks direction, not the full curve."""
    set_seed(0)
    model = build_model(seed=0)
    x, y = _synthetic_batch(1, seed=2)
    clean = true_gradient(model, x, y)
    noised = apply_dp_noise(clean, noise_multiplier=2.0, seed=3)

    recon_clean, _ = reconstruct(model, clean, y, num_images=1, iters=60, tv_weight=0.0, seed=0)
    recon_noised, _ = reconstruct(model, noised, y, num_images=1, iters=60, tv_weight=0.0, seed=0)

    mse_clean = float((recon_clean - x).pow(2).mean())
    mse_noised = float((recon_noised - x).pow(2).mean())
    assert mse_noised > mse_clean


def test_apply_dp_noise_is_a_noop_at_zero():
    set_seed(0)
    model = build_model(seed=0)
    x, y = _synthetic_batch(1, seed=4)
    grads = true_gradient(model, x, y)
    same = apply_dp_noise(grads, noise_multiplier=0.0, seed=0)
    for a, b in zip(grads, same, strict=True):
        assert torch.equal(a, b)
