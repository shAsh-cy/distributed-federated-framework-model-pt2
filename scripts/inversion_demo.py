"""Gradient inversion, honestly framed — reconstruct training images from a gradient.

The threat that motivates both defences in this repo: a single client's gradient,
the thing the server sees every round on the plaintext path, can be inverted back
into the training images that produced it. This script demonstrates it against the
small Fashion-MNIST CNN using the cosine-similarity attack of Geiping et al.
("Inverting Gradients", NeurIPS 2020), with the iDLG observation that a
single-example label is recoverable from the last-layer gradient sign.

WHAT THIS SHOWS, and — mandatory — WHAT IT DOES NOT.

* Reconstructions at batch sizes {1, 4, 8}, no defence. A batch of one inverts
  cleanly; the reconstruction degrades as the batch grows and gradients average.
* A defence curve: the same attack against a gradient carrying DP noise at
  multipliers {0, 0.3, working z=2.0}. Recognisable garment -> smeared -> mush.
* One attempt against a REALISTIC E=10 FedAvg update (ten local epochs over the
  whole shard, not one batch's gradient), which fails. That failure is the
  honest result: recent work (e.g. Huang et al. 2021, "Evaluating Gradient
  Inversion Attacks and Defenses") finds these attacks assume conditions —
  known small batch, a single step, no multi-epoch averaging — that production
  FedAvg violates. The demo illustrates the THREAT MODEL that motivates DP and
  secure aggregation; it does not claim production-realistic leakage.

Two honesty notes carried into docs/privacy_threats.md:
* Labels are treated as known (the favourable assumption in the batch>1 case);
  this is part of "idealised conditions", not a hidden strength of the attack.
* The repo's DP is CENTRAL (server-side), so it does not noise an individual
  update at all — the server sees each gradient in the clear. The defence curve
  therefore models noise added to the update *before it leaves the client* (local
  / distributed DP); it is labelled with the epsilon its multiplier corresponds
  to in the project's accountant at the working config (q=0.5, R=20, delta=1e-5).
  Hiding the individual update from an honest-but-curious server is what SECURE
  AGGREGATION does; DP bounds what the released model reveals. See the write-up.

Deterministic under --seed. Not launched by hand overnight —
scripts/run_inversion_batch.sh queues it behind the batch lock.

    python scripts/inversion_demo.py --out-dir docs/inversion --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# torch MUST be imported before TensorFlow in any process that touches both, or
# torch's std::random_device aborts the process (see docs/architecture.md). This
# script loads Fashion-MNIST through fl.data (which imports TF), so torch leads.
import torch  # noqa: F401  (import-order load-bearing)
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl.archspec import SMALL_CNN_SPEC, build_torch  # noqa: E402

LOGGER = logging.getLogger("inversion")

IMG_SHAPE = (1, 28, 28)  # torch NCHW for one 28x28 grayscale image
WORKING_NOISE_MULTIPLIER = 2.0  # configs/dp_moderate.yaml; epsilon ~= 6.23
WORKING_Q = 0.5
WORKING_ROUNDS = 20
WORKING_DELTA = 1e-5


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Model and gradients
# ---------------------------------------------------------------------------


def build_model(seed: int) -> nn.Module:
    """A seeded random-init small_cnn. Inversion leaks the input regardless of
    training state; a fixed random model keeps the demo fast and reproducible."""
    torch.manual_seed(seed)
    model = build_torch(SMALL_CNN_SPEC)
    model.eval()  # no dropout/BN-train here; deterministic forward
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def true_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    """The gradient the client would compute on one batch — the attack's target."""
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return [g.detach().clone() for g in grads]


def fedavg_update(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, batch_size: int
) -> list[torch.Tensor]:
    """A REALISTIC update: the parameter delta after ``epochs`` local epochs of
    SGD over the whole shard, returned in gradient shape (delta / -lr so it reads
    as a pseudo-gradient the attack can consume). This is what FedAvg actually
    sends — a composition of many steps over many batches, not one gradient."""
    import copy

    before = [p.detach().clone() for p in model.parameters()]
    trainer = copy.deepcopy(model)
    trainer.train()
    opt = torch.optim.SGD(trainer.parameters(), lr=lr, momentum=0.9)
    n = x.shape[0]
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(trainer(x[idx]), y[idx])
            loss.backward()
            opt.step()
    after = list(trainer.parameters())
    # delta = after - before; as a pseudo-gradient, g ~ -delta / lr.
    return [(b - a.detach()).clone() / lr for a, b in zip(before, after, strict=True)]


def apply_dp_noise(
    grads: list[torch.Tensor], noise_multiplier: float, seed: int
) -> list[torch.Tensor]:
    """Add Gaussian noise to a gradient the way client-side (local/distributed)
    DP would before it leaves the client. Calibrated to the update's own scale:
    per-coordinate stddev = ``noise_multiplier * rms(grad)``, so 0 is clean, 0.3
    is commensurate with the signal, and 2.0 dominates it — the visual defence
    curve. (Central DP, the repo's shipped mechanism, noises the AGGREGATE, not
    the individual update, and so would not blur this at all; that difference is
    the point the write-up makes.)"""
    if noise_multiplier <= 0:
        return [g.clone() for g in grads]
    flat = torch.cat([g.reshape(-1) for g in grads])
    rms = float(flat.pow(2).mean().sqrt())
    gen = torch.Generator().manual_seed(seed)
    stddev = noise_multiplier * rms
    return [g + torch.randn(g.shape, generator=gen) * stddev for g in grads]


# ---------------------------------------------------------------------------
# The attack
# ---------------------------------------------------------------------------


def _flatten(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def reconstruct(
    model: nn.Module,
    target: list[torch.Tensor],
    labels: torch.Tensor,
    num_images: int,
    iters: int = 4000,
    lr: float = 0.1,
    tv_weight: float = 1e-2,
    seed: int = 0,
) -> tuple[torch.Tensor, float]:
    """Invert ``target`` into ``num_images`` images by minimising cosine distance
    between the dummy gradient and the target, plus a total-variation prior.
    Labels are provided (see the module honesty note). Returns the reconstruction
    clamped to [0, 1] and the final objective value."""
    torch.manual_seed(seed)
    dummy = torch.rand((num_images, *IMG_SHAPE), requires_grad=True)
    opt = torch.optim.Adam([dummy], lr=lr)
    params = list(model.parameters())
    target_flat = _flatten(target)

    final = float("nan")
    for step in range(iters):
        opt.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(dummy), labels)
        dummy_grad = torch.autograd.grad(loss, params, create_graph=True)
        cosine = 1.0 - F.cosine_similarity(_flatten(dummy_grad), target_flat, dim=0)
        objective = cosine + tv_weight * total_variation(dummy)
        objective.backward()
        opt.step()
        with torch.no_grad():
            dummy.clamp_(0.0, 1.0)
        final = float(objective.detach())
        if step % max(1, iters // 5) == 0:
            LOGGER.debug("  step %d/%d objective=%.4f", step, iters, final)
    return dummy.detach().clamp(0.0, 1.0), final


def reconstruction_mse(recon: torch.Tensor, original: torch.Tensor) -> float:
    return float((recon - original).pow(2).mean())


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def save_grid(
    originals: torch.Tensor,
    recons: torch.Tensor,
    path: Path,
    title: str,
    column_labels: list[str] | None = None,
) -> None:
    """Save an originals-vs-reconstructions grid: top row real, bottom row recon."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = originals.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j in range(n):
        for row, (img, label) in enumerate(
            ((originals[j], "original"), (recons[j], "reconstruction"))
        ):
            ax = axes[row, j]
            ax.imshow(img.squeeze(0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 0 and column_labels is not None:
                ax.set_title(column_labels[j], fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", path)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_images(num_images: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic batch of real Fashion-MNIST training images as torch NCHW,
    plus their labels. Imports fl.data (TensorFlow) — torch is already loaded, so
    the import order is safe."""
    from fl.data import load_fashion_mnist

    train, _test = load_fashion_mnist()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=num_images, replace=False)
    x = train.x[idx]  # (n, 28, 28, 1) NHWC in [0, 1]
    y = train.y[idx]
    x_nchw = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    return torch.from_numpy(x_nchw).float(), torch.from_numpy(y.astype(np.int64))


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def epsilon_at_working_config(noise_multiplier: float) -> float:
    """The epsilon this multiplier buys in the project's accountant at the working
    config. Imports dp_accounting lazily; falls back to inf on the no-noise case."""
    if noise_multiplier <= 0:
        return float("inf")
    from fl.aggregation import compute_epsilon

    return compute_epsilon(noise_multiplier, WORKING_Q, WORKING_ROUNDS, WORKING_DELTA)


def phase_batch_sizes(out_dir: Path, seed: int, iters: int) -> list[dict]:
    """Reconstructions with no defence at batch sizes {1, 4, 8}."""
    results = []
    for bs in (1, 4, 8):
        set_seed(seed)
        model = build_model(seed)
        x, y = load_images(bs, seed)
        target = true_gradient(model, x, y)
        recon, obj = reconstruct(model, target, y, num_images=bs, iters=iters, seed=seed)
        mse = reconstruction_mse(recon, x)
        save_grid(
            x,
            recon,
            out_dir / f"batch_size_{bs}.png",
            title=f"No defence, batch size {bs} — reconstruction MSE {mse:.4f}",
        )
        LOGGER.info("batch size %d: MSE %.4f (objective %.4f)", bs, mse, obj)
        results.append({"batch_size": bs, "mse": mse, "objective": obj})
    return results


def phase_defense_curve(out_dir: Path, seed: int, iters: int) -> list[dict]:
    """Batch-of-one attack against a gradient with local DP noise at {0, 0.3, z}."""
    multipliers = [0.0, 0.3, WORKING_NOISE_MULTIPLIER]
    x, y = load_images(1, seed)
    originals, recons, labels, results = [], [], [], []
    for z in multipliers:
        set_seed(seed)
        model = build_model(seed)
        target = apply_dp_noise(true_gradient(model, x, y), z, seed=seed + 1)
        recon, obj = reconstruct(model, target, y, num_images=1, iters=iters, seed=seed)
        mse = reconstruction_mse(recon, x)
        eps = epsilon_at_working_config(z)
        eps_label = "inf (no privacy)" if not np.isfinite(eps) else f"{eps:.2f}"
        originals.append(x[0])
        recons.append(recon[0])
        labels.append(f"z={z}\neps={eps_label}")
        results.append({"noise_multiplier": z, "epsilon": eps, "mse": mse, "objective": obj})
        LOGGER.info("defense z=%.2f: MSE %.4f eps=%s", z, mse, eps_label)
    save_grid(
        torch.stack(originals),
        torch.stack(recons),
        out_dir / "defense_curve.png",
        title="DP-noise defence curve (noise added client-side); label = multiplier, epsilon at working config",
        column_labels=labels,
    )
    return results


def phase_fedavg_realism(out_dir: Path, seed: int, iters: int) -> dict:
    """One attempt against a realistic E=10 FedAvg update — the honest failure."""
    set_seed(seed)
    model = build_model(seed)
    x, y = load_images(64, seed)  # a small shard, not one batch
    update = fedavg_update(model, x, y, epochs=10, lr=0.01, batch_size=32)
    # Attack it as if it were a single-batch gradient for one image.
    recon, obj = reconstruct(model, update, y[:1], num_images=1, iters=iters, seed=seed)
    mse = reconstruction_mse(recon, x[:1])
    save_grid(
        x[:1],
        recon,
        out_dir / "fedavg_e10_failure.png",
        title=f"Realistic E=10 FedAvg update — attack fails (MSE {mse:.4f})",
    )
    LOGGER.info("E=10 FedAvg realism: MSE %.4f (objective %.4f) — expected to be poor", mse, obj)
    return {"epochs": 10, "shard_size": 64, "mse": mse, "objective": obj}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="docs/inversion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=4000, help="attack iterations per image set")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["batch", "defense", "fedavg"],
        default=["batch", "defense", "fedavg"],
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"seed": args.seed, "iters": args.iters}
    if "batch" in args.phases:
        summary["batch_sizes"] = phase_batch_sizes(out_dir, args.seed, args.iters)
    if "defense" in args.phases:
        summary["defense_curve"] = phase_defense_curve(out_dir, args.seed, args.iters)
    if "fedavg" in args.phases:
        summary["fedavg_realism"] = phase_fedavg_realism(out_dir, args.seed, args.iters)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
