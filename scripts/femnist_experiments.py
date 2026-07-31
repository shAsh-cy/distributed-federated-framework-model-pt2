"""FEMNIST experiments: the decoupled cohort sweep and its controls.

Why this exists
---------------
The Fashion-MNIST cohort sweep (docs/dp_diagnosis.md section 6-8) could not
attribute its own result: holding q = 0.5 there forced N = 2m, so a bigger
cohort also meant thinner shards, and accuracy moved for two unrelated reasons
at once. FEMNIST removes the confound. The population is a fixed set of real
writers, each writer's shard is what that writer actually wrote, and sweeping
the cohort m changes *only* m.

The new tension this design buys, stated up front because it is the point of
the experiment: at fixed N, raising m raises the sampling rate q = m/N, which
weakens privacy amplification by subsampling and therefore forces a larger
noise multiplier z for the same epsilon; but averaging over more clients
suppresses the applied noise as 1/m. Which effect dominates, and over what
range, is measured here -- not predicted.

Every accuracy figure this script emits is run at multiple seeds; docs report
means with ranges, never single draws. DP runs additionally vary beyond the
seed because TFF's executor draws noise unseedably (see
scripts/diagnose_dp.py's module docstring; that finding applies unchanged).

Do not pipe this script's stdout into a command that waits for EOF -- TFF
leaves an executor subprocess holding the pipe. Redirect to a file:

    python scripts/femnist_experiments.py --experiment sweep --out out.json > sweep.log 2>&1

Defaults
--------
--writers defaults to 1000: a seeded subsample of the 3,400-writer population,
sized so a 20-round m=500 cell stays CPU-feasible while still spanning
q = 0.005 .. 0.5 across the sweep. --seeds defaults to 42,43,44.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl.aggregation import (  # noqa: E402
    ClientUpdate,
    DPFedAvgAggregator,
    FedAvgAggregator,
    calibrate_noise_multiplier,
    compute_epsilon,
    l2_norm,
    subtract,
)
from fl.data import (  # noqa: E402
    FEMNIST_NUM_CLASSES,
    label_distribution,
    label_entropy,
    load_femnist,
)
from fl.models import (  # noqa: E402
    FEMNIST_CNN_PARAMS,
    build_model,
    compile_for_evaluation,
    compile_for_training,
)

LOGGER = logging.getLogger("femnist")

DEFAULT_WRITERS = 1000
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_ROUNDS = 20
DEFAULT_TARGET_EPSILON = 6.228
DEFAULT_CLIP = 0.5
SWEEP_COHORTS = (5, 20, 50, 100, 200, 500)
BRACKET_CLIPS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def seed_everything(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _reset_optimizer(model) -> None:
    import tensorflow as tf

    opt = getattr(model, "optimizer", None)
    if opt is None:
        return
    variables = opt.variables
    if callable(variables):
        variables = variables()
    for v in variables:
        v.assign(tf.zeros_like(v))


def simulate(
    *,
    train,
    test,
    shards: list[np.ndarray],
    clients_per_round: int,
    rounds: int = DEFAULT_ROUNDS,
    dp: bool,
    noise_multiplier: float = 0.0,
    l2_clip_norm: float = DEFAULT_CLIP,
    seed: int = 42,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 32,
    local_epochs: int = 1,
    label: str = "",
) -> dict:
    """One federated run over a pre-loaded population. Returns everything measured."""
    seed_everything(seed)
    started = time.monotonic()
    num_clients = len(shards)

    global_weights = build_model("femnist_cnn", seed=seed).get_weights()
    trainer = compile_for_training(build_model("femnist_cnn"), learning_rate, momentum)
    evaluator = compile_for_evaluation(build_model("femnist_cnn"))

    aggregator = (
        DPFedAvgAggregator(noise_multiplier, l2_clip_norm, clients_per_round)
        if dp
        else FedAvgAggregator()
    )

    rng = np.random.default_rng(seed)
    history: list[dict] = []

    evaluator.set_weights(global_weights)
    base_loss, base_acc = evaluator.evaluate(test.x, test.y, batch_size=512, verbose=0)

    for rnd in range(1, rounds + 1):
        round_started = time.monotonic()
        cohort = rng.choice(num_clients, size=clients_per_round, replace=False)
        updates: list[ClientUpdate] = []
        pre_clip_norms: list[float] = []

        for cid in cohort:
            idx = shards[int(cid)]
            trainer.set_weights(global_weights)
            _reset_optimizer(trainer)
            trainer.fit(
                train.x[idx],
                train.y[idx],
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
            )
            w = trainer.get_weights()
            pre_clip_norms.append(l2_norm(subtract(w, global_weights)))
            updates.append(ClientUpdate(f"c{int(cid)}", w, int(idx.size)))

        previous = global_weights
        global_weights = aggregator.aggregate(updates, global_weights)
        applied = l2_norm(subtract(global_weights, previous))
        finite = all(np.all(np.isfinite(w)) for w in global_weights)

        if finite:
            evaluator.set_weights(global_weights)
            loss, acc = evaluator.evaluate(test.x, test.y, batch_size=512, verbose=0)
        else:
            loss, acc = float("nan"), 0.0

        history.append(
            {
                "round": rnd,
                "accuracy": float(acc),
                "loss": float(loss),
                "median_pre_clip_norm": float(np.median(pre_clip_norms)),
                "clipped_fraction": float(np.mean([n > l2_clip_norm for n in pre_clip_norms])),
                "applied_delta_norm": float(applied),
                "seconds": round(time.monotonic() - round_started, 2),
                "finite": bool(finite),
            }
        )
        LOGGER.info(
            "%s round %2d: acc=%.4f median||dw||=%.3f clipped=%.2f ||applied||=%.3f %.1fs",
            label,
            rnd,
            acc,
            history[-1]["median_pre_clip_norm"],
            history[-1]["clipped_fraction"],
            applied,
            history[-1]["seconds"],
        )

    return {
        "label": label,
        "num_clients": num_clients,
        "clients_per_round": clients_per_round,
        "rounds": rounds,
        "dp": dp,
        "noise_multiplier": noise_multiplier,
        "l2_clip_norm": l2_clip_norm,
        "seed": seed,
        "sampling_rate": clients_per_round / num_clients,
        "epsilon": compute_epsilon(noise_multiplier, clients_per_round / num_clients, rounds)
        if dp
        else None,
        "untrained_accuracy": float(base_acc),
        "untrained_loss": float(base_loss),
        "final_accuracy": history[-1]["accuracy"],
        "best_accuracy": max(h["accuracy"] for h in history),
        "mean_round_seconds": float(np.mean([h["seconds"] for h in history])),
        "median_pre_clip_norm_all_rounds": float(
            np.median([h["median_pre_clip_norm"] for h in history])
        ),
        "clipped_fraction_all_rounds": float(np.mean([h["clipped_fraction"] for h in history])),
        "seconds": round(time.monotonic() - started, 1),
        "history": history,
    }


def _summary(runs: list[dict]) -> dict:
    finals = [r["final_accuracy"] for r in runs]
    return {
        "seeds": [r["seed"] for r in runs],
        "final_per_seed": finals,
        "mean_final": float(np.mean(finals)),
        "range_final": float(max(finals) - min(finals)),
        "mean_best": float(np.mean([r["best_accuracy"] for r in runs])),
        "mean_round_seconds": float(np.mean([r["mean_round_seconds"] for r in runs])),
    }


def exp_entropy(writers: int, seeds: tuple[int, ...]) -> dict:
    """Population shape: shard sizes and label skew, for docs and the tests."""
    train, test, shards = load_femnist(num_clients=writers, seed=seeds[0])
    sizes = np.array([s.size for s in shards])
    pooled = label_entropy(label_distribution(train.y, np.arange(len(train)), FEMNIST_NUM_CLASSES))
    per_writer = [
        label_entropy(label_distribution(train.y, s, FEMNIST_NUM_CLASSES)) for s in shards
    ]
    return {
        "writers": writers,
        "train_examples": len(train),
        "test_examples": len(test),
        "shard_size_min": int(sizes.min()),
        "shard_size_median": int(np.median(sizes)),
        "shard_size_max": int(sizes.max()),
        "pooled_label_entropy_nats": pooled,
        "mean_writer_label_entropy_nats": float(np.mean(per_writer)),
        "median_writer_label_entropy_nats": float(np.median(per_writer)),
        "uniform_entropy_62_nats": math.log(FEMNIST_NUM_CLASSES),
    }


def exp_baseline(writers: int, seeds: tuple[int, ...], epochs: int) -> dict:
    """Pooled centralised baseline: all shards combined, no federation.

    The true upper bound every federated number should be read against. Same
    model, same optimiser family as the clients; per-epoch test accuracy is
    recorded so the write-up can say whether the budget sufficed.
    """
    runs = []
    for seed in seeds:
        seed_everything(seed)
        train, test, _shards = load_femnist(num_clients=writers, seed=seeds[0])
        model = compile_for_training(build_model("femnist_cnn", seed=seed), 0.01, 0.9)
        per_epoch = []
        started = time.monotonic()
        for epoch in range(1, epochs + 1):
            model.fit(train.x, train.y, epochs=1, batch_size=64, verbose=0)
            loss, acc = model.evaluate(test.x, test.y, batch_size=512, verbose=0)
            per_epoch.append({"epoch": epoch, "accuracy": float(acc), "loss": float(loss)})
            LOGGER.info("baseline seed=%d epoch %d: acc=%.4f", seed, epoch, acc)
        runs.append(
            {
                "seed": seed,
                "epochs": epochs,
                "final_accuracy": per_epoch[-1]["accuracy"],
                "best_accuracy": max(e["accuracy"] for e in per_epoch),
                "per_epoch": per_epoch,
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        LOGGER.info(
            "BASELINE seed=%d -> final=%.4f best=%.4f",
            seed,
            runs[-1]["final_accuracy"],
            runs[-1]["best_accuracy"],
        )
    finals = [r["final_accuracy"] for r in runs]
    return {
        "writers": writers,
        "model_parameters": FEMNIST_CNN_PARAMS,
        "mean_final": float(np.mean(finals)),
        "range_final": float(max(finals) - min(finals)),
        "final_per_seed": finals,
        "runs": runs,
    }


def exp_nodp_control(
    writers: int,
    seeds: tuple[int, ...],
    rounds: int,
    cohorts: tuple[int, ...],
) -> dict:
    """Federated no-DP control at selected cohort sizes.

    Interpreting the DP sweep requires knowing what plain FedAvg achieves on
    the identical population, cohorts and round budget: without this, a flat
    DP curve cannot be attributed to noise as opposed to FedAvg simply not
    getting far in 20 rounds on thin natural shards. The centralised baseline
    cannot answer that -- it takes ~15,000 gradient steps per epoch, while a
    federated round takes 5 per client.
    """
    train, test, shards = load_femnist(num_clients=writers, seed=seeds[0])
    cells = []
    for m in cohorts:
        runs = [
            simulate(
                train=train,
                test=test,
                shards=shards,
                clients_per_round=m,
                rounds=rounds,
                dp=False,
                seed=seed,
                label=f"nodp/m={m}/seed={seed}",
            )
            for seed in seeds
        ]
        summary = _summary(runs)
        LOGGER.info(
            "NODP m=%d DONE: mean_final=%.4f range=%.4f",
            m,
            summary["mean_final"],
            summary["range_final"],
        )
        cells.append({"m": m, "summary": summary, "runs": runs})
    return {"writers": writers, "rounds": rounds, "cells": cells}


def exp_sweep(
    writers: int,
    seeds: tuple[int, ...],
    target_epsilon: float,
    clip: float,
    rounds: int,
) -> dict:
    """The decoupled sweep: fixed population, fixed target epsilon, m varies alone."""
    train, test, shards = load_femnist(num_clients=writers, seed=seeds[0])
    cells = []
    for m in SWEEP_COHORTS:
        q = m / writers
        z = calibrate_noise_multiplier(target_epsilon, q, rounds)
        achieved = compute_epsilon(z, q, rounds)
        LOGGER.info(
            "CELL m=%d: q=%.4f calibrated z=%.4f (achieved eps=%.4f, target %.4f)",
            m,
            q,
            z,
            achieved,
            target_epsilon,
        )
        runs = [
            simulate(
                train=train,
                test=test,
                shards=shards,
                clients_per_round=m,
                rounds=rounds,
                dp=True,
                noise_multiplier=z,
                l2_clip_norm=clip,
                seed=seed,
                label=f"sweep/m={m}/seed={seed}",
            )
            for seed in seeds
        ]
        summary = _summary(runs)
        LOGGER.info(
            "CELL m=%d DONE: mean_final=%.4f range=%.4f z=%.4f",
            m,
            summary["mean_final"],
            summary["range_final"],
            z,
        )
        cells.append(
            {
                "m": m,
                "q": q,
                "calibrated_z": z,
                "achieved_epsilon": achieved,
                "predicted_noise_norm": clip * z * math.sqrt(FEMNIST_CNN_PARAMS) / m,
                "summary": summary,
                "runs": runs,
            }
        )
    return {
        "writers": writers,
        "target_epsilon": target_epsilon,
        "delta": 1e-5,
        "clip": clip,
        "rounds": rounds,
        "cells": cells,
    }


def exp_clip_bracket(
    writers: int,
    seeds: tuple[int, ...],
    target_epsilon: float,
    m: int,
    clips: tuple[float, ...],
    rounds: int,
) -> dict:
    """Clip sweep at one cohort size, aiming to bracket the optimum.

    With natural shards the median update norm no longer moves with m, so an
    interior maximum in clip should be locatable -- the Fashion-MNIST sweep
    bottomed out at its smallest value without bracketing.
    """
    train, test, shards = load_femnist(num_clients=writers, seed=seeds[0])
    q = m / writers
    z = calibrate_noise_multiplier(target_epsilon, q, rounds)
    cells = []
    for clip in clips:
        runs = [
            simulate(
                train=train,
                test=test,
                shards=shards,
                clients_per_round=m,
                rounds=rounds,
                dp=True,
                noise_multiplier=z,
                l2_clip_norm=clip,
                seed=seed,
                label=f"bracket/S={clip}/seed={seed}",
            )
            for seed in seeds
        ]
        summary = _summary(runs)
        LOGGER.info(
            "BRACKET S=%s DONE: mean_final=%.4f range=%.4f",
            clip,
            summary["mean_final"],
            summary["range_final"],
        )
        cells.append({"clip": clip, "summary": summary, "runs": runs})
    return {
        "writers": writers,
        "m": m,
        "q": q,
        "calibrated_z": z,
        "achieved_epsilon": compute_epsilon(z, q, rounds),
        "delta": 1e-5,
        "rounds": rounds,
        "cells": cells,
    }


EXPERIMENTS = ("entropy", "baseline", "sweep", "nodp_control", "clip_bracket")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--writers", type=int, default=DEFAULT_WRITERS)
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--target-epsilon", type=float, default=DEFAULT_TARGET_EPSILON)
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    parser.add_argument("--m", type=int, default=None, help="cohort size for clip_bracket")
    parser.add_argument("--clips", default=",".join(str(c) for c in BRACKET_CLIPS))
    parser.add_argument(
        "--cohorts", default="50,500", help="cohort sizes for nodp_control (comma-separated)"
    )
    parser.add_argument("--epochs", type=int, default=5, help="baseline epochs")
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    seeds = tuple(int(s) for s in args.seeds.split(","))

    if args.experiment == "entropy":
        result = exp_entropy(args.writers, seeds)
    elif args.experiment == "baseline":
        result = exp_baseline(args.writers, seeds, args.epochs)
    elif args.experiment == "sweep":
        result = exp_sweep(args.writers, seeds, args.target_epsilon, args.clip, args.rounds)
    elif args.experiment == "nodp_control":
        cohorts = tuple(int(c) for c in args.cohorts.split(","))
        result = exp_nodp_control(args.writers, seeds, args.rounds, cohorts)
    else:
        if args.m is None:
            parser.error("--m is required for clip_bracket")
        clips = tuple(float(c) for c in args.clips.split(","))
        result = exp_clip_bracket(
            args.writers, seeds, args.target_epsilon, args.m, clips, args.rounds
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({args.experiment: result}, indent=1), encoding="utf-8")
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
