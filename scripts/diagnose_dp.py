"""Diagnostic harness for the differential-privacy accuracy collapse.

READ THIS BEFORE TRUSTING THE NUMBERS
-------------------------------------
This script does **not** modify the DP implementation. It imports
``fl.aggregation`` unchanged and drives it from an in-process simulation of the
federated loop, so the aggregation arithmetic under test is byte-for-byte the
code the real gRPC server runs. Only the transport is skipped, because these
experiments need ~20 full runs and the gRPC path costs minutes each.

Two deliberate deviations from the production client, both documented so they
can be discounted:

1. One Keras model object is reused across clients with its optimiser slots
   zeroed between clients, rather than one persistent model per client. This
   drops SGD momentum carry-over across rounds. ``validate`` below checks the
   no-DP simulation against the recorded 86.93% real run; if they agree, the
   deviation is immaterial.
2. Ablation (a) needs ``noise_multiplier = 0``, which ``DPFedAvgAggregator``
   rejects by design. ``_DiagnosticDPAggregator`` below is a copy of that class
   with the guard removed and nothing else changed -- it calls the identical
   ``tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed``.

DP RUNS ARE NOT REPRODUCIBLE, AND IT IS NOT THIS SCRIPT'S FAULT
---------------------------------------------------------------
Two consecutive ``simulate(dp=True, ...)`` calls with the same seed, the same
config, in the same process, with nothing at all in between produce different
noise. Verified by ``--experiment repeatability``, which runs each configuration
twice and never calls ``measure_applied_noise``.

The cause is in the dependency stack, not in the harness. TF Privacy's
``GaussianSumQuery.get_noised_result`` draws noise via::

    random_normal = tf.random_normal_initializer(stddev=global_state.stddev)

with no ``seed`` argument (tensorflow_privacy 0.9.0,
``privacy/dp_query/gaussian_query.py``). That initialiser *is* honoured by
``tf.random.set_seed`` in ordinary eager code and under ``tf.function`` -- both
checked -- but TFF serialises the aggregation to a computation proto and runs it
in its own executor, which never sees this process's global seed. Neither
``tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed`` nor
``GaussianSumQuery`` exposes a seed parameter, so there is no supported way to
make it deterministic from here.

Consequences for reading the numbers in docs/dp_diagnosis.md:

* Non-DP runs are exactly reproducible; ``--experiment validate`` reproduces the
  recorded 86.93% gRPC run every time.
* DP runs are reproducible *in configuration and in conclusion*, not bit-for-bit.
  Treat differences of a few accuracy points between two DP runs of the same
  config as noise, not signal. ``--experiment repeatability`` measures the spread.
* Making DP runs deterministic would require passing a per-round seed into the
  query -- i.e. changing the DP mechanism under test -- which is out of scope for
  a harness whose purpose is to measure that mechanism unmodified.

Usage:
    python scripts/diagnose_dp.py --experiment all --out docs/dp_diagnosis_data.json

Do not pipe this script's output into a command that waits for end-of-file
(``| grep ... | tail``). TFF starts a background executor subprocess that
inherits stdout and outlives the run, so the reader never sees EOF and hangs
indefinitely after the work has actually finished. Redirect to a file instead:

    python scripts/diagnose_dp.py --experiment noise_sweep --out out.json > run.log 2>&1
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
    FedAvgAggregator,
    add,
    compute_epsilon,
    l2_norm,
    subtract,
    validate_updates,
)
from fl.data import load_fashion_mnist, partition  # noqa: E402
from fl.models import build_model, compile_for_evaluation, compile_for_training  # noqa: E402

LOGGER = logging.getLogger("diagnose")

MODEL_PARAMS = 225_034
DEFAULT_ROUNDS = 20
DEFAULT_SEED = 42


class _DiagnosticDPAggregator:
    """Copy of fl.aggregation.DPFedAvgAggregator with the noise_multiplier > 0 guard removed.

    Identical in every other respect: same TFF factory, same delta-space
    aggregation, same cross-round state. The guard exists in production because
    noise_multiplier=0 provides no privacy; here it is exactly the ablation we
    need, so it is bypassed in the diagnostic only.
    """

    name = "diagnostic-dp"

    def __init__(self, noise_multiplier: float, l2_clip_norm: float, clients_per_round: int):
        self.noise_multiplier = float(noise_multiplier)
        self.l2_clip_norm = float(l2_clip_norm)
        self.clients_per_round = int(clients_per_round)
        self._process = None
        self._state = None
        self._value_type = None

    def _ensure_process(self, template):
        import tensorflow_federated as tff

        value_type = tff.to_type(
            [tff.TensorType(np.float32, np.asarray(w).shape) for w in template]
        )
        if self._process is not None and value_type == self._value_type:
            return
        factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
            noise_multiplier=self.noise_multiplier,
            clients_per_round=float(self.clients_per_round),
            clip=self.l2_clip_norm,
        )
        self._value_type = value_type
        self._process = factory.create(value_type)
        self._state = self._process.initialize()

    def aggregate(self, updates, global_weights):
        validate_updates(updates)
        deltas = [subtract(u.weights, global_weights) for u in updates]
        self._ensure_process(global_weights)
        output = self._process.next(self._state, deltas)
        self._state = output.state
        mean_delta = [np.asarray(t, dtype=np.float32) for t in output.result]
        return add(global_weights, mean_delta)


def seed_everything(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _reset_optimizer(model) -> None:
    """Zero the optimiser slot variables so one client's momentum does not leak into the next."""
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
    num_clients: int,
    clients_per_round: int,
    rounds: int = DEFAULT_ROUNDS,
    dp: bool,
    noise_multiplier: float = 0.0,
    l2_clip_norm: float = 3.0,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.5,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 32,
    local_epochs: int = 1,
    record_norms: bool = True,
    record_predictions: bool = False,
    server_lr: float = 1.0,
    label: str = "",
) -> dict:
    """Run one federated experiment in-process and return everything measured."""
    seed_everything(seed)
    started = time.monotonic()

    train, test = load_fashion_mnist()
    shards = partition(train.y, num_clients, "dirichlet", alpha, seed)

    global_weights = build_model("small_cnn", seed=seed).get_weights()

    trainer = compile_for_training(build_model("small_cnn"), learning_rate, momentum)
    evaluator = compile_for_evaluation(build_model("small_cnn"))

    aggregator = (
        _DiagnosticDPAggregator(noise_multiplier, l2_clip_norm, clients_per_round)
        if dp
        else FedAvgAggregator()
    )

    rng = np.random.default_rng(seed)
    history: list[dict] = []

    evaluator.set_weights(global_weights)
    base_loss, base_acc = evaluator.evaluate(test.x, test.y, batch_size=512, verbose=0)

    for rnd in range(1, rounds + 1):
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
            if record_norms:
                # Measured BEFORE the aggregator sees it, so clipping cannot
                # have touched it.
                pre_clip_norms.append(l2_norm(subtract(w, global_weights)))
            updates.append(ClientUpdate(f"c{int(cid)}", w, int(idx.size)))

        previous = global_weights
        try:
            global_weights = aggregator.aggregate(updates, global_weights)
            if server_lr != 1.0:
                # Diagnostic-only knob, absent from fl/: the production server
                # applies the aggregate delta as-is (server_lr == 1.0). Used to
                # test whether the clip acts as a step size when clipping binds.
                global_weights = add(
                    previous,
                    [server_lr * d for d in subtract(global_weights, previous)],
                )
            aggregated = True
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            LOGGER.warning("round %d: aggregation failed: %s", rnd, exc)
            aggregated = False

        applied = l2_norm(subtract(global_weights, previous)) if aggregated else 0.0
        finite = all(np.all(np.isfinite(w)) for w in global_weights)

        if finite:
            evaluator.set_weights(global_weights)
            loss, acc = evaluator.evaluate(test.x, test.y, batch_size=512, verbose=0)
        else:
            loss, acc = float("nan"), 0.0

        # Fraction of this round's updates whose norm exceeded the clipping
        # threshold, i.e. the updates clipping actually touched.
        clipped_fraction = (
            float(np.mean([n > l2_clip_norm for n in pre_clip_norms])) if pre_clip_norms else None
        )

        history.append(
            {
                "round": rnd,
                "accuracy": float(acc),
                "loss": float(loss),
                "pre_clip_norms": [float(n) for n in pre_clip_norms],
                "median_pre_clip_norm": float(np.median(pre_clip_norms))
                if pre_clip_norms
                else None,
                "clipped_fraction": clipped_fraction,
                "applied_delta_norm": float(applied),
                "global_weight_norm": float(l2_norm(global_weights)) if finite else None,
                "aggregated": aggregated,
                "finite": bool(finite),
            }
        )
        LOGGER.info(
            "%s round %2d: acc=%.4f  median||delta||=%s  ||applied||=%.3f",
            label,
            rnd,
            acc,
            f"{np.median(pre_clip_norms):.3f}" if pre_clip_norms else "n/a",
            applied,
        )

    result = {
        "label": label,
        "num_clients": num_clients,
        "clients_per_round": clients_per_round,
        "rounds": rounds,
        "dp": dp,
        "noise_multiplier": noise_multiplier,
        "l2_clip_norm": l2_clip_norm,
        "server_lr": server_lr,
        "sampling_rate": clients_per_round / num_clients,
        "epsilon": compute_epsilon(noise_multiplier, clients_per_round / num_clients, rounds)
        if dp and noise_multiplier > 0
        else None,
        "untrained_accuracy": float(base_acc),
        "untrained_loss": float(base_loss),
        "final_accuracy": history[-1]["accuracy"],
        "best_accuracy": max(h["accuracy"] for h in history),
        "seconds": round(time.monotonic() - started, 1),
        "history": history,
    }

    # Summary statistics over pre-clip update norms. Round 1 is reported
    # separately because once a run has destroyed its own model the clients are
    # training on wreckage and their norms say nothing about healthy training.
    r1 = history[0]["pre_clip_norms"]
    finite_norms = [n for h in history for n in h["pre_clip_norms"] if math.isfinite(n)]
    result["round1_median_pre_clip_norm"] = float(np.median(r1)) if r1 else None
    result["round1_clipped_fraction"] = history[0]["clipped_fraction"]
    result["median_pre_clip_norm_all_rounds"] = (
        float(np.median(finite_norms)) if finite_norms else None
    )
    clipped = [h["clipped_fraction"] for h in history if h["clipped_fraction"] is not None]
    result["clipped_fraction_all_rounds"] = float(np.mean(clipped)) if clipped else None

    if record_predictions and all(np.all(np.isfinite(w)) for w in global_weights):
        evaluator.set_weights(global_weights)
        preds = np.argmax(evaluator.predict(test.x, batch_size=512, verbose=0), axis=1)
        result["predicted_class_counts"] = np.bincount(preds, minlength=10).tolist()
        result["true_class_counts"] = np.bincount(test.y, minlength=10).tolist()
    return result


# ---------------------------------------------------------------------------
# Experiment 2: what noise is actually applied
# ---------------------------------------------------------------------------


_ZERO_TEMPLATE: list | None = None


def _zero_template() -> list:
    """Zero-filled weight template, built once so measurement touches no RNG.

    ``build_model`` runs Keras initialisers, which draw from the host RNG stream.
    Caching the shapes keeps ``measure_applied_noise`` from perturbing anything a
    later ``simulate`` depends on.

    This is hygiene, not a fix for the reproducibility problem -- see the module
    docstring. DP runs are non-deterministic whether or not measurement happens,
    because the noise is drawn inside TFF's executor rather than in this process.
    """
    global _ZERO_TEMPLATE
    if _ZERO_TEMPLATE is None:
        _ZERO_TEMPLATE = [np.zeros_like(w) for w in build_model("small_cnn", seed=0).get_weights()]
    return _ZERO_TEMPLATE


def measure_applied_noise(noise_multiplier: float, clip: float, m: int, trials: int = 3) -> dict:
    """Feed the aggregator all-zero deltas so the output is pure noise, and measure it."""
    template = _zero_template()
    zeros = template

    per_coord, vector_norms = [], []
    for _ in range(trials):
        agg = _DiagnosticDPAggregator(noise_multiplier, clip, m)
        updates = [ClientUpdate(f"c{i}", zeros, 100) for i in range(m)]
        out = agg.aggregate(updates, zeros)
        flat = np.concatenate([np.asarray(w).ravel() for w in out])
        per_coord.append(float(np.std(flat)))
        vector_norms.append(float(np.linalg.norm(flat)))

    d = sum(int(np.prod(np.asarray(w).shape)) for w in template)
    return {
        "noise_multiplier": noise_multiplier,
        "clip": clip,
        "clients_per_round": m,
        "dimension": d,
        "predicted_sigma_sum_per_coord": clip * noise_multiplier,
        "predicted_sigma_mean_per_coord": clip * noise_multiplier / m,
        "measured_sigma_mean_per_coord": float(np.mean(per_coord)),
        "predicted_noise_vector_norm": clip * noise_multiplier * math.sqrt(d) / m,
        "measured_noise_vector_norm": float(np.mean(vector_norms)),
        "max_signal_norm": clip,
        "signal_to_noise": clip / (clip * noise_multiplier * math.sqrt(d) / m)
        if noise_multiplier > 0
        else float("inf"),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def exp_validate() -> dict:
    """Control: the no-DP simulation must reproduce the recorded 86.93% run."""
    return simulate(
        num_clients=10, clients_per_round=5, dp=False, label="validate/no-dp", record_norms=True
    )


def exp_noise_model() -> list[dict]:
    return [
        measure_applied_noise(2.0, 3.0, 5),
        measure_applied_noise(6.0, 3.0, 5),
        measure_applied_noise(2.0, 3.0, 200),
    ]


def exp_ablation() -> list[dict]:
    """(a) clip only, (b) noise only, (c) both."""
    common = {
        "num_clients": 10,
        "clients_per_round": 5,
        "dp": True,
        "record_predictions": True,
    }
    return [
        # (a) clipping at the production norm, zero noise.
        simulate(**common, noise_multiplier=0.0, l2_clip_norm=3.0, label="ablation-a/clip-only"),
        # (b) Noise at the production magnitude with clipping effectively disabled.
        #     TFF ties stddev to clip (stddev = clip * z), so isolating noise means
        #     raising clip until it cannot bind while holding clip*z constant:
        #     3.0 * 2.0 = 6.0 = 1e6 * 6e-6. NOT a valid DP setting -- a mechanism
        #     isolation only.
        simulate(**common, noise_multiplier=6e-6, l2_clip_norm=1e6, label="ablation-b/noise-only"),
        # (c) production configuration.
        simulate(**common, noise_multiplier=2.0, l2_clip_norm=3.0, label="ablation-c/both"),
    ]


def exp_noise_sweep() -> list[dict]:
    out = []
    for z in (0.0, 0.1, 0.3, 0.5, 1.0):
        out.append(
            simulate(
                num_clients=10,
                clients_per_round=5,
                dp=True,
                noise_multiplier=z,
                l2_clip_norm=3.0,
                record_predictions=True,
                label=f"noise-sweep/z={z}",
            )
        )
    return out


def exp_cohort_sweep() -> list[dict]:
    """Hold epsilon fixed while the cohort grows.

    Epsilon depends on (z, q, rounds). Keeping z=2.0, q=0.5 and rounds=20 fixed
    holds epsilon at 6.228 for every point, so num_clients = 2 * cohort.
    Total samples trained per round stays ~30,000 regardless of cohort size.
    """
    out = []
    for m in (5, 20, 50, 100, 200):
        out.append(
            simulate(
                num_clients=2 * m,
                clients_per_round=m,
                dp=True,
                noise_multiplier=2.0,
                l2_clip_norm=3.0,
                record_predictions=True,
                label=f"cohort-sweep/m={m}",
            )
        )
    return out


CLIP_VALUES = (3.0, 1.1, 0.5)
COHORTS = (5, 20, 50, 100, 200)


def exp_epsilon_gate() -> dict:
    """Prove epsilon does not depend on the clipping norm before sweeping it.

    ``compute_epsilon`` takes no clip argument, and the underlying
    ``GaussianDpEvent`` is parameterised by the noise *multiplier* z = sigma/S,
    which is already normalised by the sensitivity S. Scaling S scales sigma
    proportionally, so z -- and therefore epsilon -- is unchanged.

    If this ever returns unequal values the clip sweep is not an
    equal-privacy comparison and must not be run.
    """
    values = {
        clip: compute_epsilon(
            noise_multiplier=2.0, sampling_rate=0.5, rounds=DEFAULT_ROUNDS, delta=1e-5
        )
        for clip in CLIP_VALUES
    }
    identical = len(set(values.values())) == 1
    if not identical:
        raise AssertionError(f"epsilon varies with clip: {values}")
    return {
        "clip_values": list(CLIP_VALUES),
        "epsilon_per_clip": {str(k): repr(v) for k, v in values.items()},
        "all_identical": identical,
        "epsilon": next(iter(values.values())),
        "delta": 1e-5,
        "noise_multiplier": 2.0,
        "sampling_rate": 0.5,
        "rounds": DEFAULT_ROUNDS,
        "reason": (
            "compute_epsilon has no clip parameter; GaussianDpEvent is parameterised by "
            "z = sigma/S, already normalised by sensitivity S = clip"
        ),
    }


def exp_clip_sweep() -> dict:
    """clip x cohort grid at fixed epsilon.

    z = 2.0, q = 0.5 and rounds = 20 are held constant, so every cell carries the
    same epsilon = 6.228 at delta = 1e-5 (asserted by exp_epsilon_gate). N = 2m
    keeps q fixed, which also keeps total samples trained per round at ~30,000.
    """
    gate = exp_epsilon_gate()
    cells = []
    for clip in CLIP_VALUES:
        for m in COHORTS:
            noise = measure_applied_noise(2.0, clip, m, trials=2)
            run = simulate(
                num_clients=2 * m,
                clients_per_round=m,
                dp=True,
                noise_multiplier=2.0,
                l2_clip_norm=clip,
                record_predictions=True,
                label=f"clip={clip}/m={m}",
            )
            run["measured_noise_norm"] = noise["measured_noise_vector_norm"]
            run["predicted_noise_norm"] = noise["predicted_noise_vector_norm"]
            cells.append(run)
            LOGGER.info(
                "CELL clip=%s m=%s -> final=%.4f best=%.4f noise=%.3f "
                "r1_median_dw=%.3f r1_clipped=%.2f",
                clip,
                m,
                run["final_accuracy"],
                run["best_accuracy"],
                run["measured_noise_norm"],
                run["round1_median_pre_clip_norm"],
                run["round1_clipped_fraction"],
            )
    return {"gate": gate, "cells": cells}


def exp_repeatability() -> dict:
    """Run §6's exact configuration twice per cohort, with nothing in between.

    Settles whether DP non-determinism comes from the diagnostic instrumentation
    or from the stack. Same seed, same config, same process, no
    ``measure_applied_noise`` call anywhere -- so any difference between pass A
    and pass B cannot be attributed to measurement.
    """
    pairs = []
    for m in COHORTS:
        runs = [
            simulate(
                num_clients=2 * m,
                clients_per_round=m,
                dp=True,
                noise_multiplier=2.0,
                l2_clip_norm=3.0,
                label=f"repeat/m={m}/pass{p}",
            )
            for p in ("A", "B")
        ]
        a, b = runs
        pairs.append(
            {
                "clients_per_round": m,
                "final_accuracy": [a["final_accuracy"], b["final_accuracy"]],
                "best_accuracy": [a["best_accuracy"], b["best_accuracy"]],
                "round1_applied_delta_norm": [
                    a["history"][0]["applied_delta_norm"],
                    b["history"][0]["applied_delta_norm"],
                ],
                "identical": a["history"][0]["applied_delta_norm"]
                == b["history"][0]["applied_delta_norm"],
                "runs": runs,
            }
        )
        LOGGER.info(
            "REPEAT m=%s -> final A=%.4f B=%.4f | r1 applied A=%.6f B=%.6f | identical=%s",
            m,
            a["final_accuracy"],
            b["final_accuracy"],
            a["history"][0]["applied_delta_norm"],
            b["history"][0]["applied_delta_norm"],
            pairs[-1]["identical"],
        )
    return {
        "any_identical": any(p["identical"] for p in pairs),
        "note": (
            "No measure_applied_noise call occurs in this experiment. If identical is "
            "False anywhere, the instrumentation is not the source of non-determinism."
        ),
        "pairs": pairs,
    }


def exp_replication() -> dict:
    """Three seeds of the winning cell and its matched control.

    A: clip=0.5, m=50, DP on (z=2.0, so epsilon=6.228 at delta=1e-5).
    B: same cohort/population, DP off, clipping disabled (FedAvgAggregator).

    DP noise is unseedable from here (see module docstring), so the seed only
    fixes the data partition, init and cohort draws; arm A still varies beyond
    it. That is fine -- the point is the distribution, not determinism. Arm B is
    exactly seed-determined.
    """
    seeds = (42, 43, 44)
    arms: dict[str, list[dict]] = {"dp": [], "no_dp": []}
    for seed in seeds:
        arms["dp"].append(
            simulate(
                num_clients=100,
                clients_per_round=50,
                dp=True,
                noise_multiplier=2.0,
                l2_clip_norm=0.5,
                seed=seed,
                label=f"replicate/dp/seed={seed}",
            )
        )
        arms["no_dp"].append(
            simulate(
                num_clients=100,
                clients_per_round=50,
                dp=False,
                seed=seed,
                label=f"replicate/no-dp/seed={seed}",
            )
        )
        LOGGER.info(
            "REPLICATE seed=%s -> dp final=%.4f best=%.4f | no-dp final=%.4f best=%.4f",
            seed,
            arms["dp"][-1]["final_accuracy"],
            arms["dp"][-1]["best_accuracy"],
            arms["no_dp"][-1]["final_accuracy"],
            arms["no_dp"][-1]["best_accuracy"],
        )
    summary = {}
    for name, runs in arms.items():
        finals = [r["final_accuracy"] for r in runs]
        summary[name] = {
            "seeds": list(seeds),
            "final_per_seed": finals,
            "mean_final": float(np.mean(finals)),
            "range_final": float(max(finals) - min(finals)),
        }
    summary["mean_gap"] = summary["no_dp"]["mean_final"] - summary["dp"]["mean_final"]
    return {"summary": summary, "runs": arms}


def exp_cohort_baseline() -> list[dict]:
    """The same cohort grid with DP switched off, to isolate the noise term.

    ``exp_clip_sweep`` sets N = 2m to hold q = 0.5, so a larger cohort also means
    a *smaller* shard per client (60,000 / 2m examples). Accuracy therefore moves
    with m for two unrelated reasons: less noise (helps) and less data per client
    (hurts). Without this control the two are inseparable and any "how many
    clients would you need" extrapolation is unattributable.
    """
    out = []
    for m in COHORTS:
        out.append(
            simulate(
                num_clients=2 * m,
                clients_per_round=m,
                dp=False,
                label=f"baseline/m={m}",
            )
        )
        LOGGER.info(
            "BASELINE m=%s -> final=%.4f best=%.4f r1_median_dw=%.3f",
            m,
            out[-1]["final_accuracy"],
            out[-1]["best_accuracy"],
            out[-1]["round1_median_pre_clip_norm"],
        )
    return out


def exp_step_size() -> list[dict]:
    """Test whether the clip acts as a step size once clipping binds.

    At m=20 both clip=1.1 and clip=0.5 clip 100% of updates, so SNR is identical
    (0.0211 either way, = m/(z*sqrt(d))), yet clip=0.5 scores ~27 points higher.
    If that gap is a step-size effect, then scaling the clip=1.1 server step by
    0.5/1.1 = 0.4545 -- which matches the clip=0.5 step magnitude while leaving
    SNR and epsilon untouched -- should recover most of the gap.

    Confirms or refutes; it does not tune. server_lr exists only in this file.
    """
    out = []
    for lr, clip in ((1.0, 1.1), (0.5 / 1.1, 1.1), (1.0, 0.5)):
        out.append(
            simulate(
                num_clients=40,
                clients_per_round=20,
                dp=True,
                noise_multiplier=2.0,
                l2_clip_norm=clip,
                server_lr=lr,
                label=f"step-size/clip={clip}/server_lr={lr:.4f}",
            )
        )
        LOGGER.info(
            "STEPSIZE clip=%s server_lr=%.4f -> final=%.4f best=%.4f",
            clip,
            lr,
            out[-1]["final_accuracy"],
            out[-1]["best_accuracy"],
        )
    return out


EXPERIMENTS = {
    "validate": exp_validate,
    "noise_model": exp_noise_model,
    "ablation": exp_ablation,
    "noise_sweep": exp_noise_sweep,
    "cohort_sweep": exp_cohort_sweep,
    "epsilon_gate": exp_epsilon_gate,
    "clip_sweep": exp_clip_sweep,
    "cohort_baseline": exp_cohort_baseline,
    "step_size": exp_step_size,
    "repeatability": exp_repeatability,
    "replication": exp_replication,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="all", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    results: dict[str, object] = {}
    for name in names:
        LOGGER.info("===== experiment: %s =====", name)
        results[name] = EXPERIMENTS[name]()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
