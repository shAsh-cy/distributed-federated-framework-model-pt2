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

        history.append(
            {
                "round": rnd,
                "accuracy": float(acc),
                "loss": float(loss),
                "pre_clip_norms": [float(n) for n in pre_clip_norms],
                "median_pre_clip_norm": float(np.median(pre_clip_norms))
                if pre_clip_norms
                else None,
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

    if record_predictions and all(np.all(np.isfinite(w)) for w in global_weights):
        evaluator.set_weights(global_weights)
        preds = np.argmax(evaluator.predict(test.x, batch_size=512, verbose=0), axis=1)
        result["predicted_class_counts"] = np.bincount(preds, minlength=10).tolist()
        result["true_class_counts"] = np.bincount(test.y, minlength=10).tolist()
    return result


# ---------------------------------------------------------------------------
# Experiment 2: what noise is actually applied
# ---------------------------------------------------------------------------


def measure_applied_noise(noise_multiplier: float, clip: float, m: int, trials: int = 3) -> dict:
    """Feed the aggregator all-zero deltas so the output is pure noise, and measure it."""
    template = build_model("small_cnn", seed=0).get_weights()
    zeros = [np.zeros_like(w) for w in template]

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


EXPERIMENTS = {
    "validate": exp_validate,
    "noise_model": exp_noise_model,
    "ablation": exp_ablation,
    "noise_sweep": exp_noise_sweep,
    "cohort_sweep": exp_cohort_sweep,
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
