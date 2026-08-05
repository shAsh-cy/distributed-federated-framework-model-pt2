"""The final batch: clip re-bracket, the FEMNIST DP arm, adaptive-vs-fixed.

Four phases, run unattended in one process, each writing its own JSON to
docs/ as it completes. A phase whose JSON already exists is skipped, so a
crashed or interrupted batch resumes by re-running the script.

  A  docs/_final_batch_a.json  FEMNIST clip re-bracket at the working budget
                               (E=10, m=50, R=20, 1 seed, S in {0.5,1,2,4},
                               z calibrated to eps=6.228 at q=0.05).
  B  docs/_final_batch_b.json  The FEMNIST DP arm: E=10, m=200, R=20,
                               3 seeds, clip chosen by A, z calibrated to
                               eps=6.228 at q=0.2. Compared in-file against
                               the recorded no-DP control at this budget.
  C  docs/_final_batch_c.json  Adaptive clipping, same budget as B, 3 seeds.
                               B is the fixed arm; it is not re-run.
  D  docs/_final_batch_d.json  Fashion-MNIST at its working budget
                               (m=50 of N=100, z=2.0): fixed S=0.5 vs
                               adaptive, 3 seeds each arm.
  E  docs/_final_batch_e.json  THE COLD-START ARM. FEMNIST, E=10, m=50,
                               R=100, 1 seed, adaptive from TFF's naive 0.1
                               default, plus a matched fixed arm at S=2.0.
                               R=100 on purpose: climbing 0.1 -> ~2.0 at
                               geometric rate 0.2 is ~15 rounds at full
                               speed, so a 20-round budget would measure
                               warm-up only. Round 20 read off the curve
                               answers the short budget; round 100 answers
                               whether adaptation FINDS the clip at all.
  F  docs/_final_batch_f.json  The Fashion quantile: adaptive at target
                               quantile {0.2, 0.35}, 3 seeds each, against
                               the recorded q=0.5 and fixed arms from D.

Phases A-D warm-started their adaptive arms from the fixed arm's clip: with
only 20 rounds and a geometric adaptation rate of 0.2, a cold start from 0.1
spends most of the run climbing toward the norm scale and the comparison
measures warm-up, not adaptation. Phase E is the deliberate exception -- the
cold start IS the experiment, testing whether adaptation finds a clip it was
never told. Total epsilon for an adaptive run equals compute_epsilon at the
nominal z (sigma-additivity; see docs/adaptive_clipping.md), and the
value/count split is recorded via adaptive_noise_breakdown in each JSON.

Seeds fix the population draw, model init and cohort sampling only; TFF draws
DP noise in its executor, unseedably, by long-standing repo rule.

Run detached (do not pipe stdout into a command that waits for EOF):

    python scripts/final_batch.py >> docs/_final_batch.log 2>&1

Heavy imports (TF/TFF via the harness modules) happen inside the phase
functions so the selection logic stays importable for unit tests.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LOGGER = logging.getLogger("final_batch")

DOCS = ROOT / "docs"
TARGET_EPSILON = 6.228
DELTA = 1e-5
ROUNDS = 20
WRITERS = 1000
SEEDS = (42, 43, 44)
BRACKET_CLIPS = (0.5, 1.0, 2.0, 4.0)
BRACKET_M = 50
DP_ARM_M = 200
LOCAL_EPOCHS = 10
FASHION_N = 100
FASHION_M = 50
FASHION_Z = 2.0
FASHION_CLIP = 0.5
ADAPTIVE_QUANTILE = 0.5
ADAPTIVE_LR = 0.2
FLAT_THRESHOLD = 0.02
COLD_START_CLIP = 0.1  # TFF's default initial estimate
COLD_START_ROUNDS = 100
COLD_START_M = 50
COLD_START_FIXED_CLIP = 2.0  # phase A's bracket answer
COLD_TARGET_NEIGHBOURHOOD = 1.8  # "reached ~2.0" = first round at or above this
FASHION_QUANTILES = (0.2, 0.35)


def pick_clip(cells: list[dict], flat_threshold: float = FLAT_THRESHOLD) -> dict:
    """Choose the bracket clip: where clipping BEGINS to bind, not one that
    binds everywhere.

    Accuracy gates, clipped fraction picks: every cell within
    ``flat_threshold`` of the best final accuracy is a contender (with one
    DP seed, small accuracy differences are noise), and among contenders the
    cell whose all-round clipped fraction is closest to 0.5 wins — the knee
    between binds-always (clip acting as a server learning rate) and
    binds-never (pure noise floor). Ties prefer the LARGER clip, the
    begins-to-bind side. If all four cells are contenders the accuracy axis
    was flat, and the JSON says so.
    """
    if not cells:
        raise ValueError("pick_clip needs at least one cell")
    accs = [c["final_accuracy"] for c in cells]
    spread = max(accs) - min(accs)
    best = max(accs)
    contenders = [c for c in cells if best - c["final_accuracy"] < flat_threshold]
    chosen = min(
        contenders,
        key=lambda c: (abs(c["clipped_fraction_all_rounds"] - 0.5), -c["l2_clip_norm"]),
    )
    flat = spread < flat_threshold
    if flat:
        reason = (
            f"accuracy flat across all cells (spread {spread:.4f} < {flat_threshold}); "
            f"picked by clipped fraction nearest 0.5, ties to the larger clip"
        )
    else:
        reason = (
            f"accuracy separates cells (spread {spread:.4f}); among cells within "
            f"{flat_threshold} of the best, picked clipped fraction nearest 0.5"
        )
    return {
        "chosen_clip": chosen["l2_clip_norm"],
        "accuracy_flat": flat,
        "accuracy_spread": float(spread),
        "reason": reason,
        "per_cell": [
            {
                "l2_clip_norm": c["l2_clip_norm"],
                "final_accuracy": c["final_accuracy"],
                "clipped_fraction_all_rounds": c["clipped_fraction_all_rounds"],
                "contender": c in contenders,
            }
            for c in cells
        ],
    }


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    LOGGER.info("wrote %s", path.relative_to(ROOT))


def _summary(runs: list[dict]) -> dict:
    import numpy as np

    finals = [r["final_accuracy"] for r in runs]
    return {
        "seeds": [r["seed"] for r in runs],
        "final_per_seed": finals,
        "mean_final": float(np.mean(finals)),
        "range_final": float(max(finals) - min(finals)),
        "mean_best": float(np.mean([r["best_accuracy"] for r in runs])),
    }


def _load_femnist_population():
    import femnist_experiments as fx

    return fx.load_femnist(num_clients=WRITERS, seed=SEEDS[0])


def phase_a(population) -> dict:
    """Re-bracket the clip at the working budget, one seed, four cells."""
    import femnist_experiments as fx

    from fl.aggregation import calibrate_noise_multiplier, compute_epsilon

    train, test, shards = population
    q = BRACKET_M / WRITERS
    z = calibrate_noise_multiplier(TARGET_EPSILON, q, ROUNDS, DELTA)
    achieved = compute_epsilon(z, q, ROUNDS, DELTA)
    LOGGER.info("PHASE A: q=%.3f calibrated z=%.4f (achieved eps=%.4f)", q, z, achieved)
    cells = []
    for clip in BRACKET_CLIPS:
        run = fx.simulate(
            train=train,
            test=test,
            shards=shards,
            clients_per_round=BRACKET_M,
            rounds=ROUNDS,
            dp=True,
            noise_multiplier=z,
            l2_clip_norm=clip,
            seed=SEEDS[0],
            local_epochs=LOCAL_EPOCHS,
            label=f"bracket/S={clip}/seed={SEEDS[0]}",
        )
        cells.append(run)
        LOGGER.info(
            "PHASE A CELL S=%.1f: final=%.4f clipped=%.2f median||dw||=%.2f",
            clip,
            run["final_accuracy"],
            run["clipped_fraction_all_rounds"],
            run["median_pre_clip_norm_all_rounds"],
        )
    selection = pick_clip(cells)
    LOGGER.info("PHASE A SELECTION: S=%s (%s)", selection["chosen_clip"], selection["reason"])
    return {
        "phase": "A",
        "writers": WRITERS,
        "m": BRACKET_M,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "target_epsilon": TARGET_EPSILON,
        "delta": DELTA,
        "calibrated_z": z,
        "achieved_epsilon": achieved,
        "selection": selection,
        "cells": cells,
    }


def phase_b(population, chosen_clip: float) -> dict:
    """The DP arm at the working budget, three seeds, clip from phase A."""
    import femnist_experiments as fx

    from fl.aggregation import calibrate_noise_multiplier, compute_epsilon

    train, test, shards = population
    q = DP_ARM_M / WRITERS
    z = calibrate_noise_multiplier(TARGET_EPSILON, q, ROUNDS, DELTA)
    achieved = compute_epsilon(z, q, ROUNDS, DELTA)
    LOGGER.info("PHASE B: q=%.3f calibrated z=%.4f (achieved eps=%.4f)", q, z, achieved)
    runs = [
        fx.simulate(
            train=train,
            test=test,
            shards=shards,
            clients_per_round=DP_ARM_M,
            rounds=ROUNDS,
            dp=True,
            noise_multiplier=z,
            l2_clip_norm=chosen_clip,
            seed=seed,
            local_epochs=LOCAL_EPOCHS,
            label=f"dp-arm/S={chosen_clip}/seed={seed}",
        )
        for seed in SEEDS
    ]
    summary = _summary(runs)
    control = _recorded_nodp_control()
    dp_cost = control["mean_final"] - summary["mean_final"] if control else None
    LOGGER.info(
        "PHASE B DONE: dp mean=%.4f range=%.4f vs no-DP control %.4f -> cost %.4f",
        summary["mean_final"],
        summary["range_final"],
        control["mean_final"] if control else float("nan"),
        dp_cost if dp_cost is not None else float("nan"),
    )
    return {
        "phase": "B",
        "writers": WRITERS,
        "m": DP_ARM_M,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "l2_clip_norm": chosen_clip,
        "target_epsilon": TARGET_EPSILON,
        "delta": DELTA,
        "calibrated_z": z,
        "achieved_epsilon": achieved,
        "summary": summary,
        "nodp_control": control,
        "dp_cost_mean": dp_cost,
        "runs": runs,
    }


def _recorded_nodp_control() -> dict | None:
    """The matched no-DP control: E=10 cell of the committed budget sweep."""
    path = DOCS / "_femnist_budget_e.json"
    if not path.exists():
        LOGGER.warning("no committed control at %s; comparison omitted", path)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for cell in data["budget"]["cells"]:
        if cell["local_epochs"] == LOCAL_EPOCHS:
            s = cell["summary"]
            return {
                "source": str(path.name),
                "mean_final": s["mean_final"],
                "range_final": s["range_final"],
                "final_per_seed": s["final_per_seed"],
            }
    LOGGER.warning("no E=%d cell in %s; comparison omitted", LOCAL_EPOCHS, path)
    return None


def _adaptive_accounting(z: float, m: int, q: float) -> dict:
    """The budget split the quantile estimate consumes, priced both ways."""
    from fl.aggregation import adaptive_noise_breakdown, compute_epsilon

    breakdown = adaptive_noise_breakdown(z, m)
    return {
        "nominal_noise_multiplier": breakdown["nominal_noise_multiplier"],
        "value_noise_multiplier": breakdown["value_noise_multiplier"],
        "count_noise_multiplier": breakdown["count_noise_multiplier"],
        "clipped_count_stddev": breakdown["clipped_count_stddev"],
        "total_epsilon": compute_epsilon(z, q, ROUNDS, DELTA),
        "epsilon_value_component": compute_epsilon(
            breakdown["value_noise_multiplier"], q, ROUNDS, DELTA
        ),
        "epsilon_count_component": compute_epsilon(
            breakdown["count_noise_multiplier"], q, ROUNDS, DELTA
        ),
        "note": (
            "dp_accounting CAN price the two Gaussian components separately "
            "(they are distinct events in TFF's ComposedDpEvent state), but "
            "the tight total comes from sigma-additivity: value and count "
            "noise compose back to exactly the nominal z, so total epsilon "
            "equals the fixed arm's. Naively summing the component epsilons "
            "over-counts relative to that total."
        ),
    }


def phase_c(population, chosen_clip: float, fixed_arm: dict) -> dict:
    """Adaptive clipping at the phase-B budget; B itself is the fixed arm."""
    import femnist_experiments as fx

    from fl.aggregation import AdaptiveDPFedAvgAggregator

    train, test, shards = population
    q = DP_ARM_M / WRITERS
    z = fixed_arm["calibrated_z"]
    runs = []
    for seed in SEEDS:
        aggregator = AdaptiveDPFedAvgAggregator(
            noise_multiplier=z,
            initial_l2_clip_norm=chosen_clip,
            clients_per_round=DP_ARM_M,
            target_quantile=ADAPTIVE_QUANTILE,
            learning_rate=ADAPTIVE_LR,
        )
        runs.append(
            fx.simulate(
                train=train,
                test=test,
                shards=shards,
                clients_per_round=DP_ARM_M,
                rounds=ROUNDS,
                dp=True,
                noise_multiplier=z,
                l2_clip_norm=chosen_clip,
                aggregator=aggregator,
                seed=seed,
                local_epochs=LOCAL_EPOCHS,
                label=f"adaptive/q={ADAPTIVE_QUANTILE}/seed={seed}",
            )
        )
    summary = _summary(runs)
    LOGGER.info(
        "PHASE C DONE: adaptive mean=%.4f range=%.4f vs fixed %.4f range=%.4f",
        summary["mean_final"],
        summary["range_final"],
        fixed_arm["summary"]["mean_final"],
        fixed_arm["summary"]["range_final"],
    )
    return {
        "phase": "C",
        "writers": WRITERS,
        "m": DP_ARM_M,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "initial_l2_clip_norm": chosen_clip,
        "target_quantile": ADAPTIVE_QUANTILE,
        "adaptation_learning_rate": ADAPTIVE_LR,
        "calibrated_z": z,
        "epsilon_accounting": _adaptive_accounting(z, DP_ARM_M, q),
        "summary": summary,
        "fixed_arm_summary": fixed_arm["summary"],
        "runs": runs,
    }


def phase_d() -> dict:
    """Fashion-MNIST at its working budget: fixed S=0.5 vs adaptive, 3 seeds each."""
    import diagnose_dp as dd

    from fl.aggregation import AdaptiveDPFedAvgAggregator, compute_epsilon

    fixed, adaptive = [], []
    for seed in SEEDS:
        fixed.append(
            dd.simulate(
                num_clients=FASHION_N,
                clients_per_round=FASHION_M,
                dp=True,
                noise_multiplier=FASHION_Z,
                l2_clip_norm=FASHION_CLIP,
                seed=seed,
                label=f"fashion/fixed/S={FASHION_CLIP}/seed={seed}",
            )
        )
        aggregator = AdaptiveDPFedAvgAggregator(
            noise_multiplier=FASHION_Z,
            initial_l2_clip_norm=FASHION_CLIP,
            clients_per_round=FASHION_M,
            target_quantile=ADAPTIVE_QUANTILE,
            learning_rate=ADAPTIVE_LR,
        )
        adaptive.append(
            dd.simulate(
                num_clients=FASHION_N,
                clients_per_round=FASHION_M,
                dp=True,
                noise_multiplier=FASHION_Z,
                l2_clip_norm=FASHION_CLIP,
                aggregator=aggregator,
                seed=seed,
                label=f"fashion/adaptive/q={ADAPTIVE_QUANTILE}/seed={seed}",
            )
        )
    fixed_summary, adaptive_summary = _summary(fixed), _summary(adaptive)
    LOGGER.info(
        "PHASE D DONE: fixed mean=%.4f range=%.4f | adaptive mean=%.4f range=%.4f",
        fixed_summary["mean_final"],
        fixed_summary["range_final"],
        adaptive_summary["mean_final"],
        adaptive_summary["range_final"],
    )
    return {
        "phase": "D",
        "num_clients": FASHION_N,
        "m": FASHION_M,
        "rounds": ROUNDS,
        "noise_multiplier": FASHION_Z,
        "fixed_clip": FASHION_CLIP,
        "initial_l2_clip_norm": FASHION_CLIP,
        "target_quantile": ADAPTIVE_QUANTILE,
        "adaptation_learning_rate": ADAPTIVE_LR,
        "epsilon": compute_epsilon(FASHION_Z, FASHION_M / FASHION_N, ROUNDS, DELTA),
        "epsilon_accounting": _adaptive_accounting(FASHION_Z, FASHION_M, FASHION_M / FASHION_N),
        "fixed_summary": fixed_summary,
        "adaptive_summary": adaptive_summary,
        "fixed_runs": fixed,
        "adaptive_runs": adaptive,
    }


def phase_e(population) -> dict:
    """The cold-start arm: does adaptation FIND a clip it was never told?

    Phase C warm-started from the bracket answer and measured whether the
    estimator HOLDS a tuned clip. Here the adaptive arm starts from TFF's
    naive 0.1 default and runs 100 rounds beside a fixed arm at the bracket
    answer S=2.0, both at eps=6.228 calibrated for the full 100 rounds.
    """
    import femnist_experiments as fx

    from fl.aggregation import (
        AdaptiveDPFedAvgAggregator,
        calibrate_noise_multiplier,
        compute_epsilon,
    )

    train, test, shards = population
    q = COLD_START_M / WRITERS
    z = calibrate_noise_multiplier(TARGET_EPSILON, q, COLD_START_ROUNDS, DELTA)
    achieved = compute_epsilon(z, q, COLD_START_ROUNDS, DELTA)
    LOGGER.info(
        "PHASE E: q=%.3f calibrated z=%.4f for R=%d (eps=%.4f)", q, z, COLD_START_ROUNDS, achieved
    )

    seed = SEEDS[0]
    common = {
        "train": train,
        "test": test,
        "shards": shards,
        "clients_per_round": COLD_START_M,
        "rounds": COLD_START_ROUNDS,
        "dp": True,
        "noise_multiplier": z,
        "seed": seed,
        "local_epochs": LOCAL_EPOCHS,
    }
    aggregator = AdaptiveDPFedAvgAggregator(
        noise_multiplier=z,
        initial_l2_clip_norm=COLD_START_CLIP,
        clients_per_round=COLD_START_M,
        target_quantile=ADAPTIVE_QUANTILE,
        learning_rate=ADAPTIVE_LR,
    )
    cold = fx.simulate(
        l2_clip_norm=COLD_START_CLIP,
        aggregator=aggregator,
        label=f"coldstart/adaptive/init={COLD_START_CLIP}/seed={seed}",
        **common,
    )
    fixed = fx.simulate(
        l2_clip_norm=COLD_START_FIXED_CLIP,
        label=f"coldstart/fixed/S={COLD_START_FIXED_CLIP}/seed={seed}",
        **common,
    )

    clips = [h["adapted_clip"] for h in cold["history"]]
    reached = next(
        (h["round"] for h in cold["history"] if h["adapted_clip"] >= COLD_TARGET_NEIGHBOURHOOD),
        None,
    )
    acc_at = lambda run, r: run["history"][r - 1]["accuracy"]  # noqa: E731
    summary = {
        "seed": seed,
        "rounds_to_reach_neighbourhood": reached,
        "neighbourhood_threshold": COLD_TARGET_NEIGHBOURHOOD,
        "adaptive_acc_round_20": acc_at(cold, 20),
        "fixed_acc_round_20": acc_at(fixed, 20),
        "adaptive_acc_round_100": acc_at(cold, 100),
        "fixed_acc_round_100": acc_at(fixed, 100),
        "adaptive_final_clip": clips[-1],
        "clip_trajectory": clips,
    }
    LOGGER.info(
        "PHASE E DONE: clip reaches %.1f at round %s | R20 adaptive=%.4f fixed=%.4f | "
        "R100 adaptive=%.4f fixed=%.4f",
        COLD_TARGET_NEIGHBOURHOOD,
        reached,
        summary["adaptive_acc_round_20"],
        summary["fixed_acc_round_20"],
        summary["adaptive_acc_round_100"],
        summary["fixed_acc_round_100"],
    )
    return {
        "phase": "E",
        "writers": WRITERS,
        "m": COLD_START_M,
        "rounds": COLD_START_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "initial_l2_clip_norm": COLD_START_CLIP,
        "fixed_clip": COLD_START_FIXED_CLIP,
        "target_quantile": ADAPTIVE_QUANTILE,
        "adaptation_learning_rate": ADAPTIVE_LR,
        "calibrated_z": z,
        "achieved_epsilon": achieved,
        "epsilon_accounting": _adaptive_accounting(z, COLD_START_M, q),
        "summary": summary,
        "adaptive_run": cold,
        "fixed_run": fixed,
    }


def phase_f() -> dict:
    """The Fashion quantile: is q=0.5 the problem, or is adaptation?

    Phase D's adaptive arm tracked the median away from a binding optimum.
    Lower targets aim the estimator at the binding regime instead. Both
    reference arms (fixed S=0.5 and adaptive q=0.5) are read from phase D's
    JSON, not re-run.
    """
    import diagnose_dp as dd

    from fl.aggregation import AdaptiveDPFedAvgAggregator

    d = json.loads((DOCS / "_final_batch_d.json").read_text(encoding="utf-8"))
    arms = {}
    for quantile in FASHION_QUANTILES:
        runs = []
        for seed in SEEDS:
            aggregator = AdaptiveDPFedAvgAggregator(
                noise_multiplier=FASHION_Z,
                initial_l2_clip_norm=FASHION_CLIP,
                clients_per_round=FASHION_M,
                target_quantile=quantile,
                learning_rate=ADAPTIVE_LR,
            )
            runs.append(
                dd.simulate(
                    num_clients=FASHION_N,
                    clients_per_round=FASHION_M,
                    dp=True,
                    noise_multiplier=FASHION_Z,
                    l2_clip_norm=FASHION_CLIP,
                    aggregator=aggregator,
                    seed=seed,
                    label=f"fashion/adaptive/q={quantile}/seed={seed}",
                )
            )
        summary = _summary(runs)
        arms[str(quantile)] = {"summary": summary, "runs": runs}
        LOGGER.info(
            "PHASE F q=%.2f DONE: mean=%.4f range=%.4f (fixed %.4f, q=0.5 %.4f)",
            quantile,
            summary["mean_final"],
            summary["range_final"],
            d["fixed_summary"]["mean_final"],
            d["adaptive_summary"]["mean_final"],
        )
    return {
        "phase": "F",
        "num_clients": FASHION_N,
        "m": FASHION_M,
        "rounds": ROUNDS,
        "noise_multiplier": FASHION_Z,
        "initial_l2_clip_norm": FASHION_CLIP,
        "adaptation_learning_rate": ADAPTIVE_LR,
        "quantiles": list(FASHION_QUANTILES),
        "reference_fixed_summary": d["fixed_summary"],
        "reference_q05_summary": d["adaptive_summary"],
        "arms": arms,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    out = {
        "A": DOCS / "_final_batch_a.json",
        "B": DOCS / "_final_batch_b.json",
        "C": DOCS / "_final_batch_c.json",
        "D": DOCS / "_final_batch_d.json",
        "E": DOCS / "_final_batch_e.json",
        "F": DOCS / "_final_batch_f.json",
    }

    def done(phase: str) -> dict | None:
        if out[phase].exists():
            LOGGER.info("PHASE %s already complete (%s exists), skipping", phase, out[phase].name)
            return json.loads(out[phase].read_text(encoding="utf-8"))
        return None

    population = None
    if not all(out[p].exists() for p in ("A", "B", "C", "E")):
        LOGGER.info("loading FEMNIST population: %d writers, seed %d", WRITERS, SEEDS[0])
        population = _load_femnist_population()

    a = done("A")
    if a is None:
        a = phase_a(population)
        _write(out["A"], a)

    chosen_clip = a["selection"]["chosen_clip"]

    b = done("B")
    if b is None:
        b = phase_b(population, chosen_clip)
        _write(out["B"], b)

    c = done("C")
    if c is None:
        c = phase_c(population, chosen_clip, b)
        _write(out["C"], c)

    d = done("D")
    if d is None:
        d = phase_d()
        _write(out["D"], d)

    e = done("E")
    if e is None:
        e = phase_e(population)
        _write(out["E"], e)

    f = done("F")
    if f is None:
        f = phase_f()
        _write(out["F"], f)

    LOGGER.info(
        "BATCH COMPLETE: A chose S=%s | B dp mean=%.4f | C adaptive mean=%.4f | "
        "D fixed=%.4f adaptive=%.4f | E cold R100=%.4f vs fixed=%.4f | "
        "F q=%s means=%s",
        chosen_clip,
        b["summary"]["mean_final"],
        c["summary"]["mean_final"],
        d["fixed_summary"]["mean_final"],
        d["adaptive_summary"]["mean_final"],
        e["summary"]["adaptive_acc_round_100"],
        e["summary"]["fixed_acc_round_100"],
        list(FASHION_QUANTILES),
        [round(f["arms"][str(q)]["summary"]["mean_final"], 4) for q in FASHION_QUANTILES],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
