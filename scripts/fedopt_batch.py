"""FedOpt and FedProx, measured. Four phases, unattended, resumable per phase.

  A  docs/_fedopt_batch_a.json  Fashion-MNIST server-LR sweep, no DP, 1 seed:
                                FedAdam and FedYogi over server_lr
                                {1e-3, 1e-2, 1e-1, 1.0}; FedAvgM over server
                                momentum {0.9, 0.99} at server_lr 1.0; plus
                                the FedAvg reference at the same budget
                                (N=100, m=50, R=20, client lr 0.01/m 0.9).
  B  docs/_fedopt_batch_b.json  Best-of-each on Fashion, 3 seeds per arm,
                                against FedAvg at the same budget, 3 seeds.
  C  docs/_fedopt_batch_c.json  Best-of-each on FEMNIST at the working budget
                                (E=10, m=200 of 1,000 writers, R=20), 3 seeds,
                                against the recorded no-DP control
                                (docs/_femnist_budget_e.json, E=10 cell).
                                Server LRs TRANSFERRED from the Fashion sweep,
                                not re-tuned -- the honest tuning cost, stated
                                in the JSON.
  D  docs/_fedopt_batch_d.json  FedProx on FEMNIST at the working budget,
                                mu in {0.001, 0.01, 0.1}, 1 seed, against the
                                same recorded control. The repo previously
                                omitted FedProx a priori; this measures it.

A phase whose JSON already exists is skipped, so a crashed or interrupted
batch resumes by re-running the script. No DP anywhere in this batch: every
run is deterministic in configuration and seed up to floating-point
summation order.

Run detached (do not pipe stdout into a command that waits for EOF):

    python scripts/fedopt_batch.py >> docs/_fedopt_batch.log 2>&1

Heavy imports (TF via the harness modules) stay inside the phase functions
so the selection logic remains importable for unit tests.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LOGGER = logging.getLogger("fedopt_batch")

DOCS = ROOT / "docs"

FASHION_N = 100
FASHION_M = 50
ROUNDS = 20
SEEDS = (42, 43, 44)
SERVER_LRS = (1e-3, 1e-2, 1e-1, 1.0)
FEDAVGM_MOMENTA = (0.9, 0.99)
FEDOPT_FAMILIES = ("fedavgm", "fedadam", "fedyogi")
FEMNIST_WRITERS = 1000
FEMNIST_M = 200
FEMNIST_LOCAL_EPOCHS = 10
FEDPROX_MUS = (0.001, 0.01, 0.1)
#: A grid cell within this of its family's best counts as "near-best" --
#: the fraction of near-best cells is the tuning-ease measure the JSON
#: records (a family where most of the grid works is easy to tune; a
#: family with one good point is a tuning cliff).
NEAR_BEST = 0.02


def select_best(cells: list[dict], near: float = NEAR_BEST) -> dict:
    """Per optimizer family: the best grid cell, and what the grid says
    about tuning cost.

    Best is highest final accuracy; ties go to the SMALLER server learning
    rate (the more conservative step -- with one seed, equal accuracies do
    not license the riskier configuration). Each family also reports its
    accuracy spread across the grid and how many cells land within ``near``
    of its best, which is the evidence for or against the Reddi et al.
    claim that the adaptive methods are easier to tune than FedAvgM.
    """
    if not cells:
        raise ValueError("select_best needs at least one grid cell")
    families: dict[str, list[dict]] = {}
    for cell in cells:
        families.setdefault(cell["server_optimizer"]["name"], []).append(cell)

    out: dict[str, dict] = {}
    for name, group in families.items():
        ranked = sorted(
            group,
            key=lambda c: (
                -c["final_accuracy"],
                c["server_optimizer"]["server_lr"],
                c["server_optimizer"].get("momentum", 0.0),
            ),
        )
        best = ranked[0]
        accs = [c["final_accuracy"] for c in group]
        near_best = [c for c in group if best["final_accuracy"] - c["final_accuracy"] <= near]
        out[name] = {
            "best": {
                "server_optimizer": best["server_optimizer"],
                "final_accuracy": best["final_accuracy"],
                "label": best.get("label", ""),
            },
            "grid_size": len(group),
            "accuracy_spread": float(max(accs) - min(accs)),
            "near_best_count": len(near_best),
            "near_best_fraction": len(near_best) / len(group),
        }
    return {"near_threshold": near, "families": out}


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


def _fedopt_aggregator(name: str, server_lr: float, momentum: float = 0.9) -> object:
    from fl.aggregation import FedOptAggregator
    from fl.server_optimizer import make_server_optimizer

    return FedOptAggregator(make_server_optimizer(name, learning_rate=server_lr, momentum=momentum))


def _fashion_run(seed: int, label: str, aggregator: object | None = None) -> dict:
    import diagnose_dp as dd

    return dd.simulate(
        num_clients=FASHION_N,
        clients_per_round=FASHION_M,
        rounds=ROUNDS,
        dp=False,
        seed=seed,
        aggregator=aggregator,
        label=label,
    )


def phase_a() -> dict:
    """The server-LR grid on Fashion, one seed, plus the FedAvg reference."""
    reference = _fashion_run(SEEDS[0], "fedavg/reference")
    reference["server_optimizer"] = {"name": "fedavg", "server_lr": 1.0}
    LOGGER.info("PHASE A REFERENCE fedavg: final=%.4f", reference["final_accuracy"])

    cells: list[dict] = []
    for name in ("fedadam", "fedyogi"):
        for slr in SERVER_LRS:
            run = _fashion_run(SEEDS[0], f"{name}/slr={slr}", _fedopt_aggregator(name, slr))
            run["server_optimizer"] = {"name": name, "server_lr": slr}
            cells.append(run)
            LOGGER.info("PHASE A CELL %s slr=%g: final=%.4f", name, slr, run["final_accuracy"])
    for beta in FEDAVGM_MOMENTA:
        run = _fashion_run(
            SEEDS[0],
            f"fedavgm/beta={beta}",
            _fedopt_aggregator("fedavgm", 1.0, momentum=beta),
        )
        run["server_optimizer"] = {"name": "fedavgm", "server_lr": 1.0, "momentum": beta}
        cells.append(run)
        LOGGER.info("PHASE A CELL fedavgm beta=%g: final=%.4f", beta, run["final_accuracy"])

    selection = select_best(cells)
    for name, family in selection["families"].items():
        LOGGER.info(
            "PHASE A BEST %s: %s final=%.4f spread=%.4f near-best %d/%d",
            name,
            family["best"]["server_optimizer"],
            family["best"]["final_accuracy"],
            family["accuracy_spread"],
            family["near_best_count"],
            family["grid_size"],
        )
    return {
        "phase": "A",
        "dataset": "fashion_mnist",
        "num_clients": FASHION_N,
        "clients_per_round": FASHION_M,
        "rounds": ROUNDS,
        "seed": SEEDS[0],
        "client_lr": 0.01,
        "client_momentum": 0.9,
        "server_lr_grid": list(SERVER_LRS),
        "fedavgm_momentum_grid": list(FEDAVGM_MOMENTA),
        "fedavg_reference": reference,
        "selection": selection,
        "cells": cells,
    }


def _best_config(selection: dict, name: str) -> dict:
    return selection["families"][name]["best"]["server_optimizer"]


def phase_b(selection: dict) -> dict:
    """Best-of-each on Fashion, three seeds, against FedAvg at the same budget."""
    arms: dict[str, dict] = {}
    for name in FEDOPT_FAMILIES:
        best = _best_config(selection, name)
        runs = []
        for seed in SEEDS:
            run = _fashion_run(
                seed,
                f"{name}/best/seed={seed}",
                _fedopt_aggregator(name, best["server_lr"], momentum=best.get("momentum", 0.9)),
            )
            run["server_optimizer"] = dict(best)
            runs.append(run)
        arms[name] = {"config": best, "runs": runs, "summary": _summary(runs)}
        LOGGER.info(
            "PHASE B ARM %s: mean=%.4f range=%.4f",
            name,
            arms[name]["summary"]["mean_final"],
            arms[name]["summary"]["range_final"],
        )

    fedavg_runs = [_fashion_run(seed, f"fedavg/seed={seed}") for seed in SEEDS]
    arms["fedavg"] = {
        "config": {"name": "fedavg", "server_lr": 1.0},
        "runs": fedavg_runs,
        "summary": _summary(fedavg_runs),
    }
    fedavg_mean = arms["fedavg"]["summary"]["mean_final"]
    deltas = {name: arms[name]["summary"]["mean_final"] - fedavg_mean for name in FEDOPT_FAMILIES}
    LOGGER.info("PHASE B DONE: fedavg mean=%.4f, deltas vs fedavg: %s", fedavg_mean, deltas)
    return {
        "phase": "B",
        "dataset": "fashion_mnist",
        "num_clients": FASHION_N,
        "clients_per_round": FASHION_M,
        "rounds": ROUNDS,
        "seeds": list(SEEDS),
        "arms": arms,
        "delta_vs_fedavg_mean": deltas,
    }


def _recorded_femnist_control() -> dict | None:
    """The matched no-DP FedAvg control: the E=10 cell of the committed
    FEMNIST budget sweep (m=200, R=20, 3 seeds)."""
    path = DOCS / "_femnist_budget_e.json"
    if not path.exists():
        LOGGER.warning("no committed control at %s; comparison omitted", path)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for cell in data["budget"]["cells"]:
        if cell["local_epochs"] == FEMNIST_LOCAL_EPOCHS:
            s = cell["summary"]
            return {
                "source": path.name,
                "mean_final": s["mean_final"],
                "range_final": s["range_final"],
                "final_per_seed": s["final_per_seed"],
            }
    LOGGER.warning("no E=%d cell in %s; comparison omitted", FEMNIST_LOCAL_EPOCHS, path)
    return None


def _femnist_run(
    population: tuple,
    seed: int,
    label: str,
    aggregator: object | None = None,
    fedprox_mu: float = 0.0,
) -> dict:
    import femnist_experiments as fx

    train, test, shards = population
    return fx.simulate(
        train=train,
        test=test,
        shards=shards,
        clients_per_round=FEMNIST_M,
        rounds=ROUNDS,
        dp=False,
        seed=seed,
        local_epochs=FEMNIST_LOCAL_EPOCHS,
        aggregator=aggregator,
        fedprox_mu=fedprox_mu,
        label=label,
    )


def phase_c(population: tuple, selection: dict) -> dict:
    """Best-of-each on FEMNIST at the working budget: the non-IID case where
    adaptive server optimizers are supposed to help most."""
    control = _recorded_femnist_control()
    arms: dict[str, dict] = {}
    for name in FEDOPT_FAMILIES:
        best = _best_config(selection, name)
        runs = []
        for seed in SEEDS:
            run = _femnist_run(
                population,
                seed,
                f"femnist/{name}/seed={seed}",
                _fedopt_aggregator(name, best["server_lr"], momentum=best.get("momentum", 0.9)),
            )
            run["server_optimizer"] = dict(best)
            runs.append(run)
        arms[name] = {"config": best, "runs": runs, "summary": _summary(runs)}
        LOGGER.info(
            "PHASE C ARM %s: mean=%.4f range=%.4f vs control %.4f",
            name,
            arms[name]["summary"]["mean_final"],
            arms[name]["summary"]["range_final"],
            control["mean_final"] if control else float("nan"),
        )
    deltas = (
        {
            name: arms[name]["summary"]["mean_final"] - control["mean_final"]
            for name in FEDOPT_FAMILIES
        }
        if control
        else None
    )
    return {
        "phase": "C",
        "dataset": "femnist",
        "writers": FEMNIST_WRITERS,
        "clients_per_round": FEMNIST_M,
        "rounds": ROUNDS,
        "local_epochs": FEMNIST_LOCAL_EPOCHS,
        "seeds": list(SEEDS),
        "tuning_note": (
            "server LRs transferred from the Fashion sweep (phase A), not re-tuned "
            "on FEMNIST; a per-dataset re-tune is part of FedOpt's real cost and "
            "this measures the transfer, not the ceiling"
        ),
        "fedavg_control": control,
        "arms": arms,
        "delta_vs_control_mean": deltas,
    }


def phase_d(population: tuple) -> dict:
    """FedProx at the working budget, one seed per mu, vs the recorded control."""
    control = _recorded_femnist_control()
    runs = []
    for mu in FEDPROX_MUS:
        run = _femnist_run(population, SEEDS[0], f"femnist/fedprox/mu={mu}", fedprox_mu=mu)
        runs.append(run)
        LOGGER.info(
            "PHASE D CELL mu=%g: final=%.4f vs control %.4f",
            mu,
            run["final_accuracy"],
            control["mean_final"] if control else float("nan"),
        )
    deltas = (
        {
            str(mu): r["final_accuracy"] - control["mean_final"]
            for mu, r in zip(FEDPROX_MUS, runs, strict=True)
        }
        if control
        else None
    )
    return {
        "phase": "D",
        "dataset": "femnist",
        "writers": FEMNIST_WRITERS,
        "clients_per_round": FEMNIST_M,
        "rounds": ROUNDS,
        "local_epochs": FEMNIST_LOCAL_EPOCHS,
        "seed": SEEDS[0],
        "mus": list(FEDPROX_MUS),
        "fedavg_control": control,
        "runs": runs,
        "delta_vs_control_mean": deltas,
        "note": (
            "one seed per mu; the control is a 3-seed mean whose range bounds "
            "what a single-seed difference can claim"
        ),
    }


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    LOGGER.info("wrote %s", path.relative_to(ROOT))


def _run_or_load(letter: str, fn) -> dict:
    path = DOCS / f"_fedopt_batch_{letter}.json"
    if path.exists():
        LOGGER.info("PHASE %s: %s already exists, skipping", letter.upper(), path.name)
        return json.loads(path.read_text(encoding="utf-8"))
    LOGGER.info("PHASE %s: starting", letter.upper())
    payload = fn()
    _write(path, payload)
    return payload


def _load_femnist_population() -> tuple:
    import femnist_experiments as fx

    return fx.load_femnist(num_clients=FEMNIST_WRITERS, seed=SEEDS[0])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    a = _run_or_load("a", phase_a)
    _run_or_load("b", lambda: phase_b(a["selection"]))

    need_population = not all(
        (DOCS / f"_fedopt_batch_{letter}.json").exists() for letter in ("c", "d")
    )
    population = _load_femnist_population() if need_population else None
    _run_or_load("c", lambda: phase_c(population, a["selection"]))
    _run_or_load("d", lambda: phase_d(population))
    LOGGER.info("BATCH COMPLETE: all four phases on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
