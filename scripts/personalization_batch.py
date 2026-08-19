"""Personalization, measured. Two phases, unattended, resumable per run.

  A  docs/_personalization_a.json  FEMNIST at the working budget (1,000 writers,
                                   m=200, R=20, E=10), FedRep vs FedAvg, 3 seeds
                                   each. The natural per-writer test split makes
                                   per-client accuracy a measurement here, not a
                                   construction. ~5 h.
  B  docs/_personalization_b.json  Fashion-MNIST, N=100, m=50, R=20, E=2, with a
                                   Dirichlet alpha=0.1 partition -- pathological
                                   label skew, the regime personalization is
                                   supposed to own. Per-client test data is
                                   synthesised by dealing the test split in the
                                   training split's own per-class proportions,
                                   which is a modelling choice and is labelled as
                                   one. ~45 min.

A phase whose JSON already exists is skipped; phase A additionally checkpoints
each completed run to ``_personalization_a_partial.json``, because five hours is
long enough that an interruption should not cost the whole phase. Neither phase
uses differential privacy, so both are deterministic in configuration and seed up
to floating-point summation order.

Two budget notes, stated here rather than discovered in the numbers:

* **Fashion runs at E=2, not the E=1 of the recorded Fashion results.** FedRep's
  local update is two stages and both must run; one epoch cannot be split into
  two. Both arms of phase B therefore get E=2, matched to each other, and phase
  B's FedAvg arm is *not* comparable to the recorded E=1 Fashion figures.
* **The population is loaded once per phase, at ``SEEDS[0]``.** The run seed
  varies model initialisation, cohort sampling and batch order; it does not vary
  which writers (or which Dirichlet draw) make up the population. That is what
  makes the per-client arrays comparable across seeds, so a client's accuracy can
  be averaged over seeds before the distribution is taken.

Run detached (do not pipe stdout into a command that waits for EOF -- TFF leaves
an executor subprocess holding it):

    python scripts/personalization_batch.py >> docs/_personalization.log 2>&1

Heavy imports stay inside the phase functions so the pairing and summary logic
below remains importable for unit tests without TensorFlow.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LOGGER = logging.getLogger("personalization_batch")

DOCS = ROOT / "docs"

SEEDS = (42, 43, 44)
ROUNDS = 20

FEMNIST_WRITERS = 1000
FEMNIST_M = 200
FEMNIST_LOCAL_EPOCHS = 10
FEMNIST_HEAD_EPOCHS = 2

FASHION_N = 100
FASHION_M = 50
FASHION_ALPHA = 0.1
FASHION_LOCAL_EPOCHS = 2
FASHION_HEAD_EPOCHS = 1

ARMS = ("fedavg", "fedrep")


# ---------------------------------------------------------------------------
# Pairing and summary (no TensorFlow; unit-tested directly)
# ---------------------------------------------------------------------------


def accuracy_column(run: dict, key: str = "per_client") -> list:
    """The per-client accuracy array of one run, ``None`` where unscorable.

    Indexed by client, in client order, always full length -- the alignment two
    arms are paired on. Compacting out the unscorable clients here would make two
    arms' arrays different lengths and pair the wrong clients together.
    """
    rows = run.get(key)
    if not rows:
        raise ValueError(f"run {run.get('label')!r} has no {key} rows")
    out: list = [None] * (max(r["client"] for r in rows) + 1)
    for row in rows:
        out[row["client"]] = row["accuracy"]
    return out


def mean_over_seeds(columns: list[list]) -> list:
    """Per-client mean across seeds; ``None`` if any seed could not score it.

    Averaging before taking the distribution is the point: a per-client figure
    from one seed carries the seed's noise into the tail, and the tail is the
    claim. A client unscorable in any seed is unscorable here, rather than being
    averaged over a different number of seeds than its neighbours.
    """
    if not columns:
        raise ValueError("mean_over_seeds needs at least one column")
    width = len(columns[0])
    if any(len(c) != width for c in columns):
        raise ValueError(f"columns disagree on client count: {[len(c) for c in columns]}")
    out: list = []
    for i in range(width):
        values = [c[i] for c in columns]
        out.append(None if any(v is None for v in values) else sum(values) / len(values))
    return out


def pair(baseline: list, personalized: list) -> tuple[list, list]:
    """Restrict two per-client columns to the clients both can score."""
    if len(baseline) != len(personalized):
        raise ValueError(
            f"columns disagree on client count: {len(baseline)} vs {len(personalized)}"
        )
    keep = [
        i
        for i in range(len(baseline))
        if baseline[i] is not None and personalized[i] is not None
    ]
    return [baseline[i] for i in keep], [personalized[i] for i in keep]


def compare_arms(runs: dict[str, list[dict]]) -> dict:
    """Everything the write-up reads: distributions, per-client pairings, tails.

    ``runs`` maps arm name to that arm's per-seed run records. Three comparisons
    come out, and the middle one is what keeps the first honest:

    * global vs personalized — the headline,
    * global vs global-plus-fine-tuned-head — how much of the headline is just
      *having* a local head, with no FedRep in it at all,
    * fine-tuned vs personalized — what FedRep's alternating training adds on
      top of that.
    """
    from fl.personalization import distribution_summary, paired_delta_summary

    global_cols = [accuracy_column(r) for r in runs["fedavg"]]
    finetuned_cols = [accuracy_column(r, "per_client_finetuned") for r in runs["fedavg"]]
    personal_cols = [accuracy_column(r) for r in runs["fedrep"]]

    columns = {
        "global": mean_over_seeds(global_cols),
        "finetuned": mean_over_seeds(finetuned_cols),
        "personalized": mean_over_seeds(personal_cols),
    }
    distributions = {
        name: distribution_summary([v for v in col if v is not None])
        for name, col in columns.items()
    }
    deltas = {}
    for name, (left, right) in {
        "personalized_vs_global": ("global", "personalized"),
        "finetuned_vs_global": ("global", "finetuned"),
        "personalized_vs_finetuned": ("finetuned", "personalized"),
    }.items():
        a, b = pair(columns[left], columns[right])
        deltas[name] = paired_delta_summary(a, b)

    return {
        "seeds": list(SEEDS),
        "clients": len(columns["global"]),
        "clients_scored": sum(1 for v in columns["global"] if v is not None),
        "per_client_seed_mean": columns,
        "distributions": distributions,
        "paired_deltas": deltas,
        "per_seed": {
            arm: [
                {
                    "seed": r["seed"],
                    "summary": r["summary"],
                    "summary_finetuned": r.get("summary_finetuned"),
                    "final_pooled_accuracy": r.get("final_pooled_accuracy"),
                    "mean_round_seconds": r.get("mean_round_seconds"),
                }
                for r in records
            ]
            for arm, records in runs.items()
        },
        "wire": runs["fedrep"][0]["wire"],
    }


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def _run_phase(
    *,
    phase: str,
    dataset: str,
    num_clients: int,
    cohort: int,
    local_epochs: int,
    head_epochs: int,
    alpha: float,
    partial_path: Path | None,
) -> dict:
    import personalization_experiments as px

    train, test, shards, test_shards, model_name = px.load_population(
        dataset, num_clients, alpha, seed=SEEDS[0]
    )
    LOGGER.info(
        "phase %s population: %d clients, %d train / %d test examples, "
        "test shard sizes min=%d median=%d max=%d, %d clients with no held-out data",
        phase,
        len(shards),
        len(train),
        len(test),
        min(s.size for s in test_shards),
        sorted(s.size for s in test_shards)[len(test_shards) // 2],
        max(s.size for s in test_shards),
        sum(1 for s in test_shards if s.size == 0),
    )

    done: dict[str, dict] = {}
    if partial_path is not None and partial_path.is_file():
        done = json.loads(partial_path.read_text(encoding="utf-8"))
        LOGGER.info("resuming phase %s with %d run(s) already recorded", phase, len(done))

    runs: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            label = f"{dataset}/{arm}/seed={seed}"
            if label in done:
                LOGGER.info("skipping completed run %s", label)
                runs[arm].append(done[label])
                continue
            record = px.simulate(
                model_name=model_name,
                train=train,
                test=test,
                shards=shards,
                test_shards=test_shards,
                method=arm,
                clients_per_round=cohort,
                rounds=ROUNDS,
                local_epochs=local_epochs,
                head_epochs=head_epochs,
                seed=seed,
                label=label,
            )
            runs[arm].append(record)
            done[label] = record
            if partial_path is not None:
                partial_path.write_text(json.dumps(done, indent=2), encoding="utf-8")

    comparison = compare_arms(runs)
    LOGGER.info(
        "PHASE %s DONE: global mean=%.4f median=%.4f worst-decile=%.4f | "
        "personalized mean=%.4f median=%.4f worst-decile=%.4f | "
        "paired delta median=%+.4f, %.1f%% of clients improved",
        phase,
        comparison["distributions"]["global"]["mean"],
        comparison["distributions"]["global"]["median"],
        comparison["distributions"]["global"]["worst_decile_mean"],
        comparison["distributions"]["personalized"]["mean"],
        comparison["distributions"]["personalized"]["median"],
        comparison["distributions"]["personalized"]["worst_decile_mean"],
        comparison["paired_deltas"]["personalized_vs_global"]["median"],
        100 * comparison["paired_deltas"]["personalized_vs_global"]["fraction_improved"],
    )
    return {
        "phase": phase,
        "dataset": dataset,
        "num_clients": num_clients,
        "clients_per_round": cohort,
        "rounds": ROUNDS,
        "local_epochs": local_epochs,
        "head_epochs": head_epochs,
        "dirichlet_alpha": alpha if dataset == "fashion_mnist" else None,
        "per_client_test_data": "natural (LEAF by-writer split)"
        if dataset == "femnist"
        else f"synthetic (test dealt in the train split's per-class proportions, alpha={alpha})",
        "comparison": comparison,
        "runs": runs,
    }


def phase_a() -> dict:
    """FEMNIST at the working budget: FedRep vs FedAvg, 3 seeds each."""
    return _run_phase(
        phase="A",
        dataset="femnist",
        num_clients=FEMNIST_WRITERS,
        cohort=FEMNIST_M,
        local_epochs=FEMNIST_LOCAL_EPOCHS,
        head_epochs=FEMNIST_HEAD_EPOCHS,
        alpha=0.5,
        partial_path=DOCS / "_personalization_a_partial.json",
    )


def phase_b() -> dict:
    """Fashion-MNIST at alpha=0.1: the pathological-heterogeneity comparison."""
    return _run_phase(
        phase="B",
        dataset="fashion_mnist",
        num_clients=FASHION_N,
        cohort=FASHION_M,
        local_epochs=FASHION_LOCAL_EPOCHS,
        head_epochs=FASHION_HEAD_EPOCHS,
        alpha=FASHION_ALPHA,
        partial_path=DOCS / "_personalization_b_partial.json",
    )


PHASES = (("A", "_personalization_a.json", phase_a), ("B", "_personalization_b.json", phase_b))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", default=None, help="Run only this phase letter (A or B); default: all."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    DOCS.mkdir(parents=True, exist_ok=True)

    for letter, filename, fn in PHASES:
        if args.only and args.only.upper() != letter:
            continue
        out = DOCS / filename
        if out.is_file():
            LOGGER.info("phase %s already recorded at %s; skipping", letter, out)
            continue
        LOGGER.info("starting phase %s", letter)
        out.write_text(json.dumps(fn(), indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
