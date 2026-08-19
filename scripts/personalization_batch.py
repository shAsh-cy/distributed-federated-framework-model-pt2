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

import numpy as np

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

#: Minimum held-out samples for a client to enter a phase's HEADLINE statistics.
#: Phase A only. FEMNIST's median writer has 18 test samples over 62 classes, so
#: its per-client accuracy is quantised into eighteenths and carries roughly
#: +/-0.11 of binomial noise at p=0.5 -- noise that averaging over seeds does not
#: touch, because every seed scores the same 18 samples. A worst-decile figure
#: computed over that population measures shard size more than it measures who
#: the model serves badly. Fashion's median client has 69 held-out samples and
#: needs no threshold, so phase B's is 0 and its headline is its full population.
#: The cost of the threshold is a selection effect, measured and reported rather
#: than assumed away -- see selection_effect() and docs/personalization.md section 9.
HEADLINE_MIN_TEST_SAMPLES = {"A": 30, "B": 0}


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


def test_samples_column(run: dict) -> list[int]:
    """Held-out sample count per client, indexed like :func:`accuracy_column`.

    Taken from one run because the population is loaded once per phase and is
    identical across arms and seeds; a mismatch would mean the phase loaded two
    different populations, which :func:`compare_arms` checks for.
    """
    rows = run.get("per_client")
    if not rows:
        raise ValueError(f"run {run.get('label')!r} has no per_client rows")
    out = [0] * (max(r["client"] for r in rows) + 1)
    for row in rows:
        out[row["client"]] = int(row["test_samples"])
    return out


def selection_effect(sizes: list[int], threshold: int, train_sizes=None, alignment=None) -> dict:
    """What restricting the headline to ``test_samples >= threshold`` selects for.

    Both halves are named, because they push the same way and a reader is owed
    both: the kept clients have MORE DATA (so their accuracy is estimated more
    precisely *and* their local head has more to fit) and MORE DISTINCTIVE LABEL
    PRIORS (so there is more for a head to exploit). The restricted subset is
    therefore the sub-population where head personalization is both most
    measurable and most favoured, and a gain measured there is an upper bound on
    the gain across the whole population, not an estimate of it.
    """
    sizes = np.asarray(sizes, dtype=float)
    keep = sizes >= threshold
    out = {
        "min_test_samples": int(threshold),
        "clients_kept": int(keep.sum()),
        "clients_total": int(sizes.size),
        "kept_fraction": float(keep.mean()) if sizes.size else 0.0,
        "test_samples_kept": int(sizes[keep].sum()),
        "test_samples_total": int(sizes.sum()),
        "test_samples_kept_fraction": (
            float(sizes[keep].sum() / sizes.sum()) if sizes.sum() else 0.0
        ),
        "median_test_samples_kept": float(np.median(sizes[keep])) if keep.any() else None,
        "median_test_samples_dropped": float(np.median(sizes[~keep])) if (~keep).any() else None,
    }
    if train_sizes is not None:
        train = np.asarray(train_sizes, dtype=float)
        out["median_train_samples_kept"] = float(np.median(train[keep])) if keep.any() else None
        out["median_train_samples_dropped"] = (
            float(np.median(train[~keep])) if (~keep).any() else None
        )
    if alignment is not None:
        align = np.asarray(alignment, dtype=float)
        out["median_label_alignment_kept"] = float(np.median(align[keep])) if keep.any() else None
        out["median_label_alignment_dropped"] = (
            float(np.median(align[~keep])) if (~keep).any() else None
        )
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


def compare_arms(
    runs: dict[str, list[dict]],
    min_test_samples: int = 0,
    train_sizes=None,
    alignment=None,
) -> dict:
    """Everything the write-up reads: distributions, per-client pairings, tails.

    ``runs`` maps arm name to that arm's per-seed run records. Three comparisons
    come out, and the middle one is what keeps the first honest:

    * global vs personalized — the headline,
    * global vs global-plus-fine-tuned-head — how much of the headline is just
      *having* a local head, with no FedRep in it at all,
    * fine-tuned vs personalized — what FedRep's alternating training adds on
      top of that.

    Each is computed twice. ``headline`` restricts to clients with at least
    ``min_test_samples`` held-out samples; ``full_population`` uses every client
    that can be scored at all. With ``min_test_samples=0`` the two are the same
    object's worth of numbers and the selection effect is empty. Both are always
    present, and ``selection_effect`` says what the restriction bought and what
    it cost — a restricted headline with no accounting of what it excluded is
    just a favourable subset with a confident name.
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
    sizes = test_samples_column(runs["fedavg"][0])
    for arm, records in runs.items():
        for record in records:
            if test_samples_column(record) != sizes:
                raise ValueError(
                    f"run {record.get('label')!r} ({arm}) scored a different population than "
                    "the first run; a phase must load its population once"
                )
    if len(sizes) != len(columns["global"]):
        raise ValueError(
            f"client count disagrees between sizes ({len(sizes)}) and "
            f"accuracies ({len(columns['global'])})"
        )

    def block(eligible: list[int]) -> dict:
        scoped = {
            name: [col[i] for i in eligible if col[i] is not None] for name, col in columns.items()
        }
        distributions = {name: distribution_summary(vals) for name, vals in scoped.items() if vals}
        deltas = {}
        for label, (left, right) in {
            "personalized_vs_global": ("global", "personalized"),
            "finetuned_vs_global": ("global", "finetuned"),
            "personalized_vs_finetuned": ("finetuned", "personalized"),
        }.items():
            a, b = pair(
                [columns[left][i] for i in eligible], [columns[right][i] for i in eligible]
            )
            if a:
                deltas[label] = paired_delta_summary(a, b)
        return {
            "clients": len(eligible),
            "clients_scored": sum(1 for i in eligible if columns["global"][i] is not None),
            "distributions": distributions,
            "paired_deltas": deltas,
        }

    everyone = list(range(len(sizes)))
    eligible = [i for i in everyone if sizes[i] >= min_test_samples]
    if not eligible:
        raise ValueError(
            f"no client has >= {min_test_samples} held-out samples; the headline threshold "
            "excludes the entire population"
        )

    return {
        "seeds": list(SEEDS),
        "clients": len(everyone),
        "clients_scored": sum(1 for v in columns["global"] if v is not None),
        "headline": {"min_test_samples": int(min_test_samples), **block(eligible)},
        "full_population": block(everyone),
        "selection_effect": selection_effect(sizes, min_test_samples, train_sizes, alignment),
        # Full population, always: the ECDF is the secondary figure and is drawn
        # over everyone, threshold or no threshold.
        "per_client_seed_mean": columns,
        "test_samples": sizes,
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

    from fl.data import dataset_num_classes
    from fl.personalization import paired_profile_alignment

    alignment = paired_profile_alignment(
        train.y, shards, test.y, test_shards, dataset_num_classes(dataset)
    ).tolist()
    comparison = compare_arms(
        runs,
        min_test_samples=HEADLINE_MIN_TEST_SAMPLES[phase],
        train_sizes=[int(s.size) for s in shards],
        alignment=alignment,
    )
    head, full = comparison["headline"], comparison["full_population"]
    LOGGER.info(
        "PHASE %s HEADLINE (>=%d held-out samples, %d of %d clients): "
        "global median=%.4f worst-decile=%.4f | personalized median=%.4f worst-decile=%.4f | "
        "paired delta median=%+.4f, %.1f%% improved",
        phase,
        comparison["selection_effect"]["min_test_samples"],
        comparison["selection_effect"]["clients_kept"],
        comparison["selection_effect"]["clients_total"],
        head["distributions"]["global"]["median"],
        head["distributions"]["global"]["worst_decile_mean"],
        head["distributions"]["personalized"]["median"],
        head["distributions"]["personalized"]["worst_decile_mean"],
        head["paired_deltas"]["personalized_vs_global"]["median"],
        100 * head["paired_deltas"]["personalized_vs_global"]["fraction_improved"],
    )
    LOGGER.info(
        "PHASE %s FULL POPULATION (secondary, %d clients): global median=%.4f "
        "worst-decile=%.4f | personalized median=%.4f worst-decile=%.4f | "
        "paired delta median=%+.4f, %.1f%% improved",
        phase,
        full["clients_scored"],
        full["distributions"]["global"]["median"],
        full["distributions"]["global"]["worst_decile_mean"],
        full["distributions"]["personalized"]["median"],
        full["distributions"]["personalized"]["worst_decile_mean"],
        full["paired_deltas"]["personalized_vs_global"]["median"],
        100 * full["paired_deltas"]["personalized_vs_global"]["fraction_improved"],
    )
    LOGGER.info(
        "PHASE %s SELECTION: kept clients hold %.1fx the training data and %.2f more "
        "label alignment than dropped ones (medians %s vs %s train, %.3f vs %.3f alignment)",
        phase,
        (comparison["selection_effect"].get("median_train_samples_kept") or 0)
        / max(1.0, comparison["selection_effect"].get("median_train_samples_dropped") or 1.0),
        (comparison["selection_effect"].get("median_label_alignment_kept") or 0.0)
        - (comparison["selection_effect"].get("median_label_alignment_dropped") or 0.0),
        comparison["selection_effect"].get("median_train_samples_kept"),
        comparison["selection_effect"].get("median_train_samples_dropped"),
        comparison["selection_effect"].get("median_label_alignment_kept") or float("nan"),
        comparison["selection_effect"].get("median_label_alignment_dropped") or float("nan"),
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
        "headline_min_test_samples": HEADLINE_MIN_TEST_SAMPLES[phase],
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
        "--phases",
        default=None,
        help=(
            "Comma-separated phase letters, IN THE ORDER THEY SHOULD RUN "
            "(e.g. 'B,A' to take the 45-minute Fashion phase before the "
            "five-hour FEMNIST one). Default: A,B."
        ),
    )
    parser.add_argument(
        "--only", default=None, help="Run only this phase letter; shorthand for --phases X."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    DOCS.mkdir(parents=True, exist_ok=True)

    known = {letter: (filename, fn) for letter, filename, fn in PHASES}
    if args.phases and args.only:
        parser.error("pass --phases or --only, not both")
    requested = args.phases or args.only or ",".join(letter for letter, _f, _fn in PHASES)
    order = [part.strip().upper() for part in requested.split(",") if part.strip()]
    unknown = [letter for letter in order if letter not in known]
    if unknown:
        parser.error(f"unknown phase(s) {unknown}; known phases are {sorted(known)}")
    if len(set(order)) != len(order):
        parser.error(f"phase list repeats a phase: {order}")
    LOGGER.info("phase order: %s", " -> ".join(order))

    for letter in order:
        filename, fn = known[letter]
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
