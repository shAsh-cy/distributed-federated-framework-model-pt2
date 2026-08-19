"""FedRep vs FedAvg on the splits this repository already has.

Why this harness and not the gRPC path
--------------------------------------
FedRep needs one piece of state the deployed client does not have: a classifier
head that survives across rounds *on the client*. ``fl/client.py`` holds nothing
between rounds except its shard, and a container that restarts reclaims its shard
and would resume with a head that is either stale, another client's, or freshly
initialised. The last of those is the dangerous one, because it is invisible: the
run completes, the numbers look plausible, and the personalization was never
there. So personalization is measured here, in the in-process harness, exactly as
the FEMNIST budget and FedOpt results are -- and ``docs/personalization.md`` says
so rather than leaving the reader to assume containers.

The two arms, and what is held equal
------------------------------------
Both arms run **this same loop**. They differ in two lines: which variables the
local optimiser is allowed to move, and which tensors are submitted.

* ``fedavg`` — one local stage over every variable for ``local_epochs`` epochs;
  the whole weight list is submitted and sample-count-weighted averaged.
* ``fedrep`` — two local stages (Collins et al., ICML 2021): ``head_epochs`` over
  the classifier head with the representation frozen, then
  ``local_epochs - head_epochs`` over the representation with the head frozen.
  Only the representation is submitted; the head goes into the
  :class:`~fl.personalization.HeadStore` under this client's id and comes back
  out next time this client is sampled.

Held equal between arms, deliberately: the total local epoch budget, the client
learning rate and momentum, the batch size, the initial global model, the cohort
sequence (same seed, same generator, same call order), and the per-client batch
shuffling (seeded from ``(seed, round, client)``, so client 7 in round 3 sees the
same batch order in both arms). What differs is the algorithm and nothing else.

The head-epoch split is **not tuned**. ``head_epochs=2`` of a 10-epoch budget on
FEMNIST is an a-priori choice -- the head is 3.45 % of the parameters and warm
starts from the previous round, so it is given a small slice -- and the write-up
reports it as an untuned constant, in the same spirit as the FedOpt phase-C
server learning rates transferred rather than re-tuned. A split sweep is the
obvious next measurement and is not in this budget.

Evaluation
----------
Per client, on that client's own held-out data:

* the **global** model's accuracy there (the ``fedavg`` arm),
* the global model with a **locally fine-tuned head** (the ``fedavg`` arm's own
  control: it separates "personalization helps" from "any local head helps"),
* the **backbone-global + local-head** accuracy (the ``fedrep`` arm).

Every one of those is recorded per client, not averaged away: the array is in the
JSON. Clients with no held-out samples are excluded and counted; clients that
were never sampled are flagged (``head_updates == 0``) so a cold head is never
silently reported as a personalized one.

Do not pipe this script's stdout into a command that waits for EOF -- TFF leaves
an executor subprocess holding the pipe. Redirect to a file.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl.aggregation import ClientUpdate, FedAvgAggregator  # noqa: E402
from fl.archspec import SPECS  # noqa: E402
from fl.personalization import (  # noqa: E402
    HeadStore,
    distribution_summary,
    paired_delta_summary,
    weighted_mean,
    wire_saving,
)

LOGGER = logging.getLogger("personalization")

METHODS = ("fedavg", "fedrep")

#: Clients below this many held-out samples give a per-client accuracy quantised
#: into very few levels (one sample can only score 0 or 1). Their figures are
#: kept in the record and additionally summarised without them, so a tail claim
#: can be checked against both.
MIN_TEST_SAMPLES = 10


def seed_everything(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _client_rng(seed: int, round_index: int, client_index: int) -> np.random.Generator:
    """Batch-order generator for one client in one round.

    Derived from the triple rather than from a single stream so that the same
    client in the same round shuffles identically in both arms -- the arms would
    otherwise diverge in batch order the moment their round costs differ, which
    adds a difference nobody asked to measure.
    """
    return np.random.default_rng([seed, round_index, client_index])


class StagedTrainer:
    """Local training over explicit variable groups.

    One traced step function per group, built once and reused for every client
    and round: a Keras ``layer.trainable = False`` plus ``recompile`` per stage
    would retrace twice per client per round, which at m=200 and R=20 is 8,000
    retraces of a graph that never changes.

    Groups are ``all`` (FedAvg's single stage), ``head`` and ``backbone``
    (FedRep's two). The grouping comes from the spec's ``personal_layers``
    marker, and the constructor checks that the head variables it collected are
    exactly the head the canonical split describes -- so a spec whose marker and
    whose Keras layers disagree fails here, not silently three phases later.
    """

    def __init__(self, model, spec, learning_rate: float, momentum: float) -> None:
        import tensorflow as tf

        self.model = model
        self.spec = spec
        head_names = set(spec.personal_layers)
        head_vars, backbone_vars = [], []
        for layer in model.layers:
            target = head_vars if layer.name in head_names else backbone_vars
            target.extend(layer.trainable_weights)

        got = [tuple(v.shape) for v in head_vars]
        want = spec.personal_shapes()
        if got != want:
            raise ValueError(
                f"head variables {got} do not match the canonical head of spec "
                f"{spec.name!r}, which is {want}. A head layer carrying non-trainable "
                "weights (BatchNorm moving statistics) would produce exactly this "
                "mismatch and needs those statistics handled explicitly."
            )

        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._groups: dict[str, tuple] = {}
        for name, variables in (
            ("all", list(model.trainable_weights)),
            ("head", head_vars),
            ("backbone", backbone_vars),
        ):
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
            opt.build(variables)
            self._groups[name] = (variables, opt, self._make_step(variables, opt))

    def _make_step(self, variables, opt):
        import tensorflow as tf

        model, loss_fn = self.model, self._loss

        @tf.function(reduce_retracing=True)
        def step(xb, yb):
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss = loss_fn(yb, logits)
            grads = tape.gradient(loss, variables)
            opt.apply_gradients(zip(grads, variables))
            preds = tf.argmax(logits, axis=1, output_type=tf.int64)
            correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(yb, tf.int64)), tf.int32))
            return loss, correct

        return step

    def reset_optimizers(self) -> None:
        """Zero every optimiser slot: local optimiser state does not cross clients."""
        import tensorflow as tf

        for _variables, opt, _step in self._groups.values():
            variables = opt.variables
            if callable(variables):
                variables = variables()
            for v in variables:
                v.assign(tf.zeros_like(v))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        stages: list[tuple[str, int]],
        batch_size: int,
        rng: np.random.Generator,
        shuffle: bool = True,
    ) -> list[dict]:
        """Run each ``(group, epochs)`` stage in order; report each stage's fit."""
        n = int(len(y))
        report: list[dict] = []
        for group, epochs in stages:
            if epochs <= 0:
                continue
            if group not in self._groups:
                raise ValueError(f"unknown variable group {group!r}; have {sorted(self._groups)}")
            _variables, _opt, step = self._groups[group]
            last_loss, correct, seen = float("nan"), 0, 0
            for _ in range(epochs):
                order = rng.permutation(n) if shuffle else np.arange(n)
                for start in range(0, n, batch_size):
                    idx = order[start : start + batch_size]
                    loss, batch_correct = step(x[idx], y[idx])
                    last_loss = float(loss)
                    correct += int(batch_correct)
                    seen += int(idx.size)
            report.append(
                {
                    "stage": group,
                    "epochs": int(epochs),
                    "loss": last_loss,
                    "accuracy": correct / max(1, seen),
                }
            )
        return report


def accuracy_of(model, x: np.ndarray, y: np.ndarray, batch: int = 512) -> float:
    """Top-1 accuracy of ``model`` on ``(x, y)``, evaluated eagerly.

    ``model.evaluate`` costs a fixed per-call overhead that dominates when the
    call is made once per client on twenty samples; this is the same number
    without it.
    """
    if len(y) == 0:
        raise ValueError("accuracy_of needs at least one sample")
    correct = 0
    for start in range(0, len(y), batch):
        logits = model(x[start : start + batch], training=False).numpy()
        correct += int((logits.argmax(axis=1) == y[start : start + batch]).sum())
    return correct / len(y)


def evaluate_per_client(
    model,
    spec,
    *,
    test,
    test_shards: list[np.ndarray],
    shared_weights: list[np.ndarray] | None = None,
    full_weights: list[np.ndarray] | None = None,
    head_store: HeadStore | None = None,
) -> list[dict]:
    """Score every client on its **own** held-out data.

    Two modes, and they are exclusive:

    * ``full_weights`` — one global model, scored on each client's test shard.
    * ``shared_weights`` + ``head_store`` — the aggregated backbone recombined
      with each client's own stored head, scored on that client's test shard.

    The client's head and the client's test shard are indexed by the same client
    id in the same expression; there is no place for one to drift from the other.
    Clients with an empty test shard are returned with ``accuracy=None`` rather
    than dropped, so the caller counts them instead of silently shrinking its
    denominator.
    """
    if (full_weights is None) == (shared_weights is None):
        raise ValueError("pass exactly one of full_weights or shared_weights")
    if shared_weights is not None and head_store is None:
        raise ValueError("shared_weights needs a head_store to recombine with")

    if full_weights is not None:
        model.set_weights(full_weights)  # the same model scores every client
    rows: list[dict] = []
    for cid, shard in enumerate(test_shards):
        row = {
            "client": int(cid),
            "test_samples": int(shard.size),
            "head_updates": int(head_store.updates(cid)) if head_store is not None else None,
        }
        if shard.size == 0:
            row["accuracy"] = None
            rows.append(row)
            continue
        if shared_weights is not None:
            model.set_weights(spec.merge_weights(shared_weights, head_store.get(cid)))
        row["accuracy"] = accuracy_of(model, test.x[shard], test.y[shard])
        rows.append(row)
    return rows


def _summarise_rows(rows: list[dict], key: str = "accuracy") -> dict:
    """Distribution over the clients that can actually be scored."""
    scored = [r for r in rows if r[key] is not None]
    if not scored:
        raise ValueError("no client has held-out data; nothing to summarise")
    values = [r[key] for r in scored]
    sizes = [r["test_samples"] for r in scored]
    big = [r for r in scored if r["test_samples"] >= MIN_TEST_SAMPLES]
    out = {
        "clients_scored": len(scored),
        "clients_unscorable": len(rows) - len(scored),
        # Only meaningful when heads were tracked: the global arm has no
        # per-client state, so counting its zeros would report "every client was
        # never sampled" for a run in which most of them were.
        "clients_never_sampled": (
            sum(1 for r in rows if r["head_updates"] == 0)
            if any(r["head_updates"] is not None for r in rows)
            else None
        ),
        "pooled_weighted_mean": weighted_mean(values, sizes),
        "per_client": distribution_summary(values),
        "min_test_samples_for_restricted": MIN_TEST_SAMPLES,
        "per_client_restricted": (
            distribution_summary([r[key] for r in big]) if big else None
        ),
        "restricted_clients": len(big),
    }
    return out


def simulate(
    *,
    model_name: str,
    train,
    test,
    shards: list[np.ndarray],
    test_shards: list[np.ndarray],
    method: str,
    clients_per_round: int,
    rounds: int = 20,
    local_epochs: int = 10,
    head_epochs: int = 2,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 32,
    seed: int = 42,
    finetune_control: bool = True,
    label: str = "",
) -> dict:
    """One federated run of ``method`` over a pre-loaded population.

    Returns everything measured, per client and per round. No differential
    privacy: personalization and DP compose (the head never enters the
    aggregate, so it is outside the sensitivity bound entirely), but measuring
    the two together at once would leave a result nobody could attribute, and
    the DP arm is not in this budget.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    spec = SPECS[model_name]
    if method == "fedrep":
        if not spec.personal_layers:
            raise ValueError(
                f"spec {model_name!r} declares no personal_layers; fedrep needs a head"
            )
        if not 1 <= head_epochs < local_epochs:
            raise ValueError(
                f"head_epochs must be in [1, local_epochs); got {head_epochs} of {local_epochs}. "
                "Both stages must run: zero head epochs is FedAvg with extra steps, and zero "
                "backbone epochs never federates anything."
            )
    if method == "fedavg" and finetune_control and head_epochs < 1:
        raise ValueError(
            f"the fine-tuning control fits each client's head for head_epochs epochs and "
            f"got {head_epochs}; pass finetune_control=False to skip the control instead "
            "of running it for zero epochs, which would silently report the global model twice"
        )

    from fl.models import build_model, compile_for_evaluation

    seed_everything(seed)
    started = time.monotonic()
    num_clients = len(shards)
    if len(test_shards) != num_clients:
        raise ValueError(
            f"train and test shard lists disagree on client count: "
            f"{num_clients} vs {len(test_shards)}"
        )

    global_full = build_model(model_name, seed=seed).get_weights()
    global_shared, initial_head = spec.split_weights(global_full)
    head_store = HeadStore(spec, initial_head)

    trainer = StagedTrainer(build_model(model_name), spec, learning_rate, momentum)
    evaluator = compile_for_evaluation(build_model(model_name))
    aggregator = FedAvgAggregator()

    stages = (
        [("head", head_epochs), ("backbone", local_epochs - head_epochs)]
        if method == "fedrep"
        else [("all", local_epochs)]
    )

    rng = np.random.default_rng(seed)
    history: list[dict] = []

    evaluator.set_weights(global_full)
    untrained_pooled = accuracy_of(evaluator, test.x, test.y)

    for rnd in range(1, rounds + 1):
        round_started = time.monotonic()
        cohort = rng.choice(num_clients, size=clients_per_round, replace=False)
        updates: list[ClientUpdate] = []
        stage_reports: list[list[dict]] = []

        for cid in cohort:
            cid = int(cid)
            idx = shards[cid]
            if method == "fedrep":
                start_weights = spec.merge_weights(global_shared, head_store.get(cid))
            else:
                start_weights = global_full
            trainer.model.set_weights(start_weights)
            trainer.reset_optimizers()
            stage_reports.append(
                trainer.fit(
                    train.x[idx],
                    train.y[idx],
                    stages,
                    batch_size,
                    _client_rng(seed, rnd, cid),
                )
            )
            trained = trainer.model.get_weights()
            if method == "fedrep":
                shared_k, head_k = spec.split_weights(trained)
                head_store.put(cid, head_k)
                updates.append(ClientUpdate(f"c{cid}", shared_k, int(idx.size)))
            else:
                updates.append(ClientUpdate(f"c{cid}", trained, int(idx.size)))

        if method == "fedrep":
            global_shared = aggregator.aggregate(updates, global_shared)
        else:
            global_full = aggregator.aggregate(updates, global_full)

        # Per-round progress, measured the same way in both arms: the cohort
        # just trained, each client on its own held-out data. The FedAvg arm
        # also reports pooled accuracy, which links these runs to every recorded
        # global-accuracy figure; the FedRep arm cannot, because there is no
        # global head to evaluate and inventing one would be a different
        # measurement wearing the same name.
        if method == "fedavg":
            evaluator.set_weights(global_full)  # one model for the whole cohort
        scored = []
        for c in cohort:
            c = int(c)
            shard = test_shards[c]
            if not shard.size:
                continue
            if method == "fedrep":
                evaluator.set_weights(spec.merge_weights(global_shared, head_store.get(c)))
            scored.append(accuracy_of(evaluator, test.x[shard], test.y[shard]))

        # The FedAvg evaluator still holds the global model here; the FedRep one
        # holds whichever client was scored last, which is why only FedAvg
        # reports a pooled figure.
        pooled = accuracy_of(evaluator, test.x, test.y) if method == "fedavg" else None

        history.append(
            {
                "round": rnd,
                "pooled_accuracy": pooled,
                "cohort_mean_accuracy": float(np.mean(scored)) if scored else None,
                "cohort_median_accuracy": float(np.median(scored)) if scored else None,
                "cohort_clients_scored": len(scored),
                "mean_stage_train_accuracy": {
                    s["stage"]: float(np.mean([r[i]["accuracy"] for r in stage_reports]))
                    for i, s in enumerate(stage_reports[0])
                }
                if stage_reports
                else {},
                "seconds": round(time.monotonic() - round_started, 2),
            }
        )
        LOGGER.info(
            "%s round %2d: cohort mean=%s median=%s pooled=%s (%.1fs)",
            label,
            rnd,
            f"{history[-1]['cohort_mean_accuracy']:.4f}"
            if history[-1]["cohort_mean_accuracy"] is not None
            else "n/a",
            f"{history[-1]['cohort_median_accuracy']:.4f}"
            if history[-1]["cohort_median_accuracy"] is not None
            else "n/a",
            f"{pooled:.4f}" if pooled is not None else "n/a (no global head)",
            history[-1]["seconds"],
        )

    # -- final per-client evaluation over the WHOLE population ---------------
    if method == "fedrep":
        rows = evaluate_per_client(
            evaluator,
            spec,
            test=test,
            test_shards=test_shards,
            shared_weights=global_shared,
            head_store=head_store,
        )
    else:
        rows = evaluate_per_client(
            evaluator, spec, test=test, test_shards=test_shards, full_weights=global_full
        )
    for row in rows:
        row["train_samples"] = int(shards[row["client"]].size)

    finetuned_rows = None
    if method == "fedavg" and finetune_control:
        finetuned_rows = _finetune_control(
            trainer,
            evaluator,
            spec,
            train=train,
            test=test,
            shards=shards,
            test_shards=test_shards,
            global_full=global_full,
            epochs=head_epochs,
            batch_size=batch_size,
            seed=seed,
        )

    result = {
        "label": label,
        "method": method,
        "model": model_name,
        "num_clients": num_clients,
        "clients_per_round": clients_per_round,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "head_epochs": head_epochs if method == "fedrep" else None,
        "backbone_epochs": local_epochs - head_epochs if method == "fedrep" else None,
        "finetune_epochs": head_epochs if (method == "fedavg" and finetune_control) else None,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "batch_size": batch_size,
        "seed": seed,
        "wire": wire_saving(spec).to_dict(),
        "untrained_pooled_accuracy": float(untrained_pooled),
        "final_pooled_accuracy": history[-1]["pooled_accuracy"] if history else None,
        "per_client": rows,
        "summary": _summarise_rows(rows),
        "per_client_finetuned": finetuned_rows,
        "summary_finetuned": _summarise_rows(finetuned_rows) if finetuned_rows else None,
        "head_participation": head_store.participation(range(num_clients))
        if method == "fedrep"
        else None,
        "mean_round_seconds": float(np.mean([h["seconds"] for h in history])) if history else None,
        "seconds": round(time.monotonic() - started, 1),
        "history": history,
    }
    if finetuned_rows is not None:
        both = [
            (r["accuracy"], f["accuracy"])
            for r, f in zip(rows, finetuned_rows, strict=True)
            if r["accuracy"] is not None and f["accuracy"] is not None
        ]
        result["finetune_delta"] = paired_delta_summary(
            [a for a, _b in both], [b for _a, b in both]
        )
    LOGGER.info(
        "%s DONE: per-client mean=%.4f median=%.4f worst-decile=%.4f (%.0fs)",
        label,
        result["summary"]["per_client"]["mean"],
        result["summary"]["per_client"]["median"],
        result["summary"]["per_client"]["worst_decile_mean"],
        result["seconds"],
    )
    return result


def _finetune_control(
    trainer: StagedTrainer,
    evaluator,
    spec,
    *,
    train,
    test,
    shards,
    test_shards,
    global_full,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict]:
    """FedAvg + local head fine-tuning, evaluated per client.

    The control that keeps the FedRep result honest. Without it a gain over the
    global model could be FedRep's alternating training or it could be the mere
    existence of a locally fitted head, and those are different claims. Each
    client starts from the *same* final global model, fits its head alone on its
    *training* shard for the same number of head epochs FedRep used per round,
    and is then scored on its held-out shard. No test sample enters the fit --
    the fit reads ``train`` and the score reads ``test``, and they are different
    objects.
    """
    rows: list[dict] = []
    for cid, shard in enumerate(test_shards):
        row = {
            "client": int(cid),
            "test_samples": int(shard.size),
            "train_samples": int(shards[cid].size),
            "head_updates": None,
            "finetune_epochs": int(epochs),
        }
        if shard.size == 0:
            row["accuracy"] = None
            rows.append(row)
            continue
        trainer.model.set_weights(global_full)
        trainer.reset_optimizers()
        idx = shards[cid]
        trainer.fit(
            train.x[idx],
            train.y[idx],
            [("head", epochs)],
            batch_size,
            _client_rng(seed, 0, cid),
        )
        evaluator.set_weights(trainer.model.get_weights())
        row["accuracy"] = accuracy_of(evaluator, test.x[shard], test.y[shard])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Population loading
# ---------------------------------------------------------------------------


def load_population(dataset: str, num_clients: int, alpha: float, seed: int):
    """``(train, test, shards, test_shards, model_name)`` for a dataset."""
    from fl.config import DATASET_MODEL, DataConfig
    from fl.data import load_federated_per_client

    cfg = DataConfig(
        dataset=dataset,
        num_clients=num_clients,
        partition="natural" if dataset == "femnist" else "dirichlet",
        dirichlet_alpha=alpha,
    )
    cfg.validate()
    train, test, shards, test_shards = load_federated_per_client(cfg, seed=seed)
    return train, test, shards, test_shards, DATASET_MODEL[dataset]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One personalization run.")
    parser.add_argument("--dataset", default="femnist", choices=("femnist", "fashion_mnist"))
    parser.add_argument("--method", default="fedrep", choices=METHODS)
    parser.add_argument("--clients", type=int, default=1000)
    parser.add_argument("--cohort", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    train, test, shards, test_shards, model_name = load_population(
        args.dataset, args.clients, args.alpha, args.seed
    )
    result = simulate(
        model_name=model_name,
        train=train,
        test=test,
        shards=shards,
        test_shards=test_shards,
        method=args.method,
        clients_per_round=args.cohort,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        head_epochs=args.head_epochs,
        seed=args.seed,
        label=f"{args.dataset}/{args.method}/seed={args.seed}",
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
