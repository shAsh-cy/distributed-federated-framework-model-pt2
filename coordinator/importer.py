"""Historical import: the repo's committed result files become runs.

The repo carries the full Fashion-MNIST experiment record (three shipped gRPC
runs, six pure-vs-mixed framework comparison runs, the diagnosis sweeps) and
40+ FEMNIST harness runs as JSON under results/ and docs/. This importer loads
them into the runs table so history and results views show real data on first
launch, without a 45-minute training job.

Mapping rules, honest by construction:

* A historical run with a genuine per-round record (``rounds`` from the gRPC
  server, ``history`` from the in-process harness, ``per_epoch`` from the
  pooled baseline) becomes a run with ``run_started`` /
  ``round_started`` / ``round_aggregated`` / ``run_completed`` events — fields
  that were never measured stay None. **No client_* events are fabricated**:
  the historical files do not record per-client participation, so the stream
  does not pretend they do.
* A record with only summary statistics becomes a completed run with final
  metrics and NO events.
* Every imported row is marked ``source="imported"``; nothing imported can be
  mistaken for a live run.
* Multi-seed cells (the FEMNIST sweep/control/bracket cells, the replication)
  keep their per-seed runs AND gain one ``is_aggregate`` summary row per cell
  carrying the mean, the range and the per-seed values — **ranges are
  preserved, never collapsed to a point value.**

The importer is idempotent per label: re-running skips labels already present.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .events import (
    RoundAggregated,
    RoundStarted,
    RunCompleted,
    RunStarted,
)
from .store import EventStore

LOGGER = logging.getLogger("coordinator.importer")


def _emit_round_events(
    store: EventStore, run_id: str, config: dict, rounds: list[dict], final: dict
) -> None:
    now = time.time()
    store.append(RunStarted(run_id=run_id, ts=now, config=config, num_classes=0, clients=[]))
    for r in rounds:
        idx = int(r.get("round", 0))
        store.append(
            RoundStarted(
                run_id=run_id, ts=now, round=idx, model_version=int(r.get("model_version", idx - 1))
            )
        )
        store.append(
            RoundAggregated(
                run_id=run_id,
                ts=now,
                round=idx,
                model_version=int(r.get("model_version", idx)),
                global_accuracy=r.get("accuracy"),
                global_loss=_finite_or_none(r.get("loss")),
                bytes_sent=r.get("bytes_sent"),
                bytes_received=r.get("bytes_received"),
                cumulative_epsilon=r.get("epsilon"),
                median_update_norm=r.get("median_pre_clip_norm"),
                clipped_fraction=r.get("clipped_fraction"),
            )
        )
    store.append(
        RunCompleted(
            run_id=run_id,
            ts=now,
            final_accuracy=final.get("final_accuracy"),
            final_loss=_finite_or_none(final.get("final_loss")),
            rounds_completed=len(rounds),
        )
    )


def _finite_or_none(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _existing_labels(store: EventStore) -> set[str]:
    return {run.label for run in store.list_runs()}


def _import_single(
    store: EventStore,
    seen: set[str],
    label: str,
    config: dict,
    rounds: list[dict],
    final: dict,
    group_key: str | None = None,
    seed: int | None = None,
) -> int:
    if label in seen:
        return 0
    run_id = store.create_run(
        config,
        source="imported",
        label=label,
        group_key=group_key,
        seed=seed,
        status="completed",
    )
    if rounds:
        _emit_round_events(store, run_id, config, rounds, final)
    store.set_status(run_id, "completed", final_metrics=final)
    seen.add(label)
    return 1


def _import_aggregate(
    store: EventStore,
    seen: set[str],
    label: str,
    group_key: str,
    summary: dict,
    config: dict,
) -> int:
    """One is_aggregate row per multi-seed cell; ranges preserved verbatim."""
    if label in seen:
        return 0
    run_id = store.create_run(
        config,
        source="imported",
        label=label,
        group_key=group_key,
        is_aggregate=True,
        status="completed",
    )
    store.set_status(run_id, "completed", final_metrics=summary)
    seen.add(label)
    return 1


# ---------------------------------------------------------------------------
# Per-file adapters
# ---------------------------------------------------------------------------


def _grpc_result_files(root: Path) -> list[Path]:
    out = sorted((root / "results").glob("*.json"))
    out += sorted((root / "results" / "compare").glob("*.json"))
    return out


def _import_grpc_run(store: EventStore, seen: set[str], path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rounds" not in data or "config" not in data:
        return 0
    label = f"grpc/{path.stem}"
    config = data["config"]
    rounds = data["rounds"]
    last = rounds[-1] if rounds else {}
    final = {
        "final_accuracy": last.get("accuracy"),
        "final_loss": _finite_or_none(last.get("loss")),
        "rounds_completed": len(rounds),
        "epsilon": last.get("epsilon"),
    }
    return _import_single(store, seen, label, config, rounds, final, seed=config.get("seed"))


def _harness_run_config(run: dict, dataset: str) -> dict:
    """Reduced config for harness runs, which never had a full Config object."""
    return {
        "seed": run.get("seed"),
        "dataset": dataset,
        "num_clients": run.get("num_clients"),
        "clients_per_round": run.get("clients_per_round"),
        "rounds": run.get("rounds"),
        "dp": run.get("dp"),
        "noise_multiplier": run.get("noise_multiplier"),
        "l2_clip_norm": run.get("l2_clip_norm"),
        "epsilon": run.get("epsilon"),
        "imported_reduced_config": True,
    }


def _import_harness_run(
    store: EventStore, seen: set[str], run: dict, dataset: str, group_key: str
) -> int:
    label = f"{group_key}/{run.get('label') or 'run'}"
    history = run.get("history") or []
    rounds = [
        {
            "round": h.get("round"),
            "accuracy": h.get("accuracy"),
            "loss": h.get("loss"),
            "median_pre_clip_norm": h.get("median_pre_clip_norm"),
            "clipped_fraction": h.get("clipped_fraction"),
            "epsilon": None,
        }
        for h in history
    ]
    final = {
        "final_accuracy": run.get("final_accuracy"),
        "best_accuracy": run.get("best_accuracy"),
        "rounds_completed": len(rounds),
        "epsilon": run.get("epsilon"),
    }
    return _import_single(
        store,
        seen,
        label,
        _harness_run_config(run, dataset),
        rounds,
        final,
        group_key=group_key,
        seed=run.get("seed"),
    )


def _iter_femnist_cells(payload: dict) -> list[tuple[str, dict, list[dict]]]:
    """Yield (cell_key, summary, runs) triples from the femnist JSON shapes."""
    cells = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        if "cells" in value:
            for cell in value["cells"]:
                ident_bits = [
                    f"{k}={cell[k]}"
                    for k in ("m", "clip", "local_epochs")
                    if k in cell and not isinstance(cell[k], dict)
                ]
                cell_key = f"{key}/{'/'.join(ident_bits) or 'cell'}"
                cells.append((cell_key, cell.get("summary") or {}, cell.get("runs") or []))
        elif "runs" in value and isinstance(value["runs"], list):
            summary = {
                k: value[k] for k in ("mean_final", "range_final", "final_per_seed") if k in value
            }
            cells.append((key, summary, value["runs"]))
        elif "pairs" in value:  # repeatability: pairs of runs per cohort
            for pair in value["pairs"]:
                cell_key = f"{key}/m={pair.get('clients_per_round')}"
                cells.append((cell_key, {}, pair.get("runs") or []))
    return cells


def _import_femnist_file(store: EventStore, seen: set[str], path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = "femnist" if "femnist" in path.stem else "fashion_mnist"
    count = 0
    for cell_key, summary, runs in _iter_femnist_cells(payload):
        group_key = f"{path.stem}/{cell_key}"
        for run in runs:
            if isinstance(run, dict) and ("history" in run or "final_accuracy" in run):
                count += _import_harness_run(store, seen, run, dataset, group_key)
        if summary and ("mean_final" in summary or "range_final" in summary):
            count += _import_aggregate(
                store,
                seen,
                label=f"{group_key}/summary",
                group_key=group_key,
                summary=summary,
                config={"imported_cell_summary": True, "source_file": path.name},
            )
    # Flat structures: baseline runs with per_epoch, plain run lists.
    for key, value in payload.items():
        if isinstance(value, dict) and "runs" in value and "cells" not in value:
            for run in value["runs"]:
                if isinstance(run, dict) and "per_epoch" in run:
                    label = f"{path.stem}/{key}/seed={run.get('seed')}"
                    rounds = [
                        {
                            "round": e.get("epoch"),
                            "accuracy": e.get("accuracy"),
                            "loss": e.get("loss"),
                        }
                        for e in run["per_epoch"]
                    ]
                    final = {
                        "final_accuracy": run.get("final_accuracy"),
                        "best_accuracy": run.get("best_accuracy"),
                        "rounds_completed": len(rounds),
                    }
                    count += _import_single(
                        store,
                        seen,
                        label,
                        {"imported_reduced_config": True, "kind": "pooled_baseline"},
                        rounds,
                        final,
                        group_key=f"{path.stem}/{key}",
                        seed=run.get("seed"),
                    )
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "history" in value[0]
        ):
            for run in value:
                count += _import_harness_run(
                    store, seen, run, dataset, group_key=f"{path.stem}/{key}"
                )
    return count


def import_history(store: EventStore, root: str | Path = ".") -> dict:
    """Import everything recognisable under results/ and docs/. Idempotent."""
    root = Path(root)
    seen = _existing_labels(store)
    imported = 0
    skipped_files: list[str] = []

    for path in _grpc_result_files(root):
        try:
            imported += _import_grpc_run(store, seen, path)
        except Exception as exc:  # noqa: BLE001 - skip-and-report per file
            skipped_files.append(f"{path.name}: {type(exc).__name__}: {exc}")

    for path in sorted((root / "docs").glob("_*.json")):
        try:
            imported += _import_femnist_file(store, seen, path)
        except Exception as exc:  # noqa: BLE001
            skipped_files.append(f"{path.name}: {type(exc).__name__}: {exc}")

    result = {"imported_runs": imported, "skipped": skipped_files}
    LOGGER.info("history import: %s", result)
    return result


def main() -> int:
    """CLI entry point: import the repo's committed results into the default DB.

    ``python -m coordinator.importer`` — idempotent; re-running skips
    already-imported labels. Exists because the README promised history
    "on first launch" while nothing ever invoked this module (audit finding
    D9, docs/audit_v0_2.md).
    """
    import logging as _logging

    from .db import create_all, make_engine

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")
    engine = make_engine()
    create_all(engine)
    result = import_history(EventStore(engine))
    LOGGER.info("done: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
