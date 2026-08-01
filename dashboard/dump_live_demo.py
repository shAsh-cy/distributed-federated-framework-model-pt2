"""Record a live-run event stream fixture without training.

The topology animation and the e2e test need a stream with client_* events;
the historical files never recorded those, and training is off-limits while
the experiment chain occupies the machine. This script drives the REAL
Runner and EventStore with a scripted executor, so sequencing, schema and
protocol shape are the real system's — while every value is taken from
measured reality rather than invented:

* label histograms: the actual Dirichlet(0.5) partition of Fashion-MNIST
  (pure numpy via fl.data; no training involved),
* payload bytes: the real serialised model size (900,136),
* global accuracy/loss per round: the recorded grpc/no_dp run's first rounds,
* client wall-clocks: the 3.6-5.6 s range measured in the compose logs.

The fixture is labelled "scripted" in its filename and in the dashboard's
mock mode; it is demo choreography over recorded measurements, not a claim
about a training run that happened.

    python dashboard/dump_live_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from coordinator.db import create_all, make_engine  # noqa: E402
from coordinator.events import (  # noqa: E402
    ClientInfo,
    ClientReported,
    ClientSampled,
    ClientDropped,
    RoundAggregated,
    RoundStarted,
    RunCompleted,
    RunStarted,
)
from coordinator.runner import RunContext, Runner  # noqa: E402
from coordinator.store import EventStore  # noqa: E402

OUT = Path("dashboard/fixtures")

#: Recorded per-round global accuracy/loss from results/no_dp.json rounds 1-6.
RECORDED_ROUNDS = [
    (0.6763, 0.9161), (0.7396, 0.7332), (0.7756, 0.6398),
    (0.7935, 0.5934), (0.8095, 0.5522), (0.8215, 0.5194),
]
MODEL_BYTES = 900_136  # serialised small_cnn, measured
WALL_CLOCKS = [4.4, 5.2, 3.8, 4.9, 5.6, 3.6, 4.1, 4.7]  # compose-log range


def scripted(run_id: str, config: dict, ctx: RunContext) -> None:
    from fl.data import label_distribution, load_fashion_mnist, partition

    train, _test = load_fashion_mnist()
    num_clients = 10
    shards = partition(train.y, num_clients, "dirichlet", 0.5, seed=42)
    rng = np.random.default_rng(42)

    ctx.emit(RunStarted(
        run_id=run_id, ts=time.time(), config=config, num_classes=10,
        clients=[
            ClientInfo(
                client_id=f"client-{i}",
                num_examples=int(s.size),
                label_histogram=label_distribution(train.y, s, 10).tolist(),
            )
            for i, s in enumerate(shards)
        ],
    ))
    for rnd, (acc, loss) in enumerate(RECORDED_ROUNDS, start=1):
        ctx.emit(RoundStarted(run_id=run_id, ts=time.time(), round=rnd, model_version=rnd - 1))
        cohort = rng.choice(num_clients, size=5, replace=False)
        # One deadline miss in round 3, so the dashboard's drop path has data.
        dropped = int(cohort[-1]) if rnd == 3 else None
        for k, cid in enumerate(cohort):
            framework = "torch" if int(cid) % 3 == 0 else "tensorflow"
            ctx.emit(ClientSampled(run_id=run_id, ts=time.time(), round=rnd,
                                   client_id=f"client-{int(cid)}", framework=framework))
        for k, cid in enumerate(cohort):
            if int(cid) == dropped:
                ctx.emit(ClientDropped(run_id=run_id, ts=time.time(), round=rnd,
                                       client_id=f"client-{int(cid)}", reason="deadline"))
                continue
            ctx.emit(ClientReported(
                run_id=run_id, ts=time.time(), round=rnd,
                client_id=f"client-{int(cid)}",
                num_examples=int(shards[int(cid)].size),
                local_accuracy=round(min(0.99, acc + rng.normal(0.04, 0.02)), 4),
                local_loss=round(max(0.05, loss + rng.normal(-0.05, 0.05)), 4),
                wall_clock_seconds=WALL_CLOCKS[(rnd + k) % len(WALL_CLOCKS)],
                bytes=MODEL_BYTES,
            ))
        reported = len(cohort) - (1 if dropped is not None else 0)
        ctx.emit(RoundAggregated(
            run_id=run_id, ts=time.time(), round=rnd, model_version=rnd,
            global_accuracy=acc, global_loss=loss,
            bytes_sent=MODEL_BYTES * len(cohort), bytes_received=MODEL_BYTES * reported,
            cumulative_epsilon=None, median_update_norm=0.98, clipped_fraction=None,
        ))
    ctx.emit(RunCompleted(run_id=run_id, ts=time.time(),
                          final_accuracy=RECORDED_ROUNDS[-1][0],
                          final_loss=RECORDED_ROUNDS[-1][1],
                          rounds_completed=len(RECORDED_ROUNDS)))
    ctx._store.set_status(run_id, "completed", final_metrics={  # noqa: SLF001
        "final_accuracy": RECORDED_ROUNDS[-1][0], "rounds_completed": len(RECORDED_ROUNDS),
    })


def main() -> int:
    engine = make_engine(":memory:")
    create_all(engine)
    store = EventStore(engine)
    runner = Runner(store, executor=scripted)
    rid = runner.start({"data": {"num_clients": 10}, "training": {"rounds": 6}})
    runner.join(rid, timeout=120)
    events = store.events_since(rid)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live_demo_scripted_events.json").write_text(
        json.dumps(events, indent=1), encoding="utf-8"
    )
    print("events:", len(events), "->", OUT / "live_demo_scripted_events.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
