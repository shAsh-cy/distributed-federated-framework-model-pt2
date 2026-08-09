"""Background run execution: the training loop as an event producer.

The API-never-blocks rule, implemented structurally: each run executes on its
own daemon thread and pushes events through :class:`EventStore.append`. The
API layer consumes those events (from the database or the hub); it holds no
reference into the loop and cannot influence its timing. The only channel in
the other direction is a stop flag, which the loop polls at its own safe
points.

Stop semantics, because "graceful" needs a definition:

* Stop arriving **between rounds** — the loop notices before emitting the next
  ``round_started``; the run ends cleanly with ``run_completed``
  (``stopped_early=True``) and no partial round exists in the stream.
* Stop arriving **mid-round** — the client currently fitting finishes (Keras
  fit is not interruptible mid-batch without corrupting optimiser state);
  remaining cohort members are recorded as ``client_dropped`` with reason
  ``stopped``; the partial round is **not aggregated** — a cohort chosen for m
  participants and aggregated over fewer would silently change the FedAvg
  weighting and, under DP, the sensitivity story. The stream shows exactly
  what happened instead.
* A crash anywhere marks the run ``failed`` and emits ``run_failed`` with the
  error; nothing is left hanging in ``running``.

For tests, the executor is injectable: lifecycle tests drive a fake executor
(crash paths, stop timings) without importing TensorFlow; the integration
test runs the real one for two rounds of Fashion-MNIST.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .events import (
    ClientDropped,
    ClientInfo,
    ClientReported,
    ClientSampled,
    RoundAggregated,
    RoundStarted,
    RunCompleted,
    RunFailed,
    RunStarted,
)
from .store import EventStore

LOGGER = logging.getLogger("coordinator.runner")

Executor = Callable[[str, dict, "RunContext"], None]


class RunContext:
    """What an executor is given: an emit function and a stop flag."""

    def __init__(self, store: EventStore, run_id: str) -> None:
        self._store = store
        self.run_id = run_id
        self.stop_requested = threading.Event()

    def emit(self, event) -> None:
        self._store.append(event)

    def should_stop(self) -> bool:
        return self.stop_requested.is_set()


class Runner:
    """Owns run threads. One instance per process, shared with the API."""

    def __init__(self, store: EventStore, executor: Executor | None = None) -> None:
        self._store = store
        self._executor: Executor = executor or _train_real
        self._active: dict[str, RunContext] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(self, config: dict) -> str:
        """Validate the config, create the run, spawn its thread, return its id.

        Validation happens *here*, synchronously, so a bad config is a 4xx at
        the API rather than an asynchronous run_failed nobody is watching yet.
        """
        from fl.config import Config

        Config.from_dict(config)  # raises ConfigError on anything invalid

        run_id = self._store.create_run(config, source="live", status="running")
        ctx = RunContext(self._store, run_id)
        thread = threading.Thread(
            target=self._guarded, args=(run_id, config, ctx), daemon=True, name=f"run-{run_id[:8]}"
        )
        with self._lock:
            self._active[run_id] = ctx
            self._threads[run_id] = thread
        thread.start()
        return run_id

    def stop(self, run_id: str) -> bool:
        """Request a graceful stop. True if the run was live to receive it."""
        with self._lock:
            ctx = self._active.get(run_id)
        if ctx is None:
            return False
        ctx.stop_requested.set()
        return True

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._active

    def join(self, run_id: str, timeout: float | None = None) -> None:
        """Test helper: wait for a run's thread."""
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is not None:
            thread.join(timeout=timeout)

    def _guarded(self, run_id: str, config: dict, ctx: RunContext) -> None:
        try:
            self._executor(run_id, config, ctx)
        except Exception as exc:  # noqa: BLE001 - converted to run_failed
            LOGGER.exception("run %s failed", run_id)
            rounds = _rounds_completed(self._store, run_id)
            ctx.emit(
                RunFailed(
                    run_id=run_id,
                    ts=time.time(),
                    error=f"{type(exc).__name__}: {exc}",
                    rounds_completed=rounds,
                )
            )
            self._store.set_status(run_id, "failed")
        finally:
            with self._lock:
                self._active.pop(run_id, None)


def _rounds_completed(store: EventStore, run_id: str) -> int:
    return sum(1 for e in store.events_since(run_id) if e.get("type") == "round_aggregated")


# ---------------------------------------------------------------------------
# The real executor
# ---------------------------------------------------------------------------


def _train_real(run_id: str, config_dict: dict, ctx: RunContext) -> None:
    """In-process federated training over fl/* — the same aggregation code the
    gRPC server runs, without the transport, pushing events as it goes."""
    import numpy as np

    from fl.aggregation import (
        ClientUpdate,
        aggregator_from_config,
        compute_epsilon,
        l2_norm,
        subtract,
    )
    from fl.checkpoint import DEFAULT_CHECKPOINT_DIR, save_checkpoint
    from fl.config import Config
    from fl.data import dataset_num_classes, label_distribution, load_federated
    from fl.models import build_model, compile_for_evaluation, compile_for_training, weights_nbytes

    store_status = ctx._store.set_status  # noqa: SLF001 - runner-internal wiring
    cfg = Config.from_dict(config_dict)
    num_classes = dataset_num_classes(cfg.data.dataset)

    train, test, shards = load_federated(cfg.data, seed=cfg.seed)
    ctx.emit(
        RunStarted(
            run_id=run_id,
            ts=time.time(),
            config=cfg.to_dict(),
            num_classes=num_classes,
            clients=[
                ClientInfo(
                    client_id=f"client-{i}",
                    num_examples=int(s.size),
                    label_histogram=label_distribution(train.y, s, num_classes).tolist(),
                )
                for i, s in enumerate(shards)
            ],
        )
    )

    global_weights = build_model(cfg.model.name, seed=cfg.seed).get_weights()
    trainer = compile_for_training(
        build_model(cfg.model.name), cfg.training.learning_rate, cfg.training.momentum
    )
    evaluator = compile_for_evaluation(build_model(cfg.model.name))
    aggregator = aggregator_from_config(cfg, cfg.clients_per_round)
    rng = np.random.default_rng(cfg.seed)
    payload_bytes = weights_nbytes(global_weights)

    final_acc: float | None = None
    final_loss: float | None = None
    completed = 0
    stopped_early = False

    for rnd in range(1, cfg.training.rounds + 1):
        if ctx.should_stop():  # between rounds: clean, no partial round
            stopped_early = True
            break
        ctx.emit(RoundStarted(run_id=run_id, ts=time.time(), round=rnd, model_version=completed))
        cohort = rng.choice(cfg.data.num_clients, size=cfg.clients_per_round, replace=False)
        updates: list[ClientUpdate] = []
        pre_norms: list[float] = []
        aborted_mid_round = False

        for pos, cid in enumerate(cohort):
            client_id = f"client-{int(cid)}"
            ctx.emit(
                ClientSampled(
                    run_id=run_id,
                    ts=time.time(),
                    round=rnd,
                    client_id=client_id,
                    framework="tensorflow",
                )
            )
            if ctx.should_stop():  # mid-round: drop the rest, do not aggregate
                for later in cohort[pos:]:
                    ctx.emit(
                        ClientDropped(
                            run_id=run_id,
                            ts=time.time(),
                            round=rnd,
                            client_id=f"client-{int(later)}",
                            reason="stopped",
                        )
                    )
                aborted_mid_round = True
                stopped_early = True
                break

            started = time.monotonic()
            idx = shards[int(cid)]
            trainer.set_weights(global_weights)
            history = trainer.fit(
                train.x[idx],
                train.y[idx],
                epochs=cfg.training.local_epochs,
                batch_size=cfg.training.batch_size,
                verbose=0,
            )
            new_weights = trainer.get_weights()
            pre_norms.append(l2_norm(subtract(new_weights, global_weights)))
            updates.append(ClientUpdate(client_id, new_weights, int(idx.size)))
            ctx.emit(
                ClientReported(
                    run_id=run_id,
                    ts=time.time(),
                    round=rnd,
                    client_id=client_id,
                    num_examples=int(idx.size),
                    local_accuracy=float(history.history.get("accuracy", [float("nan")])[-1]),
                    local_loss=float(history.history["loss"][-1]),
                    wall_clock_seconds=round(time.monotonic() - started, 3),
                    bytes=payload_bytes,
                )
            )

        if aborted_mid_round:
            break

        global_weights = aggregator.aggregate(updates, global_weights)
        evaluator.set_weights(global_weights)
        loss, acc = evaluator.evaluate(test.x, test.y, batch_size=512, verbose=0)
        completed += 1
        final_acc, final_loss = float(acc), float(loss)
        epsilon = (
            compute_epsilon(
                cfg.privacy.noise_multiplier,
                cfg.client_sampling_rate,
                completed,
                cfg.privacy.delta,
            )
            if cfg.privacy.enabled
            else None
        )
        ctx.emit(
            RoundAggregated(
                run_id=run_id,
                ts=time.time(),
                round=rnd,
                model_version=completed,
                global_accuracy=final_acc,
                global_loss=final_loss,
                bytes_sent=payload_bytes * len(cohort),
                bytes_received=payload_bytes * len(updates),
                cumulative_epsilon=epsilon,
                median_update_norm=float(np.median(pre_norms)) if pre_norms else None,
                clipped_fraction=(
                    float(np.mean([n > cfg.privacy.l2_clip_norm for n in pre_norms]))
                    if pre_norms and cfg.privacy.enabled
                    else None
                ),
            )
        )

    # The run leaves a loadable model behind, not just metrics (audit M2).
    checkpoint = save_checkpoint(
        DEFAULT_CHECKPOINT_DIR / f"{run_id}.npz",
        global_weights,
        model_name=cfg.model.name,
        config=cfg.to_dict(),
        metadata={"rounds_completed": completed, "stopped_early": stopped_early},
    )
    LOGGER.info("run %s checkpoint: %s", run_id, checkpoint)

    ctx.emit(
        RunCompleted(
            run_id=run_id,
            ts=time.time(),
            final_accuracy=final_acc,
            final_loss=final_loss,
            rounds_completed=completed,
            stopped_early=stopped_early,
        )
    )
    store_status(
        run_id,
        "stopped" if stopped_early else "completed",
        final_metrics={
            "final_accuracy": final_acc,
            "final_loss": final_loss,
            "rounds_completed": completed,
            "stopped_early": stopped_early,
            "checkpoint": str(checkpoint),
        },
    )
