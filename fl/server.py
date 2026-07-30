"""Federated learning coordinator.

Round structure
---------------
Each round the server:

1. samples a fraction ``C`` of the *registered* clients (never all of them, unless
   ``C == 1``) -- partial participation is what makes this federated rather than
   distributed;
2. publishes the current global weights together with the ``model_version`` they
   correspond to;
3. opens a barrier with a wall-clock deadline;
4. aggregates whatever arrived in time and drops the rest, logging every drop;
5. evaluates the new global model on the server-held test set;
6. increments ``model_version``.

Stragglers and staleness
------------------------
The deadline is enforced, not advisory. A client that misses it is dropped from
that round and the round proceeds without it -- a federated server that blocks on
its slowest participant has no availability story at all. Dropped clients are
logged individually so the drop rate is visible rather than inferred.

Late work is refused rather than folded in. An update trained from model *N* is
only valid input to the aggregation producing model *N+1*; applying it to model
*N+3* would mix in a gradient computed against weights that no longer exist. Such
updates are rejected with ``REJECTED_STALE_MODEL``.

The test set never leaves this process. Clients receive training indices only.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent import futures
from dataclasses import dataclass, field

import grpc
import numpy as np

from .aggregation import AggregationError, Aggregator, ClientUpdate, Weights
from .config import Config
from .proto import fl_comm_pb2, fl_comm_pb2_grpc
from .serialization import SerializationError, proto_nbytes, proto_to_weights, weights_to_proto

LOGGER = logging.getLogger("fl.server")

_PB = fl_comm_pb2


@dataclass
class RoundMetrics:
    """Everything measured about one round."""

    round: int
    model_version: int
    accuracy: float
    loss: float
    duration_seconds: float
    bytes_sent: int
    bytes_received: int
    num_selected: int
    num_reported: int
    num_dropped: int
    dropped_clients: list[str] = field(default_factory=list)
    aggregated: bool = True
    epsilon: float | None = None

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "duration_seconds": round(self.duration_seconds, 4),
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "num_selected": self.num_selected,
            "num_reported": self.num_reported,
            "num_dropped": self.num_dropped,
            "dropped_clients": list(self.dropped_clients),
            "aggregated": self.aggregated,
            "epsilon": self.epsilon,
        }


@dataclass
class _ClientRecord:
    client_id: str
    shard_index: int


class FederatedServer(fl_comm_pb2_grpc.FederatedLearningServicer):
    """gRPC servicer plus the round orchestrator that drives it."""

    def __init__(
        self,
        config: Config,
        initial_weights: Weights,
        aggregator: Aggregator,
        evaluate_fn: Callable[[Weights], tuple[float, float]],
        *,
        epsilon_fn: Callable[[int], float] | None = None,
    ) -> None:
        """
        Args:
            config: Validated run configuration.
            initial_weights: Global model weights at version 0.
            aggregator: Strategy combining client updates.
            evaluate_fn: Maps global weights to ``(loss, accuracy)`` on the
                server-held test set. Injected so tests need no real model.
            epsilon_fn: Maps a completed round count to cumulative epsilon.
                ``None`` when differential privacy is disabled.
        """
        self.config = config
        self.aggregator = aggregator
        self.evaluate_fn = evaluate_fn
        self.epsilon_fn = epsilon_fn

        self._lock = threading.Condition()
        self._clients: dict[str, _ClientRecord] = {}
        self._next_shard = 0

        self._global_weights: Weights = [
            np.array(w, dtype=np.float32, copy=True) for w in initial_weights
        ]
        self._model_version = 0
        self._round = 0
        self._cohort: set[str] = set()
        self._deadline_monotonic: float | None = None
        self._updates: dict[str, ClientUpdate] = {}
        #: Cohort members that answered this round at all, valid payload or not.
        self._responded: set[str] = set()
        self._finished = False

        self._bytes_sent = 0
        self._bytes_received = 0

        self.metrics: list[RoundMetrics] = []
        self._rng = np.random.default_rng(config.seed)
        self._grpc_server: grpc.Server | None = None
        self.port: int | None = None

    # -- properties ---------------------------------------------------------

    @property
    def num_registered(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def model_version(self) -> int:
        with self._lock:
            return self._model_version

    def global_weights(self) -> Weights:
        with self._lock:
            return [w.copy() for w in self._global_weights]

    # -- gRPC surface -------------------------------------------------------

    def Register(self, request, context):  # noqa: N802  (gRPC naming)
        if request.protocol_version != _PB.PROTOCOL_VERSION_V1:
            LOGGER.warning(
                "rejecting client with protocol_version=%s (server speaks %s)",
                request.protocol_version,
                _PB.PROTOCOL_VERSION_V1,
            )
            return _PB.RegisterResponse(
                accepted=False,
                rejection_reason=(
                    f"protocol version mismatch: client sent {request.protocol_version}, "
                    f"server speaks {_PB.PROTOCOL_VERSION_V1}"
                ),
            )

        with self._lock:
            requested = request.desired_client_id
            if requested and requested in self._clients:
                # Reconnecting client reclaims its own shard rather than being
                # handed a second one, which would double-count its data.
                record = self._clients[requested]
                LOGGER.info(
                    "client %s re-registered (shard %d)", record.client_id, record.shard_index
                )
                return _PB.RegisterResponse(
                    accepted=True,
                    client_id=record.client_id,
                    shard_index=record.shard_index,
                    num_clients=self.config.data.num_clients,
                )

            if self._next_shard >= self.config.data.num_clients:
                return _PB.RegisterResponse(
                    accepted=False,
                    rejection_reason=(f"all {self.config.data.num_clients} shards already claimed"),
                )

            client_id = requested or f"client-{self._next_shard}"
            if client_id in self._clients:
                client_id = f"{client_id}-{self._next_shard}"
            record = _ClientRecord(client_id=client_id, shard_index=self._next_shard)
            self._clients[client_id] = record
            self._next_shard += 1
            LOGGER.info(
                "registered %s -> shard %d (%d/%d)",
                client_id,
                record.shard_index,
                len(self._clients),
                self.config.data.num_clients,
            )
            self._lock.notify_all()

        return _PB.RegisterResponse(
            accepted=True,
            client_id=record.client_id,
            shard_index=record.shard_index,
            num_clients=self.config.data.num_clients,
        )

    def GetGlobalModel(self, request, context):  # noqa: N802
        with self._lock:
            if request.client_id not in self._clients:
                context.abort(grpc.StatusCode.NOT_FOUND, f"unknown client {request.client_id!r}")

            if self._finished:
                return _PB.GetGlobalModelResponse(
                    action=_PB.ROUND_ACTION_STOP,
                    round=self._round,
                    model_version=self._model_version,
                )

            # A client that is not sampled waits. So does one that has already
            # answered this round: without this it would poll, see the round still
            # open, and retrain the same round repeatedly -- burning local compute
            # and re-sending the global model on every pass.
            #
            # The check is against _responded, not _updates. A client whose update
            # was rejected has still had its turn; keying on _updates would keep
            # handing it the 900 KB model on every poll for the rest of the round.
            # Measured cost of getting this wrong: 1.5 GB of server->client
            # traffic in a 20-round DP run whose honest total is 90 MB.
            already_answered = request.client_id in self._responded
            if self._round == 0 or request.client_id not in self._cohort or already_answered:
                return _PB.GetGlobalModelResponse(
                    action=_PB.ROUND_ACTION_WAIT,
                    round=self._round,
                    model_version=self._model_version,
                )

            weights_msg = weights_to_proto(self._global_weights)
            sent = proto_nbytes(weights_msg)
            self._bytes_sent += sent
            remaining = 0.0
            if self._deadline_monotonic is not None:
                remaining = max(0.0, self._deadline_monotonic - time.monotonic())

            t = self.config.training
            return _PB.GetGlobalModelResponse(
                action=_PB.ROUND_ACTION_TRAIN,
                round=self._round,
                model_version=self._model_version,
                weights=weights_msg,
                seconds_until_deadline=remaining,
                local_epochs=t.local_epochs,
                batch_size=t.batch_size,
                learning_rate=t.learning_rate,
                momentum=t.momentum,
            )

    def SubmitUpdate(self, request, context):  # noqa: N802
        with self._lock:
            current_version = self._model_version

            if request.client_id not in self._clients:
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_UNKNOWN_CLIENT,
                    detail=f"unknown client {request.client_id!r}",
                    current_model_version=current_version,
                )

            # Staleness is checked before selection and before the deadline: an
            # update computed against a superseded model is wrong regardless of
            # whether it arrived on time.
            if request.model_version != current_version:
                LOGGER.warning(
                    "rejecting stale update from %s: trained from v%d, server holds v%d",
                    request.client_id,
                    request.model_version,
                    current_version,
                )
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_STALE_MODEL,
                    detail=(
                        f"update was trained from model_version {request.model_version}, "
                        f"server holds {current_version}"
                    ),
                    current_model_version=current_version,
                )

            if self._round == 0 or self._finished or request.round != self._round:
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_DEADLINE_PASSED,
                    detail=f"round {request.round} is not open (server is on round {self._round})",
                    current_model_version=current_version,
                )

            if request.client_id not in self._cohort:
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_NOT_SELECTED,
                    detail=f"client {request.client_id!r} was not sampled for round {self._round}",
                    current_model_version=current_version,
                )

            if self._deadline_monotonic is not None and time.monotonic() > self._deadline_monotonic:
                LOGGER.warning(
                    "rejecting update from %s: %.2fs past the round %d deadline",
                    request.client_id,
                    time.monotonic() - self._deadline_monotonic,
                    self._round,
                )
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_DEADLINE_PASSED,
                    detail=f"round {self._round} deadline has passed",
                    current_model_version=current_version,
                )

            # The client answered for this round. Record that separately from
            # whether its payload was usable: the barrier waits on *responses*,
            # so a cohort that all fail validation closes the round immediately
            # instead of every client going silent and the round burning its
            # full deadline. Only genuinely unreachable clients cost a timeout.
            self._responded.add(request.client_id)
            self._lock.notify_all()

            try:
                weights = proto_to_weights(request.weights)
            except SerializationError as exc:
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD,
                    detail=str(exc),
                    current_model_version=current_version,
                )

            if request.num_examples <= 0:
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD,
                    detail=f"num_examples must be positive, got {request.num_examples}",
                    current_model_version=current_version,
                )

            if not all(np.all(np.isfinite(w)) for w in weights):
                LOGGER.warning("rejecting non-finite update from %s", request.client_id)
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD,
                    detail="weights contain NaN or Inf",
                    current_model_version=current_version,
                )

            expected = [w.shape for w in self._global_weights]
            got = [w.shape for w in weights]
            if got != expected:
                return _PB.SubmitUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD,
                    detail=f"weight shapes {got} do not match global model {expected}",
                    current_model_version=current_version,
                )

            self._bytes_received += request.ByteSize()
            self._updates[request.client_id] = ClientUpdate(
                client_id=request.client_id,
                weights=weights,
                num_examples=int(request.num_examples),
                model_version=int(request.model_version),
            )
            LOGGER.info(
                "round %d: accepted update from %s (n=%d, %d/%d in)",
                self._round,
                request.client_id,
                request.num_examples,
                len(self._updates),
                len(self._cohort),
            )
            self._lock.notify_all()

        return _PB.SubmitUpdateResponse(
            status=_PB.UPDATE_STATUS_ACCEPTED, current_model_version=current_version
        )

    # -- lifecycle ----------------------------------------------------------

    def start(self, max_workers: int = 16) -> int:
        """Bind and start the gRPC server. Returns the bound port."""
        limit = self.config.server.max_message_mb * 1024 * 1024
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_send_message_length", limit),
                ("grpc.max_receive_message_length", limit),
            ],
        )
        fl_comm_pb2_grpc.add_FederatedLearningServicer_to_server(self, self._grpc_server)
        self.port = self._grpc_server.add_insecure_port(
            f"{self.config.server.host}:{self.config.server.port}"
        )
        if self.port == 0:
            raise RuntimeError(
                f"failed to bind {self.config.server.host}:{self.config.server.port}"
            )
        self._grpc_server.start()
        LOGGER.info("server listening on %s:%d", self.config.server.host, self.port)
        return self.port

    def stop(self, grace: float = 0.5) -> None:
        if self._grpc_server is not None:
            self._grpc_server.stop(grace)
            self._grpc_server = None

    def wait_for_clients(self, target: int | None = None, timeout: float | None = None) -> int:
        """Block until ``target`` clients register, or the timeout expires."""
        target = target if target is not None else self.config.data.num_clients
        timeout = (
            timeout if timeout is not None else self.config.server.registration_timeout_seconds
        )
        deadline = time.monotonic() + timeout
        with self._lock:
            while len(self._clients) < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._lock.wait(remaining)
            return len(self._clients)

    # -- orchestration ------------------------------------------------------

    def _sample_cohort(self) -> list[str]:
        """Sample ``ceil(C * N)`` distinct clients uniformly without replacement."""
        with self._lock:
            registered = sorted(self._clients)
        k = min(len(registered), self.config.clients_per_round)
        chosen = self._rng.choice(len(registered), size=k, replace=False)
        return [registered[int(i)] for i in sorted(chosen)]

    def run_rounds(self) -> list[RoundMetrics]:
        """Drive every configured round to completion."""
        registered = self.wait_for_clients()
        if registered < self.config.server.min_clients_per_round:
            raise RuntimeError(
                f"only {registered} client(s) registered before the timeout; "
                f"need at least {self.config.server.min_clients_per_round}"
            )
        LOGGER.info(
            "starting %d rounds with %d registered clients", self.config.training.rounds, registered
        )

        for round_index in range(1, self.config.training.rounds + 1):
            self._run_one_round(round_index)

        with self._lock:
            self._finished = True
            self._lock.notify_all()
        return self.metrics

    def _run_one_round(self, round_index: int) -> RoundMetrics:
        started = time.monotonic()
        cohort = self._sample_cohort()

        with self._lock:
            self._round = round_index
            self._cohort = set(cohort)
            self._updates = {}
            self._responded = set()
            self._deadline_monotonic = started + self.config.server.round_deadline_seconds
            bytes_sent_at_start = self._bytes_sent
            bytes_received_at_start = self._bytes_received
            self._lock.notify_all()

        LOGGER.info(
            "round %d: sampled %d/%d clients (C=%.2f): %s",
            round_index,
            len(cohort),
            self.num_registered,
            self.config.training.client_fraction,
            ", ".join(cohort),
        )

        # Barrier: release once every sampled client has answered, or at the
        # deadline. Waiting on responses rather than on accepted updates means a
        # round in which every client fails validation closes at once; only
        # genuinely silent clients cost the full timeout.
        with self._lock:
            while len(self._responded) < len(cohort):
                remaining = self._deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    break
                self._lock.wait(remaining)

            reported = dict(self._updates)
            silent = sorted(set(cohort) - self._responded)
            self._deadline_monotonic = time.monotonic()  # close the barrier

        dropped = sorted(set(cohort) - set(reported))
        for client_id in dropped:
            if client_id in silent:
                LOGGER.warning(
                    "round %d: dropping %s -- silent, no response before the %.1fs deadline",
                    round_index,
                    client_id,
                    self.config.server.round_deadline_seconds,
                )
            else:
                LOGGER.warning(
                    "round %d: dropping %s -- responded but its update was rejected",
                    round_index,
                    client_id,
                )

        aggregated = False
        if len(reported) < self.config.server.min_clients_per_round:
            LOGGER.error(
                "round %d: only %d/%d clients reported, below quorum of %d; "
                "keeping the previous global model",
                round_index,
                len(reported),
                len(cohort),
                self.config.server.min_clients_per_round,
            )
        else:
            try:
                new_weights = self.aggregator.aggregate(
                    list(reported.values()), self.global_weights()
                )
                with self._lock:
                    self._global_weights = [np.asarray(w, dtype=np.float32) for w in new_weights]
                    self._model_version += 1
                aggregated = True
            except AggregationError:
                LOGGER.exception(
                    "round %d: aggregation failed; keeping previous model", round_index
                )

        loss, accuracy = self.evaluate_fn(self.global_weights())
        duration = time.monotonic() - started

        with self._lock:
            bytes_sent = self._bytes_sent - bytes_sent_at_start
            bytes_received = self._bytes_received - bytes_received_at_start
            version = self._model_version

        epsilon = self.epsilon_fn(round_index) if self.epsilon_fn is not None else None

        metrics = RoundMetrics(
            round=round_index,
            model_version=version,
            accuracy=accuracy,
            loss=loss,
            duration_seconds=duration,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            num_selected=len(cohort),
            num_reported=len(reported),
            num_dropped=len(dropped),
            dropped_clients=dropped,
            aggregated=aggregated,
            epsilon=epsilon,
        )
        self.metrics.append(metrics)

        LOGGER.info(
            "round %d: acc=%.4f loss=%.4f %.2fs  server->clients=%s clients->server=%s  "
            "reported=%d dropped=%d%s",
            round_index,
            accuracy,
            loss,
            duration,
            _human_bytes(bytes_sent),
            _human_bytes(bytes_received),
            len(reported),
            len(dropped),
            f"  eps={epsilon:.3f}" if epsilon is not None and math.isfinite(epsilon) else "",
        )
        return metrics


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GiB"


def build_evaluator(model_name: str, test_x: np.ndarray, test_y: np.ndarray, batch_size: int = 256):
    """Return an ``evaluate_fn`` that scores weights on the server-held test set."""
    from .models import build_model, compile_for_evaluation

    model = compile_for_evaluation(build_model(model_name))

    def evaluate(weights: Weights) -> tuple[float, float]:
        model.set_weights(weights)
        loss, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
        return float(loss), float(accuracy)

    return evaluate


def build_server(config: Config) -> FederatedServer:
    """Assemble a server from a config: model, held-out test set, aggregator."""
    from .aggregation import compute_epsilon, make_aggregator
    from .data import load_fashion_mnist
    from .models import build_model

    _train, test = load_fashion_mnist()
    LOGGER.info("server holds %d held-out test examples; no client sees them", len(test))

    initial_weights = build_model(config.model.name, seed=config.seed).get_weights()
    aggregator = make_aggregator(
        dp_enabled=config.privacy.enabled,
        noise_multiplier=config.privacy.noise_multiplier,
        l2_clip_norm=config.privacy.l2_clip_norm,
        clients_per_round=config.clients_per_round,
    )

    epsilon_fn = None
    if config.privacy.enabled:

        def epsilon_fn(completed_rounds: int) -> float:
            """Cumulative epsilon after ``completed_rounds`` rounds."""
            return compute_epsilon(
                noise_multiplier=config.privacy.noise_multiplier,
                sampling_rate=config.client_sampling_rate,
                rounds=completed_rounds,
                delta=config.privacy.delta,
            )

        LOGGER.info(
            "client-level DP enabled: noise_multiplier=%.3f, l2_clip_norm=%.3f, q=%.3f, "
            "delta=%.1e -> epsilon after %d rounds will be %.3f",
            config.privacy.noise_multiplier,
            config.privacy.l2_clip_norm,
            config.client_sampling_rate,
            config.privacy.delta,
            config.training.rounds,
            epsilon_fn(config.training.rounds),
        )

    return FederatedServer(
        config=config,
        initial_weights=initial_weights,
        aggregator=aggregator,
        evaluate_fn=build_evaluator(config.model.name, test.x, test.y),
        epsilon_fn=epsilon_fn,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Federated learning server.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    parser.add_argument("--metrics-out", default=None, help="Write per-round metrics JSON here.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = Config.from_yaml(args.config)
    server = build_server(config)
    server.start()
    try:
        metrics = server.run_rounds()
    finally:
        server.stop()

    if metrics:
        final = metrics[-1]
        LOGGER.info("final accuracy %.4f after %d rounds", final.accuracy, len(metrics))

    if args.metrics_out:
        payload = {
            "config": config.to_dict(),
            "aggregator": server.aggregator.name,
            "rounds": [m.to_dict() for m in metrics],
        }
        Path(args.metrics_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("wrote metrics to %s", args.metrics_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
