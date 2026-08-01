"""Shared test scaffolding: an in-process server and a training-free fake client.

The fake client speaks the real protocol over a real gRPC channel but returns
constant weights instead of training a model. That keeps the protocol, barrier
and staleness tests measuring the coordination logic rather than Keras.
"""

from __future__ import annotations

import socket
import threading

import numpy as np

from fl.aggregation import FedAvgAggregator
from fl.config import Config
from fl.proto import fl_comm_pb2, fl_comm_pb2_grpc
from fl.serialization import weights_to_proto
from fl.server import FederatedServer

PB = fl_comm_pb2

#: Small stand-in for a model's weight list.
TEMPLATE: list[np.ndarray] = [
    np.zeros((2, 3), dtype=np.float32),
    np.zeros((3,), dtype=np.float32),
]


def free_port() -> int:
    """Reserve an ephemeral port and release it for the server to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_config(**overrides) -> Config:
    """A small, fast configuration suitable for protocol tests."""
    raw = {
        "seed": 7,
        "data": {"num_clients": 4, "partition": "dirichlet", "dirichlet_alpha": 0.5},
        "training": {"rounds": 2, "client_fraction": 0.5, "batch_size": 8},
        "server": {
            "host": "127.0.0.1",
            "port": free_port(),
            "round_deadline_seconds": 3.0,
            "min_clients_per_round": 2,
            "registration_timeout_seconds": 10.0,
        },
    }
    for section, values in overrides.items():
        if section == "seed":
            raw["seed"] = values
        else:
            raw.setdefault(section, {}).update(values)
    return Config.from_dict(raw)


class RecordingEvaluator:
    """Deterministic stand-in for real evaluation; records every call."""

    def __init__(self) -> None:
        self.calls: list[list[np.ndarray]] = []

    def __call__(self, weights):
        self.calls.append([np.array(w, copy=True) for w in weights])
        # Accuracy climbs with the mean weight so tests can assert movement.
        mean = float(np.mean([float(np.mean(w)) for w in weights]))
        return 1.0 / (1.0 + abs(mean)), min(0.99, abs(mean))


def make_server(config: Config | None = None, aggregator=None, initial=None) -> FederatedServer:
    config = config or make_config()
    return FederatedServer(
        config=config,
        initial_weights=initial if initial is not None else TEMPLATE,
        aggregator=aggregator or FedAvgAggregator(),
        evaluate_fn=RecordingEvaluator(),
    )


class FakeClient:
    """A protocol-correct client that fabricates weights instead of training."""

    def __init__(self, address: str, fill: float = 1.0, num_examples: int = 100):
        import grpc

        self.channel = grpc.insecure_channel(address)
        self.stub = fl_comm_pb2_grpc.FederatedLearningStub(self.channel)
        self.fill = fill
        self.num_examples = num_examples
        self.client_id: str | None = None
        self.shard_index: int | None = None
        self.accepted_rounds: list[int] = []
        self.rejections: list[int] = []

    def register(self, desired: str = "", protocol_version=PB.PROTOCOL_VERSION_V2):
        response = self.stub.Register(
            PB.RegisterRequest(protocol_version=protocol_version, desired_client_id=desired)
        )
        if response.accepted:
            self.client_id = response.client_id
            self.shard_index = response.shard_index
        return response

    def poll(self):
        return self.stub.GetGlobalModel(PB.GetGlobalModelRequest(client_id=self.client_id))

    def submit(self, round_index, model_version, weights=None, num_examples=None):
        weights = weights if weights is not None else [np.full_like(w, self.fill) for w in TEMPLATE]
        return self.stub.SubmitUpdate(
            PB.SubmitUpdateRequest(
                client_id=self.client_id,
                round=round_index,
                model_version=model_version,
                weights=weights_to_proto(weights),
                num_examples=self.num_examples if num_examples is None else num_examples,
            )
        )

    def serve_until_stopped(self, stop_event: threading.Event, poll_interval: float = 0.02):
        """Respond to every TRAIN instruction until told to stop."""
        while not stop_event.is_set():
            try:
                response = self.poll()
            except Exception:
                return
            if response.action == PB.ROUND_ACTION_STOP:
                return
            if response.action == PB.ROUND_ACTION_TRAIN:
                reply = self.submit(response.round, response.model_version)
                if reply.status == PB.UPDATE_STATUS_ACCEPTED:
                    self.accepted_rounds.append(response.round)
                else:
                    self.rejections.append(reply.status)
            stop_event.wait(poll_interval)

    def close(self):
        self.channel.close()


class ServerHarness:
    """Starts a server, runs its rounds on a background thread, cleans up."""

    def __init__(self, config: Config | None = None, aggregator=None, initial=None):
        self.server = make_server(config, aggregator, initial)
        self.address = f"127.0.0.1:{self.server.config.server.port}"
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def __enter__(self) -> ServerHarness:
        self.server.start()
        return self

    def run_rounds_in_background(self) -> threading.Thread:
        def target():
            try:
                self.server.run_rounds()
            except BaseException as exc:  # noqa: BLE001  (surfaced in the test)
                self.error = exc

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        return self._thread

    def __exit__(self, *exc):
        if self._thread is not None:
            self._thread.join(timeout=30)
        self.server.stop(grace=0)
        return False
