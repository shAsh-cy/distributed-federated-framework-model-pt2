"""Federated learning participant.

A client owns exactly one shard of the training set and never sees the test set.
Its loop is: register once, then poll the server -- train when sampled, wait when
not, exit when told.

Local training hyperparameters come from the server in the ``GetGlobalModel``
response rather than from the client's own config. Otherwise every client would
need to be configured consistently by hand, and one misconfigured participant
would quietly train with a different learning rate for the whole run.
"""

from __future__ import annotations

import argparse
import logging
import time

import grpc
import numpy as np

from .config import Config
from .data import load_federated
from .models import build_model, compile_for_training
from .proto import fl_comm_pb2, fl_comm_pb2_grpc
from .serialization import proto_to_weights, weights_to_proto

LOGGER = logging.getLogger("fl.client")

_PB = fl_comm_pb2


class RegistrationError(RuntimeError):
    """Raised when the server refuses to register this client."""


class FederatedClient:
    """One federated participant."""

    def __init__(self, config: Config, server_address: str, desired_client_id: str = "") -> None:
        self.config = config
        self.server_address = server_address
        self.desired_client_id = desired_client_id

        limit = config.server.max_message_mb * 1024 * 1024
        self.channel = grpc.insecure_channel(
            server_address,
            options=[
                ("grpc.max_send_message_length", limit),
                ("grpc.max_receive_message_length", limit),
            ],
        )
        self.stub = fl_comm_pb2_grpc.FederatedLearningStub(self.channel)

        self.client_id: str | None = None
        self.shard_index: int | None = None
        self.model = None
        self.x = None
        self.y = None
        # (round, model_version) pairs already attempted. A rejected submission
        # must not be retried against the same global model: the rejection is
        # deterministic, so retrying re-trains and re-sends forever, holding the
        # round open until its deadline. Seen in practice as a diverged client
        # retrying an INVALID_PAYLOAD 12 times and moving 157 MiB in one round.
        self._attempted: set[tuple[int, int]] = set()

    # -- setup --------------------------------------------------------------

    def register(self) -> str:
        """Register with the server and claim a shard."""
        response = self.stub.Register(
            _PB.RegisterRequest(
                protocol_version=_PB.PROTOCOL_VERSION_V1,
                desired_client_id=self.desired_client_id,
            )
        )
        if not response.accepted:
            raise RegistrationError(f"server refused registration: {response.rejection_reason}")
        self.client_id = response.client_id
        self.shard_index = response.shard_index
        LOGGER.info(
            "registered as %s, shard %d of %d",
            self.client_id,
            self.shard_index,
            response.num_clients,
        )
        return self.client_id

    def load_data(self, train=None, shards=None) -> None:
        """Load this client's shard. The test split is loaded and discarded.

        Args:
            train: Pre-loaded training :class:`~fl.data.Dataset`. Supplied by the
                in-process experiment runner so ten simultaneous clients share
                one copy of Fashion-MNIST instead of loading ~190 MB each.
            shards: Pre-computed partition, for the same reason. Deriving it from
                the same config and seed yields an identical split either way.
        """
        if self.shard_index is None:
            raise RuntimeError("register() must be called before load_data()")

        if train is None or shards is None:
            train, _test_held_by_server_only, shards = load_federated(
                self.config.data, seed=self.config.seed
            )
        shard = train.take(shards[self.shard_index])
        self.x, self.y = shard.x, shard.y
        LOGGER.info(
            "client %s holds %d examples across %d classes",
            self.client_id,
            len(self.y),
            len(np.unique(self.y)),
        )

    def _ensure_model(self, learning_rate: float, momentum: float):
        if self.model is None:
            self.model = compile_for_training(
                build_model(self.config.model.name), learning_rate, momentum
            )
        return self.model

    # -- one round ----------------------------------------------------------

    def train_one_round(self, response) -> tuple[list[np.ndarray], int, float, float]:
        """Run local training from the published global weights."""
        model = self._ensure_model(response.learning_rate, response.momentum)
        model.set_weights(proto_to_weights(response.weights))

        history = model.fit(
            self.x,
            self.y,
            epochs=max(1, response.local_epochs),
            batch_size=max(1, response.batch_size),
            verbose=0,
        )
        loss = float(history.history["loss"][-1])
        accuracy = float(history.history.get("accuracy", [float("nan")])[-1])
        return model.get_weights(), int(len(self.y)), loss, accuracy

    def submit(self, round_index: int, model_version: int, weights, num_examples, loss, accuracy):
        request = _PB.SubmitUpdateRequest(
            client_id=self.client_id,
            round=round_index,
            model_version=model_version,
            weights=weights_to_proto(weights),
            num_examples=num_examples,
            train_loss=loss,
            train_accuracy=accuracy,
        )
        return self.stub.SubmitUpdate(request)

    # -- main loop ----------------------------------------------------------

    def run(
        self,
        poll_interval: float = 0.5,
        max_idle_polls: int = 100_000,
        max_unreachable_polls: int = 20,
    ) -> None:
        """Poll the server until it says to stop, or until it goes away."""
        if self.client_id is None:
            self.register()
        if self.x is None:
            self.load_data()

        idle = 0
        unreachable = 0
        while True:
            try:
                response = self.stub.GetGlobalModel(
                    _PB.GetGlobalModelRequest(client_id=self.client_id)
                )
                unreachable = 0
            except ValueError:
                # close() was called while this loop was running. Exiting quietly
                # is correct: a shut-down client has nothing to report, and
                # raising here would surface a spurious error from a daemon
                # thread during teardown.
                LOGGER.info("client %s: channel closed, exiting", self.client_id)
                return
            except grpc.RpcError as exc:
                # The server going away is the normal end of a run, not a crash:
                # it stops serving once the last round is aggregated, and a
                # client polling a moment later sees UNAVAILABLE rather than
                # STOP. Treating that as a failure makes every client exit
                # non-zero, and under `restart: on-failure` they then crash-loop
                # forever after a perfectly successful run. Retry briefly to ride
                # out a genuine blip, then exit cleanly.
                unreachable += 1
                if unreachable > max_unreachable_polls:
                    LOGGER.info(
                        "client %s: server unreachable after %d attempts (%s); exiting",
                        self.client_id,
                        unreachable,
                        exc.code(),
                    )
                    return
                time.sleep(poll_interval)
                continue

            if response.action == _PB.ROUND_ACTION_STOP:
                LOGGER.info("client %s: server signalled stop", self.client_id)
                return

            attempt_key = (response.round, response.model_version)
            already_tried = attempt_key in self._attempted

            if response.action != _PB.ROUND_ACTION_TRAIN or already_tried:
                idle += 1
                if idle > max_idle_polls:
                    LOGGER.warning("client %s: idle limit reached, exiting", self.client_id)
                    return
                time.sleep(poll_interval)
                continue

            idle = 0
            self._attempted.add(attempt_key)
            started = time.monotonic()
            weights, n, loss, accuracy = self.train_one_round(response)
            elapsed = time.monotonic() - started

            # Local divergence (training on a global model already destroyed by
            # excessive DP noise, say) is reported rather than hidden. Going
            # silent would look identical to a crashed client and would hold the
            # round open until its deadline; submitting lets the server reject
            # the payload and close the barrier immediately.
            if not all(np.all(np.isfinite(w)) for w in weights):
                LOGGER.error(
                    "client %s: local training diverged to NaN/Inf in round %d; "
                    "submitting so the server can reject it and close the round",
                    self.client_id,
                    response.round,
                )

            reply = self.submit(response.round, response.model_version, weights, n, loss, accuracy)
            status_name = _PB.UpdateStatus.Name(reply.status)
            if reply.status == _PB.UPDATE_STATUS_ACCEPTED:
                LOGGER.info(
                    "client %s: round %d update accepted (n=%d, loss=%.4f, acc=%.4f, %.1fs)",
                    self.client_id,
                    response.round,
                    n,
                    loss,
                    accuracy,
                    elapsed,
                )
            else:
                LOGGER.warning(
                    "client %s: round %d update %s -- %s",
                    self.client_id,
                    response.round,
                    status_name,
                    reply.detail,
                )
            # Wait for the server to publish the next model before polling again,
            # otherwise a fast client spins re-training the same round.
            time.sleep(poll_interval)

    def close(self) -> None:
        self.channel.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Federated learning client.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    parser.add_argument(
        "--server", default=None, help="host:port of the server (default: from config)."
    )
    parser.add_argument(
        "--client-id", default="", help="Stable client id, so a restart reclaims its shard."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = Config.from_yaml(args.config)
    address = args.server or f"{config.server.host}:{config.server.port}"

    client = FederatedClient(config, address, desired_client_id=args.client_id)
    try:
        client.register()
        client.load_data()
        client.run()
    finally:
        client.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
