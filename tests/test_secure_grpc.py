"""Secure aggregation over REAL gRPC: 5 clients, 2 rounds, one induced dropout.

This is the integration test HALF 1.5 asks for. It exercises the whole V3 secure
path — announce, roster, share routing, masked submit, deadline dropout,
Shamir-backed recovery, unmask — over a real gRPC server and real channels. It is
deliberately TF-free: each client uses a constant-weight fake trainer, so the
test measures the secure protocol and its plumbing, not Keras. It needs the
generated gRPC stubs, so it runs in CI / the Docker image, not on a host without
grpc (see the module note in tests/test_secure_aggregation.py).

The dropped client distributes its shares and then goes silent before its masked
update — the realistic dropout the protocol recovers from, since survivors then
hold the key-seed shares that cancel its orphaned pairwise masks.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from fl.secure_client import SecureFederatedClient  # noqa: E402
from fl.secure_live import WeightUpdate, quantized_weighted_average  # noqa: E402
from fl.secure_server import SecureFederatedServer  # noqa: E402
from tests.helpers import TEMPLATE, RecordingEvaluator, make_config  # noqa: E402


class _FakeTrainer:
    """Returns a fixed weight list regardless of input — the training-free stand-in."""

    def __init__(self, weights: list[np.ndarray]) -> None:
        self._weights = [np.array(w, dtype=np.float32) for w in weights]

    def fit(self, weights, x, y, epochs, batch_size):
        del weights, x, y, epochs, batch_size
        return [w.copy() for w in self._weights], 0.1, 0.9


def _fixed_weights(fill: float) -> list[np.ndarray]:
    return [np.full_like(w, fill, dtype=np.float32) for w in TEMPLATE]


class _DropRoundOneClient(SecureFederatedClient):
    """Distributes its shares every round, but withholds its masked update in
    round 1 — a client that drops after setup, then rejoins in round 2."""

    def _participate(self, response) -> None:
        if response.round == 1:
            roster, my_order = self._roster_from(response)
            from fl.secure_round import ParticipantSession

            participant = ParticipantSession(self.client_id, my_order, seed=self._mask_seed)
            self._send_shares(participant, roster, int(response.threshold), response.round)
            return  # go silent before the masked submit
        super()._participate(response)


def _prime(client: SecureFederatedClient, fill: float, num_examples: int) -> None:
    """Attach a fake trainer and a dummy shard so the client never touches TF."""
    client.trainer = _FakeTrainer(_fixed_weights(fill))
    client.x = np.zeros((num_examples, 1), dtype=np.float32)
    client.y = np.zeros((num_examples,), dtype=np.int64)


def test_five_clients_two_rounds_one_dropout_over_grpc():
    config = make_config(
        data={"num_clients": 5},
        training={"client_fraction": 1.0, "rounds": 2},
        server={"round_deadline_seconds": 2.0, "min_clients_per_round": 2},
    )
    evaluator = RecordingEvaluator()
    server = SecureFederatedServer(
        config=config,
        initial_weights=TEMPLATE,
        evaluate_fn=evaluator,
        threshold=3,
    )
    server.start()
    address = f"127.0.0.1:{server.config.server.port}"

    fills = {"c0": 0.2, "c1": 0.4, "c2": 0.6, "c3": 0.8, "c4": 1.0}
    counts = {"c0": 3000, "c1": 4000, "c2": 5000, "c3": 6000, "c4": 7000}
    clients: list[SecureFederatedClient] = []
    for cid, fill in fills.items():
        cls = _DropRoundOneClient if cid == "c4" else SecureFederatedClient
        client = cls(config, address, desired_client_id=cid)
        client.register()
        _prime(client, fill, counts[cid])
        clients.append(client)

    server_thread = threading.Thread(target=server.run_rounds, daemon=True)
    client_threads = [
        threading.Thread(target=c.run, kwargs={"poll_interval": 0.05}, daemon=True) for c in clients
    ]
    try:
        server_thread.start()
        for t in client_threads:
            t.start()
        server_thread.join(timeout=60)
        assert not server_thread.is_alive(), "secure run did not finish in time"
    finally:
        for c in clients:
            c.close()
        server.stop(grace=0)

    metrics = server.metrics
    assert len(metrics) == 2

    # Round 1: c4 dropped after sharing; the other four recover and aggregate.
    r1 = metrics[0]
    assert r1.num_selected == 5
    assert r1.num_reported == 4
    assert r1.num_dropped == 1
    assert r1.dropped_clients == ["c4"]
    assert r1.aggregated

    # Round 2: c4 rejoins; nobody drops.
    r2 = metrics[1]
    assert r2.num_reported == 5
    assert r2.num_dropped == 0
    assert r2.aggregated
    assert server.model_version == 2

    # The recovered round-1 model is the secure weighted mean of the SURVIVORS,
    # bit-exact to the maskless quantised computation. evaluator.calls[0] is the
    # global model handed to evaluation right after round 1's aggregation.
    survivors = [
        WeightUpdate(cid, _fixed_weights(fills[cid]), counts[cid])
        for cid in ("c0", "c1", "c2", "c3")
    ]
    expected = quantized_weighted_average(survivors)
    for got, want in zip(evaluator.calls[0], expected, strict=True):
        np.testing.assert_array_equal(got, want)
