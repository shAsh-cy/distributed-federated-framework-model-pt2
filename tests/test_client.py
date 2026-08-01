"""The real FederatedClient, end to end against a real server.

Every other suite drives the protocol with ``FakeClient``, which fabricates
weights instead of training. That keeps the coordination tests fast but leaves
``fl.client`` itself unexercised, so this module runs the genuine client: it
registers, receives global weights over gRPC, trains a real Keras model, and
submits.

Shards are injected directly rather than loaded from Fashion-MNIST, so the tests
cover the client's control flow without paying for a 190 MB dataset load per
client.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fl.client import FederatedClient, RegistrationError
from fl.data import Dataset
from fl.models import build_small_cnn
from tests.helpers import PB, FakeClient, ServerHarness, make_config

pytestmark = pytest.mark.slow

RNG = np.random.default_rng(0)


def _tiny_shard(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """A small batch of Fashion-MNIST-shaped noise, enough to train one step on."""
    x = RNG.random((n, 28, 28, 1)).astype(np.float32)
    y = RNG.integers(0, 10, size=n).astype(np.int64)
    return x, y


def _real_client(address, config, client_id, n=48, framework="tensorflow") -> FederatedClient:
    client = FederatedClient(config, address, desired_client_id=client_id, framework=framework)
    client.register()
    client.x, client.y = _tiny_shard(n)
    return client


def _model_config(**overrides):
    return make_config(
        data={"num_clients": 2},
        training={
            "client_fraction": 1.0,
            "rounds": 1,
            "local_epochs": 1,
            "batch_size": 16,
        },
        server={"round_deadline_seconds": 120.0, "min_clients_per_round": 1},
        **overrides,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_real_client_registers_and_receives_a_shard_index():
    config = _model_config()
    with ServerHarness(config, initial=build_small_cnn(seed=0).get_weights()) as h:
        client = FederatedClient(config, h.address, desired_client_id="real-0")
        assert client.register() == "real-0"
        assert client.shard_index == 0
        client.close()


def test_real_client_raises_when_registration_is_refused():
    config = make_config(
        data={"num_clients": 1},
        training={"client_fraction": 1.0},
        server={"min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        first = FederatedClient(config, h.address, desired_client_id="a")
        first.register()
        second = FederatedClient(config, h.address, desired_client_id="b")
        with pytest.raises(RegistrationError, match="already claimed"):
            second.register()
        first.close()
        second.close()


def test_load_data_requires_registration_first():
    config = _model_config()
    with ServerHarness(config) as h:
        client = FederatedClient(config, h.address)
        with pytest.raises(RuntimeError, match="register\\(\\) must be called"):
            client.load_data()
        client.close()


def test_load_data_takes_only_its_own_shard_from_preloaded_data():
    """The client must slice the shard it was assigned, and nothing else."""
    config = make_config(data={"num_clients": 2}, training={"client_fraction": 1.0})
    with ServerHarness(config) as h:
        train = Dataset(
            x=np.arange(20 * 28 * 28, dtype=np.float32).reshape(20, 28, 28, 1),
            y=np.arange(20, dtype=np.int64) % 10,
        )
        shards = [np.arange(0, 12), np.arange(12, 20)]

        a = FederatedClient(config, h.address, desired_client_id="a")
        a.register()
        a.load_data(train=train, shards=shards)
        assert len(a.y) == 12

        b = FederatedClient(config, h.address, desired_client_id="b")
        b.register()
        b.load_data(train=train, shards=shards)
        assert len(b.y) == 8

        # Disjoint: no overlap between the two clients' images.
        assert not (set(a.x[:, 0, 0, 0].tolist()) & set(b.x[:, 0, 0, 0].tolist()))
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# A real training round
# ---------------------------------------------------------------------------


def test_real_client_trains_and_its_update_is_accepted():
    """Full path: poll -> receive weights -> Keras fit -> submit -> accepted."""
    initial = build_small_cnn(seed=0).get_weights()
    config = _model_config()
    with ServerHarness(config, initial=initial) as h:
        client = _real_client(h.address, config, "real-0")
        filler = FakeClient(h.address)
        filler.register(desired="filler")

        h.run_rounds_in_background()
        deadline = time.monotonic() + 30
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        response = client.stub.GetGlobalModel(PB.GetGlobalModelRequest(client_id=client.client_id))
        assert response.action == PB.ROUND_ACTION_TRAIN
        assert response.local_epochs == 1
        assert response.batch_size == 16

        weights, n, loss, accuracy = client.train_one_round(response)
        assert n == 48
        assert np.isfinite(loss)
        assert 0.0 <= accuracy <= 1.0
        assert [w.shape for w in weights] == [w.shape for w in initial]
        # Training moved the weights.
        assert any(not np.array_equal(a, b) for a, b in zip(weights, initial, strict=True))

        reply = client.submit(response.round, response.model_version, weights, n, loss, accuracy)
        assert reply.status == PB.UPDATE_STATUS_ACCEPTED

        # The filler is in the cohort too; let it answer so the barrier closes
        # instead of waiting out the deadline.
        filler.submit(response.round, response.model_version)
        h._thread.join(timeout=60)
        assert h.server.metrics, "round did not complete"
        assert h.server.metrics[0].num_reported >= 1
        client.close()
        filler.close()


def test_real_client_run_loop_completes_a_full_experiment():
    """Two real clients, two rounds, driven entirely by FederatedClient.run()."""
    initial = build_small_cnn(seed=0).get_weights()
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 2, "local_epochs": 1, "batch_size": 16},
        server={"round_deadline_seconds": 120.0, "min_clients_per_round": 2},
    )
    with ServerHarness(config, initial=initial) as h:
        clients = [_real_client(h.address, config, f"real-{i}") for i in range(2)]
        threads = [
            threading.Thread(target=c.run, kwargs={"poll_interval": 0.05}, daemon=True)
            for c in clients
        ]
        for t in threads:
            t.start()

        h.run_rounds_in_background()
        h._thread.join(timeout=180)
        for t in threads:
            t.join(timeout=60)

        assert h.error is None
        assert len(h.server.metrics) == 2
        assert all(m.num_reported == 2 for m in h.server.metrics)
        assert all(m.num_dropped == 0 for m in h.server.metrics)
        assert h.server.model_version == 2
        # The global model actually moved away from its initial weights.
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(h.server.global_weights(), initial, strict=True)
        )
        for c in clients:
            c.close()


def test_client_stops_when_the_server_says_stop():
    initial = build_small_cnn(seed=0).get_weights()
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1, "batch_size": 16},
        server={"round_deadline_seconds": 60.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config, initial=initial) as h:
        client = _real_client(h.address, config, "real-0")
        filler = FakeClient(h.address)
        filler.register(desired="filler")

        thread = threading.Thread(target=client.run, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        h.run_rounds_in_background()
        h._thread.join(timeout=120)

        thread.join(timeout=30)
        assert not thread.is_alive(), "client did not exit after ROUND_ACTION_STOP"
        client.close()
        filler.close()


def test_client_does_not_retrain_a_round_it_already_attempted():
    """The guard that stopped a diverged client re-sending 157 MiB in one round."""
    initial = build_small_cnn(seed=0).get_weights()
    config = _model_config()
    with ServerHarness(config, initial=initial) as h:
        client = _real_client(h.address, config, "real-0")
        filler = FakeClient(h.address)
        filler.register(desired="filler")

        h.run_rounds_in_background()
        deadline = time.monotonic() + 30
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        response = client.stub.GetGlobalModel(PB.GetGlobalModelRequest(client_id=client.client_id))
        key = (response.round, response.model_version)
        client._attempted.add(key)
        assert key in client._attempted

        # run() must now skip this round rather than training it again.
        thread = threading.Thread(target=client.run, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        time.sleep(1.0)
        assert client.client_id not in h.server._updates

        filler.submit(response.round, response.model_version)
        h._thread.join(timeout=60)
        thread.join(timeout=30)
        client.close()
        filler.close()


# -- torch clients -----------------------------------------------------------


def test_torch_client_trains_and_its_update_is_accepted():
    """The same full path as the Keras client, trained on PyTorch.

    The server in this test is the unmodified FederatedServer; if it needed to
    know the framework, this test could not pass.
    """
    initial = build_small_cnn(seed=0).get_weights()
    config = _model_config()
    with ServerHarness(config, initial=initial) as h:
        client = _real_client(h.address, config, "torch-0", framework="torch")
        filler = FakeClient(h.address)
        filler.register(desired="filler")

        h.run_rounds_in_background()
        deadline = time.monotonic() + 30
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        response = client.stub.GetGlobalModel(PB.GetGlobalModelRequest(client_id=client.client_id))
        assert response.action == PB.ROUND_ACTION_TRAIN

        weights, n, loss, accuracy = client.train_one_round(response)
        assert n == 48
        assert np.isfinite(loss)
        assert 0.0 <= accuracy <= 1.0
        # Canonical shapes on the wire, regardless of torch's native layout.
        assert [w.shape for w in weights] == [w.shape for w in initial]
        assert any(not np.array_equal(a, b) for a, b in zip(weights, initial, strict=True))

        reply = client.submit(response.round, response.model_version, weights, n, loss, accuracy)
        assert reply.status == PB.UPDATE_STATUS_ACCEPTED

        filler.submit(response.round, response.model_version)
        h._thread.join(timeout=60)
        assert h.server.metrics, "round did not complete"
        client.close()
        filler.close()


def test_framework_travels_at_registration_and_unknown_framework_rejected_locally():
    config = _model_config()
    with ServerHarness(config) as h:
        client = FederatedClient(config, h.address, framework="torch")
        client.register()
        assert client.client_id  # a torch-announcing client registers normally
        # Constructor validation is local: a typo'd framework never reaches the wire.
        with pytest.raises(ValueError, match="framework must be one of"):
            FederatedClient(config, h.address, framework="jax")
        client.close()
