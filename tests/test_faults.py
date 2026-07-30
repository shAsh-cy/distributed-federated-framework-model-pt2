"""Fault scenarios.

Covers: a client that disconnects mid-round, a client that times out, a client
that returns NaN weights, and a server restarted while clients are connected.

The property under test throughout is that one participant's failure degrades
that round rather than the run.
"""

from __future__ import annotations

import threading
import time

import grpc
import numpy as np
import pytest

from fl.aggregation import AggregationError, ClientUpdate, FedAvgAggregator, weighted_average
from fl.server import FederatedServer
from tests.helpers import (
    PB,
    TEMPLATE,
    FakeClient,
    RecordingEvaluator,
    ServerHarness,
    make_config,
)

pytestmark = pytest.mark.slow


def _wait_for_round(server, target: int = 1, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while server._round < target and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server._round >= target, f"server never reached round {target}"


# ---------------------------------------------------------------------------
# Client disconnects mid-round
# ---------------------------------------------------------------------------


def test_client_disconnecting_mid_round_does_not_stall_the_round():
    """Closing a channel after being selected must not hold the barrier open."""
    config = make_config(
        data={"num_clients": 3},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 2.0, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(3)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)

        clients[0].submit(1, 0)
        clients[1].submit(1, 0)
        clients[2].close()  # vanishes without reporting

        h._thread.join(timeout=30)
        assert h.error is None
        m = h.server.metrics[0]
        assert m.num_reported == 2
        assert m.num_dropped == 1
        assert m.aggregated is True
        for c in clients[:2]:
            c.close()


def test_disconnected_client_is_still_registered_and_can_return():
    """A dropped round must not evict the client from the population."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 2},
        server={"round_deadline_seconds": 1.5, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        a = FakeClient(h.address)
        a.register(desired="a")
        b = FakeClient(h.address)
        b.register(desired="b")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)
        a.submit(1, 0)
        b.close()  # misses round 1

        _wait_for_round(h.server, 2)
        assert h.server.num_registered == 2

        # It reconnects and reclaims the same shard.
        returning = FakeClient(h.address)
        response = returning.register(desired="b")
        assert response.accepted
        assert response.shard_index == 1

        h._thread.join(timeout=30)
        a.close()
        returning.close()


def test_aggregation_uses_only_the_clients_that_reported():
    """The weighting denominator must exclude dropped clients."""
    config = make_config(
        data={"num_clients": 3},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 1.5, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        present_a = FakeClient(h.address, fill=1.0, num_examples=10)
        present_b = FakeClient(h.address, fill=3.0, num_examples=1000)
        absent = FakeClient(h.address, fill=100.0, num_examples=999999)
        present_a.register(desired="a")
        present_b.register(desired="b")
        absent.register(desired="absent")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)
        present_a.submit(1, 0)
        present_b.submit(1, 0)
        h._thread.join(timeout=30)

        # Only a and b count: (10*1 + 1000*3) / 1010, absent's 100.0 excluded.
        expected = (10 * 1.0 + 1000 * 3.0) / 1010
        np.testing.assert_allclose(
            h.server.global_weights()[0], np.full((2, 3), expected), rtol=1e-5
        )
        for c in (present_a, present_b, absent):
            c.close()


# ---------------------------------------------------------------------------
# Client times out
# ---------------------------------------------------------------------------


def test_slow_client_missing_the_deadline_is_dropped_and_refused_afterwards():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 1.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        prompt = FakeClient(h.address)
        slow = FakeClient(h.address)
        prompt.register(desired="prompt")
        slow.register(desired="slow")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)
        prompt.submit(1, 0)

        h._thread.join(timeout=30)

        m = h.server.metrics[0]
        assert m.dropped_clients == ["slow"]

        # Arriving after the round closed: refused, not quietly folded in.
        late = slow.submit(1, 0)
        assert late.status in (
            PB.UPDATE_STATUS_REJECTED_DEADLINE_PASSED,
            PB.UPDATE_STATUS_REJECTED_STALE_MODEL,
        )
        prompt.close()
        slow.close()


def test_every_client_timing_out_leaves_the_model_untouched():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 2},
        server={"round_deadline_seconds": 0.5, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        for i in range(2):
            c = FakeClient(h.address)
            c.register(desired=f"c{i}")
            c.close()

        before = h.server.global_weights()
        h.run_rounds_in_background()
        h._thread.join(timeout=30)

        assert h.error is None
        assert len(h.server.metrics) == 2
        assert all(m.num_reported == 0 for m in h.server.metrics)
        assert all(m.aggregated is False for m in h.server.metrics)
        assert h.server.model_version == 0
        for x, y in zip(h.server.global_weights(), before, strict=False):
            np.testing.assert_array_equal(x, y)


def test_client_cancelling_a_call_does_not_corrupt_server_state():
    """A client that abandons an in-flight RPC must not damage the server.

    Cancellation is used rather than a tiny deadline: against a loopback server
    a sub-millisecond deadline often completes anyway, so the test would pass or
    fail depending on machine speed.
    """
    with ServerHarness() as h:
        c = FakeClient(h.address)
        c.register(desired="c0")

        future = c.stub.GetGlobalModel.future(PB.GetGlobalModelRequest(client_id="c0"))
        future.cancel()
        assert future.cancelled()
        with pytest.raises(grpc.FutureCancelledError):
            future.result()

        # The server is unharmed: it still answers, and the abandoned call left
        # no residue in the registry.
        assert c.poll().action in (PB.ROUND_ACTION_WAIT, PB.ROUND_ACTION_TRAIN)
        assert h.server.num_registered == 1
        assert h.server._updates == {}
        c.close()


# ---------------------------------------------------------------------------
# NaN weights
# ---------------------------------------------------------------------------


def test_nan_update_is_rejected_over_the_wire():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        a, b = FakeClient(h.address), FakeClient(h.address)
        a.register(desired="a")
        b.register(desired="b")
        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)

        nan_weights = [np.full_like(w, np.nan) for w in TEMPLATE]
        reply = a.submit(1, 0, weights=nan_weights)
        assert reply.status == PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD
        assert "NaN" in reply.detail
        assert "a" not in h.server._updates

        b.submit(1, 0)
        h._thread.join(timeout=30)
        a.close()
        b.close()


def test_nan_from_one_client_does_not_poison_the_global_model():
    """The decisive property: one bad client must not turn the model into NaN."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        poisoner = FakeClient(h.address)
        honest = FakeClient(h.address, fill=2.0, num_examples=50)
        poisoner.register(desired="poisoner")
        honest.register(desired="honest")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)
        poisoner.submit(1, 0, weights=[np.full_like(w, np.nan) for w in TEMPLATE])
        honest.submit(1, 0)
        h._thread.join(timeout=30)

        result = h.server.global_weights()
        assert all(np.all(np.isfinite(w)) for w in result), "NaN reached the global model"
        np.testing.assert_allclose(result[0], np.full((2, 3), 2.0), rtol=1e-5)
        poisoner.close()
        honest.close()


def test_inf_update_is_rejected_too():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        a, b = FakeClient(h.address), FakeClient(h.address)
        a.register(desired="a")
        b.register(desired="b")
        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)
        reply = a.submit(1, 0, weights=[np.full_like(w, np.inf) for w in TEMPLATE])
        assert reply.status == PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD
        b.submit(1, 0)
        h._thread.join(timeout=30)
        a.close()
        b.close()


def test_aggregator_refuses_nan_even_if_it_somehow_reaches_it():
    """Defence in depth: the arithmetic layer rejects NaN independently of the RPC layer."""
    updates = [
        ClientUpdate("good", [np.ones((2, 2), np.float32)], 10),
        ClientUpdate("bad", [np.full((2, 2), np.nan, np.float32)], 10),
    ]
    with pytest.raises(AggregationError, match="non-finite"):
        weighted_average(updates)
    with pytest.raises(AggregationError, match="non-finite"):
        FedAvgAggregator().aggregate(updates, [np.zeros((2, 2), np.float32)])


# ---------------------------------------------------------------------------
# Server restart
# ---------------------------------------------------------------------------


def test_clients_survive_a_server_restart_and_reclaim_their_shards():
    """A restarted server re-admits returning clients onto their original shards."""
    config = make_config(data={"num_clients": 3}, training={"client_fraction": 1.0})
    port = config.server.port

    first = FederatedServer(
        config=config,
        initial_weights=TEMPLATE,
        aggregator=FedAvgAggregator(),
        evaluate_fn=RecordingEvaluator(),
    )
    first.start()
    clients = [FakeClient(f"127.0.0.1:{port}") for _ in range(3)]
    original = {}
    for i, c in enumerate(clients):
        response = c.register(desired=f"c{i}")
        original[f"c{i}"] = response.shard_index
    first.stop(grace=0)

    # Calls against the dead server fail rather than hanging forever.
    with pytest.raises(grpc.RpcError):
        clients[0].stub.GetGlobalModel(PB.GetGlobalModelRequest(client_id="c0"), timeout=2.0)

    second = FederatedServer(
        config=config,
        initial_weights=TEMPLATE,
        aggregator=FedAvgAggregator(),
        evaluate_fn=RecordingEvaluator(),
    )
    second.start()
    try:
        assert second.num_registered == 0  # state is not persisted across a restart
        reclaimed = {}
        for i, c in enumerate(clients):
            # Existing channels are in gRPC's backoff state after the outage;
            # wait for the reconnect rather than racing it. A real client does
            # the same thing, which is the behaviour under test.
            grpc.channel_ready_future(c.channel).result(timeout=20)
            response = c.register(desired=f"c{i}")
            assert response.accepted
            reclaimed[f"c{i}"] = response.shard_index
        assert reclaimed == original
        assert second.num_registered == 3
    finally:
        second.stop(grace=0)
        for c in clients:
            c.close()


def test_restarted_server_starts_from_model_version_zero():
    """Clients holding a pre-restart version must be refused, not silently accepted."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 1},
    )
    port = config.server.port

    first = FederatedServer(
        config=config,
        initial_weights=TEMPLATE,
        aggregator=FedAvgAggregator(),
        evaluate_fn=RecordingEvaluator(),
    )
    first.start()
    a = FakeClient(f"127.0.0.1:{port}")
    b = FakeClient(f"127.0.0.1:{port}")
    a.register(desired="a")
    b.register(desired="b")
    thread = threading.Thread(target=first.run_rounds, daemon=True)
    thread.start()
    _wait_for_round(first, 1)
    a.submit(1, 0)
    b.submit(1, 0)
    thread.join(timeout=30)
    assert first.model_version == 1
    first.stop(grace=0)

    second = FederatedServer(
        config=config,
        initial_weights=TEMPLATE,
        aggregator=FedAvgAggregator(),
        evaluate_fn=RecordingEvaluator(),
    )
    second.start()
    try:
        assert second.model_version == 0
        grpc.channel_ready_future(a.channel).result(timeout=20)
        a.register(desired="a")
        reply = a.submit(1, model_version=1)  # version from before the restart
        assert reply.status == PB.UPDATE_STATUS_REJECTED_STALE_MODEL
        assert reply.current_model_version == 0
    finally:
        second.stop(grace=0)
        a.close()
        b.close()
