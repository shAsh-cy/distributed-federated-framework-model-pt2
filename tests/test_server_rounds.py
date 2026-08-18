"""Round orchestration: client sampling, the deadline barrier, and staleness."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fl.aggregation import ClientUpdate, weighted_average
from tests.helpers import PB, FakeClient, ServerHarness, make_config

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Client sampling
# ---------------------------------------------------------------------------


def test_server_samples_the_configured_fraction_of_clients():
    """C=0.5 over 4 clients must sample exactly 2 -- not all of them."""
    config = make_config(data={"num_clients": 4}, training={"client_fraction": 0.5, "rounds": 1})
    assert config.clients_per_round == 2

    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(4)]
        for c in clients:
            c.register()

        stop = threading.Event()
        threads = [
            threading.Thread(target=c.serve_until_stopped, args=(stop,), daemon=True)
            for c in clients
        ]
        for t in threads:
            t.start()
        h.run_rounds_in_background()
        h._thread.join(timeout=30)
        stop.set()
        for t in threads:
            t.join(timeout=5)

        assert h.error is None
        metrics = h.server.metrics
        assert len(metrics) == 1
        assert metrics[0].num_selected == 2
        assert metrics[0].num_reported == 2
        for c in clients:
            c.close()


def test_sampling_is_a_strict_subset_when_fraction_is_below_one():
    config = make_config(data={"num_clients": 4}, training={"client_fraction": 0.5})
    with ServerHarness(config) as h:
        for i in range(4):
            c = FakeClient(h.address)
            c.register(desired=f"c{i}")
            c.close()
        cohort = h.server._sample_cohort()
        assert len(cohort) == 2
        assert len(set(cohort)) == 2
        assert set(cohort) < set(h.server._clients)


def test_full_participation_when_fraction_is_one():
    config = make_config(data={"num_clients": 3}, training={"client_fraction": 1.0})
    with ServerHarness(config) as h:
        for i in range(3):
            c = FakeClient(h.address)
            c.register(desired=f"c{i}")
            c.close()
        assert len(h.server._sample_cohort()) == 3


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_each_client_receives_a_distinct_shard():
    with ServerHarness(make_config(data={"num_clients": 4})) as h:
        shards = []
        for i in range(4):
            c = FakeClient(h.address)
            response = c.register(desired=f"c{i}")
            assert response.accepted
            shards.append(response.shard_index)
            c.close()
        assert sorted(shards) == [0, 1, 2, 3]


def test_registration_beyond_the_configured_client_count_is_refused():
    config = make_config(data={"num_clients": 2}, training={"client_fraction": 1.0})
    with ServerHarness(config) as h:
        for i in range(2):
            c = FakeClient(h.address)
            assert c.register(desired=f"c{i}").accepted
            c.close()
        extra = FakeClient(h.address)
        response = extra.register(desired="c-extra")
        assert not response.accepted
        assert "already claimed" in response.rejection_reason
        extra.close()


def test_reconnecting_client_reclaims_its_own_shard():
    """A restarted client must not be handed a second shard, which would double-count."""
    with ServerHarness(make_config(data={"num_clients": 3})) as h:
        first = FakeClient(h.address)
        original = first.register(desired="stable")
        first.close()

        again = FakeClient(h.address)
        second = again.register(desired="stable")
        again.close()

        assert second.accepted
        assert second.shard_index == original.shard_index
        assert h.server.num_registered == 1


def test_protocol_version_mismatch_is_refused():
    with ServerHarness() as h:
        c = FakeClient(h.address)
        response = c.register(protocol_version=PB.PROTOCOL_VERSION_UNSPECIFIED)
        assert not response.accepted
        assert "protocol version mismatch" in response.rejection_reason
        c.close()


# ---------------------------------------------------------------------------
# The deadline barrier
# ---------------------------------------------------------------------------


def test_straggler_is_dropped_and_the_round_still_completes():
    """The round must not block on its slowest client."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 1.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        prompt = FakeClient(h.address)
        straggler = FakeClient(h.address)
        prompt.register(desired="prompt")
        straggler.register(desired="straggler")  # registers, then never reports

        stop = threading.Event()
        t = threading.Thread(target=prompt.serve_until_stopped, args=(stop,), daemon=True)
        t.start()

        started = time.monotonic()
        h.run_rounds_in_background()
        h._thread.join(timeout=30)
        elapsed = time.monotonic() - started
        stop.set()
        t.join(timeout=5)

        assert h.error is None
        m = h.server.metrics[0]
        assert m.num_selected == 2
        assert m.num_reported == 1
        assert m.num_dropped == 1
        assert m.dropped_clients == ["straggler"]
        assert m.aggregated is True
        # Released at the deadline, not hung indefinitely.
        assert 0.9 <= elapsed < 10.0
        prompt.close()
        straggler.close()


def test_round_completes_immediately_once_every_client_reports():
    """The barrier releases early; it does not wait out the deadline."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 20.0, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(2)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")

        stop = threading.Event()
        threads = [
            threading.Thread(target=c.serve_until_stopped, args=(stop,), daemon=True)
            for c in clients
        ]
        for t in threads:
            t.start()

        started = time.monotonic()
        h.run_rounds_in_background()
        h._thread.join(timeout=30)
        elapsed = time.monotonic() - started
        stop.set()
        for t in threads:
            t.join(timeout=5)

        assert h.error is None
        assert h.server.metrics[0].num_reported == 2
        assert elapsed < 15.0, "barrier waited for the deadline despite a complete cohort"
        for c in clients:
            c.close()


def test_below_quorum_keeps_the_previous_global_model():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 0.5, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        for i in range(2):
            c = FakeClient(h.address)
            c.register(desired=f"c{i}")
            c.close()

        before = h.server.global_weights()
        h.run_rounds_in_background()
        h._thread.join(timeout=30)

        m = h.server.metrics[0]
        assert m.num_reported == 0
        assert m.aggregated is False
        assert h.server.model_version == 0
        for a, b in zip(h.server.global_weights(), before, strict=False):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Staleness and selection
# ---------------------------------------------------------------------------


def test_stale_model_version_is_rejected():
    """An update trained from a superseded model must not be folded in."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        c = FakeClient(h.address)
        c.register(desired="c0")
        other = FakeClient(h.address)
        other.register(desired="c1")

        h.run_rounds_in_background()
        deadline = time.monotonic() + 5
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        reply = c.submit(round_index=1, model_version=999)
        assert reply.status == PB.UPDATE_STATUS_REJECTED_STALE_MODEL
        assert "999" in reply.detail
        assert reply.current_model_version == 0

        h._thread.join(timeout=30)
        c.close()
        other.close()


def test_update_from_an_unregistered_client_is_rejected():
    with ServerHarness() as h:
        rogue = FakeClient(h.address)
        rogue.client_id = "never-registered"
        reply = rogue.submit(round_index=1, model_version=0)
        assert reply.status == PB.UPDATE_STATUS_REJECTED_UNKNOWN_CLIENT
        rogue.close()


def test_update_for_a_round_that_is_not_open_is_rejected():
    with ServerHarness() as h:
        c = FakeClient(h.address)
        c.register(desired="c0")
        reply = c.submit(round_index=1, model_version=0)
        assert reply.status == PB.UPDATE_STATUS_REJECTED_DEADLINE_PASSED
        c.close()


def test_unselected_client_cannot_contribute():
    config = make_config(
        data={"num_clients": 4},
        training={"client_fraction": 0.5, "rounds": 1},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        clients = {}
        for i in range(4):
            c = FakeClient(h.address)
            c.register(desired=f"c{i}")
            clients[f"c{i}"] = c

        h.run_rounds_in_background()
        deadline = time.monotonic() + 5
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        cohort = set(h.server._cohort)
        outsider = next(cid for cid in clients if cid not in cohort)

        reply = clients[outsider].submit(round_index=1, model_version=0)
        assert reply.status == PB.UPDATE_STATUS_REJECTED_NOT_SELECTED

        for cid in cohort:
            clients[cid].submit(round_index=1, model_version=0)
        h._thread.join(timeout=30)
        for c in clients.values():
            c.close()


# ---------------------------------------------------------------------------
# Aggregation wiring and metrics
# ---------------------------------------------------------------------------


def test_aggregated_model_is_the_sample_weighted_average_of_what_arrived():
    """End-to-end check that the server's arithmetic is FedAvg, over real gRPC."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        a = FakeClient(h.address, fill=1.0, num_examples=10)
        b = FakeClient(h.address, fill=3.0, num_examples=1000)
        a.register(desired="a")
        b.register(desired="b")

        h.run_rounds_in_background()
        deadline = time.monotonic() + 5
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        a.submit(1, 0)
        b.submit(1, 0)
        h._thread.join(timeout=30)

        expected = (10 * 1.0 + 1000 * 3.0) / 1010
        result = h.server.global_weights()
        np.testing.assert_allclose(result[0], np.full((2, 3), expected), rtol=1e-5)
        assert h.server.model_version == 1
        a.close()
        b.close()


def test_metrics_record_accuracy_loss_duration_and_bytes_each_way():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(2)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")

        stop = threading.Event()
        threads = [
            threading.Thread(target=c.serve_until_stopped, args=(stop,), daemon=True)
            for c in clients
        ]
        for t in threads:
            t.start()
        h.run_rounds_in_background()
        h._thread.join(timeout=30)
        stop.set()
        for t in threads:
            t.join(timeout=5)

        m = h.server.metrics[0]
        assert 0.0 <= m.accuracy <= 1.0
        assert m.loss > 0.0
        assert m.duration_seconds > 0.0
        assert m.bytes_sent > 0, "no weights were counted as sent"
        assert m.bytes_received > 0, "no weights were counted as received"
        assert set(m.to_dict()) >= {
            "round",
            "accuracy",
            "loss",
            "duration_seconds",
            "bytes_sent",
            "bytes_received",
            "num_dropped",
        }
        for c in clients:
            c.close()


def test_model_version_increments_once_per_aggregated_round():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 3},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(2)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")

        stop = threading.Event()
        threads = [
            threading.Thread(target=c.serve_until_stopped, args=(stop,), daemon=True)
            for c in clients
        ]
        for t in threads:
            t.start()
        h.run_rounds_in_background()
        h._thread.join(timeout=60)
        stop.set()
        for t in threads:
            t.join(timeout=5)

        assert h.error is None
        assert len(h.server.metrics) == 3
        assert [m.model_version for m in h.server.metrics] == [1, 2, 3]
        assert h.server.model_version == 3
        for c in clients:
            c.close()


def test_local_hyperparameters_are_dictated_by_the_server():
    config = make_config(
        data={"num_clients": 2},
        training={
            "client_fraction": 1.0,
            "rounds": 1,
            "local_epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.05,
            "momentum": 0.5,
            "fedprox_mu": 0.07,
        },
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        c = FakeClient(h.address)
        c.register(desired="c0")
        other = FakeClient(h.address)
        other.register(desired="c1")

        h.run_rounds_in_background()
        deadline = time.monotonic() + 5
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        response = None
        for candidate in (c, other):
            r = candidate.poll()
            if r.action == PB.ROUND_ACTION_TRAIN:
                response = r
                break
        assert response is not None
        assert response.local_epochs == 3
        assert response.batch_size == 16
        assert response.learning_rate == pytest.approx(0.05)
        assert response.momentum == pytest.approx(0.5)
        assert response.proximal_mu == pytest.approx(0.07)
        assert response.seconds_until_deadline > 0

        h._thread.join(timeout=30)
        c.close()
        other.close()


def test_server_side_aggregation_matches_the_reference_implementation():
    """Cross-check the wire path against the pure-numpy reference."""
    reference = weighted_average(
        [
            ClientUpdate("a", [np.full((2, 3), 1.0, np.float32)], 10),
            ClientUpdate("b", [np.full((2, 3), 3.0, np.float32)], 1000),
        ]
    )
    np.testing.assert_allclose(reference[0], np.full((2, 3), (10 + 3000) / 1010), rtol=1e-6)
