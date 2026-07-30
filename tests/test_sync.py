"""Client synchronisation.

Covers: updates trained from a superseded global model are rejected, the round
barrier holds until either the cohort completes or the deadline expires, and
concurrent registration is race-free.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tests.helpers import PB, FakeClient, ServerHarness, make_config

pytestmark = pytest.mark.slow


def _wait_for_round(server, target: int = 1, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while server._round < target and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server._round >= target, f"server never reached round {target}"


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_update_from_a_superseded_model_version_is_rejected():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 2},
        server={"round_deadline_seconds": 5.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        a, b = FakeClient(h.address), FakeClient(h.address)
        a.register(desired="a")
        b.register(desired="b")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)

        # Complete round 1 so the server advances to model version 1.
        a.submit(1, 0)
        b.submit(1, 0)
        _wait_for_round(h.server, 2)
        assert h.server.model_version == 1

        # Now replay round 1's work: correct client, but version 0.
        stale = a.submit(round_index=1, model_version=0)
        assert stale.status == PB.UPDATE_STATUS_REJECTED_STALE_MODEL
        assert stale.current_model_version == 1
        assert "model_version 0" in stale.detail

        h._thread.join(timeout=30)
        a.close()
        b.close()


def test_stale_update_does_not_change_the_global_model():
    """Rejection must be real, not merely a status code on an applied update."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 2.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        a, b = FakeClient(h.address, fill=99.0), FakeClient(h.address)
        a.register(desired="a")
        b.register(desired="b")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)

        before = h.server.global_weights()
        reply = a.submit(round_index=1, model_version=12345)
        assert reply.status == PB.UPDATE_STATUS_REJECTED_STALE_MODEL
        assert h.server._updates == {}
        for x, y in zip(h.server.global_weights(), before, strict=False):
            np.testing.assert_array_equal(x, y)

        h._thread.join(timeout=30)
        a.close()
        b.close()


def test_future_model_version_is_also_rejected():
    """Staleness is an equality check; a version ahead of the server is wrong too."""
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 2.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        a, b = FakeClient(h.address), FakeClient(h.address)
        a.register(desired="a")
        b.register(desired="b")
        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)

        reply = a.submit(round_index=1, model_version=999)
        assert reply.status == PB.UPDATE_STATUS_REJECTED_STALE_MODEL
        h._thread.join(timeout=30)
        a.close()
        b.close()


def test_server_reports_its_version_so_a_stale_client_can_resynchronise():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 2.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        a, b = FakeClient(h.address), FakeClient(h.address)
        a.register(desired="a")
        b.register(desired="b")
        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)
        reply = a.submit(1, model_version=42)
        assert reply.current_model_version == h.server.model_version
        h._thread.join(timeout=30)
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# The round barrier
# ---------------------------------------------------------------------------


def test_barrier_holds_until_the_cohort_is_complete():
    """The round must not close while a selected client is still outstanding."""
    config = make_config(
        data={"num_clients": 3},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 20.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(3)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)

        clients[0].submit(1, 0)
        clients[1].submit(1, 0)
        time.sleep(0.5)
        # Two of three in: the round is still open and the model is unchanged.
        assert h.server.model_version == 0
        assert len(h.server.metrics) == 0

        clients[2].submit(1, 0)
        h._thread.join(timeout=30)

        assert len(h.server.metrics) == 1
        assert h.server.metrics[0].num_reported == 3
        assert h.server.model_version == 1
        for c in clients:
            c.close()


def test_barrier_releases_at_the_deadline_when_the_cohort_never_completes():
    config = make_config(
        data={"num_clients": 3},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 1.5, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(3)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)
        clients[0].submit(1, 0)  # only one of three ever reports

        started = time.monotonic()
        h._thread.join(timeout=30)
        elapsed = time.monotonic() - started

        m = h.server.metrics[0]
        assert m.num_reported == 1
        assert m.num_dropped == 2
        assert 0.5 < elapsed < 10.0
        for c in clients:
            c.close()


def test_quorum_is_evaluated_against_arrivals_not_the_cohort_size():
    """Enough arrivals means aggregate, even though others were dropped."""
    config = make_config(
        data={"num_clients": 4},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 1.5, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(4)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)
        clients[0].submit(1, 0)
        clients[1].submit(1, 0)

        h._thread.join(timeout=30)
        m = h.server.metrics[0]
        assert m.num_reported == 2
        assert m.num_dropped == 2
        assert m.aggregated is True
        assert h.server.model_version == 1
        for c in clients:
            c.close()


def test_a_client_that_already_reported_is_told_to_wait():
    """Otherwise it re-trains the same round for as long as the round stays open."""
    config = make_config(
        data={"num_clients": 3},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 10.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(3)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")
        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)

        assert clients[0].poll().action == PB.ROUND_ACTION_TRAIN
        clients[0].submit(1, 0)
        assert clients[0].poll().action == PB.ROUND_ACTION_WAIT
        # A client that has not reported is still told to train.
        assert clients[1].poll().action == PB.ROUND_ACTION_TRAIN

        clients[1].submit(1, 0)
        clients[2].submit(1, 0)
        h._thread.join(timeout=30)
        for c in clients:
            c.close()


def test_clients_are_told_to_stop_once_every_round_is_done():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 1.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        c = FakeClient(h.address)
        c.register(desired="c0")
        other = FakeClient(h.address)
        other.register(desired="c1")
        h.run_rounds_in_background()
        h._thread.join(timeout=30)
        assert c.poll().action == PB.ROUND_ACTION_STOP
        c.close()
        other.close()


# ---------------------------------------------------------------------------
# Concurrent registration
# ---------------------------------------------------------------------------


def test_concurrent_registration_assigns_distinct_shards():
    """Twelve clients registering at once must not collide on a shard."""
    num = 12
    config = make_config(data={"num_clients": num}, training={"client_fraction": 0.5})
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(num)]
        barrier = threading.Barrier(num)
        results: list = [None] * num

        def register(i: int) -> None:
            barrier.wait()  # maximise contention
            results[i] = clients[i].register()

        threads = [threading.Thread(target=register, args=(i,)) for i in range(num)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert all(r is not None and r.accepted for r in results)
        shards = sorted(r.shard_index for r in results)
        assert shards == list(range(num)), f"shard collision: {shards}"
        assert len({r.client_id for r in results}) == num
        assert h.server.num_registered == num
        for c in clients:
            c.close()


def test_concurrent_registration_respects_the_client_cap():
    """Twice as many registrants as shards: exactly num_clients get in."""
    num = 6
    config = make_config(data={"num_clients": num}, training={"client_fraction": 1.0})
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(num * 2)]
        barrier = threading.Barrier(num * 2)
        results: list = [None] * (num * 2)

        def register(i: int) -> None:
            barrier.wait()
            results[i] = clients[i].register()

        threads = [threading.Thread(target=register, args=(i,)) for i in range(num * 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        accepted = [r for r in results if r.accepted]
        refused = [r for r in results if not r.accepted]
        assert len(accepted) == num
        assert len(refused) == num
        assert sorted(r.shard_index for r in accepted) == list(range(num))
        assert all("already claimed" in r.rejection_reason for r in refused)
        for c in clients:
            c.close()


def test_concurrent_submissions_are_all_recorded():
    """Simultaneous updates must not lose one another."""
    num = 6
    config = make_config(
        data={"num_clients": num},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 20.0, "min_clients_per_round": 2},
    )
    with ServerHarness(config) as h:
        clients = [FakeClient(h.address) for _ in range(num)]
        for i, c in enumerate(clients):
            c.register(desired=f"c{i}")

        h.run_rounds_in_background()
        _wait_for_round(h.server, 1)

        barrier = threading.Barrier(num)
        statuses: list = [None] * num

        def submit(i: int) -> None:
            barrier.wait()
            statuses[i] = clients[i].submit(1, 0).status

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(num)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert all(s == PB.UPDATE_STATUS_ACCEPTED for s in statuses)
        h._thread.join(timeout=30)
        assert h.server.metrics[0].num_reported == num
        for c in clients:
            c.close()
