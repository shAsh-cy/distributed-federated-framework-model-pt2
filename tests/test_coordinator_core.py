"""Coordinator core: event schema, store ordering/replay, runner lifecycle.

Everything here runs without TensorFlow: lifecycle tests inject scripted
executors, so stop/crash semantics are asserted deterministically rather than
raced against a real training loop.
"""

from __future__ import annotations

import threading
import time

import pytest

from coordinator.db import create_all, make_engine
from coordinator.events import (
    SCHEMA_VERSION,
    ClientDropped,
    ClientSampled,
    RoundAggregated,
    RoundStarted,
    RunCompleted,
    RunStarted,
    UnknownEventError,
    parse_event,
)
from coordinator.runner import RunContext, Runner
from coordinator.store import EventStore


def make_store(tmp_path=None) -> EventStore:
    engine = make_engine(":memory:" if tmp_path is None else tmp_path / "db.sqlite")
    create_all(engine)
    return EventStore(engine)


def _started(run_id: str) -> RunStarted:
    return RunStarted(run_id=run_id, ts=time.time(), config={}, num_classes=10, clients=[])


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestEventSchema:
    def test_round_trip_through_dict(self):
        e = RoundAggregated(
            run_id="r",
            ts=1.0,
            round=3,
            model_version=3,
            global_accuracy=0.5,
            cumulative_epsilon=6.228,
            median_update_norm=0.16,
            clipped_fraction=0.99,
        )
        parsed = parse_event(e.dict())
        assert parsed == e
        assert parsed.schema_version == SCHEMA_VERSION

    def test_unknown_type_rejected(self):
        with pytest.raises(UnknownEventError, match="unknown event type"):
            parse_event({"type": "telemetry_v9", "schema_version": SCHEMA_VERSION})

    def test_unknown_schema_version_rejected_not_guessed(self):
        e = _started("r").dict()
        e["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(UnknownEventError, match="unsupported schema_version"):
            parse_event(e)

    def test_unexpected_fields_rejected(self):
        payload = _started("r").dict()
        payload["surprise"] = 1
        with pytest.raises(Exception, match="extra fields not permitted"):
            parse_event(payload)

    def test_drop_reasons_are_closed_set(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ClientDropped(run_id="r", ts=1.0, round=1, client_id="c", reason="vibes")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStoreOrderingAndReplay:
    def test_sequences_are_contiguous_from_zero(self):
        store = make_store()
        rid = store.create_run({})
        for _ in range(5):
            store.append(_started(rid))
        seqs = [e["seq"] for e in store.events_since(rid)]
        assert seqs == [0, 1, 2, 3, 4]

    def test_replay_from_arbitrary_sequence(self):
        store = make_store()
        rid = store.create_run({})
        for _ in range(6):
            store.append(_started(rid))
        assert [e["seq"] for e in store.events_since(rid, since=4)] == [4, 5]

    def test_sequences_survive_restart(self, tmp_path):
        """A new store over the same file continues the sequence, gap-free."""
        engine_path = tmp_path
        store = make_store(engine_path)
        rid = store.create_run({})
        store.append(_started(rid))
        store.append(_started(rid))

        reopened = EventStore(make_engine(engine_path / "db.sqlite"))
        reopened.append(_started(rid))
        assert [e["seq"] for e in reopened.events_since(rid)] == [0, 1, 2]

    def test_runs_are_isolated(self):
        store = make_store()
        a, b = store.create_run({}), store.create_run({})
        store.append(_started(a))
        store.append(_started(b))
        assert [e["seq"] for e in store.events_since(a)] == [0]
        assert [e["seq"] for e in store.events_since(b)] == [0]

    def test_appends_from_many_threads_stay_gap_free(self):
        store = make_store()
        rid = store.create_run({})
        errors: list[Exception] = []

        def spam():
            try:
                for _ in range(25):
                    store.append(_started(rid))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=spam) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert [e["seq"] for e in store.events_since(rid)] == list(range(100))


class TestMigrationsMatchModels:
    def test_alembic_schema_equals_metadata_schema(self, tmp_path):
        """`alembic upgrade head` and Base.metadata must agree, or migrations
        have drifted from the models."""
        import os

        from alembic.config import Config as AlembicConfig
        from sqlalchemy import inspect

        from alembic import command

        db = tmp_path / "migrated.sqlite"
        cfg = AlembicConfig("alembic.ini")
        os.environ["COORDINATOR_DB"] = str(db)
        try:
            command.upgrade(cfg, "head")
        finally:
            os.environ.pop("COORDINATOR_DB", None)

        migrated = make_engine(db)
        modeled = make_engine(tmp_path / "modeled.sqlite")
        create_all(modeled)

        def schema(engine):
            insp = inspect(engine)
            return {
                table: sorted(c["name"] for c in insp.get_columns(table))
                for table in sorted(insp.get_table_names())
                if table != "alembic_version"
            }

        assert schema(migrated) == schema(modeled)


# ---------------------------------------------------------------------------
# Runner lifecycle via scripted executors
# ---------------------------------------------------------------------------


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


VALID_CONFIG = {"data": {"num_clients": 4}, "training": {"rounds": 2}}


class TestRunnerLifecycle:
    def test_crash_mid_round_marks_run_failed_not_hanging(self):
        store = make_store()

        def crashing(run_id: str, config: dict, ctx: RunContext) -> None:
            ctx.emit(_started(run_id))
            ctx.emit(RoundStarted(run_id=run_id, ts=time.time(), round=1, model_version=0))
            raise ValueError("aggregation exploded")

        runner = Runner(store, executor=crashing)
        rid = runner.start(VALID_CONFIG)
        runner.join(rid, timeout=10)

        run = store.get_run(rid)
        assert run.status == "failed"
        types = [e["type"] for e in store.events_since(rid)]
        assert types == ["run_started", "round_started", "run_failed"]
        failed = store.events_since(rid)[-1]
        assert "aggregation exploded" in failed["error"]
        assert failed["rounds_completed"] == 0
        assert not runner.is_active(rid)

    def test_stop_between_rounds_ends_cleanly_with_no_partial_round(self):
        store = make_store()
        round_done = threading.Event()

        def executor(run_id: str, config: dict, ctx: RunContext) -> None:
            ctx.emit(_started(run_id))
            for rnd in (1, 2):
                if ctx.should_stop():
                    break
                ctx.emit(RoundStarted(run_id=run_id, ts=time.time(), round=rnd, model_version=0))
                ctx.emit(
                    RoundAggregated(run_id=run_id, ts=time.time(), round=rnd, model_version=rnd)
                )
                round_done.set()
                # Hold between rounds until the test has issued its stop.
                assert ctx.stop_requested.wait(timeout=10)
            ctx.emit(
                RunCompleted(run_id=run_id, ts=time.time(), rounds_completed=1, stopped_early=True)
            )
            ctx._store.set_status(run_id, "stopped")  # noqa: SLF001

        runner = Runner(store, executor=executor)
        rid = runner.start(VALID_CONFIG)
        assert round_done.wait(timeout=10)
        assert runner.stop(rid)
        runner.join(rid, timeout=10)

        types = [e["type"] for e in store.events_since(rid)]
        assert types == ["run_started", "round_started", "round_aggregated", "run_completed"]
        assert store.events_since(rid)[-1]["stopped_early"] is True
        assert store.get_run(rid).status == "stopped"

    def test_stop_mid_round_drops_remaining_cohort_and_skips_aggregation(self):
        store = make_store()
        first_client_done = threading.Event()

        def executor(run_id: str, config: dict, ctx: RunContext) -> None:
            ctx.emit(_started(run_id))
            ctx.emit(RoundStarted(run_id=run_id, ts=time.time(), round=1, model_version=0))
            cohort = ["c0", "c1", "c2"]
            for pos, cid in enumerate(cohort):
                ctx.emit(ClientSampled(run_id=run_id, ts=time.time(), round=1, client_id=cid))
                if pos == 0:
                    first_client_done.set()
                    assert ctx.stop_requested.wait(timeout=10)
                if ctx.should_stop():
                    for later in cohort[pos + 1 :]:
                        ctx.emit(
                            ClientDropped(
                                run_id=run_id,
                                ts=time.time(),
                                round=1,
                                client_id=later,
                                reason="stopped",
                            )
                        )
                    break
            ctx.emit(
                RunCompleted(run_id=run_id, ts=time.time(), rounds_completed=0, stopped_early=True)
            )
            ctx._store.set_status(run_id, "stopped")  # noqa: SLF001

        runner = Runner(store, executor=executor)
        rid = runner.start(VALID_CONFIG)
        assert first_client_done.wait(timeout=10)
        assert runner.stop(rid)
        runner.join(rid, timeout=10)

        types = [e["type"] for e in store.events_since(rid)]
        assert types == [
            "run_started",
            "round_started",
            "client_sampled",
            "client_dropped",
            "client_dropped",
            "run_completed",
        ]
        assert "round_aggregated" not in types  # partial round never aggregated

    def test_stop_on_unknown_or_finished_run_reports_false(self):
        store = make_store()

        def instant(run_id: str, config: dict, ctx: RunContext) -> None:
            ctx.emit(_started(run_id))
            ctx.emit(RunCompleted(run_id=run_id, ts=time.time(), rounds_completed=0))
            ctx._store.set_status(run_id, "completed")  # noqa: SLF001

        runner = Runner(store, executor=instant)
        rid = runner.start(VALID_CONFIG)
        runner.join(rid, timeout=10)
        assert _wait(lambda: not runner.is_active(rid))
        assert runner.stop(rid) is False
        assert runner.stop("no-such-run") is False

    def test_invalid_config_rejected_synchronously(self):
        from fl.config import ConfigError

        runner = Runner(make_store())
        with pytest.raises(ConfigError):
            runner.start({"data": {"num_clients": 0}})


class TestOrphanedRunRecovery:
    def test_startup_fails_runs_a_dead_process_left_running(self):
        """Audit finding M1 (docs/audit_v0_2.md): a killed coordinator left
        its live runs at status 'running' forever — verified against a real
        docker restart. Startup recovery marks them failed; completed and
        imported rows are untouched."""
        store = make_store()
        dead = store.create_run({}, status="running")
        pending = store.create_run({}, status="pending")
        done = store.create_run({}, status="completed")
        imported = store.create_run({}, source="imported", status="running")

        orphaned = store.fail_orphaned_runs()

        assert set(orphaned) == {dead, pending}
        assert store.get_run(dead).status == "failed"
        assert store.get_run(pending).status == "failed"
        assert store.get_run(done).status == "completed"
        assert store.get_run(imported).status == "running"
        assert store.fail_orphaned_runs() == []  # idempotent
