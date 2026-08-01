"""Coordinator HTTP + WebSocket surface, via TestClient.

The WebSocket tests assert the two properties the frontend depends on:
strict seq ordering on the live stream, and replay correctness across
reconnects (no gaps, no duplicates). Lifecycle endpoints are driven with
scripted executors so nothing here trains a model.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from coordinator.app import create_app
from coordinator.events import (
    RoundAggregated,
    RoundStarted,
    RunCompleted,
    RunStarted,
)
from coordinator.runner import RunContext, Runner
from tests.test_coordinator_core import VALID_CONFIG, make_store


def _started(run_id: str) -> RunStarted:
    return RunStarted(run_id=run_id, ts=time.time(), config={}, num_classes=10, clients=[])


def make_client(executor=None):
    store = make_store()
    runner = Runner(store, executor=executor) if executor else Runner(store)
    app = create_app(store=store, runner=runner)
    return TestClient(app), store, runner


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_datasets_lists_both_with_partitions_and_models(self):
        client, _, _ = make_client()
        with client:
            data = {d["name"]: d for d in client.get("/datasets").json()}
        assert data["fashion_mnist"]["num_classes"] == 10
        assert data["fashion_mnist"]["partition_schemes"] == ["iid", "dirichlet"]
        assert data["femnist"]["num_classes"] == 62
        assert data["femnist"]["partition_schemes"] == ["natural"]
        assert data["femnist"]["model"] == "femnist_cnn"

    def test_algorithms_declare_dp(self):
        client, _, _ = make_client()
        with client:
            algos = {a["name"]: a for a in client.get("/algorithms").json()}
        assert algos["fedavg"]["differentially_private"] is False
        assert algos["dp-fedavg"]["differentially_private"] is True

    def test_architectures_carry_parameter_counts_and_tensors(self):
        client, _, _ = make_client()
        with client:
            archs = {a["name"]: a for a in client.get("/architectures").json()}
        assert archs["small_cnn"]["parameter_count"] == 225_034
        assert archs["femnist_cnn"]["parameter_count"] == 231_742
        names = [t["name"] for t in archs["small_cnn"]["tensors"]]
        assert names[0] == "conv1/kernel" and names[-1] == "logits/bias"


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------


def instant_executor(run_id: str, config: dict, ctx: RunContext) -> None:
    ctx.emit(_started(run_id))
    ctx.emit(RoundStarted(run_id=run_id, ts=time.time(), round=1, model_version=0))
    ctx.emit(
        RoundAggregated(
            run_id=run_id, ts=time.time(), round=1, model_version=1, global_accuracy=0.5
        )
    )
    ctx.emit(RunCompleted(run_id=run_id, ts=time.time(), rounds_completed=1, final_accuracy=0.5))
    ctx._store.set_status(
        run_id,
        "completed",  # noqa: SLF001
        final_metrics={"final_accuracy": 0.5},
    )


class TestRunEndpoints:
    def test_post_runs_starts_and_returns_id(self):
        client, store, runner = make_client(executor=instant_executor)
        with client:
            response = client.post("/runs", json={"config": VALID_CONFIG})
            assert response.status_code == 201
            rid = response.json()["run_id"]
            runner.join(rid, timeout=10)

            detail = client.get(f"/runs/{rid}").json()
        assert detail["status"] == "completed"
        assert detail["source"] == "live"
        assert detail["final_metrics"] == {"final_accuracy": 0.5}
        assert detail["num_events"] == 4
        assert detail["config"]["data"]["num_clients"] == 4

    def test_invalid_config_is_422_with_the_validators_reason(self):
        client, _, _ = make_client(executor=instant_executor)
        with client:
            response = client.post("/runs", json={"config": {"data": {"num_clients": 0}}})
        assert response.status_code == 422
        assert "num_clients" in response.json()["detail"]

    def test_runs_list_and_404_detail(self):
        client, store, runner = make_client(executor=instant_executor)
        with client:
            rid = client.post("/runs", json={"config": VALID_CONFIG}).json()["run_id"]
            runner.join(rid, timeout=10)
            listed = client.get("/runs").json()
            assert [r["id"] for r in listed] == [rid]
            assert client.get("/runs/nope").status_code == 404
            assert client.post("/runs/nope/stop").status_code == 404

    def test_stop_on_finished_run_says_nothing_to_stop(self):
        client, store, runner = make_client(executor=instant_executor)
        with client:
            rid = client.post("/runs", json={"config": VALID_CONFIG}).json()["run_id"]
            runner.join(rid, timeout=10)
            reply = client.post(f"/runs/{rid}/stop").json()
        assert reply["stopping"] is False
        assert "nothing to stop" in reply["detail"]

    def test_stop_requests_graceful_cancellation_of_live_run(self):
        release = threading.Event()

        def held_executor(run_id: str, config: dict, ctx: RunContext) -> None:
            ctx.emit(_started(run_id))
            assert ctx.stop_requested.wait(timeout=10)
            release.set()
            ctx.emit(
                RunCompleted(run_id=run_id, ts=time.time(), rounds_completed=0, stopped_early=True)
            )
            ctx._store.set_status(run_id, "stopped")  # noqa: SLF001

        client, store, runner = make_client(executor=held_executor)
        with client:
            rid = client.post("/runs", json={"config": VALID_CONFIG}).json()["run_id"]
            reply = client.post(f"/runs/{rid}/stop").json()
            assert reply["stopping"] is True
            assert release.wait(timeout=10)
            runner.join(rid, timeout=10)
            assert client.get(f"/runs/{rid}").json()["status"] == "stopped"


# ---------------------------------------------------------------------------
# WebSocket: ordering, replay, reconnect, multiple consumers
# ---------------------------------------------------------------------------


def _recv(ws) -> dict:
    return json.loads(ws.receive_text())


class TestWebSocket:
    def test_unknown_run_closed_with_4404(self):
        client, _, _ = make_client()
        with client, client.websocket_connect("/runs/ghost/events") as ws:
            # close frame surfaces as a disconnect on receive
            with pytest.raises(Exception):  # noqa: B017 - close semantics vary
                ws.receive_text()

    def test_full_stream_is_ordered_and_complete_for_terminal_run(self):
        client, store, runner = make_client(executor=instant_executor)
        with client:
            rid = client.post("/runs", json={"config": VALID_CONFIG}).json()["run_id"]
            runner.join(rid, timeout=10)
            with client.websocket_connect(f"/runs/{rid}/events") as ws:
                events = [_recv(ws) for _ in range(4)]
        assert [e["seq"] for e in events] == [0, 1, 2, 3]
        assert [e["type"] for e in events] == [
            "run_started",
            "round_started",
            "round_aggregated",
            "run_completed",
        ]

    def test_replay_from_sequence_number_no_gaps_no_duplicates(self):
        client, store, runner = make_client(executor=instant_executor)
        with client:
            rid = client.post("/runs", json={"config": VALID_CONFIG}).json()["run_id"]
            runner.join(rid, timeout=10)

            # First connection consumes only part of the stream.
            with client.websocket_connect(f"/runs/{rid}/events") as ws:
                first = [_recv(ws) for _ in range(2)]
            # Reconnect where we left off, exactly like a browser would.
            since = first[-1]["seq"] + 1
            with client.websocket_connect(f"/runs/{rid}/events?since={since}") as ws:
                rest = [_recv(ws) for _ in range(2)]

        seqs = [e["seq"] for e in first + rest]
        assert seqs == [0, 1, 2, 3]  # complete, ordered, no overlap

    def test_live_events_reach_the_socket_in_order(self):
        gate = threading.Event()

        def gated_executor(run_id: str, config: dict, ctx: RunContext) -> None:
            ctx.emit(_started(run_id))
            assert gate.wait(timeout=10)
            for rnd in (1, 2):
                ctx.emit(
                    RoundStarted(run_id=run_id, ts=time.time(), round=rnd, model_version=rnd - 1)
                )
                ctx.emit(
                    RoundAggregated(run_id=run_id, ts=time.time(), round=rnd, model_version=rnd)
                )
            ctx.emit(RunCompleted(run_id=run_id, ts=time.time(), rounds_completed=2))
            ctx._store.set_status(run_id, "completed")  # noqa: SLF001

        client, store, runner = make_client(executor=gated_executor)
        with client:
            rid = client.post("/runs", json={"config": VALID_CONFIG}).json()["run_id"]
            with client.websocket_connect(f"/runs/{rid}/events") as ws:
                assert _recv(ws)["type"] == "run_started"
                gate.set()  # events emitted only while we are connected: pure live path
                live = [_recv(ws) for _ in range(5)]
            runner.join(rid, timeout=10)
        assert [e["seq"] for e in live] == [1, 2, 3, 4, 5]
        assert live[-1]["type"] == "run_completed"

    def test_two_browsers_watch_one_run_and_see_identical_streams(self):
        gate = threading.Event()

        def gated_executor(run_id: str, config: dict, ctx: RunContext) -> None:
            ctx.emit(_started(run_id))
            assert gate.wait(timeout=10)
            ctx.emit(RoundStarted(run_id=run_id, ts=time.time(), round=1, model_version=0))
            ctx.emit(RunCompleted(run_id=run_id, ts=time.time(), rounds_completed=1))
            ctx._store.set_status(run_id, "completed")  # noqa: SLF001

        client, store, runner = make_client(executor=gated_executor)
        with client:
            rid = client.post("/runs", json={"config": VALID_CONFIG}).json()["run_id"]
            with client.websocket_connect(f"/runs/{rid}/events") as a:
                with client.websocket_connect(f"/runs/{rid}/events") as b:
                    assert _recv(a)["type"] == "run_started"
                    assert _recv(b)["type"] == "run_started"
                    gate.set()
                    stream_a = [_recv(a) for _ in range(2)]
                    stream_b = [_recv(b) for _ in range(2)]
            runner.join(rid, timeout=10)
        assert stream_a == stream_b
        assert [e["seq"] for e in stream_a] == [1, 2]

    def test_connecting_after_terminal_replays_and_closes(self):
        client, store, runner = make_client(executor=instant_executor)
        with client:
            rid = client.post("/runs", json={"config": VALID_CONFIG}).json()["run_id"]
            runner.join(rid, timeout=10)
            with client.websocket_connect(f"/runs/{rid}/events") as ws:
                events = [_recv(ws) for _ in range(4)]
                # After the terminal event the server closes; further receives fail.
                with pytest.raises(Exception):  # noqa: B017 - close exception type varies
                    ws.receive_text()
        assert events[-1]["type"] == "run_completed"


class TestOpenApiSurface:
    def test_all_required_paths_present(self):
        client, _, _ = make_client()
        with client:
            spec = client.get("/openapi.json").json()
        for path in (
            "/runs",
            "/runs/{run_id}",
            "/runs/{run_id}/stop",
            "/datasets",
            "/algorithms",
            "/architectures",
        ):
            assert path in spec["paths"], path


class TestCommittedSchema:
    def test_committed_openapi_matches_the_live_app(self):
        """docs/openapi.json is the frontend's contract; drift fails here,
        not in a broken generated client. Regenerate with
        scripts/export_openapi.py."""
        import json
        from pathlib import Path

        from scripts.export_openapi import build_schema

        committed = json.loads(Path("docs/openapi.json").read_text(encoding="utf-8"))
        assert committed == build_schema()
