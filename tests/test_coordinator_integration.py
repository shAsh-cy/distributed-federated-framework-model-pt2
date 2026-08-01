"""Integration: two real training rounds through the full coordinator stack.

Fashion-MNIST, deliberately NOT FEMNIST — this must stay fast. A small
config (10 clients, 2 per round, 2 rounds) exercises the real executor:
data loading, real Keras training, real FedAvg aggregation, real evaluation,
with the complete event sequence asserted for order and content.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from coordinator.app import create_app
from coordinator.runner import Runner
from tests.test_coordinator_core import make_store

pytestmark = pytest.mark.slow

INTEGRATION_CONFIG = {
    "seed": 42,
    "data": {"dataset": "fashion_mnist", "num_clients": 10, "partition": "dirichlet"},
    "model": {"name": "small_cnn"},
    "training": {
        "rounds": 2,
        "client_fraction": 0.2,  # m = 2
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.01,
        "momentum": 0.9,
    },
    "server": {"min_clients_per_round": 2},
}


@pytest.fixture(scope="module")
def completed_run():
    store = make_store()
    runner = Runner(store)  # the REAL executor
    app = create_app(store=store, runner=runner)
    client = TestClient(app)
    with client:
        rid = client.post("/runs", json={"config": INTEGRATION_CONFIG}).json()["run_id"]
        runner.join(rid, timeout=600)
        detail = client.get(f"/runs/{rid}").json()
        with client.websocket_connect(f"/runs/{rid}/events") as ws:
            events = []
            while True:
                import json as _json

                event = _json.loads(ws.receive_text())
                events.append(event)
                if event["type"] in ("run_completed", "run_failed"):
                    break
    return detail, events


class TestTwoRealRounds:
    def test_run_completes(self, completed_run):
        detail, events = completed_run
        assert detail["status"] == "completed"
        assert events[-1]["type"] == "run_completed"
        assert events[-1]["rounds_completed"] == 2

    def test_event_sequence_is_complete_and_ordered(self, completed_run):
        _, events = completed_run
        assert [e["seq"] for e in events] == list(range(len(events)))

        types = [e["type"] for e in events]
        assert types[0] == "run_started"
        # Each round: round_started, then m x (client_sampled, client_reported),
        # then round_aggregated. m = 2.
        per_round = [
            "round_started",
            "client_sampled",
            "client_reported",
            "client_sampled",
            "client_reported",
            "round_aggregated",
        ]
        assert types == ["run_started"] + per_round + per_round + ["run_completed"]

    def test_run_started_carries_population_and_histograms(self, completed_run):
        _, events = completed_run
        started = events[0]
        assert started["num_classes"] == 10
        assert len(started["clients"]) == 10
        for client_info in started["clients"]:
            histogram = client_info["label_histogram"]
            assert len(histogram) == 10
            assert sum(histogram) == client_info["num_examples"]
        # Dirichlet split: 60k examples split exhaustively.
        assert sum(c["num_examples"] for c in started["clients"]) == 60_000

    def test_round_aggregated_carries_real_measurements(self, completed_run):
        _, events = completed_run
        aggs = [e for e in events if e["type"] == "round_aggregated"]
        assert len(aggs) == 2
        for agg in aggs:
            assert 0.0 <= agg["global_accuracy"] <= 1.0
            assert agg["global_loss"] > 0
            assert agg["bytes_sent"] == 900_136 * 2  # model nbytes x cohort
            assert agg["bytes_received"] == 900_136 * 2
            assert agg["median_update_norm"] > 0
            assert agg["cumulative_epsilon"] is None  # no-DP run: no fake epsilon
        # Training moved accuracy off the untrained floor within two rounds.
        assert aggs[-1]["global_accuracy"] > 0.3

    def test_client_reports_carry_real_local_metrics(self, completed_run):
        _, events = completed_run
        reports = [e for e in events if e["type"] == "client_reported"]
        assert len(reports) == 4
        for report in reports:
            assert report["num_examples"] > 0
            assert report["wall_clock_seconds"] > 0
            assert report["bytes"] == 900_136
            assert 0.0 <= report["local_accuracy"] <= 1.0

    def test_replay_equals_live_history(self, completed_run):
        """The stream a late consumer replays is exactly the run's history."""
        detail, events = completed_run
        assert detail["num_events"] == len(events)
