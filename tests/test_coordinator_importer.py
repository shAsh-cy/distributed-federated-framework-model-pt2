"""Historical importer, run against the repo's real committed result files.

These tests import the actual JSON artifacts in results/ and docs/ — the same
files a first launch would load — and pin the properties the dashboard
depends on: real per-round events where history exists, no fabricated client
events, ranges preserved on multi-seed summaries, imported runs
distinguishable from live ones, and idempotency.
"""

from __future__ import annotations

import pytest

from coordinator.importer import import_history
from tests.test_coordinator_core import make_store


@pytest.fixture(scope="module")
def imported():
    store = make_store()
    report = import_history(store, root=".")
    return store, report


class TestHistoricalImport:
    def test_imports_a_meaningful_volume_of_history(self, imported):
        store, report = imported
        runs = store.list_runs()
        # 3 shipped gRPC runs + 6 compare runs + the FEMNIST/diagnosis
        # harness runs. The exact count grows with committed history; the
        # floor asserts the importer is not silently skipping whole families.
        assert report["imported_runs"] >= 50
        assert len(runs) == report["imported_runs"]

    def test_nothing_skipped_with_errors(self, imported):
        _, report = imported
        assert report["skipped"] == []

    def test_grpc_runs_have_full_round_event_streams(self, imported):
        store, _ = imported
        run = next(r for r in store.list_runs() if r.label == "grpc/no_dp")
        events = store.events_since(run.id)
        types = [e["type"] for e in events]
        assert types[0] == "run_started"
        assert types[-1] == "run_completed"
        assert types.count("round_aggregated") == 20  # the recorded 20-round run
        # Real measured values, not placeholders:
        final = [e for e in events if e["type"] == "run_completed"][-1]
        assert final["final_accuracy"] == pytest.approx(0.8693, abs=1e-4)

    def test_dp_run_events_carry_epsilon(self, imported):
        store, _ = imported
        run = next(r for r in store.list_runs() if r.label == "grpc/dp_moderate")
        aggs = [e for e in store.events_since(run.id) if e["type"] == "round_aggregated"]
        assert aggs[-1]["cumulative_epsilon"] == pytest.approx(6.228, abs=1e-2)

    def test_no_client_events_are_fabricated(self, imported):
        store, _ = imported
        for run in store.list_runs():
            for event in store.events_since(run.id):
                assert not event["type"].startswith("client_"), (
                    f"{run.label} fabricated {event['type']}"
                )

    def test_all_imported_runs_marked_imported(self, imported):
        store, _ = imported
        assert all(r.source == "imported" for r in store.list_runs())

    def test_multiseed_summaries_preserve_ranges_not_points(self, imported):
        store, _ = imported
        aggregates = [r for r in store.list_runs() if r.is_aggregate]
        assert aggregates, "no multi-seed summary rows were imported"
        for run in aggregates:
            metrics = run.final_metrics()
            assert "range_final" in metrics, f"{run.label} collapsed its range"
            assert "mean_final" in metrics

    def test_femnist_sweep_cell_has_seed_runs_and_summary(self, imported):
        store, _ = imported
        cells = [
            r
            for r in store.list_runs()
            if r.group_key
            and r.group_key.endswith("m=50")  # endswith: "m=500" must not match
            and "_femnist_sweep" in r.group_key
        ]
        seed_runs = [r for r in cells if not r.is_aggregate]
        summaries = [r for r in cells if r.is_aggregate]
        assert len(seed_runs) == 3  # three seeds, each its own run with events
        assert len(summaries) == 1
        summary = summaries[0].final_metrics()
        assert len(summary["final_per_seed"]) == 3

    def test_import_is_idempotent(self, imported):
        store, first_report = imported
        count_before = len(store.list_runs())
        second_report = import_history(store, root=".")
        assert second_report["imported_runs"] == 0
        assert len(store.list_runs()) == count_before

    def test_imported_run_events_replayable_from_zero(self, imported):
        """Imported streams honour the same replay contract as live ones."""
        store, _ = imported
        run = next(r for r in store.list_runs() if r.label == "grpc/pure_42")
        seqs = [e["seq"] for e in store.events_since(run.id)]
        assert seqs == list(range(len(seqs)))
