"""The batch's pairing and figure logic, tested without TensorFlow.

scripts/personalization_batch.py and scripts/plot_personalization.py keep their
heavy work behind function-local imports (and the plotter has no dependencies at
all) precisely so the part that decides *what the write-up says* -- which client
is compared with which, what counts as improved, what the headline figure draws
-- is testable in milliseconds against hand-made records.

The failure these guard against is the quiet one. A pairing bug does not crash;
it produces a plausible number for the wrong comparison, and a plausible number
is exactly what nobody re-checks.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import personalization_batch as pb  # noqa: E402
import plot_personalization as pp  # noqa: E402


def _run(
    method: str,
    seed: int,
    accuracies: list,
    finetuned: list | None = None,
    test_samples: list | None = None,
) -> dict:
    sizes = test_samples or [10 if a is not None else 0 for a in accuracies]
    rows = [
        {"client": i, "accuracy": a, "test_samples": sizes[i], "head_updates": 1}
        for i, a in enumerate(accuracies)
    ]
    record = {
        "label": f"x/{method}/seed={seed}",
        "method": method,
        "seed": seed,
        "per_client": rows,
        "summary": {"per_client": {"mean": 0.0}},
        "wire": {"spec": "femnist_cnn", "parameters_head": 7998},
    }
    if finetuned is not None:
        record["per_client_finetuned"] = [
            {"client": i, "accuracy": a, "test_samples": sizes[i], "head_updates": 1}
            for i, a in enumerate(finetuned)
        ]
    return record


class TestAccuracyColumn:
    def test_column_is_indexed_by_client_not_by_row_order(self):
        """Rows arrive in whatever order the evaluator produced them; the column
        must be keyed by client id, or two arms would pair different clients."""
        run = _run("fedavg", 42, [0.1, 0.2, 0.3])
        run["per_client"] = list(reversed(run["per_client"]))
        assert pb.accuracy_column(run) == [0.1, 0.2, 0.3]

    def test_unscorable_clients_keep_their_slot(self):
        """Compacting them out here would shorten one arm's column and silently
        shift every later client against the other arm."""
        run = _run("fedavg", 42, [0.1, None, 0.3])
        assert pb.accuracy_column(run) == [0.1, None, 0.3]

    def test_a_run_without_the_requested_rows_is_an_error(self):
        with pytest.raises(ValueError, match="has no per_client_finetuned rows"):
            pb.accuracy_column(_run("fedavg", 42, [0.1]), "per_client_finetuned")


class TestMeanOverSeeds:
    def test_per_client_means_are_taken_before_the_distribution(self):
        assert pb.mean_over_seeds([[0.2, 0.4], [0.4, 0.8]]) == [
            pytest.approx(0.3),
            pytest.approx(0.6),
        ]

    def test_a_client_unscorable_in_any_seed_is_unscorable_overall(self):
        """Averaging it over two seeds while its neighbours average over three
        would put a differently-noisy number into the same distribution."""
        assert pb.mean_over_seeds([[0.2, 0.4], [None, 0.8]]) == [None, pytest.approx(0.6)]

    def test_columns_of_different_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="disagree on client count"):
            pb.mean_over_seeds([[0.1, 0.2], [0.1]])

    def test_no_columns_is_an_error(self):
        with pytest.raises(ValueError, match="at least one column"):
            pb.mean_over_seeds([])


class TestPairing:
    def test_pairing_keeps_only_clients_both_arms_can_score(self):
        left, right = pb.pair([0.1, None, 0.3, 0.4], [0.2, 0.5, None, 0.9])
        assert left == [0.1, 0.4] and right == [0.2, 0.9]

    def test_pairing_rejects_columns_that_cannot_correspond(self):
        with pytest.raises(ValueError, match="disagree on client count"):
            pb.pair([0.1, 0.2], [0.1])


class TestCompareArms:
    @staticmethod
    def _runs(test_samples=None):
        # Three seeds, four clients. The personalized arm helps clients 0 and 1
        # a lot, leaves client 2 alone and HURTS client 3 -- a shape a mean
        # would flatten and the paired report must not.
        global_acc = [0.20, 0.40, 0.60, 0.80]
        finetuned = [0.25, 0.45, 0.60, 0.78]
        personal = [0.50, 0.60, 0.60, 0.70]
        return {
            "fedavg": [_run("fedavg", s, global_acc, finetuned, test_samples) for s in pb.SEEDS],
            "fedrep": [_run("fedrep", s, personal, None, test_samples) for s in pb.SEEDS],
        }

    def test_the_three_comparisons_are_all_reported(self):
        out = pb.compare_arms(self._runs())
        for scope in ("headline", "full_population"):
            assert set(out[scope]["paired_deltas"]) == {
                "personalized_vs_global",
                "finetuned_vs_global",
                "personalized_vs_finetuned",
            }
            assert set(out[scope]["distributions"]) == {"global", "finetuned", "personalized"}
        assert out["clients"] == out["clients_scored"] == 4

    def test_the_paired_report_shows_the_client_that_was_hurt(self):
        headline = pb.compare_arms(self._runs())["headline"]["paired_deltas"][
            "personalized_vs_global"
        ]
        assert headline["fraction_improved"] == pytest.approx(0.5)
        assert headline["fraction_worsened"] == pytest.approx(0.25)
        assert headline["fraction_unchanged"] == pytest.approx(0.25)
        assert headline["min"] == pytest.approx(-0.10)
        assert headline["max"] == pytest.approx(0.30)

    def test_the_fine_tuning_control_is_separated_from_the_fedrep_effect(self):
        """The number that says how much of the headline is FedRep, and how
        much is merely having any local head at all."""
        deltas = pb.compare_arms(self._runs())["full_population"]["paired_deltas"]
        # global -> fine-tuned: +0.05, +0.05, 0.00, -0.02  => +0.02 mean
        # fine-tuned -> FedRep:  +0.25, +0.15, 0.00, -0.08  => +0.08 mean
        assert deltas["finetuned_vs_global"]["mean"] == pytest.approx(0.02)
        assert deltas["personalized_vs_finetuned"]["mean"] == pytest.approx(0.08)

    def test_the_worst_decile_is_carried_through_to_the_summary(self):
        dists = pb.compare_arms(self._runs())["full_population"]["distributions"]
        assert dists["global"]["worst_decile_mean"] == pytest.approx(0.20)
        assert dists["personalized"]["worst_decile_mean"] == pytest.approx(0.50)

    def test_with_no_threshold_headline_and_full_population_agree(self):
        out = pb.compare_arms(self._runs(), min_test_samples=0)
        assert out["headline"]["clients"] == out["full_population"]["clients"] == 4
        assert (
            out["headline"]["distributions"]["personalized"]
            == out["full_population"]["distributions"]["personalized"]
        )
        assert out["selection_effect"]["kept_fraction"] == pytest.approx(1.0)

    def test_the_headline_threshold_restricts_and_the_full_population_does_not(self):
        """The reporting change phase A exists to carry: the headline is the set
        of clients that can actually be measured, the ECDF is still everyone."""
        # Clients 0 and 1 are small-shard; 2 and 3 clear the bar.
        runs = self._runs(test_samples=[5, 8, 40, 60])
        out = pb.compare_arms(runs, min_test_samples=30)
        assert out["headline"]["min_test_samples"] == 30
        assert out["headline"]["clients"] == 2
        assert out["full_population"]["clients"] == 4
        # Global accuracies of the kept clients are 0.60 and 0.80.
        assert out["headline"]["distributions"]["global"]["median"] == pytest.approx(0.70)
        assert out["full_population"]["distributions"]["global"]["median"] == pytest.approx(0.50)
        # The ECDF column stays full-population regardless of the threshold.
        assert len(out["per_client_seed_mean"]["global"]) == 4
        assert out["test_samples"] == [5, 8, 40, 60]

    def test_the_headline_threshold_can_flip_the_sign_of_the_story(self):
        """Exactly why the selection effect must be reported beside it: here
        personalization helps the small-shard clients and hurts the large-shard
        ones, so a >=30 headline reports a loss and the full population a gain."""
        runs = self._runs(test_samples=[5, 8, 40, 60])
        out = pb.compare_arms(runs, min_test_samples=30)
        assert out["headline"]["paired_deltas"]["personalized_vs_global"]["mean"] < 0
        assert out["full_population"]["paired_deltas"]["personalized_vs_global"]["mean"] > 0

    def test_a_threshold_excluding_everyone_is_an_error(self):
        with pytest.raises(ValueError, match="excludes the entire population"):
            pb.compare_arms(self._runs(test_samples=[5, 8, 9, 10]), min_test_samples=30)

    def test_selection_effect_names_both_halves(self):
        out = pb.compare_arms(
            self._runs(test_samples=[5, 8, 40, 60]),
            min_test_samples=30,
            train_sizes=[50, 60, 400, 500],
            alignment=[0.01, 0.02, 0.30, 0.40],
        )
        effect = out["selection_effect"]
        assert effect["clients_kept"] == 2 and effect["clients_total"] == 4
        assert effect["test_samples_kept"] == 100 and effect["test_samples_total"] == 113
        # (a) more data, and (b) sharper label priors -- both must be visible.
        assert effect["median_train_samples_kept"] > effect["median_train_samples_dropped"]
        assert effect["median_label_alignment_kept"] > effect["median_label_alignment_dropped"]
        assert effect["median_test_samples_kept"] == pytest.approx(50.0)

    def test_a_phase_that_scored_two_different_populations_is_rejected(self):
        """Guards the assumption seed-averaging rests on: one population, loaded
        once, scored by every arm and every seed."""
        runs = self._runs(test_samples=[5, 8, 40, 60])
        runs["fedrep"][1]["per_client"][0]["test_samples"] = 99
        with pytest.raises(ValueError, match="different population"):
            pb.compare_arms(runs, min_test_samples=30)

    def test_unscorable_clients_survive_as_gaps_and_are_excluded_from_deltas(self):
        runs = self._runs()
        for record in runs["fedavg"]:
            record["per_client"][2]["accuracy"] = None
        out = pb.compare_arms(runs)
        assert out["clients"] == 4 and out["clients_scored"] == 3
        assert out["per_client_seed_mean"]["global"][2] is None
        assert out["full_population"]["paired_deltas"]["personalized_vs_global"]["n"] == 3


class TestFigure:
    def test_ecdf_is_one_step_per_observation_ending_at_one(self):
        points = pp.ecdf([0.3, 0.1, 0.2])
        assert [x for x, _f in points] == [0.1, 0.2, 0.3]
        assert [f for _x, f in points] == [pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]

    def test_the_path_steps_rather_than_interpolating(self):
        """An ECDF drawn as a smooth line would pass through accuracies no
        client has. Every segment must be axis-aligned."""
        path = pp.step_path(pp.ecdf([0.2, 0.8]), lambda x, f: (100 * x, 100 * (1 - f)))
        commands = [c.split() for c in path.replace("M", "|M").replace("L", "|L").split("|") if c]
        points = [(float(c[1]), float(c[2])) for c in commands]
        for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
            assert x0 == pytest.approx(x1) or y0 == pytest.approx(y1)

    def test_render_produces_valid_self_contained_svg_with_one_path_per_series(self):
        svg = pp.render(
            {
                "global": [0.2, 0.5, 0.7],
                "finetuned": [0.3, 0.5, 0.75],
                "personalized": [0.4, 0.6, 0.9],
            },
            "title",
            "subtitle",
        )
        root = ET.fromstring(svg)  # parses, so it is well-formed XML
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        assert len(paths) == 3
        assert "http://" not in svg.replace("http://www.w3.org/2000/svg", ""), "no external refs"

    def test_a_missing_series_is_simply_not_drawn(self):
        """A phase run without the fine-tuning control must still plot."""
        svg = pp.render({"global": [0.2, 0.5], "personalized": [0.4, 0.6]}, "t", "s")
        assert len(ET.fromstring(svg).findall(".//{http://www.w3.org/2000/svg}path")) == 2

    def test_nothing_to_plot_is_an_error_not_an_empty_chart(self):
        with pytest.raises(ValueError, match="no series carried any values"):
            pp.render({"global": [], "personalized": []}, "t", "s")

    def test_titles_are_escaped_so_a_label_cannot_break_the_document(self):
        svg = pp.render({"global": [0.2, 0.5]}, "alpha < 1 & \"tight\"", "s")
        ET.fromstring(svg)
        assert "alpha &lt; 1 &amp;" in svg

    def test_end_to_end_from_a_phase_record(self, tmp_path):
        phase = {
            "dataset": "femnist",
            "num_clients": 4,
            "clients_per_round": 2,
            "rounds": 20,
            "per_client_test_data": "natural (LEAF by-writer split)",
            "comparison": pb.compare_arms(TestCompareArms._runs()),
        }
        phase_path = tmp_path / "phase.json"
        phase_path.write_text(json.dumps(phase), encoding="utf-8")
        out = tmp_path / "ecdf.svg"
        assert pp.main(["--phase", str(phase_path), "--out", str(out)]) == 0
        ET.fromstring(out.read_text(encoding="utf-8"))
