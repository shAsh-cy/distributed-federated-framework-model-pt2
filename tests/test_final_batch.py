"""The phase-A clip selection rule, tested on synthetic bracket cells.

scripts/final_batch.py keeps its heavy imports inside the phase functions
precisely so this logic stays importable without TF/TFF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from final_batch import pick_clip  # noqa: E402


def cell(clip: float, acc: float, frac: float) -> dict:
    return {"l2_clip_norm": clip, "final_accuracy": acc, "clipped_fraction_all_rounds": frac}


class TestPickClip:
    def test_flat_accuracy_picks_by_clipped_fraction(self):
        """All cells within 2pp: the fraction nearest 0.5 wins and the
        selection says the accuracy axis was flat."""
        cells = [
            cell(0.5, 0.700, 1.00),
            cell(1.0, 0.705, 0.95),
            cell(2.0, 0.702, 0.55),
            cell(4.0, 0.698, 0.05),
        ]
        picked = pick_clip(cells)
        assert picked["chosen_clip"] == 2.0
        assert picked["accuracy_flat"] is True
        assert "flat" in picked["reason"]

    def test_separated_accuracy_gates_the_contenders(self):
        """A cell 5pp behind is not a contender even with the ideal fraction."""
        cells = [
            cell(0.5, 0.70, 0.50),  # ideal fraction, but far behind on accuracy
            cell(1.0, 0.75, 0.90),
            cell(2.0, 0.749, 0.70),
            cell(4.0, 0.73, 0.05),
        ]
        picked = pick_clip(cells)
        assert picked["chosen_clip"] == 2.0  # within 2pp of best, fraction nearest 0.5
        assert picked["accuracy_flat"] is False

    def test_everywhere_binding_winner_loses_to_close_runner_up(self):
        """The user rule verbatim: pick where clipping BEGINS to bind, not a
        cell that binds everywhere. A clip that binds on every update only
        survives if nothing within 2pp binds less."""
        cells = [
            cell(0.5, 0.752, 1.00),  # best accuracy, but clipped=1.00 throughout
            cell(1.0, 0.748, 0.60),  # within 2pp, actually at the knee
            cell(2.0, 0.720, 0.20),
            cell(4.0, 0.700, 0.00),
        ]
        picked = pick_clip(cells)
        assert picked["chosen_clip"] == 1.0

    def test_fraction_tie_prefers_the_larger_clip(self):
        """binds-always and binds-never are equidistant from 0.5; the tie
        goes to the larger clip, the begins-to-bind side."""
        cells = [cell(0.5, 0.70, 1.00), cell(4.0, 0.70, 0.00)]
        picked = pick_clip(cells)
        assert picked["chosen_clip"] == 4.0

    def test_per_cell_report_marks_contenders(self):
        cells = [cell(0.5, 0.70, 0.50), cell(1.0, 0.75, 0.90)]
        picked = pick_clip(cells)
        by_clip = {c["l2_clip_norm"]: c for c in picked["per_cell"]}
        assert by_clip[0.5]["contender"] is False
        assert by_clip[1.0]["contender"] is True

    def test_empty_cells_raise(self):
        with pytest.raises(ValueError, match="at least one cell"):
            pick_clip([])
