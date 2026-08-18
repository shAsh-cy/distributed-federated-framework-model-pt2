"""The FedOpt batch's selection rule, unit-tested before any training run.

scripts/fedopt_batch.py keeps its heavy imports inside the phase functions
precisely so this logic stays importable without TF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fedopt_batch import select_best  # noqa: E402


def _cell(name: str, server_lr: float, acc: float, momentum: float | None = None) -> dict:
    opt: dict = {"name": name, "server_lr": server_lr}
    if momentum is not None:
        opt["momentum"] = momentum
    return {
        "server_optimizer": opt,
        "final_accuracy": acc,
        "label": f"{name}/slr={server_lr}",
    }


def test_picks_the_best_cell_per_family():
    cells = [
        _cell("fedadam", 0.001, 0.60),
        _cell("fedadam", 0.01, 0.75),
        _cell("fedadam", 0.1, 0.72),
        _cell("fedyogi", 0.01, 0.70),
        _cell("fedyogi", 0.1, 0.74),
    ]
    result = select_best(cells)
    assert result["families"]["fedadam"]["best"]["server_optimizer"]["server_lr"] == 0.01
    assert result["families"]["fedyogi"]["best"]["server_optimizer"]["server_lr"] == 0.1


def test_tie_goes_to_the_smaller_server_lr():
    """With one seed, equal accuracies do not license the riskier step."""
    cells = [
        _cell("fedadam", 0.1, 0.75),
        _cell("fedadam", 1.0, 0.75),
        _cell("fedadam", 0.01, 0.74),
    ]
    result = select_best(cells)
    assert result["families"]["fedadam"]["best"]["server_optimizer"]["server_lr"] == 0.1


def test_spread_and_near_best_measure_the_tuning_cliff():
    """A family where one grid point works and the rest crater is a tuning
    cliff, and the near-best fraction must say so."""
    cliff = [
        _cell("fedadam", 0.001, 0.20),
        _cell("fedadam", 0.01, 0.25),
        _cell("fedadam", 0.1, 0.75),
        _cell("fedadam", 1.0, 0.30),
    ]
    plateau = [
        _cell("fedyogi", 0.001, 0.72),
        _cell("fedyogi", 0.01, 0.74),
        _cell("fedyogi", 0.1, 0.75),
        _cell("fedyogi", 1.0, 0.74),
    ]
    result = select_best(cliff + plateau, near=0.02)
    adam = result["families"]["fedadam"]
    yogi = result["families"]["fedyogi"]
    assert adam["accuracy_spread"] == pytest.approx(0.55)
    assert adam["near_best_count"] == 1
    assert yogi["near_best_count"] == 3  # the 0.74s sit within 0.02 of 0.75; 0.72 does not
    assert yogi["near_best_fraction"] == pytest.approx(0.75)
    assert yogi["accuracy_spread"] == pytest.approx(0.03)


def test_momentum_breaks_ties_deterministically():
    cells = [
        _cell("fedavgm", 1.0, 0.75, momentum=0.99),
        _cell("fedavgm", 1.0, 0.75, momentum=0.9),
    ]
    result = select_best(cells)
    assert result["families"]["fedavgm"]["best"]["server_optimizer"]["momentum"] == 0.9


def test_empty_grid_is_an_error():
    with pytest.raises(ValueError, match="at least one grid cell"):
        select_best([])


def test_near_threshold_is_recorded():
    result = select_best([_cell("fedadam", 0.1, 0.7)], near=0.05)
    assert result["near_threshold"] == 0.05
    assert result["families"]["fedadam"]["grid_size"] == 1
    assert result["families"]["fedadam"]["near_best_fraction"] == 1.0
