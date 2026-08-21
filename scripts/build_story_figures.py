"""Derive dashboard/fixtures/story_figures.json from the committed results.

The explainer simplifies language, never data. Every number it puts on screen
comes from here, and every entry here carries the file and JSON pointer it was
read from, so the claim is checkable rather than asserted:

  * `tests/test_explainer_assets.py` regenerates this file and fails if the
    committed copy has drifted;
  * `dashboard/tests/story.test.tsx` re-resolves every pointer against the
    real results JSON and then asserts that no number appears on a story
    screen that is not backed by one of these values.

Adding a figure means adding a row to FIGURES below. There is deliberately no
way to hand-write a display string into a story component.

Run:  python scripts/build_story_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dashboard" / "fixtures" / "story_figures.json"


def resolve(document: Any, pointer: str) -> Any:
    """RFC 6901 JSON pointer, restricted to the shapes these files actually use."""
    node = document
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def pct1(value: float) -> str:
    return f"{value * 100:.1f} %"


def points1(value: float) -> str:
    return f"{value * 100:.1f}"


def fixed3(value: float) -> str:
    return f"{value:.3f}"


def fixed1(value: float) -> str:
    return f"{value:.1f}"


def integer(value: float) -> str:
    return str(int(value))


def grouped(value: float) -> str:
    return f"{int(value):,}"


def exponent(value: float) -> str:
    return f"1e{round(math.log10(value))}"


FORMATS = {
    "pct1": pct1,
    "points1": points1,
    "fixed3": fixed3,
    "fixed1": fixed1,
    "integer": integer,
    "grouped": grouped,
    "exponent": exponent,
}

BATCH_B = "docs/_final_batch_b.json"
R_CURVE = "docs/_femnist_r_curve.json"
BASELINE = "docs/_femnist_baseline.json"
BUDGET_E = "docs/_femnist_budget_e.json"
FASHION = "results/no_dp.json"

# name, file, pointer, format, what it means in one line
FIGURES: list[tuple[str, str, str, str, str]] = [
    (
        "nodpFinal",
        BATCH_B,
        "/nodp_control/mean_final",
        "pct1",
        "FEMNIST, 200 clients per round, 20 rounds, no privacy. Mean of 3 seeds.",
    ),
    (
        "dpFinal",
        BATCH_B,
        "/summary/mean_final",
        "pct1",
        "The same configuration with client-level DP. Mean of 3 seeds.",
    ),
    (
        "dpCost",
        BATCH_B,
        "/dp_cost_mean",
        "points1",
        "The gap between those two, in percentage points.",
    ),
    (
        "epsilon",
        BATCH_B,
        "/achieved_epsilon",
        "fixed3",
        "Privacy budget actually spent over the 20 rounds, from dp_accounting.",
    ),
    ("delta", BATCH_B, "/delta", "exponent", "The delta the epsilon is quoted at."),
    (
        "clipNorm",
        BATCH_B,
        "/l2_clip_norm",
        "fixed1",
        "L2 clipping norm S: every update is trimmed to at most this length.",
    ),
    (
        "noiseZ",
        BATCH_B,
        "/calibrated_z",
        "fixed3",
        "Noise multiplier z, calibrated to hit the target epsilon.",
    ),
    ("cohort", BATCH_B, "/m", "integer", "Clients sampled per round."),
    ("rounds", BATCH_B, "/rounds", "integer", "Rounds in the headline comparison."),
    ("writers", BATCH_B, "/writers", "integer", "FEMNIST writers in the population."),
    ("localEpochs", BATCH_B, "/local_epochs", "integer", "Local epochs per selected client."),
    (
        "longRoundsAcc",
        R_CURVE,
        "/budget/cells/0/summary/mean_final",
        "pct1",
        "The same federation trained for longer. Mean of 3 seeds.",
    ),
    ("longRounds", R_CURVE, "/budget/rounds", "integer", "Rounds behind longRoundsAcc."),
    (
        "pooled",
        BASELINE,
        "/baseline/mean_final",
        "pct1",
        "All the data in one place: the ceiling federation gives up. Mean of 3 seeds.",
    ),
    (
        "modelParams",
        BASELINE,
        "/baseline/model_parameters",
        "grouped",
        "Parameters in the FEMNIST model.",
    ),
    (
        "fashionDense",
        FASHION,
        "/final/accuracy",
        "pct1",
        "Fashion-MNIST dense baseline, no privacy.",
    ),
]

# name, file, pointer to a list of per-seed runs, field inside each history row
SERIES: list[tuple[str, str, str, str, str]] = [
    (
        "dpCurve",
        BATCH_B,
        "/runs",
        "accuracy",
        "Per-round accuracy with DP, averaged over the recorded seeds.",
    ),
    (
        "nodpCurve",
        BUDGET_E,
        "/budget/cells/2/runs",
        "accuracy",
        "Per-round accuracy without DP, same cohort and rounds, averaged over seeds.",
    ),
]


def load(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def mean_over_seeds(runs: list[dict[str, Any]], field: str) -> list[float]:
    lengths = {len(run["history"]) for run in runs}
    if len(lengths) != 1:
        raise ValueError(f"histories of unequal length: {sorted(lengths)}")
    length = lengths.pop()
    return [fmean(run["history"][i][field] for run in runs) for i in range(length)]


def main() -> int:
    used = {file for _, file, *_ in FIGURES} | {file for _, file, *_ in SERIES}
    documents = {path: load(path) for path in used}

    figures: dict[str, Any] = {}
    for name, file, pointer, fmt, meaning in FIGURES:
        value = resolve(documents[file], pointer)
        figures[name] = {
            "value": value,
            "display": FORMATS[fmt](value),
            "format": fmt,
            "meaning": meaning,
            "source": {"file": file, "pointer": pointer},
        }

    series: dict[str, Any] = {}
    for name, file, pointer, field, meaning in SERIES:
        runs = resolve(documents[file], pointer)
        series[name] = {
            "points": mean_over_seeds(runs, field),
            "seeds": [run["seed"] for run in runs],
            "meaning": meaning,
            "source": {
                "file": file,
                "pointer": pointer,
                "field": field,
                "reduce": "mean_over_seeds",
            },
        }

    payload = {
        "schema_version": 1,
        "note": (
            "Generated by scripts/build_story_figures.py — do not edit by hand. "
            "Every entry carries the committed results file and JSON pointer it "
            "was read from; the story components may display no number that is "
            "not backed by an entry here."
        ),
        "figures": figures,
        "series": series,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} — {len(figures)} figures, {len(series)} series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
