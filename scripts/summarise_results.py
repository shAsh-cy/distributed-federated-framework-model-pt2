"""Print the recorded results as a comparison table.

Reads results/*.json and reports final accuracy against the epsilon actually
achieved, plus the noise-to-signal diagnostic that explains the numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ORDER = [("no_dp", "no DP"), ("dp_moderate", "moderate noise"), ("dp_high", "high noise")]


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for key, label in ORDER:
        report = load(key)
        if report is None:
            print(f"(missing results/{key}.json -- run scripts/run_all_experiments.sh)")
            continue
        privacy = report["privacy"]
        eps = privacy["epsilon"]
        d = report["model"]["parameters"]
        m = privacy["clients_per_round"]
        z = privacy["noise_multiplier"]
        ratio = (z * math.sqrt(d) / m) if privacy["enabled"] else 0.0
        rows.append(
            {
                "label": label,
                "z": z if privacy["enabled"] else 0.0,
                "epsilon": eps,
                "accuracy": report["final"]["accuracy"],
                "loss": report["final"]["loss"],
                "baseline": report["baseline_untrained"]["accuracy"],
                "ratio": ratio,
                "seconds": report["wall_clock_seconds"],
            }
        )

    if not rows:
        return 1

    print()
    print(
        f"{'configuration':<16} {'z':>5} {'epsilon':>10} {'accuracy':>10} {'loss':>12} "
        f"{'noise/signal':>13} {'seconds':>8}"
    )
    print("-" * 80)
    for r in rows:
        eps = "inf (none)" if r["epsilon"] is None else f"{r['epsilon']:.3f}"
        ratio = "-" if r["ratio"] == 0 else f"{r['ratio']:.1f}x"
        print(
            f"{r['label']:<16} {r['z']:>5.1f} {eps:>10} {r['accuracy']:>10.4f} "
            f"{r['loss']:>12.4f} {ratio:>13} {r['seconds']:>8.1f}"
        )
    print("-" * 80)
    print(f"untrained baseline accuracy: {rows[0]['baseline']:.4f}")
    print()
    print("noise/signal = z*sqrt(d)/m, the ratio of injected Gaussian noise to the")
    print("largest possible aggregate signal. Above ~1 the update is mostly noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
