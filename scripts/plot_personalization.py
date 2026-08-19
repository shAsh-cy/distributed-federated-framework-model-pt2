"""Draw the per-client accuracy ECDF from a recorded personalization phase.

The headline figure, and the reason it is an ECDF rather than a bar chart of
means: personalization's whole claim is about the shape of the per-client
distribution, and two methods with the same mean can have completely different
shapes. On an ECDF the claim is legible directly -- a curve shifted right helps
everyone, a curve whose *lower* tail lifts helps the clients the global model
serves worst, and a curve that crosses the baseline helps some clients by hurting
others. The last of those is the one a mean would hide, and it is the one worth
knowing about.

Pure standard library, deliberately. matplotlib is not in ``requirements.txt``
and adding it would mean re-resolving a dependency set pinned around
tensorflow-federated's exact ``typing-extensions==4.5.*``; an SVG is a few dozen
lines of text and costs nothing to generate inside the pinned container.

Usage:
    python scripts/plot_personalization.py --phase docs/_personalization_a.json \
        --out docs/personalization_ecdf_femnist.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WIDTH, HEIGHT = 760, 460
MARGIN_LEFT, MARGIN_RIGHT = 64, 24
MARGIN_TOP, MARGIN_BOTTOM = 44, 64

#: Series drawn, in draw order, with the label the legend shows. The baseline is
#: drawn first so the personalized curve sits on top of it where they overlap.
SERIES = (
    ("global", "global model", "#8c8c8c"),
    ("finetuned", "global + local head (fine-tuned)", "#d4801a"),
    ("personalized", "FedRep (shared backbone + local head)", "#1f6fb2"),
)


def ecdf(values: list[float]) -> list[tuple[float, float]]:
    """Step points ``(x, F(x))`` of the empirical CDF, one per observation."""
    ordered = sorted(values)
    n = len(ordered)
    return [(x, (i + 1) / n) for i, x in enumerate(ordered)]


def step_path(points: list[tuple[float, float]], to_px) -> str:
    """SVG path for the right-continuous step function through ``points``."""
    if not points:
        return ""
    x0, y0 = to_px(points[0][0], 0.0)
    commands = [f"M {x0:.2f} {y0:.2f}"]
    prev_y = y0
    for x, f in points:
        px, py = to_px(x, f)
        commands.append(f"L {px:.2f} {prev_y:.2f}")
        commands.append(f"L {px:.2f} {py:.2f}")
        prev_y = py
    return " ".join(commands)


def render(series: dict[str, list[float]], title: str, subtitle: str) -> str:
    """Return a complete, self-contained SVG document."""
    present = [(key, label, colour) for key, label, colour in SERIES if series.get(key)]
    if not present:
        raise ValueError("nothing to plot: no series carried any values")

    all_values = [v for key, _l, _c in present for v in series[key]]
    x_lo = max(0.0, min(all_values) - 0.02)
    x_hi = min(1.0, max(all_values) + 0.02)
    if x_hi - x_lo < 0.05:  # a degenerate range would divide by ~0 below
        x_lo, x_hi = max(0.0, x_lo - 0.05), min(1.0, x_hi + 0.05)

    plot_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    def to_px(x: float, f: float) -> tuple[float, float]:
        return (
            MARGIN_LEFT + (x - x_lo) / (x_hi - x_lo) * plot_w,
            MARGIN_TOP + (1.0 - f) * plot_h,
        )

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{_escape(title)}">',
        "<style>",
        "  .bg { fill: #ffffff; }",
        "  .ink { fill: #24292f; }",
        "  .muted { fill: #57606a; }",
        "  .axis { stroke: #d0d7de; }",
        "  .frame { stroke: #8c959f; }",
        "  text { font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, sans-serif; }",
        "  @media (prefers-color-scheme: dark) {",
        "    .bg { fill: #0d1117; }",
        "    .ink { fill: #e6edf3; }",
        "    .muted { fill: #9198a1; }",
        "    .axis { stroke: #30363d; }",
        "    .frame { stroke: #6e7681; }",
        "  }",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{WIDTH}" height="{HEIGHT}"/>',
        f'<text class="ink" x="{MARGIN_LEFT}" y="20" font-size="14" '
        f'font-weight="600">{_escape(title)}</text>',
        f'<text class="muted" x="{MARGIN_LEFT}" y="36" font-size="11">{_escape(subtitle)}</text>',
    ]

    # Grid and axes.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        _gx, gy = to_px(x_lo, frac)
        out.append(
            f'<line class="axis" x1="{MARGIN_LEFT}" y1="{gy:.2f}" '
            f'x2="{MARGIN_LEFT + plot_w}" y2="{gy:.2f}"/>'
        )
        out.append(
            f'<text class="muted" x="{MARGIN_LEFT - 8}" y="{gy + 4:.2f}" font-size="10" '
            f'text-anchor="end">{frac:.2f}</text>'
        )
    ticks = _x_ticks(x_lo, x_hi)
    for tick in ticks:
        tx, _ty = to_px(tick, 0.0)
        out.append(
            f'<line class="axis" x1="{tx:.2f}" y1="{MARGIN_TOP}" '
            f'x2="{tx:.2f}" y2="{MARGIN_TOP + plot_h}"/>'
        )
        out.append(
            f'<text class="muted" x="{tx:.2f}" y="{MARGIN_TOP + plot_h + 16:.2f}" '
            f'font-size="10" text-anchor="middle">{tick:.2f}</text>'
        )
    out.append(
        f'<rect class="frame" fill="none" x="{MARGIN_LEFT}" y="{MARGIN_TOP}" '
        f'width="{plot_w}" height="{plot_h}"/>'
    )
    out.append(
        f'<text class="muted" x="{MARGIN_LEFT + plot_w / 2:.0f}" '
        f'y="{HEIGHT - 26}" font-size="11" text-anchor="middle">'
        "per-client test accuracy</text>"
    )
    out.append(
        f'<text class="muted" x="{14}" y="{MARGIN_TOP + plot_h / 2:.0f}" font-size="11" '
        f'text-anchor="middle" transform="rotate(-90 14 {MARGIN_TOP + plot_h / 2:.0f})">'
        "fraction of clients at or below</text>"
    )

    for key, label, colour in present:
        values = series[key]
        out.append(
            f'<path fill="none" stroke="{colour}" stroke-width="1.9" '
            f'stroke-linejoin="round" d="{step_path(ecdf(values), to_px)}"/>'
        )
        median = sorted(values)[len(values) // 2]
        mx, my = to_px(median, 0.5)
        out.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="3" fill="{colour}"/>')

    legend_y = MARGIN_TOP + 14
    for i, (key, label, colour) in enumerate(present):
        y = legend_y + i * 16
        x = MARGIN_LEFT + plot_w - 250
        out.append(
            f'<line x1="{x}" y1="{y - 4}" x2="{x + 22}" y2="{y - 4}" '
            f'stroke="{colour}" stroke-width="2.4"/>'
        )
        n = len(series[key])
        med = sorted(series[key])[n // 2]
        out.append(
            f'<text class="ink" x="{x + 28}" y="{y}" font-size="10">'
            f"{_escape(label)} (median {med:.3f})</text>"
        )

    out.append(
        f'<text class="muted" x="{MARGIN_LEFT}" y="{HEIGHT - 8}" font-size="9">'
        "dots mark medians; each curve is one client per step, "
        "accuracies averaged over seeds before the distribution is taken</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _x_ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    step = 0.05 if span <= 0.35 else (0.1 if span <= 0.7 else 0.2)
    first = round(lo / step) * step
    ticks = []
    t = first
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            ticks.append(round(t, 4))
        t += step
    return ticks


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def series_from_phase(phase: dict) -> dict[str, list[float]]:
    """Pull the seed-averaged per-client columns out of a phase record."""
    columns = phase["comparison"]["per_client_seed_mean"]
    return {key: [v for v in columns.get(key, []) if v is not None] for key in columns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, help="A recorded phase JSON.")
    parser.add_argument("--out", required=True, help="SVG to write.")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    phase = json.loads(Path(args.phase).read_text(encoding="utf-8"))
    dataset = phase["dataset"]
    title = args.title or (
        f"Per-client accuracy, {dataset}: global vs personalized "
        f"(N={phase['num_clients']}, m={phase['clients_per_round']}, R={phase['rounds']})"
    )
    # The ECDF is the SECONDARY figure and is drawn over the whole population,
    # threshold or no threshold; the headline statistics live in the JSON. Saying
    # so on the figure keeps the two from being read as the same number.
    threshold = phase.get("headline_min_test_samples", 0)
    scope = (
        f"all {phase['comparison']['clients_scored']} scorable clients; "
        f"headline statistics use the {phase['comparison']['selection_effect']['clients_kept']} "
        f"with >= {threshold} held-out samples"
        if threshold
        else f"all {phase['comparison']['clients_scored']} scorable clients"
    )
    subtitle = (
        f"{scope}, {len(phase['comparison']['seeds'])} seeds, "
        f"per-client test data: {phase['per_client_test_data']}"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(series_from_phase(phase), title, subtitle), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
