"""Generate docs/how-it-works.html — the whole explainer as one file.

One file, no build step to read it, no external request of any kind: CSS, JS,
SVG and images are all inline, images as data URIs. It has to render correctly
opened from file:// by double-click, because it is going to be attached to
emails and opened by people who will not run a server.

Every number comes from dashboard/fixtures/story_figures.json (see
scripts/build_story_figures.py), which carries the results file and JSON
pointer behind each one. Every image comes from docs/inversion/manifest.json.
Nothing here is typed by hand except the prose.

Regenerate after changing either of those:

    python scripts/build_how_it_works.py

tests/test_explainer_assets.py fails if the committed page has drifted from
what this script produces.
"""

from __future__ import annotations

import base64
import html
import json
import math
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "how-it-works.html"

REPO = "https://github.com/shAsh-cy/distributed-federated-framework-model-pt2"
BLOB = REPO + "/blob/main/"

FIGURES_JSON = ROOT / "dashboard" / "fixtures" / "story_figures.json"
FIXTURE_JSON = ROOT / "dashboard" / "fixtures" / "live_demo_scripted_events.json"
MANIFEST_JSON = ROOT / "docs" / "inversion" / "manifest.json"

DATA = json.loads(FIGURES_JSON.read_text(encoding="utf-8"))
EVENTS = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
MANIFEST = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------


def fig(name: str) -> str:
    """A sourced figure, with its provenance in the tooltip."""
    entry = DATA["figures"][name]
    source = f"{entry['source']['file']} {entry['source']['pointer']}"
    return f'<span class="n" title="{html.escape(source)}">{html.escape(entry["display"])}</span>'


def raw(name: str) -> float:
    return float(DATA["figures"][name]["value"])


def points(name: str) -> list[float]:
    return [float(x) for x in DATA["series"][name]["points"]]


# --------------------------------------------------------------------------
# the recorded run, for the two classroom diagrams
# --------------------------------------------------------------------------

RUN_STARTED = next(e for e in EVENTS if e["type"] == "run_started")
CLIENTS = RUN_STARTED["clients"]
DROP = next(e for e in EVENTS if e["type"] == "client_dropped")
DROP_ROUND = int(DROP["round"])
DROP_CLIENT = str(DROP["client_id"])
SAMPLED = [
    str(e["client_id"])
    for e in EVENTS
    if e["type"] == "client_sampled" and e["round"] == DROP_ROUND
]
REPORTED = {
    str(e["client_id"]): e
    for e in EVENTS
    if e["type"] == "client_reported" and e["round"] == DROP_ROUND
}
TOTAL_ROUNDS = sum(1 for e in EVENTS if e["type"] == "round_started")


def ring(count: int, radius: float) -> list[tuple[float, float]]:
    return [
        (
            radius * math.cos(2 * math.pi * i / count - math.pi / 2),
            radius * math.sin(2 * math.pi * i / count - math.pi / 2),
        )
        for i in range(count)
    ]


def histogram_marks(counts: list[int], size: float) -> str:
    peak = max(counts + [1])
    bar = size / len(counts)
    parts = []
    for i, count in enumerate(counts):
        height = max(1 if count > 0 else 0, count / peak * size * 0.55)
        parts.append(
            f'<rect x="{-size / 2 + i * bar:.2f}" y="{size / 2 - height:.2f}" '
            f'width="{max(0.5, bar - 0.6):.2f}" height="{height:.2f}"/>'
        )
    return "".join(parts)


def classroom_svg(*, show_round: bool) -> str:
    """The class. Optionally mid-round, with the recorded cohort and the drop."""
    size, radius = 460, 166
    places = ring(len(CLIENTS), radius)
    out = [
        f'<svg viewBox="{-size / 2} {-size / 2} {size} {size}" '
        f'role="img" aria-label="{"One round of the class at work" if show_round else "A class of ten students, each with their own notebook"}">'
    ]

    if show_round:
        for (x, y), client in zip(places, CLIENTS, strict=True):
            cid = client["client_id"]
            if cid not in SAMPLED:
                continue
            dropped = cid == DROP_CLIENT
            report = REPORTED.get(cid)
            byte_count = int(report["bytes"]) if report and report.get("bytes") else 900_136
            width = 1.0 if dropped else min(4.0, 0.75 + byte_count / 600_000)
            dash = ' stroke-dasharray="4 4"' if dropped else ""
            colour = "var(--slate)" if dropped else "var(--client)"
            out.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="0" y2="0" stroke="{colour}" '
                f'stroke-width="{width:.2f}"{dash} opacity="0.7"/>'
            )
    else:
        wall = radius - 46
        out.append(
            f'<circle r="{wall}" fill="none" stroke="var(--ink)" stroke-width="1.25" '
            'stroke-dasharray="2 5" opacity="0.55"/>'
        )
        # A notebook stopped dead at the wall, on the way to the middle.
        nx, ny = places[len(places) // 4]
        length = math.hypot(nx, ny) or 1
        px, py = nx / length * wall, ny / length * wall
        out.append(
            f'<g transform="translate({px:.1f} {py:.1f})">'
            '<rect x="-9" y="-11" width="18" height="22" fill="var(--ground-raised)" '
            'stroke="var(--client)" stroke-width="1.5"/>'
            '<line x1="-9" y1="-4" x2="9" y2="-4" stroke="var(--client)"/>'
            '<line x1="-9" y1="1" x2="9" y2="1" stroke="var(--client)"/>'
            '<line x1="-9" y1="6" x2="4" y2="6" stroke="var(--client)"/>'
            "</g>"
        )

    out.append(
        '<rect x="-14" y="-14" width="28" height="28" fill="var(--global)"/>'
        '<text y="30" text-anchor="middle" class="lbl" fill="var(--global)">THE TEACHER</text>'
    )

    for (x, y), client in zip(places, CLIENTS, strict=True):
        cid = client["client_id"]
        active = show_round and cid in SAMPLED
        dropped = show_round and cid == DROP_CLIENT
        colour = "var(--slate)" if (dropped or not active) else "var(--client)"
        stroke_width = 2 if active and not dropped else 1
        opacity = "0.45" if dropped else "1"
        out.append(
            f'<g transform="translate({x:.1f} {y:.1f})" opacity="{opacity}">'
            f'<circle r="16" fill="var(--ground-raised)" stroke="{colour}" stroke-width="{stroke_width}"/>'
            f'<g fill="{colour}" opacity="0.85">{histogram_marks(client["label_histogram"], 20)}</g>'
            f'<text y="26" text-anchor="middle" class="lbl" fill="var(--ink)" opacity="0.75">'
            f"s{cid.rsplit('-', 1)[1]}</text>"
            "</g>"
        )

    out.append(
        f'<text x="{-size / 2 + 4}" y="{-size / 2 + 14}" class="lbl" fill="var(--slate)">'
        "each circle is a student · the bars are their notebook</text>"
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# the notebooks at three settings
# --------------------------------------------------------------------------

STUDENTS, CHAPTERS, PER_CHAPTER = 10, 10, 600


def deal(alpha: float, seed: int = 42) -> list[list[int]]:
    """Dirichlet(alpha) proportions per chapter, dealt to the students.

    Same construction as fl/data.py::partition_dirichlet — these are three
    still frames, so the story-mode coupling that makes dragging smooth is not
    needed and stdlib gammavariate is exactly the right sampler.
    """
    rng = random.Random(seed)
    grid = [[0] * CHAPTERS for _ in range(STUDENTS)]
    for chapter in range(CHAPTERS):
        weights = [rng.gammavariate(alpha, 1.0) for _ in range(STUDENTS)]
        total = sum(weights) or 1.0
        dealt = 0
        for student in range(STUDENTS - 1):
            share = round(weights[student] / total * PER_CHAPTER)
            grid[student][chapter] = share
            dealt += share
        grid[STUDENTS - 1][chapter] = max(0, PER_CHAPTER - dealt)
    return grid


def notebook_grid_svg(alpha: float, label: str) -> str:
    grid = deal(alpha)
    cell_w, cell_h, gap = 74.0, 40.0, 8.0
    cols = 5
    rows = (STUDENTS + cols - 1) // cols
    width = cols * cell_w + (cols - 1) * gap
    height = rows * (cell_h + 14) + (rows - 1) * gap
    out = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="Ten notebooks when the sharing is {html.escape(label)}">'
    ]
    for index, counts in enumerate(grid):
        col, row = index % cols, index // cols
        ox = col * (cell_w + gap)
        oy = row * (cell_h + 14 + gap)
        peak = max(counts + [1])
        bar = cell_w / CHAPTERS
        out.append(
            f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="var(--ground-raised)"/>'
        )
        for chapter, count in enumerate(counts):
            bh = max(1 if count > 0 else 0, count / peak * (cell_h - 3))
            out.append(
                f'<rect x="{ox + chapter * bar + 0.4:.2f}" y="{oy + cell_h - bh:.2f}" '
                f'width="{bar - 0.8:.2f}" height="{bh:.2f}" fill="var(--client)"/>'
            )
        out.append(
            f'<text x="{ox + cell_w / 2:.1f}" y="{oy + cell_h + 11:.1f}" text-anchor="middle" '
            f'class="lbl" fill="var(--slate)">s{index + 1}</text>'
        )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# trimming, and the two curves
# --------------------------------------------------------------------------


def trim_svg() -> str:
    bars = 20
    trim = raw("clipNorm")
    share = raw("clippedFinal")
    median = raw("medianNormFinal")
    over = round(bars * share)
    lengths = []
    for i in range(bars):
        rank = (i + 0.5) / bars
        lengths.append(
            trim + 0.15 + 1.5 * (1 - rank) if i < over else median * (0.45 + 0.75 * (1 - rank))
        )
    width, height = 460, 170
    longest = max(lengths + [trim * 1.6])
    scale = (width - 90) / longest
    trim_x = 60 + trim * scale
    row = (height - 40) / bars
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Twenty corrections; '
        f"{html.escape(DATA['figures']['clippedFinal']['display'])} of them cross the trim line "
        'and are cut back to it">',
        '<text x="0" y="11" class="lbl" fill="var(--slate)">corrections</text>',
        f'<line x1="{trim_x:.1f}" y1="16" x2="{trim_x:.1f}" y2="{height - 18}" '
        'stroke="var(--ink)" stroke-width="1.25"/>',
        f'<text x="{trim_x + 4:.1f}" y="{height - 5}" class="lbl" fill="var(--ink)">trim line</text>',
    ]
    for i, length in enumerate(lengths):
        y = 24 + i * row
        full = length * scale
        cut = min(length, trim) * scale
        if full > cut:
            out.append(
                f'<line x1="{60 + cut:.1f}" y1="{y:.1f}" x2="{60 + full:.1f}" y2="{y:.1f}" '
                'stroke="var(--slate)" stroke-width="1" stroke-dasharray="2 3"/>'
            )
        out.append(
            f'<line x1="60" y1="{y:.1f}" x2="{60 + cut:.1f}" y2="{y:.1f}" '
            'stroke="var(--client)" stroke-width="2.5"/>'
        )
    out.append("</svg>")
    return "".join(out)


def budget_svg() -> str:
    width, height = 460, 34
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="The ink budget, spent to {html.escape(DATA["figures"]["epsilon"]["display"])}">'
        f'<rect x="0" y="8" width="{width}" height="12" fill="var(--ground-raised)" '
        'stroke="var(--rule)"/>'
        f'<rect x="1" y="9" width="{width - 2}" height="10" fill="var(--budget)"/>'
        f'<text x="0" y="{height - 2}" class="lbl" fill="var(--slate)">'
        "empty at the start of training · full after "
        f"{html.escape(DATA['figures']['rounds']['display'])} rounds</text>"
        "</svg>"
    )


def curves_svg() -> str:
    nodp = points("nodpCurve")
    dp = points("dpCurve")
    pooled = raw("pooled")
    width, height = 520, 300
    left, bottom, top, right = 46, 40, 16, 12
    plot_w = width - left - right
    plot_h = height - top - bottom

    def px(round_index: int) -> float:
        return left + plot_w * round_index / (len(nodp) - 1)

    def py(accuracy: float) -> float:
        return top + plot_h * (1 - accuracy)

    def path(values: list[float]) -> str:
        return " ".join(
            ("M" if i == 0 else "L") + f"{px(i):.1f} {py(v):.1f}" for i, v in enumerate(values)
        )

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Accuracy over '
        f'{len(nodp)} rounds, with and without privacy, against the pooled baseline">'
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = py(tick)
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="var(--rule)" stroke-dasharray="2 4"/>'
            f'<text x="{left - 6}" y="{y + 3:.1f}" text-anchor="end" class="lbl" '
            f'fill="var(--slate)">{int(tick * 100)}%</text>'
        )
    for round_index in (0, 4, 9, 14, 19):
        x = px(round_index)
        out.append(
            f'<text x="{x:.1f}" y="{height - bottom + 14}" text-anchor="middle" class="lbl" '
            f'fill="var(--slate)">{round_index + 1}</text>'
        )
    out.append(
        f'<text x="{width - right}" y="{height - bottom + 30}" text-anchor="end" class="lbl" '
        'fill="var(--slate)">rounds</text>'
    )
    out.append(
        f'<line x1="{left}" y1="{py(pooled):.1f}" x2="{width - right}" y2="{py(pooled):.1f}" '
        'stroke="var(--ink)" stroke-dasharray="4 4"/>'
        f'<text x="{left + 4}" y="{py(pooled) - 5:.1f}" class="lbl" fill="var(--ink)">'
        "all the data in one place "
        f"{html.escape(DATA['figures']['pooled']['display'])}</text>"
    )
    out.append(
        f'<path d="{path(nodp)}" fill="none" stroke="var(--global)" stroke-width="2"/>'
        f'<path d="{path(dp)}" fill="none" stroke="var(--budget)" stroke-width="2"/>'
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# the inversion panels
# --------------------------------------------------------------------------


def data_uri(filename: str) -> str:
    payload = (MANIFEST_JSON.parent / filename).read_bytes()
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def arm(noise_zero: bool) -> dict[str, Any]:
    for entry in MANIFEST["entries"]:
        if (entry["noise_multiplier"] == 0) == noise_zero:
            return entry
    raise SystemExit(
        "docs/inversion/manifest.json needs one entry with noise_multiplier 0 and one above it"
    )


def panels(entry: dict[str, Any], right_title: str) -> str:
    condition = f"batch {entry['batch_size']} · noise ×{entry['noise_multiplier']:.3f} · " + (
        "no privacy budget spent" if entry["epsilon"] is None else f"ε {entry['epsilon']:.3f}"
    )
    return (
        '<div class="panels">'
        f"<figure><figcaption>The page</figcaption>"
        f'<img src="{data_uri(entry["original_png"])}" alt="The original page, {html.escape(condition)}"></figure>'
        f"<figure><figcaption>{html.escape(right_title)}</figcaption>"
        f'<img src="{data_uri(entry["reconstruction_png"])}" alt="The reconstruction, {html.escape(condition)}"></figure>'
        "</div>"
        f'<p class="cond">{html.escape(condition)}</p>'
        f"<p>{html.escape(entry['caption'])}</p>"
    )


PENDING = any(entry.get("placeholder") for entry in MANIFEST["entries"])
PENDING_NOTICE = (
    '<p class="warn">Those two pictures are placeholders. The real ones are still being '
    "made, and rather than show you something that merely looks like a result, the panels "
    "say so. Every number and every other picture on this page is measured.</p>"
    if PENDING
    else ""
)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

CSS = """
:root {
  --ground:#eceee9; --ground-raised:#f0f2ee; --ink:#3a3d3c; --global:#1f3a5f;
  --client:#a64b2a; --budget:#c9922e; --slate:#6b7a82; --rule:#c6cbc2;
  --head:Bahnschrift,"Arial Narrow","Roboto Condensed",sans-serif;
  --mono:"Cascadia Mono","JetBrains Mono",Consolas,ui-monospace,monospace;
  --prose:system-ui,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ground:#1a1d1c; --ground-raised:#222624; --ink:#d4d8d2; --global:#7fa3d4;
    --client:#d98a63; --budget:#e0b566; --slate:#8a99a1; --rule:#3b403d;
  }
}
* { box-sizing:border-box; }
html { background:var(--ground); }
body {
  margin:0; padding:0 1rem 4rem; background:var(--ground); color:var(--ink);
  font-family:var(--prose); font-size:17px; line-height:1.6;
}
.wrap { max-width:44rem; margin:0 auto; }
header { border-bottom:2px solid var(--ink); padding:2rem 0 1rem; }
h1 { font-family:var(--head); font-size:1.9rem; text-transform:uppercase;
     letter-spacing:.02em; margin:0 0 .4rem; }
.sub { color:var(--slate); margin:0; }
nav { position:sticky; top:0; background:var(--ground); border-bottom:1px solid var(--rule);
      padding:.6rem 0; margin-bottom:2rem; z-index:5; }
nav ol { list-style:none; display:flex; flex-wrap:wrap; gap:.9rem; margin:0; padding:0;
         font-family:var(--mono); font-size:.72rem; }
nav a { color:var(--slate); text-decoration:none; border-bottom:1px solid transparent; }
nav a:hover, nav a.here { color:var(--ink); border-bottom-color:var(--ink); }
section { margin:0 0 3.5rem; scroll-margin-top:6.5rem; }
h2 { font-family:var(--head); font-size:1.35rem; text-transform:uppercase;
     letter-spacing:.02em; margin:0 0 .2rem; }
h3 { font-family:var(--head); font-size:1rem; text-transform:uppercase;
     letter-spacing:.02em; margin:1.6rem 0 .4rem; }
.step { font-family:var(--mono); font-size:.75rem; color:var(--slate); }
p { margin:0 0 1rem; }
figure { margin:0; }
svg { display:block; width:100%; height:auto; max-width:34rem; margin:1.2rem auto; }
.lbl { font-family:var(--mono); font-size:9px; }
.n { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.cap { border-left:2px solid var(--ink); padding-left:.8rem; margin:1.2rem 0; }
.panels { display:flex; flex-wrap:wrap; gap:1rem; margin:1.2rem 0 .6rem; }
.panels figure { flex:1 1 12rem; min-width:0; }
.panels figcaption { font-family:var(--head); font-size:.72rem; text-transform:uppercase;
                     letter-spacing:.02em; color:var(--slate); margin-bottom:.3rem; }
.panels img { width:100%; max-width:15rem; height:auto; border:1px solid var(--rule);
              background:var(--ground-raised); }
.cond { font-family:var(--mono); font-size:.75rem; color:var(--slate); }
.warn { border:1px solid var(--client); color:var(--client); padding:.6rem .8rem;
        font-size:.9rem; }
.note { color:var(--slate); font-size:.82rem; }
.legend { list-style:none; display:flex; flex-wrap:wrap; gap:1.4rem; padding:0;
          font-size:.9rem; }
.legend li { display:flex; align-items:center; gap:.5rem; }
.swatch { width:1.6rem; height:2px; display:inline-block; }
.three { display:grid; gap:1.4rem; }
.three figcaption { font-family:var(--mono); font-size:.75rem; color:var(--slate); }
.three svg { margin:.4rem 0 0; max-width:none; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; vertical-align:top; padding:.5rem .6rem .5rem 0;
         border-bottom:1px solid var(--rule); }
th { font-family:var(--head); text-transform:uppercase; letter-spacing:.02em;
     font-size:.75rem; color:var(--slate); }
td.an { font-family:var(--head); text-transform:uppercase; letter-spacing:.02em; }
code { font-family:var(--mono); font-size:.88em; }
a { color:inherit; text-decoration:underline; text-underline-offset:3px;
    text-decoration-color:var(--rule); }
a:hover { text-decoration-color:var(--ink); }
footer { border-top:1px solid var(--rule); padding-top:1rem; color:var(--slate);
         font-size:.82rem; }
/* On a phone the nav is seven lines; sticking it to the top eats the page. */
@media (max-width: 40rem) {
  nav { position:static; }
  section { scroll-margin-top:1rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""

# The only script on the page: underline the section you are reading. Nothing
# depends on it, and it is inert if scripting is off.
JS = """
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('nav a'));
  if (!('IntersectionObserver' in window) || !links.length) return;
  var byId = {};
  links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
  var seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      var link = byId[entry.target.id];
      if (link && entry.isIntersecting) {
        links.forEach(function (a) { a.classList.remove('here'); });
        link.classList.add('here');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });
  Object.keys(byId).forEach(function (id) {
    var section = document.getElementById(id);
    if (section) seen.observe(section);
  });
})();
"""

SECTIONS = [
    ("problem", "Nobody hands over their notebook"),
    ("round", "What one round looks like"),
    ("different", "Everybody knows different things"),
    ("attack", "Homework can be read backwards"),
    ("defence", "Trimming and smudging"),
    ("cost", "What being careful costs"),
    ("engineers", "For engineers"),
]

ANALOGIES = [
    (
        "Trimming",
        "L2 clipping at norm S",
        f"Every client update is scaled down to a maximum L2 norm before it is used. Here S = {DATA['figures']['clipNorm']['display']}.",
        [
            (
                "docs/adaptive_clipping.md",
                "the clip bracket and why the fixed clip stays the default",
            ),
            ("docs/_final_batch_b.json", "the run this page quotes"),
        ],
    ),
    (
        "Ink",
        "Gaussian noise at multiplier z, added to the aggregate",
        f"Noise with standard deviation z·S is added to the summed update, not to each client's. Here z = {DATA['figures']['noiseZ']['display']}.",
        [
            ("docs/dp_diagnosis.md", "the noise actually applied, measured, and the sweep over z"),
            ("docs/_noise_sweep.json", "the sweep"),
        ],
    ),
    (
        "The ink budget",
        "ε, computed by dp_accounting — not chosen",
        f"ε = {DATA['figures']['epsilon']['display']} at δ = {DATA['figures']['delta']['display']} over {DATA['figures']['rounds']['display']} rounds, from the sampled Gaussian accountant. z is calibrated to hit a target ε; ε is never asserted.",
        [
            ("docs/_epsilon_gate.json", "the gate that proves ε does not move with the clip"),
            ("docs/femnist_budget.md", "spending the budget on FEMNIST"),
        ],
    ),
    (
        "The merge",
        "Sample-weighted FedAvg over the sampled cohort",
        "Each client's update is weighted by how many examples it trained on, summed over the cohort that reported in time, and applied to the global model.",
        [("docs/architecture.md", "hand-written FedAvg, TFF for the DP aggregation only")],
    ),
    (
        "The class textbook",
        "The global model",
        f"{DATA['figures']['modelParams']['display']} parameters, broadcast to every client each round — including the ones that were not sampled.",
        [("docs/architecture.md", "one canonical wire format, conversions at the edges")],
    ),
    (
        "Sealed envelopes",
        "Pairwise additive masking (secure aggregation)",
        "Each pair of clients derives a shared mask; one adds it, the other subtracts it, so the masks cancel exactly in the sum. The server sees uniformly-masked vectors and a bit-identical total. A teaching implementation of Bonawitz et al., not production cryptography — the module lists its own elisions.",
        [
            (
                "docs/secure_aggregation.md",
                "the protocol, the dropout recovery, and what is elided",
            ),
            ("docs/architecture.md", "why secure aggregation and DP are complementary"),
        ],
    ),
]


def analogy_rows() -> str:
    rows = []
    for analogy, mechanism, detail, docs in ANALOGIES:
        links = " · ".join(
            f'<a href="{BLOB}{path}"><code>{path}</code></a> — {html.escape(what)}'
            for path, what in docs
        )
        rows.append(
            f'<tr><td class="an">{html.escape(analogy)}</td>'
            f"<td><strong>{html.escape(mechanism)}</strong><br>{html.escape(detail)}"
            f'<br><span class="note">{links}</span></td></tr>'
        )
    return "".join(rows)


def nav_html() -> str:
    items = "".join(
        f'<li><a href="#{slug}">{html.escape(title)}</a></li>' for slug, title in SECTIONS
    )
    return f"<nav><ol>{items}</ol></nav>"


def build() -> str:
    undefended = arm(noise_zero=True)
    defended = arm(noise_zero=False)
    sources = sorted(
        {entry["source"]["file"] for entry in DATA["figures"].values()}
        | {entry["source"]["file"] for entry in DATA["series"].values()}
    )
    source_links = ", ".join(f'<a href="{BLOB}{f}"><code>{f}</code></a>' for f in sources)

    body = f"""
<div class="wrap">
<header>
  <h1>How a computer learns from your data without ever seeing it</h1>
  <p class="sub">A class of students, a teacher, and ten notebooks that never leave the room.
  About ten minutes. Every number on this page was measured; hover any of them to see which
  file it came from.</p>
</header>

{nav_html()}

<section id="problem">
  <p class="step">One</p>
  <h2>Nobody hands over their notebook</h2>
  <p>Imagine a classroom. Every student has a notebook, and each notebook is different: one
  student has pages and pages on chapter three, another has almost nothing on it. The bars
  drawn on each desk below are real — they are the actual notebooks from a real run of this
  system.</p>
  <p>Now the rule. The notebooks are not allowed to leave the room. They might be hospital
  records, or everything you have ever typed on your phone. Throw one at the teacher and it
  bounces off the wall, which is what the dashed circle is.</p>
  {classroom_svg(show_round=False)}
  <p class="cap">The notebooks never leave. Only the lessons learned from them do.</p>
  <p>So the teacher has a problem. They want to write one textbook that is good for the whole
  class, and they are never allowed to read a single notebook.</p>
</section>

<section id="round">
  <p class="step">Two</p>
  <h2>What one round looks like</h2>
  <p>Here is the trick, and it is the whole idea. Instead of collecting notebooks, the teacher
  sends out the current textbook and asks for <em>corrections</em> — the marked-up homework that
  says &ldquo;this bit is wrong, it should say this instead&rdquo;.</p>
  <p>A round goes like this. The teacher picks a handful of students, not the whole class. Each
  of them studies their own notebook and works out what the textbook gets wrong for them. Each
  sends back only their corrections. The teacher merges all the corrections into a new edition
  and hands that new edition to <em>everyone</em>, including the students who were not asked
  this time.</p>
  {classroom_svg(show_round=True)}
  <p>The picture above is a real round — round <span class="n">{DROP_ROUND}</span> of a recorded
  <span class="n">{TOTAL_ROUNDS}</span>-round run. The thickness of each line is how much that
  student's corrections actually weighed. One line is dashed and its student is faded: that
  student was too slow, their corrections arrived after the teacher had already merged, and they
  were thrown away. That happens, and the system is built to expect it.</p>
  <p class="cap">Corrections travel. Notebooks stay put.</p>
</section>

<section id="different">
  <p class="step">Three</p>
  <h2>Everybody knows different things</h2>
  <p>Here is what makes this hard, and it is worth slowing down for. Nobody chose who got which
  chapters. A hospital sees the patients who live near it. A phone sees the words its owner
  types. So the notebooks are not just different, they are lopsided in ways nobody planned.</p>
  <p>Three classrooms, all with ten students. On the left, everybody has a bit of everything. On
  the right, most students have notes on one chapter and nothing else.</p>
  <div class="three">
    <figure><figcaption>Sharing is even</figcaption>{notebook_grid_svg(10.0, "even")}</figure>
    <figure><figcaption>Sharing is lopsided</figcaption>{notebook_grid_svg(0.5, "lopsided")}</figure>
    <figure><figcaption>Almost one chapter each</figcaption>{notebook_grid_svg(0.05, "one chapter each")}</figure>
  </div>
  <p class="cap">Some students only have notes on a few chapters. That is the hard part of this
  whole field.</p>
  <p class="note">Those three pictures are drawn the same way the real classrooms are dealt out,
  but they are examples of the shape rather than a measurement of one particular run.</p>
</section>

<section id="attack">
  <p class="step">Four</p>
  <h2>Homework can be read backwards</h2>
  <p>Now the uncomfortable part. The teacher never saw a notebook — only corrections. It turns
  out that corrections can be read backwards: with enough patience you can work out roughly what
  page must have produced them.</p>
  {panels(undefended, "What someone rebuilt from the corrections")}
  <p class="cap">From nothing but the corrections, someone rebuilt a page of a student's
  notebook.</p>
  <p>Be careful how much that proves. It is one student, one very small piece of homework, one
  study step, and nothing standing in the way. It gets much harder as any of those grow. But
  &ldquo;the data never left the room&rdquo; is not, by itself, a promise that the data is
  private — which is why the next part exists.</p>
  {PENDING_NOTICE}
</section>

<section id="defence">
  <p class="step">Five</p>
  <h2>Trimming and smudging</h2>
  <h3>First, trim every correction to the same length</h3>
  <p>No student is allowed to shout. Every set of corrections is cut back to the same maximum
  size — trimmed — so no single student can swing the textbook on their own. Below, twenty
  corrections; the ones that stick out past the line get cut back to it.</p>
  {trim_svg()}
  <p>In the last round of the run this page quotes, {fig("clippedFinal")} of corrections were
  long enough to be trimmed. The middle one was {fig("medianNormFinal")} long against a trim
  line set at {fig("clipNorm")}.</p>

  <h3>Then, spill a little ink on the pile</h3>
  <p>Before the teacher reads the pile of corrections, a small random smudge is added to it —
  ink. Not enough to ruin the lesson, but enough that no single student's handwriting can be
  picked out. Here the smudge is {fig("noiseZ")} times the size of the trim line.</p>

  <h3>The ink runs out</h3>
  <p>Here is the part that makes this a promise rather than a hope. There is a fixed budget of
  ink. Every round spends some of it, and when it runs out, training stops.</p>
  {budget_svg()}
  <p>After {fig("rounds")} rounds, this run had spent a budget of {fig("epsilon")}, at a
  δ of {fig("delta")}. That number is not a choice made afterwards — it is calculated by an
  accountant that knows how much noise was added and how often, and it is the honest answer
  whether you like it or not.</p>

  <h3>Now run the same attack again</h3>
  {panels(defended, "What the attack rebuilt this time")}
  <p class="cap">Same attack, same student, same homework. Trimmed and smudged, there is
  nothing left to read.</p>
  {PENDING_NOTICE}
  <p>There is one more trick in this project that is not in the pictures. Students can seal their
  corrections in envelopes that only cancel out when you add every envelope together — so the
  teacher can read the total of the class's corrections but never a single student's. That is
  secure aggregation, and it is in the table at the bottom.</p>
</section>

<section id="cost">
  <p class="step">Six</p>
  <h2>What being careful costs</h2>
  <p>Trimming and smudging are not free. They make the textbook a bit worse. The honest thing to
  do is to measure how much worse, and print the number.</p>
  {curves_svg()}
  <ul class="legend">
    <li><span class="swatch" style="background:var(--global)"></span> no privacy, ends at {fig("nodpFinal")}</li>
    <li><span class="swatch" style="background:var(--budget)"></span> with privacy, ends at {fig("dpFinal")}</li>
  </ul>
  <p class="cap">Privacy cost {fig("dpCost")} points of accuracy here. We measured it rather than
  hiding it.</p>
  <p>And now the wider frame, which is the thing most explanations leave out. Splitting the class
  up costs <em>more</em> than the privacy does. If every notebook could sit in one pile on one
  desk, the textbook would score {fig("pooled")}. Spread across the class it scores
  {fig("nodpFinal")}. Spread across the class and made private, {fig("dpFinal")}.</p>
  <p>That gap is not fixed either. Let the same class work for {fig("longRounds")} rounds instead
  of {fig("rounds")} and the no-privacy line climbs to {fig("longRoundsAcc")}. Time buys back
  some of what splitting up costs.</p>
  <p>On an easier subject — sorting photographs of clothes, instead of reading handwriting —
  the same system reaches {fig("fashionDense")} without privacy.</p>
</section>

<section id="engineers">
  <p class="step">Appendix</p>
  <h2>For engineers</h2>
  <p class="note">Every analogy above, mapped back to the mechanism, with the document each one
  is measured in.</p>
  <table>
    <thead><tr><th>In the story</th><th>In the code</th></tr></thead>
    <tbody>{analogy_rows()}</tbody>
  </table>
  <h3>The configuration behind the numbers</h3>
  <p>FEMNIST, {fig("writers")} writers, {fig("cohort")} clients sampled per round,
  {fig("rounds")} rounds, {fig("localEpochs")} local epochs, client-level DP with the noise
  multiplier calibrated to a target ε. Both arms are means over three seeds; the accuracy curves
  above are the per-round means of those same three runs.</p>
</section>

<footer>
  <p>Generated by <code>scripts/build_how_it_works.py</code> from {source_links} and
  <code>docs/inversion/manifest.json</code>. No number on this page was typed by hand; hover any
  of them for the file and JSON pointer it came from.</p>
  <p><a href="{REPO}">The repository</a> has the code, the configs, and every result quoted
  here — including the runs that went badly.</p>
</footer>
</div>
"""

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>How a computer learns from your data without ever seeing it</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body.strip()}\n"
        f"<script>{JS}</script>\n"
        "</body>\n</html>\n"
    )


def main() -> int:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT} — {OUT.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
