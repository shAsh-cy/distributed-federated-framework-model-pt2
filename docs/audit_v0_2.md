# v0.2 freeze audit — findings

Branch `audit/v0.2-freeze` at `abcd216` (merged main). Method: the live
dashboard path driven against a real coordinator (uvicorn in the dev image,
real training, real WebSockets, Playwright in live mode — the packaged e2e
pins `VITE_MOCK=1` and had never exercised any of this); a fresh clone from
GitHub taken through cold install, full test suite, lint, the verbatim
Quickstart compose command, and the dashboard's ci/build/test/e2e; all four
result documents read as one; static sweeps over config reachability, dead
code, and the full git history. Findings are reported here first; fixes land
as separate commits, each referencing its finding ID. Severity: **C**ritical
/ **M**ajor / **D**ocumentation / **S**cope-minor.

## C1 — The dashboard's WebSocket cannot connect to a real server at all

`requirements.txt` pins bare `uvicorn==0.23.2` with neither `websockets` nor
`wsproto`. Every WS test passes through Starlette's in-process TestClient,
which needs no WS server protocol; a real uvicorn rejects the upgrade:

    WARNING: No supported WebSocket library detected. …
    "GET /runs/test/events HTTP/1.1" 404 Not Found

Reproduced against the dev image. With `websockets==12.0` installed the full
contract holds end to end (verified over a real socket, real training):
contiguous seqs, live streaming, `run_started` carrying 5 clients × 10-class
label histograms with real counts, `cumulative_epsilon` rising per round
(3.25 → 4.56 → 5.55 on a 3-round DP run), `run_completed`, clean server
close. Fix: pin `websockets` in requirements.

## C2 — The coordinator can run exactly ONE DP run per process lifetime

Nothing in the repo installs a TFF execution context; it works only because
importing TFF installs one — in the importing thread. `fl/aggregation.py`
imports TFF lazily inside the first DP aggregation, so the context lands in
whichever Runner thread hosts the first DP run. TFF's context stack is
`threading.local`; every later DP run gets a fresh thread and dies:

    RuntimeError: No default context installed.   (fl/aggregation.py:266)

Observed live: the process's first DP run completed; the second (driven
through the dashboard's Configure form) failed at round 1's aggregation, and
the live Playwright run failed with it. No test starts two DP runs in one
process, so nothing caught it. The gRPC server and the one-shot harness
scripts aggregate on a single thread, which is why every recorded experiment
was immune. Fix: an explicit per-thread context guard in the DP aggregators.

## M1 — A killed coordinator leaves its runs "running" forever

Crash→failed is handled only for in-process exceptions. `docker restart`
mid-run left the run at `status: running` permanently (verified; it is still
there). A restarted coordinator should mark orphaned live runs failed on
startup. (Related minor: the run-status column updates after the terminal
event is broadcast, so a client reading status at stream-close can briefly
see `running` — observed once; settles correctly.)

## M2 — No completed run saves a model anyone can use

Audit item 2's question has a flat answer: no. All three execution paths
(gRPC server, coordinator runner, in-process harness) write metrics JSON and
discard the trained weights. The repo trains models nobody can load. Fix: a
checkpoint write at run end plus a load-and-classify script exercising both
the TF and torch paths.

## M3 — `privacy.adaptive_clipping: true` is silently ignored on two of three paths

The four `adaptive_*` config fields are forwarded to `make_aggregator` only
by `fl/server.py:658-661`. `coordinator/runner.py:187-192` and
`scripts/run_experiment.py:83-88` pass neither, and the dispatch defaults to
fixed clipping — so a coordinator/API run configured for adaptive clipping
runs fixed, with no warning, while reporting the config it did not honour.

## M4 — The README quotes two different results for the same Fashion arm

Fashion-MNIST, m=50, S=0.5, z=2.0, ε=6.228, three seeds — measured twice,
months of commits apart: **73.4 % (range 1.1 pp)** in dp_diagnosis §8.2 and
**72.4 % (range 3.2 pp)** as the fixed arm of the adaptive comparison
(`_final_batch_d.json`). Both are legitimate draws of an unseedable-DP
configuration, but README Results quotes both (`:327/:440` vs `:511/:520`)
with no note that they are independent replications of the same cell, and
adaptive_clipping.md imports "the ~3.5 pp DP cost" from the first while its
own table implies 4.5 pp against the same ceiling. Needs an explicit
reconciliation, not a silent choice of one.

## D1 — dp_diagnosis corrects a README claim that no longer exists

`dp_diagnosis.md:819-837` ("Correction to a claim in README.md") quotes and
corrects the README's "m ≳ z·√d, roughly 950 clients per round". No form of
that text remains in README. The correction reads as if the README still
carries the error.

## D2 — femnist_cohort still carries a retracted magnitude

`femnist_cohort.md:203-204`: "the same 6× clip change moved final accuracy by
**63 points**" — dp_diagnosis `:689-693` explicitly retired 63.5 pp (measured
against a low draw) in favour of ≈ 55 pp against the four-draw mean. The
sibling was never updated.

## D3 — femnist_cohort's forward pointers predate the work that answered them

`femnist_cohort.md:209-210` still says bracketing the clip "first requires a
budget at which FEMNIST trains (see the roadmap)" — femnist_budget.md found
that budget and adaptive_clipping.md phase A did that bracket. And §5's "no
clip optimum to locate" carries no marker that femnist_budget.md §5
re-scoped it to the stalled regime only; a reader entering here first leaves
misinformed.

## D4 — §7.0's ε-invariance gate has no committed artefact

`dp_diagnosis.md:915` lists `docs/_epsilon_gate.json` in Reproducing; the
file does not exist in the repo. The bitwise-ε claim it backs is pure
accountant arithmetic (no training), so the artefact is regenerable.

## D5 — The FEMNIST no-DP control table is unmapped

`femnist_cohort.md:152-156` — load-bearing for the document's conclusion —
has no Reproducing command and no named data file (`docs/_femnist_nodp.json`
exists and is evidently the source; nothing says so).

## D6 — The phase-F table breaks the repo's own ε-marking rule

`adaptive_clipping.md:157-161` (quantile recovery) shows DP accuracies with
no ε/δ or marking in or near the table; the equal-ε basis sits ~125 lines
earlier.

## D7 — The README's test figure is stale and unprovenanced

`README.md:619`: "342 tests, 92 % statement coverage" — the fresh clone
collects **378 tests** and measures **93 % statement coverage** (FEMNIST
real-data checks skip without the cache, as documented), and no provenance
is cited for the number anywhere.

## D8 — "docs/femnist_cohort.md for all experiments" is an overreach

`README.md:610` — budget, adaptive, and final-batch experiments live in
other documents; femnist_cohort's Reproducing block covers only four.

## D9 — "History on first launch" describes behaviour that does not exist

`README.md:254`: nothing invokes `coordinator.importer.import_history` — not
app startup, not a CLI (`coordinator/importer.py` has no entry point). A
fresh API serves an empty history until someone calls the function from
Python. The "97 runs on the current tree" count is also stale: running the
importer against the current tree imports **141 runs**. Fix: give the
importer a real entry point, state the true count and the command.

## D10 — The final batch is not reproducible from any document

`scripts/final_batch.py` — source of the six `_final_batch_*.json` files
behind the adaptive write-up and the FEMNIST DP headline — is named in no
README section and no doc's Reproducing block. (CI, further, executes no
script from scripts/ at all: they are linted, never run.)

## R1 — Roadmap vs reality

- "**More datasets and architectures**, beyond the single
  `fashion_mnist`/`small_cnn` pair the config currently accepts"
  (`README.md:772`) — stale: the config has accepted `femnist`/`femnist_cnn`
  since the cohort work; `/datasets` serves both. The bullet contradicts the
  repo's own Results section.
- Every other roadmap bullet names genuinely unbuilt work, and every
  limitation checked remains true (secure-aggregation wiring, robust
  aggregation, auth/TLS, re-tuned default clip — configs still ship S=3.0 —
  multi-host, persistent server state, the FEMNIST scoping items).

## S1 — Two tests assert nothing

`tests/test_config.py:26` (relies on ConfigError propagating; can also skip
silently) and `tests/test_rpc.py:35` (relies on a timeout raising). Both are
exception-reliant by design; each should say so with an assertion or
comment.

## S2 — Three dead public symbols

`fl/adapters.py:148 make_adapter`, `fl/aggregation.py:139 uniform_average`,
`fl/serialization.py:80 weights_fingerprint` — referenced only by tests,
documented nowhere.

**Resolution (recorded after re-reading the code):** the finding was
over-called for two of the three. `uniform_average`'s docstring states it is
"kept as a named function so the difference from weighted_average is
explicit and directly testable" — it is the executable half of the README's
weighted-vs-unweighted signature test, deliberate test-facing API.
`weights_fingerprint` is the declared bit-identity primitive of the wire
round-trip tests. Both stay. `make_adapter` gains a production caller:
`scripts/predict.py` uses it to load checkpoints into the torch model (M2's
fix), which is what a factory like it was for.

## What passed (verified, no findings)

- **Fresh clone** (GitHub, `abcd216`): mode bits exactly right (three `.sh`
  at 100755, nothing else executable); **cold `pip install` on clean
  python:3.10-slim succeeds** and imports torch-first cleanly (the
  documented torch-before-TF abort reproduces on a cold env exactly as the
  repo warns); **full suite green — 378 tests collected, 93 % coverage,
  ruff lint+format clean** (FEMNIST real-data checks skip without the
  cache, as documented); **dashboard `npm ci`/build/test/e2e all green**;
  **the verbatim Quickstart compose command trains five containers to
  79.03 % in 5 rounds** — inside the documented 77–79 % band — writes
  `/app/results/docker_run.json` and every container exits 0, exactly as
  README states.
- **Live event contract** (post-C1): everything item 1 asked, at API level —
  histograms, curves data, ε accumulation, clean close; stop-mid-round
  drops the un-reported cohort as `client_dropped` (2 observed live) and the
  partial round is not aggregated; run status settles to `stopped`.
- **Live UI** (Playwright in live mode, real API, real DP training —
  `dashboard/e2e-live/`, committed with this audit): the full item-1
  checklist passes once C1 and C2 are fixed — WS connects, topology renders
  and animates 5 real clients from `run_started`'s histograms, the round
  counter advances through 003, curves draw from real `round_aggregated`
  events, the event log follows with per-round ε, the privacy budget meter
  shows the accumulating ε, and `run_completed` closes the stream with the
  console reading "status completed · stream closed". One cosmetic
  curiosity, recorded not chased: the charts render one degenerate
  1-px-wide curve path alongside the real ones, so DOM assertions on the
  *first* path mislead. Charts need two aggregated rounds before any path
  draws — correct behaviour, worth knowing when watching short runs.
- **History hygiene**: no secrets in tree or history (all 182 files ever
  added reviewed by pattern; one historical `.env.example` held a localhost
  address only); largest object ever 915 KB (committed experiment JSON);
  no datasets, checkpoints, databases, node_modules, or build artifacts
  tracked, ever.
- **Config surface**: every non-adaptive config field is read on at least
  one production path (adaptive: see M3). No TODO/FIXME anywhere. No
  unreachable code, no commented-out blocks.

## The verdict (audit item 7)

A senior engineer with thirty minutes would clone it, watch the compose demo
train five containers to ~78 %, skim a README that states its own
retractions more honestly than most papers, spot-check a claim against a
committed JSON and find it exact, and conclude this is a rare thing: a
learning repo whose evidence discipline is real, whose tests are green from
a cold clone, and whose negative results are written up as carefully as its
positive ones. The single weakest thing they would find, if they clicked
one level deeper, is that the coordinator-and-dashboard layer — the repo's
most demo-able surface — had never once been run against real training: the
WebSocket endpoint could not accept a connection from a real browser, the
second DP run in any server's lifetime crashed, and no run leaves behind a
model you can load. The system's observability layer was proven against
recorded fixtures and inferred to work live; the audit's one-line lesson is
that the inference was three bugs short of true.
