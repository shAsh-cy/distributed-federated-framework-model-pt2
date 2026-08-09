# Adaptive clipping: implementation, accounting, and the measured comparison

**Status: measured, including the cold start and the quantile follow-ups.**
Six phases, unattended, raw data in `docs/_final_batch_{a..f}.json`. The
verdict after all of them: **adaptive clipping is a reliable quantile
tracker, and that is both its strength and its limit.** Warm-started at a
tuned clip it holds it (FEMNIST: a match). Cold-started it finds *the
median* — which on every configuration measured here where the tuned
optimum was a *binding* clip is the wrong target (Fashion at q=0.5; the
FEMNIST cold start). A lower target quantile recovers Fashion's fixed
performance — which relocates the tuning problem rather than removing it.

## What is implemented

`privacy.adaptive_clipping: true` switches the DP aggregator to TFF's
quantile-based adaptive clipping (`gaussian_adaptive`, Andrew et al. 2021).
`l2_clip_norm` becomes the initial estimate; `adaptive_target_quantile`
(default 0.5 — track the median), `adaptive_learning_rate` (default 0.2,
geometric) and `adaptive_clipped_count_stddev` (default `m/20`) are
configurable and validated. The fixed-norm path is unchanged and remains
the default. The adapted clip is recorded per round (`adapted_clip` in the
metrics JSON), which is what the trajectory tables below are plotted from.

## The privacy accounting, settled and now measured

The quantile estimate consumes budget and the split is explicit: TF Privacy
noises the clipped-count bit (sensitivity ½) at stddev σ_b and inflates the
value noise to `z_v = (z⁻² − (2σ_b)⁻²)^(−1/2)`; the two compose back to
**exactly the nominal z** by Gaussian σ-additivity, so **total ε for an
adaptive run equals the fixed arm's** — confirmed in the recorded runs:

| Dataset | nominal z | value z | count z | ε total (= fixed arm) | ε value alone | ε count alone |
|---|---|---|---|---|---|---|
| FEMNIST (m=200) | 1.1141 | 1.1159 | 20.0 | **6.228** | 6.211 | 0.168 |
| Fashion (m=50) | 2.0 | 2.1822 | 5.0 | **6.228** | 5.559 | 2.021 |

dp_accounting CAN price the components separately (they are distinct
Gaussian events in TFF's `ComposedDpEvent` state) — the columns above do —
but the parts do not naively add: 6.211 + 0.168 and 5.559 + 2.021 both
exceed 6.228, because independent composition over-counts relative to the
σ-additive total. At m=200 the quantile estimate is nearly free (value
noise 1.1159 vs 1.1141); at m=50 it is visibly not.

## The fixed arm it was measured against (phases A and B)

The old FEMNIST clip bracket was measured at the E=1 stall and re-scoped
([femnist_budget.md](femnist_budget.md) §5), so phase A re-bracketed at the
working budget (E=10, m=50, R=20, one seed, z calibrated to ε=6.228):

| S | final acc | clipped fraction |
|---|---|---|
| 0.5 | 0.503 | 1.00 |
| 1.0 | 0.618 | 0.92 |
| **2.0** | **0.624** | **0.69** |
| 4.0 | 0.546 | 0.14 |

Textbook shape: binds-everywhere hurts (the Fashion-era S=0.5 costs 12pp),
binds-never hurts (noise bought for nothing), the begins-to-bind knee wins.
Phase B then ran the DP arm proper — E=10, m=200, R=20, S=2.0, z=1.1141,
three seeds: **0.6815 (range 0.007)** against the recorded no-DP control
0.7279 (range 0.002). **Client-level DP at ε=6.228 costs 4.6pp on FEMNIST
at the working budget** — against the 12.8pp the federated setup itself
costs relative to the 85.6% pooled baseline, and beside the ~3.5pp DP cost
on Fashion. On both datasets, DP costs roughly a third of federation.

## Adaptive vs fixed, measured (phases C and D)

**Which arms started where — this matters for reading the table.** Both
adaptive arms were WARM-STARTED from the fixed arm's clip, not TFF's 0.1
default: the FEMNIST arm at 2.0 (phase A's bracket answer), the Fashion arm
at 0.5 (the tuned sweep winner). With 20 rounds and geometric rate 0.2 a
cold start would measure warm-up, not adaptation — so these phases measure
whether the estimator *holds* a tuned clip, not whether it *finds* one.
The finding question is phase E, below.

| | fixed | adaptive | per-seed (adaptive) |
|---|---|---|---|
| FEMNIST, E=10, m=200, ε=6.228 | 0.6815 (range 0.007) | **0.6830 (range 0.006)** | 0.684 / 0.680 / 0.685 |
| Fashion, m=50, ε=6.228 | **0.7240 (range 0.032)** | 0.7006 (range 0.051) | 0.722 / 0.671 / 0.709 |

**Reconciliation the reader deserves** (audit finding M4): this Fashion
fixed arm is the SAME cell dp_diagnosis §8.2 recorded at **0.7336 (range
1.1 pp)** — an independent three-seed re-measurement months apart, under
noise that TFF draws unseedably. The two draws differ by 1.0 pp with
overlapping ranges; both are committed (`_replication.json`,
`_final_batch_d.json`), neither is "the" number, and the DP-cost figure
quoted from §8.2 (~3.5 pp) would read ~4.5 pp against this draw. Same
mechanism, same ε; the spread is the unseedable noise doing what this repo
has always said it does.

**FEMNIST: a match inside seed ranges** (+0.15pp on the mean). An earlier
revision of this document claimed the match was achieved "without knowing
the bracket answer in advance" — that was wrong, and is corrected here: the
arm was warm-started AT the bracket answer, so it demonstrates holding, not
finding. **Fashion: adaptive trails by 2.3pp on the mean** with
overlapping seed ranges (fixed 0.703–0.735, adaptive 0.671–0.722); by this
repo's resolution standards the gap is not fully resolved, but adaptive
never beats fixed on any seed pairing, and the mechanism (next section)
says the direction is real. A negative result on a principled method is
still a result.

## The trajectories: the mechanism visibly works — and that is the problem on Fashion

Adapted clip vs measured median pre-clip norm, seed 42, all rounds:

| round | FEMNIST clip | FEMNIST median ‖Δw‖ | frac | | Fashion clip | Fashion median ‖Δw‖ | frac |
|---|---|---|---|---|---|---|---|
| 1 | 1.81 | 1.30 | 0.04 | | 0.55 | 0.88 | 0.92 |
| 3 | 1.55 | 0.93 | 0.07 | | 0.66 | 1.03 | 0.88 |
| 5 | 1.64 | 2.41 | 0.93 | | 0.72 | 0.81 | 0.70 |
| 7 | 1.99 | 2.66 | 0.98 | | 0.79 | 0.90 | 0.74 |
| 9 | 2.32 | 2.57 | 0.69 | | 0.78 | 0.79 | 0.50 |
| 12 | 2.38 | 2.40 | 0.51 | | 0.74 | 0.76 | 0.56 |
| 15 | 2.25 | 2.14 | 0.43 | | 0.76 | 0.66 | 0.36 |
| 18 | 2.16 | 2.01 | 0.39 | | 0.73 | 0.72 | 0.48 |
| 20 | 2.08 | 1.91 | 0.41 | | 0.72 | 0.76 | 0.60 |

On FEMNIST the clip dips while early norms are small, chases the norm jump
at rounds 4–7 with a two-round lag, and settles just above the median with
the clipped fraction at ~0.41 ≈ the 0.5 target: the estimator tracks. On
Fashion it tracks just as faithfully — **away from the optimum**. The
Fashion working configuration S=0.5 was chosen by sweep and sits in the
*binding* regime (norms ~0.8, clip 0.5, fraction ~0.9): the repo's own
clip-as-step-size finding says that clip functions as beneficial server
learning-rate control. A median-targeting estimator dutifully raises the
clip to ~0.75, un-binding it — more noise (noise scales with the clip),
less implicit step-size control, slightly worse accuracy. Adaptive
clipping finds the quantile it is told to find; when the tuned optimum
lies *below* the median, that is the wrong target.

## The cold start (phase E): adaptation finds *a* clip — not *the* clip

One seed, stated as such. FEMNIST, E=10, m=50, R=100, ε=6.228 recalibrated
for 100 rounds (z=0.8232); adaptive from TFF's naive 0.1 default beside a
matched fixed arm at the bracket answer S=2.0:

| | round 20 | round 100 |
|---|---|---|
| fixed S=2.0 | **0.593** | **0.624** |
| adaptive from 0.1 | 0.339 | 0.548 |

The clip needs **31 rounds** to reach the neighbourhood of 2.0 (0.26 at
round 10, 0.68 at 20, 1.78 at 30) — matching the theoretical maximum climb
rate of e^(0.1) per round when everything clips, so no faster start was
available at this learning rate. The warm-up half of the verdict was
expected: at round 20 the cold arm is 25pp behind, and a 20-round budget
cold-started from 0.1 would have measured nothing else.

The unexpected half: **it never catches up.** From round ~35 the clip
overshoots to 2.6–3.0 and settles at the median (final clip 2.81, clipped
fraction 0.50 — the estimator doing exactly its job), and accuracy
plateaus at ~0.55, still 7.6pp behind the fixed arm at round 100. Note the
feedback: the adaptive arm's own median norms run 2.6–3.1 while the fixed
arm's run ~2.6 — a larger clip buys proportionally larger noise, a noisier
model produces larger local updates, and the median the estimator chases
is partly its own noise. The equilibrium "clip ≈ median" is *above* the
bracketed optimum, which sits deliberately in the binding regime below the
median. Same lesson as Fashion, now on FEMNIST: tracking the median is
only the right move when the optimum happens to be there.

## The Fashion quantile (phase F): recovery, at the price of the point

Aiming the estimator below the median instead — target quantile 0.2 and
0.35, 3 seeds each, warm-started at 0.5 like the recorded q=0.5 arm. Every
arm in this table is DP at the working budget: z = 2.0, m = 50, giving
**ε = 6.228 at δ = 1×10⁻⁵**, identical across arms by the σ-additivity
identity above:

| arm | mean | range | per-seed |
|---|---|---|---|
| fixed S=0.5 | **0.7240** | 0.032 | 0.733 / 0.735 / 0.703 |
| adaptive q=0.2 | 0.7184 | 0.041 | 0.725 / 0.695 / 0.736 |
| adaptive q=0.35 | 0.7201 | 0.034 | 0.732 / 0.730 / 0.698 |
| adaptive q=0.5 | 0.7006 | 0.051 | 0.722 / 0.671 / 0.709 |

**Yes: a lower quantile recovers the fixed arm's performance** — both 0.2
and 0.35 land within half a point of fixed, inside overlapping ranges, and
the q=0.2 trajectory steers the clip to ~0.49–0.56, i.e. onto the tuned
value the sweep found. Insensitivity between 0.2 and 0.35 is worth noting:
the quantile knob is coarse, at least here.

**And the uncomfortable half, stated without softening: this is
relocation, not removal.** The lower quantile had to be *chosen*, and it
was chosen by already knowing the fixed sweep's answer — that the optimum
binds. Adaptive clipping with a tuned target quantile is a tuned method.
Combined with phase E — where the untuned default (track the median)
converged confidently to the wrong operating point — the value proposition
is weaker than the phase-C match alone suggested: the estimator reliably
delivers whatever quantile it is aimed at, and aiming it is the same kind
of problem that bracketing the clip was, one level up. No rule for
choosing the target quantile from first principles is offered here,
because these experiments do not provide one.

## Decision, per the criterion stated before the data

The stub's criterion: match-or-beat the bracketed optimum within seed
ranges while removing the bracket → recommended default; trail materially →
opt-in. After all six phases: **the fixed path stays the default**, and
the earlier "recommended starting point when no bracket exists" claim is
withdrawn — the cold start is the no-bracket scenario, and it lost by
7.6pp at five times the short budget (one seed). What survives honestly:
adaptive clipping *holds* a known-good clip through drift in the norms
(phase C), and with a target quantile tuned against known behaviour it
matches fixed (phase F). Both uses presuppose the knowledge the method was
hoped to replace.

## Reproducing

```bash
python scripts/final_batch.py >> docs/_final_batch.log 2>&1
```

One unattended script, all six phases (A–F → `docs/_final_batch_{a..f}.json`),
resumable: a phase whose JSON already exists is skipped, so re-running after
a crash continues where it stopped. Run detached; do not pipe stdout into a
command that waits for EOF (TFF holds the pipe). Wall-clock ~9 h for all six
phases from scratch on CPU. (Previously documented nowhere — audit finding
D10.)
