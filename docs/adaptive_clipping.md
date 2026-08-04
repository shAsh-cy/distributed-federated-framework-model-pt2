# Adaptive clipping: implementation, accounting, and the measured comparison

**Status: measured.** The comparison this document was stubbed for has run —
four phases, unattended, raw data in `docs/_final_batch_{a,b,c,d}.json`.
The one-line verdict, stated the way the decision criterion below demands:
**adaptive clipping matches the bracketed fixed clip on FEMNIST while
removing the need for the bracket, and trails it on Fashion-MNIST.**

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

Both adaptive arms start from the fixed arm's clip, not TFF's 0.1 default:
with 20 rounds and geometric rate 0.2, a cold start from 0.1 would measure
warm-up, not adaptation (a deliberate deviation from this document's
original stub, recorded in the batch script's docstring).

| | fixed | adaptive | per-seed (adaptive) |
|---|---|---|---|
| FEMNIST, E=10, m=200, ε=6.228 | 0.6815 (range 0.007) | **0.6830 (range 0.006)** | 0.684 / 0.680 / 0.685 |
| Fashion, m=50, ε=6.228 | **0.7240 (range 0.032)** | 0.7006 (range 0.051) | 0.722 / 0.671 / 0.709 |

**FEMNIST: a match inside seed ranges** (+0.15pp on the mean), achieved
without knowing the bracket answer in advance — which is the entire value
proposition. **Fashion: adaptive trails by 2.3pp on the mean** with
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

## Decision, per the criterion stated before the data

The stub's criterion: match-or-beat the bracketed optimum within seed
ranges while removing the bracket → recommended default; trail materially →
opt-in. The split result gives a split answer, resolved conservatively:
**the fixed path stays the default.** Adaptive clipping is the recommended
*starting point when no bracket exists* — on FEMNIST it recovered the
bracket's answer for free — but where a tuned clip already exists it should
be kept, and where the tuned optimum is a binding clip, a 0.5-quantile
target is actively mismatched (a lower `adaptive_target_quantile` would be
the thing to try, and has not been).
