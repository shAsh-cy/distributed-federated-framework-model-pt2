# Adaptive clipping: implementation notes and the runs still to do

**Status: implemented and unit-tested; NOT yet compared experimentally.**
No figure in this document is a result — the comparison runs below have not
been run, and nothing here should be quoted as evidence that adaptive
clipping helps or hurts on this codebase.

## What is implemented

`privacy.adaptive_clipping: true` switches the DP aggregator to TFF's
quantile-based adaptive clipping (`gaussian_adaptive`, Andrew et al. 2021).
`l2_clip_norm` becomes the initial estimate; `adaptive_target_quantile`
(default 0.5 — track the median), `adaptive_learning_rate` (default 0.2,
geometric) and `adaptive_clipped_count_stddev` (default `m/20`) are
configurable and validated. The fixed-norm path is unchanged and remains the
default. The adapted clip is recorded per round (`adapted_clip` in the
metrics JSON) so its trajectory can be plotted against the measured median
update norm.

## The privacy accounting, settled

The quantile estimate consumes budget, and the split is explicit rather than
hidden: TF Privacy noises the clipped-count bit (sensitivity ½) at stddev
σ_b and inflates the value noise to `z_v = (z⁻² − (2σ_b)⁻²)^(−1/2)`, so the
two components compose back to **exactly the nominal z** by Gaussian
σ-additivity — TFF's own aggregator state carries the `ComposedDpEvent` of
both Gaussians (verified empirically on TFF 0.87: z=2, m=50 → z_v=2.182179,
z_count=5.0, matching hand computation). Consequently:

- **Total ε for an adaptive run = `compute_epsilon` at the nominal
  multiplier — identical to the fixed-clip path.** The recorded ε=6.228
  anchor is unaffected by switching clipping modes.
- `fl.aggregation.adaptive_noise_breakdown()` reports the split
  (value-noise multiplier, count-noise multiplier, σ_b); per-component ε's
  are computable individually, with the caveat — documented in tests — that
  naively summing them over-counts relative to the σ-additive total.
- dp_accounting **can** separate the components (they are distinct Gaussian
  events); the tight total, however, comes from the σ-additivity identity,
  not from composing the parts.

## Comparison runs to be done (later session; machine currently occupied)

All at 3 seeds per arm, means with ranges, per the repo rule.

1. **Fashion-MNIST at its working budget** — the corrected configuration
   (m = 50 of N = 100, 20 rounds, z = 2.0, ε = 6.228):
   adaptive (initial clip 0.1, quantile 0.5) vs the best fixed clip
   (S = 0.5, mean 73.4 % from the recorded replication).
2. **FEMNIST at E = 10** (m = 200 of N = 1000, 20 rounds): adaptive vs best
   fixed clip. NOTE: the "best fixed clip" at this budget is **unknown** —
   the earlier bracket ran at the E = 1 stall and is explicitly re-scoped
   (femnist_budget.md §5); the fixed arm therefore needs its own bracket
   first, at E = 10 where median update norms are ~1.9–2.6 and S = 0.5
   binds on every update.
3. **Per-run artefacts to record**: final/best accuracy per seed, the
   adapted-clip trajectory overlaid on the measured median pre-clip norm
   (the plot the per-round logging exists for), clipped fraction, achieved
   ε (must equal the fixed arm's — same nominal z), and wall-clock.

Expected decision criterion, stated before the data so it cannot drift: if
adaptive clipping matches or beats the *bracketed* best fixed clip within
seed ranges while removing the need for the bracket itself, it becomes the
recommended default; if it trails the bracketed optimum materially, it stays
opt-in with the trade-off documented.
