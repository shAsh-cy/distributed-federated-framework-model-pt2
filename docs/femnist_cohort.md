# FEMNIST: the decoupled cohort experiment

**Goal.** The Fashion-MNIST cohort sweep ([dp_diagnosis.md](dp_diagnosis.md)
§6–§8) could not attribute its own result: holding q = 0.5 there forced
N = 2m, so growing the cohort also thinned every shard, and accuracy moved for
two unrelated reasons at once. This experiment removes the confound with a
real fixed client population — LEAF-derived federated EMNIST, partitioned by
writer — and re-runs the cohort-size question in the only design that can
answer it: **shard size constant, cohort size the sole variable.**

All accuracy figures in this document are means over three seeds (42, 43, 44)
with the range stated. DP cells vary beyond the seed because TFF's executor
draws noise unseedably (dp_diagnosis.md §10); the range column absorbs both.

---

## 1. The population

TFF's `emnist.load_data(only_digits=False)` — the federated EMNIST that TFF
documents as derived from the LEAF benchmark's preprocessing. 62 classes
(10 digits, 26 upper, 26 lower), 3,400 writers, partitioned by who actually
wrote each character. Nothing about the split is synthesised: each client *is*
one writer.

Experiments run on a seeded 1,000-writer subsample (`--writers 1000`, the
documented default), sized so the largest sweep cell (m = 500, 20 rounds)
stays CPU-feasible while the sweep still spans sampling rates q = 0.005 to
0.5. The full 3,400-writer set is available behind the same flag.

Measured population shape (1,000 writers, seed 42):

| Quantity | Value |
|---|---|
| Train examples | 196,241 |
| Test examples (pooled, same writers) | 22,639 |
| Shard size min / median / max | 30 / 159 / 392 |
| Pooled label entropy | 3.664 nats |
| **Mean per-writer label entropy** | **3.377 nats** |
| Median per-writer label entropy | 3.345 nats |
| Uniform entropy over 62 classes | 4.127 nats |

The per-writer distributions are measurably non-uniform — mean per-writer
entropy sits 0.29 nats below the pooled entropy (asserted in
`tests/test_femnist.py` with a loose margin; these are the measured values).
The skew is natural: writers genuinely differ in what and how much they wrote.

**Dataset caveat, found while testing:** 649 of 77,483 full-population test
images (0.84 %) are byte-identical to some train image after uint8
quantisation — all label-consistent, minimal-ink glyphs (mean ink fraction
3.9 %). The train split internally contains 2,732 byte-duplicates, so this is
upstream LEAF/TFF preprocessing, not a packing artefact of this repo; the
writer-level train/test split is taken from upstream verbatim. A test pins the
rate below 1 % so a packing regression would fail loudly.

## 2. Model and baselines

`femnist_cnn`: the same backbone as `small_cnn` with the logits layer widened
to 62 classes — **231,742 parameters** (hand-computed breakdown asserted in
tests), within 3 % of the Fashion model's 225,034, so DP noise magnitudes are
directly comparable across the two datasets. The LEAF reference CNN (~6.6 M
parameters) was deliberately not used: DP noise scales with √d and CPU round
time with d.

**Pooled centralised baseline** — all shards combined, no federation, the true
upper bound every federated number below should be read against (the repo
previously had only a no-DP *federated* control, which is not the same
ceiling):

| Seed | Epoch 1 | 2 | 3 | 4 | 5 (final) |
|---|---|---|---|---|---|
| 42 | 0.8194 | 0.8372 | 0.8484 | 0.8518 | 0.8561 |
| 43 | 0.8099 | 0.8355 | 0.8448 | 0.8504 | 0.8525 |
| 44 | 0.8153 | 0.8405 | 0.8523 | 0.8551 | 0.8592 |

**Pooled baseline: 85.6 % (mean of 3 seeds: 85.61 / 85.25 / 85.92, range
0.7 pp).** SGD lr 0.01, momentum 0.9, batch 64, 5 epochs over the pooled
196,241 examples (~174 s per seed on CPU). The curve is still rising ~0.4 pp
per epoch at epoch 5, so 85.6 % is a slight *under*-estimate of the
centralised ceiling — a conservative bound in the right direction for reading
federated numbers against. Per-writer test accuracy is naturally capped below
Fashion-MNIST levels: 62 classes with confusable pairs (O/0, l/1/I) put even
centralised EMNIST-62 in the mid-80s in the literature.

## 3. The decoupled sweep — design

Population fixed at N = 1,000 writers. Cohort m ∈ {5, 20, 50, 100, 200, 500}.
Rounds fixed at 20, δ = 1×10⁻⁵, clip S = 0.5 (the value the Fashion sweep
established against S = 3.0; §5 brackets it properly). Because shard size no
longer moves with m, **any accuracy change across cells is attributable to
cohort size alone.**

**The tension.** Holding ε fixed while m varies is no longer free, and this is
the substantive difference from the Fashion design: at fixed N, raising m
raises the sampling rate q = m/N, which *weakens privacy amplification by
subsampling* and therefore demands a larger noise multiplier z for the same
ε. But averaging over more clients suppresses the applied noise as 1/m. The
two effects pull in opposite directions, and which dominates over what range
is measured below, not derived.

z is calibrated per cell by bisection on the accountant
(`fl.aggregation.calibrate_noise_multiplier`); ε is still computed, never
chosen — each cell reports the ε its calibrated mechanism actually achieves:

| m | q = m/N | calibrated z | achieved ε | applied-noise norm S·z·√d/m |
|---|---|---|---|---|
| 5 | 0.005 | 0.4539 | 6.2286 | 21.85 |
| 20 | 0.020 | 0.5544 | 6.2280 | 6.67 |
| 50 | 0.050 | 0.6723 | 6.2277 | 3.24 |
| 100 | 0.100 | 0.8273 | 6.2275 | 1.99 |
| 200 | 0.200 | 1.1141 | 6.2283 | 1.34 |
| 500 | 0.500 | 2.0001 | 6.2281 | 0.96 |

Two mechanism-level observations worth noting before any accuracy is seen:
z rises 4.4× across the sweep while the applied noise still falls 23× — the
1/m averaging outpaces the amplification loss *in noise magnitude*. At
q = 0.5 the calibration recovers z = 2.000, the exact recorded Fashion-MNIST
pair, which anchors the calibrator to a known point. What this does to
*accuracy* is §4's question.

## 4. The decoupled sweep — results

Final accuracy, mean of three seeds with range; chance on 62 classes is 1.6 %:

| m | q | z | achieved ε | noise ‖·‖ | med ‖Δw‖ | clipped | **mean final** | range | per-seed | s/round |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.005 | 0.454 | 6.2286 | 21.85 | 0.42 | 0.31 | **0.0766** | 0.0385 | 0.094/0.080/0.056 | 1.7 |
| 20 | 0.020 | 0.554 | 6.2280 | 6.67 | 0.19 | 0.05 | **0.0540** | 0.0060 | 0.056/0.050/0.055 | 3.8 |
| 50 | 0.050 | 0.672 | 6.2277 | 3.24 | 0.18 | 0.02 | **0.0761** | 0.0510 | 0.053/0.071/0.104 | 8.0 |
| 100 | 0.100 | 0.827 | 6.2275 | 1.99 | 0.17 | 0.02 | **0.0745** | 0.0411 | 0.069/0.057/0.098 | 15.6 |
| 200 | 0.200 | 1.114 | 6.2283 | 1.34 | 0.16 | 0.02 | **0.0693** | 0.0310 | 0.088/0.057/0.063 | 30.8 |
| 500 | 0.500 | 2.000 | 6.2281 | 0.96 | 0.16 | 0.02 | **0.0662** | 0.0188 | 0.061/0.059/0.078 | 129.1 |

Mean per-round trajectories (rounds 1 → 5 → 10 → 15 → 20) rise to ≈ 0.06 by
round 5 and stall there in every cell.

**The design worked; the effect isn't there.** Median update norms are constant
across cells (0.16–0.19 for m ≥ 20 — exactly what fixing the shard size was
meant to achieve, against Fashion-MNIST's 4× drift), the applied noise falls
22.7× from m = 5 to m = 500, and final accuracy does not move: every cell sits
at 5–8 % with overlapping seed ranges. **A 100× change in cohort size, with
shard size genuinely held constant and the privacy budget honestly
re-calibrated per cell, produced no measurable accuracy change at ε = 6.228.**

**The tension, resolved at the mechanism level but mooted at the accuracy
level.** In noise magnitude, 1/m averaging beats amplification loss decisively:
z must rise 4.4× as q goes 0.005 → 0.5, yet the applied noise still falls
22.7×. But none of that reached accuracy, because of what the control shows:

**Federated no-DP control (identical population, cohorts, rounds), both ends
of the sweep:**

| m | No-DP mean final | range | DP mean final (from the sweep) |
|---|---|---|---|
| 50 | **0.0749** | 0.0643 | 0.0761 |
| 500 | **0.0755** | 0.0443 | 0.0662 |

Statistically identical to the DP cells at both ends — and the no-DP curve is
*itself flat in m*. Plain FedAvg is exactly as stuck as DP-FedAvg, with or
without 10× more clients averaged per round. The aggregate no-noise signal per
round has norm ≈ 0.02–0.16 and shrinks as the model settles into a shallow
plateau; twenty such steps go nowhere on a 62-class problem. **The binding
constraint at this operating point is the optimisation budget — 20 rounds ×
1 local epoch × ~159-example shards — not the noise, and not the cohort.**
DP is "free" here only in the degenerate sense that there is nothing yet for
the noise to destroy.

What this does and does not establish:

- It **does** establish that the Fashion-MNIST cohort curve cannot be read as
  "cohort size fixes DP": with shards fixed and amplification honestly
  accounted, cohort size alone moved nothing here.
- It does **not** establish that cohort size never matters: a cohort benefit
  could emerge at operating points where FedAvg itself makes real progress
  per round (more rounds, more local epochs, larger learning rate). This
  sweep measures the configured operating point — deliberately matched to the
  Fashion-MNIST budget (20 rounds, 1 epoch, lr 0.01) for comparability — and
  reports what it found there.

## 5. Clip bracket at the winning cohort

No cohort won §4 (all cells statistically flat), so the bracket ran at
**m = 200**: within 28 % of the best cell's noise conditions at a quarter of
m = 500's compute, making a full five-clip × three-seed grid feasible. z is
calibrated for q = 0.2 (z = 1.114, achieved ε = 6.228); the clip does not move
ε (dp_diagnosis.md §7.0), so every row is an equal-privacy comparison.

| Clip S | mean final | range | per-seed | mean clipped fraction |
|---|---|---|---|---|
| 1.0 | 0.0577 | 0.0176 | 0.049/0.057/0.067 | 0.00 |
| 0.5 | 0.0600 | 0.0051 | 0.062/0.057/0.061 | 0.02 |
| 0.25 | 0.0616 | 0.0047 | 0.063/0.059/0.063 | 0.28 |
| 0.125 | 0.0628 | 0.0159 | 0.073/0.058/0.057 | 0.60 |
| 0.0625 | 0.0643 | 0.0344 | 0.085/0.051/0.058 | 0.99 |

**The bracket is mechanically sound and finds nothing — which is itself the
result.** The grid genuinely straddles the update norm: clipping binds on 0 %
of updates at S = 1.0 and 99 % at S = 0.0625, exactly the instrumented range
the Fashion sweep lacked. Yet accuracy is clip-independent: the end-to-end
drift (+0.7 pp as S falls 16×) is smaller than the seed ranges. **There is no
clip optimum to locate at an operating point where noise is not the binding
constraint** — the no-DP control (§4) already showed noiseless FedAvg equally
stuck, so no amount of noise reduction via the clip can help. Contrast
Fashion-MNIST, where noise *was* binding and the same 6× clip change moved
final accuracy by 63 points.

The Fashion finding ("the optimum lies below 0.5, unbracketed") is therefore
neither confirmed nor refuted here: it was a claim about a noise-bound regime,
and this operating point never enters that regime. Bracketing the clip
optimum on FEMNIST first requires a budget at which FEMNIST trains (see the
roadmap).

## 6. What this says about the Fashion-MNIST curve

Placed side by side, the two cohort curves answer the question the Fashion
sweep raised. Same model size (±3 %), same round budget, same optimiser, same
target ε, both at S = 0.5:

| | Fashion-MNIST (N = 2m, shards thin as m grows) | FEMNIST (N = 1000 fixed, shards constant) |
|---|---|---|
| Shard size across sweep | 6,000 → 150 | ~159 median, constant |
| Median update norm across sweep | 0.98 → 0.29 (falls 3.4×) | 0.16–0.19 (flat) |
| z across sweep | 2.0 fixed (q fixed) | 0.454 → 2.000 (calibrated) |
| DP final accuracy across sweep | 11 % → 73 % → 57 % (moves 62 pp) | 5–8 % (flat) |
| No-DP control across sweep | 87 % → 57 % (falls 29 pp) | ≈ 7 % at m = 50 (flat, = DP) |

**Which effect drove the Fashion m = 50 optimum: thin shards or aggregate
noise? Thin shards — through the update-norm and ceiling channels — with
noise mattering only at the destructive extreme.** The FEMNIST design removes
the shard-size channel entirely, and with it removed, the cohort axis does
nothing: no rise toward an optimum, no decline past one. Everything that made
the Fashion curve move — the falling no-DP ceiling, the shrinking update
norms that changed where the clip binds, and the m ≤ 20 regime where
per-round noise physically destroyed the model (NaN) — was either a shard
effect or an extreme-noise effect. The "averaging over more clients fixes DP"
reading of the Fashion curve does not survive the controlled version of its
own experiment at this operating point.

**The k = 0.828 exponent is retired.** dp_diagnosis.md §7.3 fitted
g ∝ m^−k to the Fashion update norms and derived SNR ∝ m^0.172; that document
already flagged k as conflating shard size with cohort size and refused to
extrapolate it. The FEMNIST measurement closes the question: with shards
fixed, the update norm is flat in m (0.16–0.19 across 25×), i.e. **k ≈ 0 once
the confound is removed — k was measuring shard size all along.** It survives
only as a footnote in dp_diagnosis.md quantifying that design's confound.

---

## Reproducing

```bash
python scripts/prepare_femnist.py     # one-time: ~170 MB download, packs to data/
python scripts/femnist_experiments.py --experiment entropy      --out docs/_femnist_entropy.json  > e.log 2>&1
python scripts/femnist_experiments.py --experiment baseline     --out docs/_femnist_baseline.json > b.log 2>&1
python scripts/femnist_experiments.py --experiment sweep        --out docs/_femnist_sweep.json    > s.log 2>&1
python scripts/femnist_experiments.py --experiment clip_bracket --m <winner> --out docs/_femnist_bracket.json > c.log 2>&1
```

Raw per-round data for every run is committed as `docs/_femnist_*.json`.
Do not pipe the script's stdout into a command that waits for EOF (TFF leaves
an executor subprocess holding the pipe); redirect to a file as above.
