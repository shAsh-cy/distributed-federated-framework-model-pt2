# Diagnosis: why the DP runs collapse to chance

**Question.** Both DP configurations report exactly 10.00 % test accuracy — chance
on a balanced 10-class problem, and below the 12.25 % untrained baseline. Is the
DP implementation broken, or is this the mechanism behaving correctly at a scale
where it cannot work?

**Verdict.** The implementation is not broken. The collapse is caused by the
**magnitude of the Gaussian noise**, not by clipping and not by a bug in the
aggregation. At the configured cohort size the noise added to the aggregate is
roughly **570× larger in L2 norm than the useful signal**, so the first DP round
replaces the model with noise and every later round adds more.

**Second verdict, from §7–§9 — the configuration was also wrong, and this document
originally said it could not be.** At the same ε = 6.228, lowering the clipping
norm from 3.0 to 0.5 and raising the cohort to 50 reaches **73.4 % final accuracy
(mean of three seeds, range 1.1 pp) against a 76.9 % matched non-private ceiling
(mean of the same three seeds, range 2.4 pp) — a DP cost of 3.5 pp (§8.2)**.
Normalised against ceilings
measured at the same cohort size, client-level DP costs **3–5 % of achievable
accuracy, not 88 %**. The claim that lowering the clip "does not help" is
**retracted** — see the retraction under Conclusions. It holds only where clipping
binds, and the production setting was in the opposite regime.

**No DP code was changed to produce this document.** The harness
(`scripts/diagnose_dp.py`) imports `fl.aggregation` unchanged and drives it
directly.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Clipping norm far below update norms, annihilating updates | **Ruled out** | §1 — median update norm 0.984, clip 3.0; clip is 3× *above* the median |
| Gaussian noise magnitude destroys the signal | **Confirmed** | §2, §3 — noise ‖·‖ = 569.7 vs signal ‖·‖ ≈ 1.0 |
| Model degenerates to predicting one class | **Confirmed** | §4 — 84.8 % then 97.4 % of all test images given the same label |
| Bug in the DP aggregation path | **Ruled out** | §2 — measured noise matches TFF's documented formula to 4 significant figures |
| Clipping norm *too high*, buying noise for nothing | **Confirmed** | §7, §8.2 — `S` 3.0 → 0.5 at `m` = 50 moves final accuracy from 18.8 % (mean of 4 draws) to 73.4 % (mean of 3 seeds) at identical ε |
| The cohort-size result is confounded by shrinking shards | **Confirmed** | §8 — the no-DP ceiling itself falls 86.93 % → 57.42 % across the same grid |
| Once clipping binds, the clip acts as a server step size | **Confirmed** | §9 — a 0.4545× step at `S` = 1.1 recovers the full 27-point gap to `S` = 0.5 |
| DP non-determinism is caused by the diagnostic instrumentation | **Ruled out** | §10 — two passes with no measurement anywhere still differ; the noise is drawn inside TFF's executor, which never sees `tf.random.set_seed` |

**Read every DP accuracy figure here as a single draw.** Run-to-run spread is
4.7–29.5 accuracy points depending on cohort size (§10.3), and cannot be removed
from the harness. Only effects larger than that spread are treated as findings; §10.3
tabulates which of this document's claims survive and which do not.

---

## Method

`scripts/diagnose_dp.py` runs the federated loop in-process, calling the same
`fl.aggregation` code the gRPC server calls. Only the transport is skipped, since
these experiments need ~20 full runs.

**Control.** The no-DP simulation must reproduce the recorded real run, or none of
the rest is trustworthy:

| | Final accuracy | Best | Untrained baseline |
|---|---|---|---|
| Recorded gRPC run (`results/no_dp.json`) | 0.8693 | 0.8769 | 0.1225 |
| This harness, same seed | **0.8693** | 0.8758 | 0.1225 |

Final accuracy matches exactly. The harness is faithful.

**Reporting convention.** Headline figures throughout this document are **final
accuracy** — the reading at round 20 — because that is what a deployment gets.
Where best accuracy is also shown the column is labelled; best is diagnostic
(it distinguishes "never trained" from "trained and then destabilised") and is
never the headline.

**Two documented deviations.** (1) One Keras model is reused across clients with
optimiser slots zeroed between them, rather than one persistent model per client;
this drops SGD momentum carry-over, and the control above shows it is immaterial.
(2) The harness has no server-side NaN rejection, so a destroyed run continues to
NaN and reports 0.0000, whereas production rejects NaN updates and *freezes* on
the last finite model, reporting exactly 0.1000. Both are the same collapse; the
production number is the frozen destroyed model, as §4 shows.

---

## 1. Update norms before clipping, versus the clipping norm

Measured on every sampled client in every round, taken **before** the aggregator
sees the update, so clipping cannot have touched it.

Non-private run (`z = 0`), 100 client-updates over 20 rounds:

| Statistic | Value |
|---|---|
| Median ‖Δw‖₂ | **0.984** |
| Mean | 1.206 |
| Min / Max | 0.640 / 4.599 |
| **Configured clipping norm `S`** | **3.0** |

Per-round medians: 3.99, 1.33, 1.10, 1.00, 0.90, 0.96, 0.99, 0.89, 1.13, 0.77,
0.91, 1.18, 0.94, 0.98, 1.26, 1.03, 0.98, 0.89, 0.89, 1.20.

**Finding — the clipping norm is not the problem.** The test proposed was "if the
clip norm is far below the median update norm, updates are being annihilated".
The opposite holds: the clip (3.0) sits about **3× above** the median update norm
(0.984). Clipping binds only in round 1, where the median is 3.99 because the
first update departs from a random initialisation. From round 2 onward almost no
update is clipped at all.

This also corrects the basis on which `l2_clip_norm: 3.0` was originally chosen.
That value came from measuring update norms **from the initial random model
only** (observed 1.96–4.65, median 2.96) — i.e. round-1 behaviour, which is not
representative. Across training the median is 0.984. The clip is loose, not tight.

---

## 2. The Gaussian noise actually applied

TFF's `DifferentiallyPrivateFactory.gaussian_fixed` builds
(`tensorflow_federated/python/aggregators/differential_privacy.py:250`):

```python
query = tfp.NormalizedQuery(
    tfp.GaussianSumQuery(l2_norm_clip=clip, stddev=clip * noise_multiplier),
    denominator=clients_per_round,
)
```

So each client's delta is clipped to `S`, the clipped deltas are summed, Gaussian
noise of per-coordinate standard deviation `S·z` is added **to the sum**, and the
result is divided by `m`:

```
σ_per-coordinate (on the mean)  =  S · z / m
E‖noise vector‖₂ (on the mean)  =  S · z · √d / m        d = 225,034 parameters
```

Measured by feeding the aggregator all-zero deltas, so the output is pure noise:

| `z` | `S` | `m` | σ/coord predicted | σ/coord **measured** | ‖noise‖ predicted | ‖noise‖ **measured** |
|---|---|---|---|---|---|---|
| 2.0 | 3.0 | 5 | 1.2 | **1.2009** | 569.3 | **569.7** |
| 6.0 | 3.0 | 5 | 3.6 | **3.599** | 1708 | **1707** |
| 2.0 | 3.0 | 200 | 0.03 | **0.02996** | 14.23 | **14.21** |

Measurement matches theory to four significant figures. **The DP aggregation is
doing exactly what it is specified to do.**

### Signal-to-noise ratio of the aggregated update

The signal is the mean of the clipped client deltas. Its norm is bounded by `S`
= 3.0, but §1 shows the actual updates are ≈ 1.0 in norm and clipping rarely
binds, so the realistic signal is ≈ 1.0.

| Configuration | ‖signal‖ | ‖noise‖ | **SNR** |
|---|---|---|---|
| Moderate, `z = 2`, `m = 5` | ≈ 1.0 (bound 3.0) | 569.7 | **0.0018** (bound 0.0053) |
| High, `z = 6`, `m = 5` | ≈ 1.0 (bound 3.0) | 1707 | **0.0006** (bound 0.0018) |

For scale: the **entire weight vector** has norm 17.34 at initialisation and grows
only to 21.02 over 20 non-private rounds. A single moderate-DP round adds a noise
vector of norm 569.7 — about **30× the whole model**. Measured global weight norm
after each DP round: 17.34 → 570.2 → 805.7 → 985.7. The model is not perturbed;
it is replaced by a random walk.

---

## 3. Ablation — clipping alone, noise alone, both

Three runs. Every arm's full condition is tabulated so none has to be inferred:

| Run | `z` | `S` (clip) | `m` | `N` | Rounds | Seed | **Final accuracy** | Best accuracy |
|---|---|---|---|---|---|---|---|---|
| **(a) clipping only** | 0 | 3.0 | 5 | 10 | 20 | 42 | **0.8512** | 0.8634 |
| **(b) noise only** (clip cannot bind) | 6 × 10⁻⁶ | 10⁶ | 5 | 10 | 20 | 42 | **0.0000** | 0.0096 |
| **(c) both** — production config | 2.0 | 3.0 | 5 | 10 | 20 | 42 | **0.0000** | 0.1005 |

Run (b) needs explanation. TFF ties the noise standard deviation to the clipping
norm (`stddev = clip · z`), so noise cannot be varied independently through the
public API. Isolating it means raising `clip` until it can never bind while
holding the product constant: `3.0 × 2.0 = 6.0 = 10⁶ × 6×10⁻⁶`. Run (b) therefore
applies **the same absolute noise as production with clipping effectively
switched off**. It is not a valid DP configuration — its `z` implies a vacuous
ε — it is a mechanism isolation only.

**Finding.** Run (a) recovers to 85.12 %, within 1.8 points of the 86.93 %
non-private baseline. The residual gap is the cost of clipping the round-1 update
(the only round where clipping binds) plus run-to-run variance. Run (b), noise
alone with no clipping at all, collapses just as completely as production.

This is the decisive experiment: **clipping is fine, noise is the cause.**

**Resolving an apparent conflict with the recorded results.** Run (c) reports
final accuracy 0.0000 while `results/dp_moderate.json` — the real gRPC run at the
same nominal condition (`z = 2.0`, `S = 3.0`, `m = 5`, `N = 10`, 20 rounds, seed
42) — recorded 10.00 %. These are not a contradiction; they are **two draws from
a bimodal outcome distribution at the edge of trainability**. At this
configuration the noise either drives the weights to NaN (the harness reads
0.0000) or leaves a finite model that predicts a single class for nearly every
input (reads exactly 10.00 % on the balanced test set, §4). Both endpoints are
reachable from the identical config and seed because the DP noise itself is not
seedable (§10) — the repeatability run reproduces exactly this split at `m = 5`,
one pass ending at 0.0000 and the other at 0.1000 (§10.1). A reporting difference
stacks on top: production rejects NaN updates and freezes on the last finite
model, so it maps *both* endpoints to ≈ 10 %, while the harness lets NaN runs
read 0.0000 (see *Method*). "Collapse" in this document means either endpoint;
neither number is more correct than the other.

---

## 4. What the collapsed model predicts

Production reports exactly 10.00 % because the server rejects NaN updates and
freezes on the last finite model. Reproducing that state and asking the model what
it outputs for all 10,000 test images:

| Round | Accuracy | ‖global weights‖ | Predicted-class counts | Distinct classes | Largest class share |
|---|---|---|---|---|---|
| 0 (untrained) | 0.1225 | 17.34 | `[0, 0, 62, 1020, 0, 2481, 6092, 0, 345, 0]` | 5 | 60.9 % |
| 1 | 0.1044 | 570.2 | `[0, 0, 8476, 0, 0, 4, 0, 1322, 198, 0]` | 4 | **84.8 %** |
| 2 | 0.1003 | 805.7 | `[51, 0, 9737, 0, 0, 2, 0, 200, 10, 0]` | 5 | **97.4 %** |
| 3 | 0.1399 | 985.7 | `[170, 0, 5394, 0, 0, 0, 0, 4436, 0, 0]` | 3 | 53.9 % |

True class counts are 1,000 each.

**Finding — confirmed.** The model degenerates to near-constant output. By round 2
it assigns class 2 to **97.4 %** of all inputs; accuracy 0.1003 is simply the
1,000 genuine class-2 images it happens to get right. This is the classic
signature of a saturated network whose logits are dominated by one unit, exactly
what a weight vector consisting of large random values produces. Accuracy landing
on 10.00 % is not coincidence: predicting a single class on a balanced 10-class
test set gives exactly 1000/10000.

---

## 5. Noise-multiplier sweep

Cohort held at `m = 5` (the production setting), `S = 3.0`, 20 rounds, seed 42.
ε at δ = 10⁻⁵.

| `z` | ε | ‖noise‖ | **Final accuracy** | Best accuracy | Predicted-class counts | Largest share |
|---|---|---|---|---|---|---|
| 0.0 | ∞ (none) | 0.0 | **0.8512** | 0.8634 | `[1197, 979, 898, 965, 1632, 1035, 327, 978, 990, 999]` | 16 % |
| 0.1 | 1060.1 | 28.5 | **0.7009** | 0.7917 | `[1285, 891, 1417, 1120, 1062, 2083, 184, 742, 931, 285]` | 21 % |
| 0.3 | 128.3 | 85.4 | **0.0985** | 0.1686 | `[29, 1, 1, 2, 9447, 1, 35, 484, 0, 0]` | **94 %** |
| 0.5 | 51.0 | 142.3 | **0.0867** | 0.1871 | `[167, 87, 19, 22, 580, 7958, 107, 627, 128, 305]` | **80 %** |
| 1.0 | 16.6 | 284.6 | **0.0000** | 0.1646 | (weights non-finite) | — |

**Where accuracy departs from the baseline:** immediately. Even `z = 0.1` costs
15 points (85.1 % → 70.1 %), at a noise norm of 28.5 against a signal of ≈ 1.0.

**Where it hits chance:** between `z = 0.1` and `z = 0.3`. By `z = 0.3` accuracy is
9.85 % and the model puts 94 % of test images in a single class.

**The damning column is ε.** The only setting that retains useful accuracy,
`z = 0.1`, carries ε = 1060 — a vacuous guarantee. The lowest ε in the sweep,
16.6 at `z = 1.0`, has already destroyed the model. At `m = 5` there is no noise
multiplier that buys both a meaningful privacy budget and a working classifier.
The production settings (ε = 6.2 and ε = 1.6) sit far beyond the right-hand end of
this table.

---

## 6. Cohort-size sweep at fixed ε — the structural experiment

ε depends on `(z, q, rounds)`, so holding `z = 2.0`, `q = 0.5` and `rounds = 20`
fixed keeps **ε = 6.228 at every point** — the production moderate setting.
Cohort `m` grows with the population as `N = 2m`, so shards shrink and the total
samples trained per round stays ≈ 30,000 throughout. Only `m` changes.

| `m` | `N` | Shard | ‖noise‖ | Round-1 median ‖Δw‖ | SNR | **Final acc** | **Best acc** | Mean acc, first 5 → last 5 rounds |
|---|---|---|---|---|---|---|---|---|
| 5 | 10 | 6,000 | 569.25 | 3.994 | 0.0070 | 0.1319 | 0.1319 | 0.087 → 0.132 |
| 20 | 40 | 1,500 | 142.31 | 2.033 | 0.0143 | 0.0536 | 0.1646 | 0.118 → 0.095 |
| 50 | 100 | 600 | 56.93 | 0.876 | 0.0154 | 0.1085 | 0.3870 | 0.215 → 0.123 |
| 100 | 200 | 300 | 28.46 | 0.466 | 0.0164 | **0.3737** | **0.5074** | 0.193 → **0.420** |
| 200 | 400 | 150 | 14.23 | 0.180 | 0.0127 | **0.4120** | **0.5582** | 0.250 → **0.485** |

Round-1 medians are quoted because they are the last measurement taken before a
collapsing run destroys its own model; the all-rounds median is meaningless once
clients are training on wreckage (it reaches 1.4 × 10¹⁵ at `m = 5`).

> **Confound — read this before drawing a cohort-size conclusion from this table.**
> Holding `q = 0.5` requires `N = 2m`, so a larger cohort *necessarily* means a
> smaller shard: 6,000 examples per client at `m = 5` down to 150 at `m = 200`.
> Accuracy therefore moves along this table for two unrelated reasons — less noise
> (helps) and less data per client (hurts). This table cannot separate them.
> §8 runs the identical grid with DP switched off and does separate them; the
> conclusion below survives, but weaker than stated here, and the "4.2×" framing
> is superseded.

Per-round accuracy, showing the qualitative change:

| `m` | Accuracy by round |
|---|---|
| 5 | 0.10, 0.06, 0.06, 0.12, 0.10, 0.13, 0.13, 0.13 … 0.13 — **flat, dead from round 1** |
| 20 | 0.12, 0.11, 0.10, 0.13, 0.12, 0.10, 0.15, 0.17 … 0.05 — **noise, no trend** |
| 50 | 0.10, 0.11, 0.22, 0.25, 0.39, 0.17, 0.16, 0.10 … 0.11 — **unstable, no convergence** |
| 100 | 0.10, 0.17, 0.22, 0.24, 0.24, 0.26, 0.35, 0.39, 0.49 … 0.37 — **learning** |
| 200 | 0.12, 0.20, 0.26, 0.31, 0.37, 0.39, 0.41, 0.39 … 0.56, 0.49, 0.41 — **learning** |

**Finding — accuracy recovers as the cohort grows, at constant ε.** Privacy is
held identical across every row; only the number of clients averaged per round
changes. Accuracy goes from dead-flat chance at `m = 5` to a clear learning curve
reaching 55.8 % at `m = 200`. That is a **4.2× improvement in best accuracy for
exactly the same privacy guarantee**, obtained purely by averaging more clients.

This is the structural explanation. The noise added per round is fixed by
`S · z` regardless of `m`, but it is divided by `m`, while the signal is not. More
clients means the same privacy cost spread over more useful updates.

**A caveat worth stating**: the recovery is not solely an SNR effect. The SNR
column barely moves (0.007 → 0.016 → 0.013) because shards shrink as `m` grows,
so individual updates get smaller too. What changes is that the noise is
**zero-mean and independent across rounds**, so it partially cancels over 20
rounds while the consistent signal accumulates. That is why `m = 100` and
`m = 200` learn steadily at a per-round SNR of only ≈ 0.015, whereas `m = 5`
— at half that SNR — is destroyed in a single round and never recovers, because
once the weights are 570 in norm the model is outside the regime where gradients
mean anything.

---

## 7. Clipping-norm sweep at fixed ε

§6 varied the cohort at one clipping norm. This varies both: `S ∈ {3.0, 1.1, 0.5}`
crossed with `m ∈ {5, 20, 50, 100, 200}`, 15 cells.

### 7.0 Gate — the clip does not move the privacy budget

Checked before running anything, because the sweep is only meaningful if every
cell carries the same ε. It does, three ways:

1. `compute_epsilon(noise_multiplier, sampling_rate, rounds, delta)` has **no clip
   parameter**. There is no path from `S` into the calculation.
2. The composed event is
   `SelfComposedDpEvent(PoissonSampledDpEvent(q=0.5, GaussianDpEvent(z=2.0)), count=20)`.
   `GaussianDpEvent` carries only `z = σ/S`, already normalised by the sensitivity
   `S = clip`. TFF sets `stddev = clip · z`, so scaling `S` scales `σ` in
   proportion and leaves `z` — hence ε — untouched.
3. Computed ε is **bitwise identical** at all three clips:
   `6.2284173254307244604888182948343455791473388671875`.

**ε = 6.228 at δ = 1 × 10⁻⁵ in every cell below.** The sweep is an equal-privacy
comparison. `exp_epsilon_gate` raises `AssertionError` and aborts the sweep if this
ever stops holding.

### 7.1 The grid

SNR is `min(median‖Δw‖, S) / ‖applied noise‖` — the signal that survives clipping,
over the noise actually added. Both terms are measured, not modelled.

| `S` | `m` | **Final acc** | Best acc | ‖noise‖ | Median ‖Δw‖ (r1) | Median ‖Δw‖ (all) | Clipped (r1) | Clipped (all) | **SNR** |
|---|---|---|---|---|---|---|---|---|---|
| 3.0 | 5 | 0.0000 | 0.0594 | 569.25 | 3.994 | *diverged* | 0.80 | 0.09 | 0.00527 |
| 3.0 | 20 | 0.1000 | 0.1400 | 142.52 | 2.033 | 56.26 | 0.00 | 0.94 | 0.02105 |
| 3.0 | 50 | 0.1000 | 0.4507 | 56.88 | 0.876 | 1.947 | 0.00 | 0.18 | 0.03422 |
| 3.0 | 100 | 0.3962 | 0.5087 | 28.46 | 0.466 | 0.951 | 0.00 | 0.00 | 0.03343 |
| 3.0 | 200 | 0.4651 | 0.5887 | 14.26 | 0.180 | 0.388 | 0.00 | 0.00 | 0.02719 |
| 1.1 | 5 | 0.0000 | 0.1421 | 208.47 | 3.994 | *diverged* | 1.00 | 0.79 | 0.00528 |
| 1.1 | 20 | 0.1238 | 0.2891 | 52.19 | 2.033 | 4.260 | 1.00 | 1.00 | 0.02108 |
| 1.1 | 50 | 0.6528 | 0.6954 | 20.90 | 0.876 | 0.907 | 0.28 | 0.29 | 0.04340 |
| 1.1 | 100 | 0.6434 | 0.6832 | 10.42 | 0.466 | 0.513 | 0.02 | 0.02 | 0.04926 |
| 1.1 | 200 | 0.5694 | 0.5970 | 5.22 | 0.180 | 0.304 | 0.00 | 0.00 | 0.05816 |
| 0.5 | 5 | 0.1123 | 0.1821 | 94.91 | 3.994 | *diverged* | 1.00 | 0.96 | 0.00527 |
| 0.5 | 20 | 0.5219 | 0.5638 | 23.74 | 2.033 | 2.098 | 1.00 | 1.00 | 0.02106 |
| **0.5** | **50** | **0.7348** | **0.7348** | 9.49 | 0.876 | 0.670 | 0.96 | 0.82 | 0.05269 |
| 0.5 | 100 | 0.6815 | 0.6891 | 4.74 | 0.466 | 0.455 | 0.45 | 0.41 | 0.09603 |
| 0.5 | 200 | 0.5745 | 0.6348 | 2.374 | 0.180 | 0.289 | 0.04 | 0.09 | 0.12177 |

Measured noise matches the closed form `S·z·√d / m` to 3–4 significant figures in
all 15 cells.

**All three `m = 5` runs diverge to NaN.** The applied noise (95–569) dwarfs the
weight scale, the weights explode, and `0.0000` is a NaN model rather than a
prediction. Their all-rounds medians are not real measurements and are marked
*diverged* rather than tabulated. `m = 5` is excluded from every conclusion below.

**The final-vs-best gap is itself diagnostic.** The winning cell (`S = 0.5`,
`m = 50`) has `final == best` exactly — 0.7348 at round 20 — meaning it was still
improving when the round budget ran out and never destabilised. Contrast
`S = 3.0`, `m = 50`: best 0.4507 mid-run, final 0.1000 — it reached 45 % and then
collapsed back to chance. Same ε, same cohort; the difference between "still
climbing" and "climbed and was destroyed" is the clipping norm.

### 7.2 Two regimes, and the boundary moves

Where clipping binds (`S < ‖Δw‖`), signal and noise both scale with `S`, so SNR
reduces to `m / (z·√d)` — independent of the clip. That is confirmed to three
digits across a 6× change in `S`:

| `m` | SNR @ 3.0 | SNR @ 1.1 | SNR @ 0.5 | `m / (z·√d)` |
|---|---|---|---|---|
| 5 | 0.00527 | 0.00528 | 0.00527 | 0.005271 |
| 20 | 0.02105 | 0.02108 | 0.02106 | 0.021081 |

Where the clip is slack (`S > ‖Δw‖`), the signal is untouched and only the noise
scales with `S`, so SNR = `‖Δw‖·m / (S·z·√d)` and **falls as the clip rises**.

The boundary between the regimes is not fixed, because the update norm shrinks as
shards shrink. At `m = 200` the median update is 0.289, so even `S = 0.5` is
slack (9% clipped) and SNR is still climbing when the sweep runs out of values —
**the sweep does not bracket the optimum at large `m`**.

> **Accuracy in this grid is a single run per cell, and the run-to-run spread is
> 4.7–29.5 accuracy points (§10).** The `S = 3.0` → `S = 0.5` improvement at
> `m = 50` is roughly 2× the spread and survives; the `S = 1.1` vs `S = 0.5`
> ordering at `m ≥ 50` does **not**, and should be read as "no distinguishable
> difference" rather than as a ranking. SNR and noise columns are unaffected —
> those are measurements of the noise distribution, not of a training outcome.

### 7.3 Fitted exponent for the signal decay

Round-1 median pre-clip norms are identical across all three clips — 3.994,
2.033, 0.876, 0.466, 0.180 at `m` = 5, 20, 50, 100, 200 — because round 1 starts
from the same seeded initialisation before any clip or noise applies. That makes
them a clip-independent measurement of how the update norm scales.

Least squares on `ln g = ln A − k·ln m`:

| Quantity | Value |
|---|---|
| **k** | **0.828 ± 0.095** (std err) |
| A | 19.06 |
| **R²** | **0.962** |
| RMSE (log space) | 0.211 |
| Residuals (log) | −0.230, +0.242, +0.160, +0.101, −0.273 |

Noise falls as `1/m`, so net **SNR ∝ m^(1−k) = m^0.172**.

Fit quality is good but not exact: the residual signs run −, +, +, +, − , which is
systematic curvature, so a single power law is a description rather than a law.

**What `k` actually measures — and why the extrapolation from it is worthless.**
Because the sweep holds `q = 0.5` by setting `N = 2m`, each client's shard is
`60,000 / 2m`. So `k ≈ 0.83` is very nearly "update norm scales with shard size",
not a property of cohort size at all. Taken at face value, `SNR ∝ m^0.172`
extrapolates to ~735,000 clients for SNR = 0.5 at `S = 0.5` — a meaningless
number, since 60,000 examples cannot be split among 1.47 million clients. The
exponent is reported because it was asked for and because it quantifies the
confound; it is not a basis for a client-count estimate. §8 gives the answer that is.

---

## 8. No-DP control at every cohort size — separating noise from thin shards

The identical grid with `noise_multiplier = 0` and clipping disabled. Same `N = 2m`,
same seed, same partitioning, same round count. This is the ceiling each cell of
§7 was actually competing against.

| `m` | `N` | Shard | **Final acc** | Best acc |
|---|---|---|---|---|
| 5 | 10 | 6,000 | **0.8693** | 0.8758 |
| 20 | 40 | 1,500 | **0.8304** | 0.8304 |
| 50 | 100 | 600 | **0.7716** | 0.7716 |
| 100 | 200 | 300 | **0.7154** | 0.7183 |
| 200 | 400 | 150 | **0.5742** | 0.6539 |

`m = 5` reproduces the recorded non-private gRPC run (86.93%) exactly, which
validates the in-process simulation against the real transport.

**The ceiling falls by 29 points from `m = 5` to `m = 200` with no DP involved at
all.** That is the confound, measured: growing the cohort in this design costs
more accuracy than the noise does past `m ≈ 100`.

### 8.1 Normalised — the price of DP alone

DP accuracy over its matched ceiling, at the winning clip `S = 0.5`, final accuracy
throughout:

| `m` | DP final | No-DP ceiling (final) | **Ratio** | Gap |
|---|---|---|---|---|
| 5 | 0.1123 | 0.8693 | **0.129** | 75.7 pp |
| 20 | 0.5219 | 0.8304 | **0.628** | 30.8 pp |
| 50 | 0.7348 | 0.7716 | **0.952** | 3.7 pp * |
| 100 | 0.6815 | 0.7154 | **0.953** | 3.4 pp |
| 200 | 0.5745 | 0.5742 | 1.001 † | −0.0 pp † |

\* Single draw from the grid. The `m = 50` cell is replicated at three seeds in
§8.2, which is where the headline gap comes from: mean 3.5 pp, per-seed 4.3 / 4.6
/ 1.6 pp.

† The `m = 200` ratio is not a real measurement of DP's cost. That control is the
only cell in §8 with a large final-vs-best gap (final 0.5742, best 0.6539) — it was
still oscillating at round 20, so its *final* reading is a draw, not a ceiling.
Final-to-final gives 1.001; against the control's best it is 0.5745 / 0.6539 ≈
**0.88**. The honest statement is **no measurable penalty at `m = 200`, though the
non-private control was itself unstable at this cohort size** — not that DP is
free there.

The same ratio at every clip, which is where the clipping norm's real effect shows:

| `m` | No-DP | Ratio @ 3.0 | Ratio @ 1.1 | Ratio @ 0.5 |
|---|---|---|---|---|
| 5 | 0.8693 | 0.000 | 0.000 | 0.129 |
| 20 | 0.8304 | 0.120 | 0.149 | 0.628 |
| 50 | 0.7716 | 0.130 | 0.846 | **0.952** |
| 100 | 0.7154 | 0.554 | 0.899 | **0.953** |
| 200 | 0.5742 | 0.810 | 0.992 | 1.001 † |

**This is the only number that speaks to DP's price rather than the experimental
design's.** The claim rests on `m = 50` and `m = 100`, where both arms are stable:
two cohorts independently landing at ratio ≈ 0.95, with `m = 200` consistent with
the same value (its ratio brackets 0.88–1.00 depending on which reading of its
unstable control is used). **Three cohorts agreeing at ≈ 0.95** is the finding —
client-level DP at ε = 6.228 costs on the order of 5 % of achievable accuracy once
the cohort and clip are right — not any single gap figure.

> **Precision warning.** Denominators here are exactly reproducible; numerators are
> single DP runs, and DP runs have a measured spread of 4.7–29.5 accuracy points at
> `S = 3.0` (§10). At `S = 0.5`, `m = 50` the spread is now measured directly —
> **1.1 pp across three seeds (§8.2)** — confirming that cells inside the trainable
> regime are far more stable than cells at its edge. The other `S = 0.5` cohorts
> remain single draws. Read these ratios as "≈ 0.95 at `m = 50`–100, and consistent
> with that at `m = 200`", not to three significant figures.
> The qualitative claim (DP's cost is single-digit percent once the cohort and clip
> are right, versus the 88% implied by the shipped configuration) is far larger than
> any plausible spread and is not at risk.

The headline collapse was never the price of privacy. It was the price of running
DP at `m = 5` with a clipping norm 3× above the median update.

### 8.2 Replication of the winning cell — three seeds, both arms

Single draws are not a basis for a headline, so the winning cell and its matched
control were each run at three seeds (42, 43, 44). Identical configuration
otherwise: `m = 50`, `N = 100`, 20 rounds; arm A with DP on (`S = 0.5`,
`z = 2.0`, ε = 6.228, δ = 10⁻⁵), arm B with DP off and clipping disabled.

| Seed | A: DP final acc | B: no-DP final acc | Gap |
|---|---|---|---|
| 42 | 0.7285 | 0.7716 | 4.3 pp |
| 43 | 0.7328 | 0.7788 | 4.6 pp |
| 44 | 0.7395 | 0.7553 | 1.6 pp |
| **Mean** | **0.7336** | **0.7686** | **3.5 pp** |
| Range | 1.1 pp | 2.4 pp | — |

**The headline is the mean: DP 73.4 % vs matched ceiling 76.9 %, a gap of
3.5 pp (ratio 0.954).** No seed was selected; all three are reported. The range
on the DP arm (1.1 pp) is under the ~5 pp threshold at which a point value would
be misleading, so quoting ≈ 73 % is fair — but the mean is the number this
document uses. The original single sweep draw (0.7348) sits inside the replicated
range, and arm B at seed 42 reproduces the §8 control (0.7716) exactly,
re-confirming that the non-DP path is deterministic.

Two useful by-products:

- **Cross-seed spread at `S = 0.5`, `m = 50` is now measured: 1.1 pp** on the DP
  arm — two orders smaller than the 29.5 pp spread of `S = 3.0` at the same
  cohort (§10.3). A configuration inside the trainable regime is stable; one at
  the edge of trainability is bimodal. (The 1.1 pp conflates seed variation with
  DP-noise variation — the noise is unseedable (§10) — but their sum bounds each.)
- **Every seed's DP run has `final ≈ best`** (43 and 44 exactly, 42 within
  0.4 pp): all three were still improving at round 20 and none destabilised.

**How many clients would you need?** Properly posed, the question separates. The
DP penalty is already down to ≈ 5 % by `m = 50` and stays there through the
largest cohort tested (`m = 200` shows no measurable penalty, though its control
was unstable — see †). What limits accuracy past that point is data per client,
not cohort size — this
population is 60,000 examples no matter how it is divided. With shard size held
fixed instead (each new client bringing its own data), noise falls as `1/m` while
`g` stays put, so SNR grows as `m¹` and the DP penalty closes faster still. The
honest answer is **on the order of 50–200 clients per round for the noise to stop
mattering at ε ≈ 6** — not the thousands implied by any SNR = 1 criterion.

---

## 9. Is the clipping norm acting as a server step size?

§7.2 leaves one cell unexplained. At `m = 20`, `S = 1.1` and `S = 0.5` both clip
100% of updates, so their SNR is identical (0.02108 vs 0.02106) — yet `S = 0.5`
scores **27 points higher**. Identical privacy, identical SNR, very different
accuracy. SNR cannot be the whole story.

Hypothesis: SNR is scale-invariant but the *step* is not. The server applies the
aggregate delta with no server-side learning rate, so when clipping binds the
per-round step magnitude is proportional to `S`. At `S = 1.1` the model moves 2.2×
further per round than at `S = 0.5` — same signal-to-noise ratio, larger absolute
excursion through weight space.

Test: scale the `S = 1.1` server step by `0.5/1.1 = 0.4545`, matching the `S = 0.5`
step magnitude while leaving ε, `z` and SNR untouched. All three arms in one
process, so they share RNG state and are directly comparable.

| Clip | Server LR | ε | SNR | **Final acc** | Best acc |
|---|---|---|---|---|---|
| 1.1 | 1.0 | 6.228 | 0.0211 | 0.2726 | 0.3160 |
| 1.1 | **0.4545** | 6.228 | 0.0211 | **0.5934** | 0.5934 |
| 0.5 | 1.0 | 6.228 | 0.0211 | 0.5169 | 0.5890 |

**Confirmed.** Scaling the step recovers the entire gap — 0.2726 → 0.5934, matching
and slightly exceeding the `S = 0.5` arm at 0.5169. The `m = 20` cell is no longer
an anomaly: **once clipping binds, lowering the clip is a server learning-rate
reduction, not an SNR improvement.** It is SNR-neutral and stability-positive,
which is also why `S = 0.5` was the only `m = 5` configuration still finite at
round 20.

`server_lr` exists only in `scripts/diagnose_dp.py` and defaults to 1.0; `fl/` is
unchanged and the production server still applies the aggregate delta as-is.

---

## 10. Reproducibility — and a correction to what this document blamed

An earlier version of this document attributed the fact that §7's `S = 3.0` column
does not match §6 to the noise instrumentation: `clip_sweep` calls
`measure_applied_noise` before each cell, which consumes host RNG state.
**That attribution was wrong.** It was a plausible guess presented as an
explanation, and testing it took one experiment that had not been run.

### 10.1 The test

`--experiment repeatability` runs each `S = 3.0` configuration **twice in the same
process, with nothing in between** — no `measure_applied_noise` call anywhere in
the experiment. If instrumentation were the cause, the two passes would agree.

| `m` | Final acc, pass A | Final acc, pass B | Round-1 ‖applied‖ A | Round-1 ‖applied‖ B | Identical? |
|---|---|---|---|---|---|
| 5 | 0.0000 | 0.1000 | 569.108277 | 569.882132 | **No** |
| 20 | 0.0960 | 0.1002 | 142.181407 | 142.765687 | **No** |
| 50 | 0.1479 | 0.3952 | 56.907197 | 56.884553 | **No** |
| 100 | 0.4973 | 0.5377 | 28.450752 | 28.528936 | **No** |
| 200 | 0.5886 | 0.4976 | 14.201963 | 14.230995 | **No** |

DP runs do not reproduce even without any measurement. The instrumentation is
exonerated; the non-determinism is in the stack.

### 10.2 The actual cause

TF Privacy 0.9.0 (`privacy/dp_query/gaussian_query.py`, `get_noised_result`) draws
the Gaussian noise via

```python
random_normal = tf.random_normal_initializer(stddev=global_state.stddev)
```

with **no `seed` argument**. That initialiser is in fact honoured by
`tf.random.set_seed` in ordinary eager code and under `tf.function` — both checked
directly, both reproducible — so the initialiser alone is not the problem. The
problem is that TFF serialises the aggregation to a computation proto and executes
it **in its own executor, which never sees this process's global seed**. Verified:
constructing the TFF DP aggregator twice with `tf.random.set_seed(42)` before each
gives different noise, while the same initialiser under a plain `tf.function` gives
identical noise.

Neither `tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed` (signature:
`noise_multiplier, clients_per_round, clip`) nor `GaussianSumQuery` exposes a seed
parameter. **There is no supported way to make this deterministic from the
harness.** Doing it anyway would mean passing a per-round seed into the query —
changing the DP mechanism this document exists to measure, and risking a silent
break of the across-round independence the privacy analysis assumes. Not done.

The measurement path was still isolated from the host RNG stream (the zero
template is now built once and cached) — correct hygiene, but it fixes nothing,
and it is labelled as such in the code.

### 10.3 Measured spread, and which conclusions survive it

Four independent runs of the identical `S = 3.0` configuration now exist: §6, §7,
and passes A and B. Final accuracy:

| `m` | Final acc, §6 | Final acc, §7 | Final acc, A | Final acc, B | **Range** |
|---|---|---|---|---|---|
| 5 | 0.1319 | 0.0000 | 0.0000 | 0.1000 | 13.2 pp |
| 20 | 0.0536 | 0.1000 | 0.0960 | 0.1002 | 4.7 pp |
| 50 | 0.1085 | 0.1000 | 0.1479 | 0.3952 | **29.5 pp** |
| 100 | 0.3737 | 0.3962 | 0.4973 | 0.5377 | 16.4 pp |
| 200 | 0.4120 | 0.4651 | 0.5886 | 0.4976 | 17.7 pp |

Median range **16.4 pp**, maximum **29.5 pp**. This is large, and it means several
comparisons made earlier in this document are not supported by single runs.
Testing each reported effect against the spread at its own cohort size:

| Comparison | Effect | Spread | Verdict |
|---|---|---|---|
| clip 3.0 → 0.5 @ `m` = 50 | ≈ 55 pp ‡ | 29.5 pp | **Survives** |
| clip 1.1 → 0.5 @ `m` = 20 | 39.8 pp | 4.7 pp | **Survives** |
| Step-size 0.4545× @ `m` = 20 (§9) | 32.1 pp | 4.7 pp | **Survives** |
| clip 3.0 → 1.1 @ `m` = 50 | 55.3 pp | 29.5 pp | Marginal |
| clip 3.0 → 0.5 @ `m` = 100 | 28.5 pp | 16.4 pp | Marginal |
| clip 3.0 → 0.5 @ `m` = 200 | 10.9 pp | 17.7 pp | **Within noise** |
| clip 1.1 → 0.5 @ `m` = 50 | 8.2 pp | 29.5 pp | **Within noise** |
| clip 1.1 → 0.5 @ `m` = 100 | 3.8 pp | 16.4 pp | **Within noise** |
| clip 1.1 → 0.5 @ `m` = 200 | 0.5 pp | 17.7 pp | **Within noise** |

‡ Computed against the **mean of the four recorded `S = 3.0`, `m = 50` draws**
(0.1085, 0.1000, 0.1479, 0.3952 → 0.188), not against any single draw, so the
figure does not move if someone re-runs it. An earlier version quoted 63.5 pp,
which happened to be measured against one of the low draws. Other rows in this
table are single-draw effects (only `S = 3.0`, `m = 50` has four draws on record)
and carry the spread caveat accordingly.

**What still stands:** the central claim — the shipped `S = 3.0` is badly chosen and
lowering it is worth tens of accuracy points at no privacy cost — survives at
`m = 50` by better than 2×, and the step-size mechanism in §9 survives by ~7×. The
ε-invariance gate (§7.0) and every noise measurement (§2, §7.1) are unaffected:
they are measurements of the noise distribution itself, and match closed form to
3–4 significant figures regardless of seed.

**What does not stand:** the distinction between `S = 1.1` and `S = 0.5` at
`m ≥ 50` is **within noise** and should not have been reported as a ranking. The
"plateaus as predicted" reading in §7.2, and the ordering of the ratio columns in
§8.1, rest on differences smaller than the run-to-run spread. The recommendation
of `S = 0.5` over `S = 1.1` is supported at `m = 20` and *not* at `m ≥ 50`.

**Caveat on the caveat:** the spread above was measured at `S = 3.0`, the row that
was asked for. Spread is clearly heterogeneous — it is largest at `m = 50`, where
that configuration sits exactly at the edge of trainability and outcomes are
bimodal (a run either takes off or does not), and smallest at `m = 20`, where every
run is uniformly dead. Runs that train steadily are far more stable, and for the
winning cell this is now **measured, not expected**: the three-seed replication
(§8.2) puts cross-seed spread at `S = 0.5`, `m = 50` at **1.1 pp** on the DP arm —
against 29.5 pp for `S = 3.0` at the same cohort — and the non-DP control (§8) is
*exactly* reproducible at every cohort. The §8.1 ratios at other cohorts remain
single draws and should still not be read to three significant figures.

### 10.4 What would make this reproducible

Not applied, and not applicable from the harness:

1. **A seedable DP query.** Pass a per-round seed derived from a master seed into
   `GaussianSumQuery`, so noise is independent across rounds but reproducible
   across runs. Requires changing the mechanism under test and getting the
   independence right; a wrong implementation silently invalidates the ε.
2. **Report medians over repeats** instead of single runs. Cheapest honest fix, and
   the right one for any figure meant to be compared — cost is a linear multiple of
   compute.
3. **Upstream.** Neither TFF nor TF Privacy exposes a seed on this path; a fix
   belongs there.

Until one of those, **DP figures in this document are single draws from a
distribution with a spread of order 15–30 accuracy points at the marginal cohort
sizes.** Conclusions are drawn only from effects larger than that, as tabulated
above.

---

## Conclusions

**1. The DP implementation is correct.** Measured noise matches TFF's documented
formula to four significant figures (§2). Nothing in `fl/aggregation.py` is
misbehaving.

**2. Clipping is not the cause, and was exonerated by two independent tests.**
The clipping norm (3.0) is ~3× *above* the median update norm (0.984), so it
rarely binds (§1); and clipping with zero noise recovers 85.12 % accuracy against
an 86.93 % non-private baseline (§3a).

**3. The cause is the magnitude of the Gaussian noise relative to the cohort
size.** Noise on the aggregate has norm `S·z·√d / m` = 569.7 at production
settings, against a signal of ≈ 1.0 and a whole-model weight norm of 17.3. The
first DP round replaces the model with a random vector ~30× its own size (§2, §4).

**4. The 10.00 % figure is a degenerate classifier, not a scoring artefact.** The
model assigns one class to 84.8 % then 97.4 % of all test images; 10.00 % is
exactly the 1,000 images of that class in a balanced 10,000-sample test set (§4).

**5. The collapse is structural, not a scale of noise that can be tuned away at
`m = 5`.** The noise sweep (§5) shows the only usable setting, `z = 0.1`, carries
ε = 1060 — vacuous. Every ε below ~50 has already destroyed the model.

**6. The fix is cohort size *and* clipping norm together.** At an identical
ε = 6.228, growing the cohort from 5 to 200 clients turns a flat dead line into a
learning curve (§6). But §8 shows this was overstated: some of that gain is the
cohort, and the `N = 2m` design penalises large cohorts with thin shards. The
combined fix — `m = 50` with `S = 0.5` — reaches **73.4 % final accuracy at
ε = 6.228** (mean of three seeds, range 1.1 pp), against a 76.9 % matched
non-private ceiling (mean of the same seeds, range 2.4 pp): a DP cost of 3.5 pp
(§7, §8.2).

**7. The clipping norm was the larger error, and the earlier claim that it could
not matter was wrong.** Reducing `S` from 3.0 to 0.5 at `m = 50` moves final
accuracy from 18.8 % (mean of the four recorded draws) to 73.4 % (mean of three
seeds) at identical ε — ≈ 55 pp. See the retraction below.

**8. Client-level DP costs ≈ 5 % of achievable accuracy here, not 88 %.**
Normalised against matched no-DP ceilings, ratios at `S = 0.5` are 0.95 at
`m = 50` and 0.95 at `m = 100`, the two cohorts where both arms are stable, and
`m = 200` is consistent with the same value (its control was unstable; ratio
0.88–1.00 depending on the reading — §8.1 †). The headline collapse measured a
badly chosen configuration, not the price of privacy.

**9. Once clipping binds, the clip is a step size, not an SNR lever.** Scaling the
`S = 1.1` server step by 0.4545 recovers the entire 27-point gap to `S = 0.5` at
identical ε and identical SNR (§9).

### Retraction — "lowering `l2_clip_norm` alone does not help"

The previous version of this document closed with:

> Note that lowering `l2_clip_norm` alone does **not** help: `stddev = clip · z`,
> so the noise-to-signal ratio `z·√d/m` is independent of the clipping norm.

**That is wrong, and it was the most consequential error in the diagnosis.** The
algebra holds only in the regime it silently assumed — where clipping *binds*.
When the clip is slack, the signal is unchanged and only the noise scales with
`S`, so SNR = `‖Δw‖·m / (S·z·√d)` and lowering the clip is a direct, unqualified
improvement.

The production setting was in the slack regime and the document said so two
sections earlier — §1 records the clip at 3.0 against a median update norm of
0.984 — so the conclusion contradicted its own evidence. Measured cost of the
error, at fixed ε = 6.228:

| `m` | Final acc @ `S` = 3.0 | Final acc @ `S` = 0.5 |
|---|---|---|
| 50 | 0.188 (mean of 4 draws) | **0.7336** (mean of 3 seeds, §8.2) |
| 100 | 0.3962 (single draw) | **0.6815** (single draw) |
| 200 | 0.4651 (single draw) | **0.5745** (single draw) |

Lowering the clip alone, changing nothing else and spending no additional privacy
budget, was worth **≈ 55 accuracy points** at `m = 50` — computed against the mean
of the four recorded `S = 3.0` draws rather than any one of them (§10.3 ‡), so the
figure is stable under re-running.

### Correction to a claim in README.md

The README states that usable client-level DP here "needs **m ≳ z·√d**, roughly
950 clients per round". That figure came from setting the per-round
signal-to-noise ratio to 1, using the clipping norm (3.0) as the signal.

Both parts are now measurable and both are wrong in interesting ways:

- The realistic signal is the *actual* update norm (≈ 1.0), not the clipping
  bound, which would make the SNR = 1 threshold ≈ 2,850 clients, not 950.
- But SNR = 1 is the wrong criterion. §6 shows genuine learning at a per-round SNR
  of ≈ 0.015, because independent zero-mean noise partially cancels across rounds.
  Useful accuracy appears at `m ≈ 100`, an order of magnitude below either
  analytic figure.

The honest statement is the measured one: **at ε ≈ 6, this model needs on the
order of 100+ clients per round before it learns at all, and 200 clients still
reach only ~56 % against an 86.9 % non-private baseline.** The README's analytic
threshold should be replaced with that.

**Superseded by §7–§8.** The statement above is still an improvement on the
analytic threshold, but it was measured at `S = 3.0` — a clip that never binds at
those cohort sizes and therefore inflates the noise for nothing. At `S = 0.5` the
model learns at `m = 20` (52.19 %) and reaches ≈ 73 % at `m = 50` (mean of three
seeds, §8.2). And the "56 %
at 200 clients" comparison was against the wrong baseline: the matched non-private
ceiling at `m = 200` is 57.42 %, not 86.93 %, because those clients hold 150
examples each. Use the normalised ratios in §8.1.

### What would fix it — not applied, per instruction

No parameter was adjusted to improve any number in this document. For the record,
the options the evidence supports, in order of expected effect:

1. **More clients per round** — the only change demonstrated to work at constant ε
   (§6). Requires a larger simulated population.
2. **A smaller model.** Noise scales as `√d`; the 225,034-parameter CNN is the
   dominant term. A model with 10× fewer parameters cuts the noise norm by ~3.2×.
3. **Adaptive clipping** (`DifferentiallyPrivateFactory.gaussian_adaptive`), which
   tracks a quantile of the update norms instead of a fixed 3.0 chosen from
   unrepresentative round-1 data (§1). This mainly improves the signal side; on
   its own it will not close a 570:1 gap.
4. **Re-derive `l2_clip_norm` from the corrected measurement.** The current 3.0 was
   set from round-1 norms; the across-training median is 0.984.

~~Note that lowering `l2_clip_norm` alone does **not** help: `stddev = clip · z`, so
the noise-to-signal ratio `z·√d/m` is independent of the clipping norm. That is
worth knowing before anyone tries it as the obvious first fix.~~ **Retracted — see
the retraction under Conclusions.** Lowering the clip is in fact the single
highest-value change available, worth ≈ 55 accuracy points at `m = 50` (against
the four-draw mean, §10.3 ‡) at no privacy cost. Item 4 above should be read as
item 1.

### Recommended clipping norm

**`l2_clip_norm = 0.5`, with two caveats: it is the best value tested rather than a
located optimum, and at `m ≥ 50` it is not statistically separable from 1.1.**

It wins or ties in every column of §7 and never underperforms. The support is
uneven, and §10 is the reason to state which parts are load-bearing:

- **Against `S = 3.0`: decisive.** ≈ 55 pp at `m = 50` against the four-draw mean
  of the 3.0 side (§10.3 ‡), roughly 2× the worst-case single-run spread.
- **Against `S = 1.1` at `m = 20`: decisive.** 39.8 pp against a 4.7 pp spread, and
  §9 independently confirms the mechanism at ~7× the spread.
- **Against `S = 1.1` at `m ≥ 50`: not established.** Differences of 0.5–8.2 pp
  against a 16–30 pp spread. Choose 0.5 over 1.1 there because it is never worse
  and is much better where the two do separate, not because the sweep showed it
  winning.

At `m ≤ 20`, where clipping binds and SNR is fixed at `m/(z·√d)` regardless, a
smaller clip is a smaller step and the runs stay numerically alive (§9) —
`S = 0.5` was the only `m = 5` configuration not fully NaN by round 20. The theoretical justification
is that the clip should sit near the typical update norm: the across-training
median is 0.984 at `m = 5` but falls to 0.670, 0.455 and 0.289 at `m = 50`, 100 and
200, so 0.5 sits sensibly inside that range, whereas the current 3.0 is slack
everywhere past `m = 20` and buys pure noise. The honest caveat is that the sweep
bottoms out at 0.5 and SNR is still rising there at `m = 100` and `m = 200`
(clipped fraction only 0.41 and 0.09), so the optimum is probably below 0.5 and
this sweep does not bracket it. There must be a turning point — once clipping
binds, step size ∝ `S`, so `S → 0` means no learning within a fixed round budget —
but it was not found. Adaptive clipping
(`DifferentiallyPrivateFactory.gaussian_adaptive`) tracks a quantile of the update
norms and would locate it per-round without a sweep; that is the better long-term
answer than any fixed constant.

---

## Reproducing

```bash
python scripts/diagnose_dp.py --experiment noise_model  --out docs/_noise_model.json  > ns.log 2>&1
python scripts/diagnose_dp.py --experiment validate     --out docs/_validate.json     > v.log  2>&1
python scripts/diagnose_dp.py --experiment ablation     --out docs/_ablation.json     > a.log  2>&1
python scripts/diagnose_dp.py --experiment noise_sweep  --out docs/_noise_sweep.json  > n.log  2>&1
python scripts/diagnose_dp.py --experiment cohort_sweep --out docs/_cohort_sweep.json > c.log  2>&1
python scripts/diagnose_dp.py --experiment epsilon_gate    --out docs/_epsilon_gate.json    > g.log  2>&1
python scripts/diagnose_dp.py --experiment clip_sweep      --out docs/_clip_sweep.json      > cs.log 2>&1
python scripts/diagnose_dp.py --experiment cohort_baseline --out docs/_cohort_baseline.json > cb.log 2>&1
python scripts/diagnose_dp.py --experiment step_size       --out docs/_step_size.json       > ss.log 2>&1
```

```bash
python scripts/diagnose_dp.py --experiment repeatability --out docs/_repeatability.json > r.log 2>&1
python scripts/diagnose_dp.py --experiment replication   --out docs/_replication.json   > rp.log 2>&1
```

Non-DP runs are exactly reproducible. **DP runs are not, and cannot be made so
from this harness** — the noise is drawn inside TFF's executor, which never sees
`tf.random.set_seed`. Measured spread is 4.7–29.5 accuracy points depending on
cohort size. See §10 for the mechanism, the evidence, and which conclusions in
this document survive that spread; read no DP figure here to three significant
figures.

Raw per-round data for every run in this document is committed alongside it as
`docs/_*.json`. Do not pipe the script's stdout into `grep`/`tail`: TFF leaves a
background executor subprocess holding the pipe, so the reader never sees EOF.
