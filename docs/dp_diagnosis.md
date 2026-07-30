# Diagnosis: why the DP runs collapse to chance

**Question.** Both DP configurations report exactly 10.00 % test accuracy — chance
on a balanced 10-class problem, and below the 12.25 % untrained baseline. Is the
DP implementation broken, or is this the mechanism behaving correctly at a scale
where it cannot work?

**Verdict.** The implementation is not broken. The collapse is caused by the
**magnitude of the Gaussian noise**, not by clipping, not by a bug in the
aggregation, and not by a misconfigured clipping norm. At the configured cohort
size the noise added to the aggregate is roughly **570× larger in L2 norm than
the model's entire weight vector's useful signal**, so the first DP round
replaces the model with noise and every later round adds more.

**No DP code was changed to produce this document.** The harness
(`scripts/diagnose_dp.py`) imports `fl.aggregation` unchanged and drives it
directly.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Clipping norm far below update norms, annihilating updates | **Ruled out** | §1 — median update norm 0.984, clip 3.0; clip is 3× *above* the median |
| Gaussian noise magnitude destroys the signal | **Confirmed** | §2, §3 — noise ‖·‖ = 569.7 vs signal ‖·‖ ≈ 1.0 |
| Model degenerates to predicting one class | **Confirmed** | §4 — 84.8 % then 97.4 % of all test images given the same label |
| Bug in the DP aggregation path | **Ruled out** | §2 — measured noise matches TFF's documented formula to 4 significant figures |

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

Three runs, identical seed, 10 clients, `m = 5`, 20 rounds.

| Run | `z` | `S` | Final accuracy | Best |
|---|---|---|---|---|
| **(a) clipping only** (`z = 0`, `S = 3.0`) | 0 | 3.0 | **0.8512** | 0.8634 |
| **(b) noise only** (clipping disabled) | 6 × 10⁻⁶ | 10⁶ | **0.0000** | 0.0096 |
| **(c) both** — production config | 2.0 | 3.0 | **0.0000** | 0.1005 |

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

| `z` | ε | ‖noise‖ | **Final accuracy** | Best | Predicted-class counts | Largest share |
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

**6. The fix is cohort size, and it is demonstrated, not asserted.** At an
identical ε = 6.228, growing the cohort from 5 to 200 clients takes best accuracy
from 13.2 % to 55.8 % and turns a flat dead line into a learning curve (§6).

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

Note that lowering `l2_clip_norm` alone does **not** help: `stddev = clip · z`, so
the noise-to-signal ratio `z·√d/m` is independent of the clipping norm. That is
worth knowing before anyone tries it as the obvious first fix.

---

## Reproducing

```bash
python scripts/diagnose_dp.py --experiment noise_model  --out docs/_noise_model.json  > ns.log 2>&1
python scripts/diagnose_dp.py --experiment validate     --out docs/_validate.json     > v.log  2>&1
python scripts/diagnose_dp.py --experiment ablation     --out docs/_ablation.json     > a.log  2>&1
python scripts/diagnose_dp.py --experiment noise_sweep  --out docs/_noise_sweep.json  > n.log  2>&1
python scripts/diagnose_dp.py --experiment cohort_sweep --out docs/_cohort_sweep.json > c.log  2>&1
```

Raw per-round data for every run in this document is committed alongside it as
`docs/_*.json`. Do not pipe the script's stdout into `grep`/`tail`: TFF leaves a
background executor subprocess holding the pipe, so the reader never sees EOF.
