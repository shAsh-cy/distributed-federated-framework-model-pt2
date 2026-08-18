# Server optimizers: FedOpt and FedProx, measured

FedAvg applies the aggregated client delta to the global model verbatim. FedOpt
(Reddi et al. 2021, [arXiv:2003.00295](https://arxiv.org/abs/2003.00295)) reads
that delta as a *pseudo-gradient* and feeds it to a server optimizer with state
that persists across rounds — momentum for FedAvgM, first and second moments
for FedAdam/FedYogi. FedProx (Li et al. 2020) attacks the same problem from the
client side, adding a proximal term `(mu/2)||w - w_global||^2` to the local
objective so client drift is penalised rather than corrected after the fact.

Both were previously omitted from this repo *a priori*. This document records
what happened when they were measured instead. The implementation is
`fl/server_optimizer.py` and `fl/aggregation.py` (`FedOptAggregator`), and
`fl/fedprox.py`; the batch that produced every number here is
`scripts/fedopt_batch.py`.

The identity case is load-bearing: SGD at server learning rate 1.0 with no
momentum reproduces `FedAvgAggregator`'s output bit-for-bit (asserted in
`tests/test_server_optimizer.py`), so FedOpt is a strict generalisation of the
existing path rather than a parallel one.

---

## Phase A — a tuning map, not a result

**One seed. This section ranks nothing.** Its only job is to locate the region
each family works in, so phases B and C had somewhere to spend three seeds.
These runs are non-DP and therefore exactly reproducible, so the uncertainty
here is not run-to-run noise — it is seed-to-seed variation in the partition and
cohort draws, which phase B went on to measure directly on this same task at
**0.9–2.9 pp across three seeds**. Differences between single-seed cells smaller
than that are not evidence of anything, which rules out reading any of the
close pairs below as an ordering.

Fashion-MNIST, N = 100, m = 50, R = 20, client lr 0.01 / momentum 0.9, seed 42.
FedAvg reference at the same budget: **0.7716**.

| Family | Swept | Cell | Final acc |
|---|---|---|---|
| FedAdam | server lr | 0.001 | 0.6403 |
| | | **0.01** | **0.7898** |
| | | 0.1 | 0.7894 |
| | | 1.0 | 0.1000 (diverged; best-round 0.2798) |
| FedYogi | server lr | 0.001 | 0.6391 |
| | | 0.01 | 0.7871 |
| | | **0.1** | **0.7949** |
| | | 1.0 | 0.1000 (diverged; best-round 0.2798) |
| FedAvgM | momentum @ server lr 1.0 | **0.9** | **0.8086** (best-round 0.8127) |
| | | 0.99 | 0.7909 (best-round 0.7964) |

Two things the map does establish, because they are not small effects:

- **FedAdam and FedYogi collapse to chance at server lr 1.0.** 0.1000 on ten
  balanced classes is the untrained prior. This is a cliff, not a gradient.
- **They tolerate a decade either side of it.** 0.01 and 0.1 land within 0.6 pp
  of each other for FedAdam and within 0.8 pp for FedYogi.

The selected cells (FedAdam 0.01, FedYogi 0.1, FedAvgM momentum 0.9) carried
into phases B and C.

### The `accuracy_spread` field in the JSON is not a tuning-difficulty measure

`_fedopt_batch_a.json` records `accuracy_spread` and `near_best_fraction` per
family. Read them with care: FedAdam's spread is 0.690 and FedAvgM's is 0.018,
which looks like an enormous difference in robustness and is mostly an artifact
of what was swept. FedAdam's number spans four server learning rates across
three decades *including the diverged cell*; FedAvgM's spans two momentum values
at a single fixed server learning rate. They are not the same experiment. See
[the Reddi question](#does-this-agree-with-reddi-et-al-on-tuning-difficulty)
below.

---

## Phase B — Fashion-MNIST, three seeds

N = 100, m = 50, R = 20, seeds 42/43/44. Ranges are max − min over seeds.

| Arm | Per-seed final | Mean | Range | vs FedAvg |
|---|---|---|---|---|
| FedAvg | 0.7716, 0.7788, 0.7553 | **0.7686** | 0.0235 | — |
| FedAvgM (slr 1.0, β 0.9) | 0.8086, 0.8213, 0.8058 | **0.8119** | 0.0155 | **+4.33 pp** |
| FedYogi (slr 0.1) | 0.7949, 0.8238, 0.7992 | **0.8060** | 0.0289 | **+3.74 pp** |
| FedAdam (slr 0.01) | 0.7898, 0.7861, 0.7809 | **0.7856** | 0.0089 | **+1.70 pp** |

All three beat FedAvg. FedAvgM's +4.33 pp is the largest margin and its seed
range (0.0155) does not overlap FedAvg's band, so the ordering FedAvgM > FedAvg
is real at this budget. FedAvgM vs FedYogi (+0.59 pp, ranges 0.0155 and 0.0289)
is **not** separated by three seeds — those two are tied on this evidence.

---

## Phase C — FEMNIST, three seeds, at borrowed hyperparameters

**This is the limitation, stated before the numbers rather than after them: the
server learning rates were transferred from the Fashion sweep and never re-tuned
on FEMNIST.** A per-dataset re-tune is part of FedOpt's real cost, and it was not
paid here. These numbers measure *transfer*, not the methods' ceiling, and they
may understate all three arms. The JSON says so in its own `tuning_note` field.

FEMNIST, 1,000 writers, m = 200, E = 10, R = 20, seeds 42/43/44. Control is the
recorded no-DP arm from `docs/_femnist_budget_e.json`.

| Arm | Per-seed final | Mean | Range | vs control |
|---|---|---|---|---|
| FedAvg control | 0.7288, 0.7283, 0.7267 | **0.7279** | 0.0021 | — |
| FedYogi (slr 0.1) | 0.7367, 0.7283, 0.7358 | **0.7336** | 0.0083 | +0.57 pp |
| FedAvgM (slr 1.0, β 0.9) | 0.7131, 0.7219, 0.7264 | **0.7205** | 0.0133 | −0.74 pp |
| FedAdam (slr 0.01) | 0.6709, 0.6656, 0.6660 | **0.6675** | 0.0053 | **−6.04 pp** |

The honest reading:

- **FedYogi's +0.57 pp is not a win.** Its lowest seed (0.7283) sits inside the
  control's own three-seed band (0.7267–0.7288). Indistinguishable.
- **FedAvgM's −0.74 pp is not a loss** for the same reason in the other
  direction — its range (0.0133) covers the gap.
- **FedAdam's −6.04 pp is real.** Three seeds spanning 0.0053, more than 28×
  the control's range away from it. Something about FedAdam at server lr 0.01
  genuinely does not transfer to this population.

So the Fashion result did not survive the move to natural per-writer
heterogeneity — and the one thing that did travel was the *failure*, not the
gain. Whether a FEMNIST re-tune would recover the Fashion margins is untested
and this document does not claim either way.

---

## Does this agree with Reddi et al. on tuning difficulty?

**Neither. The experiment as designed cannot adjudicate that claim**, and
reporting it as agreement or contradiction would be the error.

What the paper actually claims is narrower than "adaptive methods are easier to
tune". It is a claim about the *shape of the good region in the joint (client
lr, server lr) plane* (§5.2): *"For FedAvgM, there are only a few good values of
η_l for each η, while for FedAdam and FedYogi, there are many good values of η_l
for a range of η. Thus FedAdam and FedYogi are arguably easier to tune in this
setting."* Appendix E.3 sharpens it — adaptive methods show **rectangular**
(separable) good regions, FedAvgM a **triangular** (coupled) one, meaning its two
learning rates must be tuned together. The claim is hedged in the paper itself
("arguably", "in this setting", "in many tasks"), and Figure 2 is one task.

Three facts about our grid make it the wrong instrument:

1. **We never varied client lr.** It was fixed at 0.01 throughout. The claim is
   about the *coupling* between client lr and server lr; a single vertical line
   through the plane cannot reveal whether the region is a rectangle or a
   triangle.
2. **We swept FedAvgM on an axis the paper holds fixed.** Reddi et al. fix
   server momentum at 0.9 and never sweep it — it is the peer of β₁/β₂, not a
   tuning axis. We swept momentum for FedAvgM and server lr for the adaptive
   arms, which compares two different kinds of knob.
3. **The budgets are unequal.** Reddi et al. give *every* optimizer the same
   8 × 9 = 72-point grid at half-decade spacing. Ours gives the adaptive arms
   four cells and FedAvgM two.

What our data *can* support, stated as the configuration comparison it is: with
server lr pinned to 1.0 and momentum swept, FedAvgM reached 0.8086 on Fashion,
while FedAdam and FedYogi with server lr swept over three decades reached 0.7898
and 0.7949 — and both adaptive arms collapse to chance at the server lr FedAvgM
runs at. That is an end-to-end result under a stated, unequal search budget. It
is not a test of Reddi et al.

One observation that is worth recording without dressing it as a verdict: in
this repo's experience the adaptive methods were the ones that needed the
tuning. FedAvgM worked at server lr 1.0 — the FedAvg default, requiring no
search on that axis at all — while FedAdam and FedYogi had to have their decade
found or they produced a model no better than the untrained prior. Reddi et al.
also report FedAvgM competitive once tuned on dense-gradient tasks (CIFAR-10
77.4 vs FedYogi 78.0), so this is not in tension with the paper.

A minimal legitimate replication would sweep the same 2-D grid for every arm
(client lr and server lr each ≥ 3 decades at √10 spacing), fix τ at 1e-3 and
momentum at 0.9, and report the *shape* of each method's near-best region rather
than peak accuracy. That is a substantially larger experiment than this one.

---

## Phase D — FedProx: a measured null

FEMNIST at the working budget (1,000 writers, m = 200, E = 10, R = 20), **one
seed per µ**, against the same three-seed control.

| µ | Final acc | vs control (0.7279) |
|---|---|---|
| 0.001 | 0.7287 | +0.08 pp |
| 0.01 | 0.7264 | −0.15 pp |
| 0.1 | 0.7086 | **−1.93 pp** |

The control's own three-seed range is 0.0021 (0.21 pp), and that is the yardstick
a single-seed difference has to clear. µ = 0.001 (+0.08 pp) and µ = 0.01
(−0.15 pp) are both *inside* it: indistinguishable from noise. µ = 0.1 is 9×
the control range below it, and that one is a real degradation — the proximal
term is strong enough to hold clients near a global model that has not finished
moving.

**The claim this licenses, and no more: no benefit from FedProx at the tested
budgets on our splits, and a measurable cost at µ = 0.1.** It is *not* "FedProx
doesn't help". One seed per cell, one dataset, one budget, three µ values, and
E = 10 local epochs — FedProx is motivated by client drift under many local
steps and systems heterogeneity, and this budget may simply not be where it
earns its keep.

What changed is the epistemic status, not the default. FedProx was previously
absent from this repo because it had been ruled out in advance; it is now absent
from the defaults because it was measured and did not pay. That is the upgrade.

---

## Phase E — FedOpt under differential privacy

*Running. This section is filled in from `docs/_fedopt_batch_e.json` when the
batch lands.*

The composition is sound and this is the part worth stating carefully. When
`privacy.enabled`, the server optimizer does not replace the DP aggregator — it
**wraps** it. The DP aggregator clips each client's delta, adds Gaussian noise
and averages uniformly; the optimizer then consumes that *privatized* delta as
its pseudo-gradient. Its moments at round *t* are functions of privatized
outputs from rounds 1..*t*−1 alone, so this is post-processing and the reported
epsilon at a fixed noise multiplier is unchanged. `compute_epsilon` is a
function of (z, q, R) only, and `tests/test_server_optimizer.py` asserts the
equality directly rather than arguing it.

This replaced a blanket refusal. The refusal's stated reason — that the FedOpt
path applies to the weighted mean, whose per-client sensitivity is unbounded —
was a true description of the *old implementation*, in which `FedOptAggregator`
computed its own sample-count-weighted mean, and not a barrier in principle. The
combination that would genuinely break accounting is still unreachable, but by
construction rather than by validation: under DP `make_aggregator` always
supplies a DP inner aggregator, so `weighted_average` is never called.
`tests/test_server_optimizer.py::TestDpPostProcessing` asserts that too, by
making `weighted_average` raise and running a DP round through FedAdam.

The mechanism the phase tests: DP noise is zero-mean, and server momentum
averages across rounds, which is exactly the setting where momentum should
recover signal. The counter-consideration, and the reason FedAvgM is run
alongside FedAdam rather than FedAdam alone: FedAdam's second moment estimates
E[Δ²] = E[Δ]² + σ², so injected noise variance inflates *v* permanently, and
the step `1/(√v + τ)` is a nonlinear map of a noisy input — a biased estimate of
the noiseless Adam step. Momentum has no such term. Without both arms a null
result could not distinguish "server adaptivity does not recover DP cost" from
"the second moment is the part the noise breaks".

Arms, matched to the recorded fixed-clip DP arm on everything except the server
step (S = 2.0, target ε = 6.228 at δ = 1e-5, z = 1.1141230964660644, m = 200,
E = 10, R = 20, seeds 42/43/44):

- **FedAdam** at server lr 0.01 — the arm the question was asked about.
- **FedAvgM** at server lr 1.0, momentum 0.9 — the mechanism control.

Against two recorded arms: the fixed-clip DP arm at **0.6815**
(`docs/_final_batch_b.json`) and the no-DP control at **0.7279**
(`docs/_femnist_budget_e.json`) — a DP cost of 4.64 pp.

Two limitations that will apply to whatever the numbers say. The server learning
rates are borrowed *twice over* — transferred from Fashion, then applied under
noise that changes the loss surface the server step sees. And DP noise is drawn
unseedably by TFF, so unlike phases A–D these runs reproduce in distribution
rather than bit-for-bit.
