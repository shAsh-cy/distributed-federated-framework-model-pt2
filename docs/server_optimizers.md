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

**Both arms lose badly, and the reason is a coupling nobody had priced: the
clipping norm and the server optimizer are not independent knobs.**

FEMNIST, 1,000 writers, m = 200, E = 10, R = 20, seeds 42/43/44, matched to the
recorded fixed-clip DP arm on everything except the server step — S = 2.0,
target ε = 6.228 at δ = 1e-5, z = 1.1141230964660644.

| Arm | Per-seed final | Mean | Range | vs DP arm | Residual DP cost |
|---|---|---|---|---|---|
| No-DP control | 0.7288, 0.7283, 0.7267 | **0.7279** | 0.21 pp | — | — |
| Fixed-clip DP | 0.6841, 0.6769, 0.6835 | **0.6815** | 0.72 pp | — | 4.64 pp |
| DP + FedAdam (slr 0.01) | 0.5636, 0.5527, 0.5268 | **0.5477** | 3.68 pp | **−13.38 pp** | **18.02 pp** |
| DP + FedAvgM (slr 1.0, β 0.9) | 0.5335, 0.5280, 0.5337 | **0.5317** | 0.57 pp | **−14.98 pp** | **19.62 pp** |

So the answer to the question the phase was built to ask — does adaptive server
optimization recover part of the 4.64 pp DP cost? — is **no, emphatically**. It
roughly quadruples it. Both arms land near 0.53–0.55 where the plain DP arm
reaches 0.68.

**ε is 6.228256677985603 on all six runs, identical to the fixed-clip arm's, to
every recorded digit.** That is the post-processing claim holding on the real
operating point rather than in a unit test: the server optimizer changed the
model by 15 accuracy points and moved the privacy accounting by nothing.

### The mechanism: a server optimizer invalidates the clip bracket

The update norms say what happened, and they were recorded per run:

| Arm | Median ‖Δw‖ over all rounds | vs baseline | Clipped fraction (S = 2.0) |
|---|---|---|---|
| Fixed-clip DP | 2.174 | — | 0.569 |
| DP + FedAdam | 2.546 | +17 % | 0.628 |
| DP + FedAvgM | 3.062 | **+41 %** | **0.818** |

S = 2.0 was not arbitrary — it is the begins-to-bind knee, re-bracketed against
the *FedAvg* trajectory's measured update norms at this exact budget
([docs/adaptive_clipping.md](adaptive_clipping.md) §phase A), where it clips
57 % of updates. A server optimizer moves the global model further per round,
clients then train from a model further from where they started, their deltas
grow, and a larger share of each delta is thrown away by a clip chosen for a
gentler trajectory. FedAvgM inflates the median norm by 41 % and ends up
discarding the tails of **82 %** of all updates.

Two existing results in this repo predicted this and nobody connected them to
FedOpt. [docs/dp_diagnosis.md](dp_diagnosis.md) §9 asks "is the clipping norm
acting as a server step size?" and finds that at these budgets it partly is —
so changing the server step and changing the clip are not independent
interventions. And [docs/femnist_budget.md](femnist_budget.md) §5 states flatly
that when the budget moved to E = 10, "the clip must be re-bracketed from
scratch". Phase E is the same lesson with the *server optimizer* as the thing
that moved: **under DP, the clip bracket does not transfer across server
optimizers**, and both arms here ran with a clip bracketed for neither.

**The two arms fail differently, which is only visible because both were run.**
FedAvgM is *consistently* bad — 0.57 pp across seeds, tighter than the fixed
arm's own 0.72 pp — the signature of a systematic overshoot that happens the
same way every time. FedAdam's norm inflation is milder (+17 %) but its seed
range blows out to **3.68 pp, five times the fixed arm's**, which is the
signature of erratic step scaling rather than uniform overshoot: its second
moment estimates E[Δ²] = E[Δ]² + σ², so the injected noise variance inflates *v*
and `1/(√v + τ)` becomes a nonlinear function of a noisy quantity. Had only
FedAdam been run — the arm the question was originally posed about — the result
would have been a bare number with an unattributable cause, and the
second-moment story would have looked like the whole explanation rather than
half of one.

### What this does and does not license

**It licenses:** at this operating point, with server learning rates
transferred from a noiseless Fashion sweep and a clip bracketed for FedAvg,
server-side adaptivity does not recover DP cost — it multiplies it.

**It does not license "FedOpt does not work under DP."** The confound is
identified and measured, not hypothesised: the clip is wrong for these
trajectories. The hyperparameters here are borrowed *twice over* — tuned on
Fashion without noise, then applied to FEMNIST under noise that changes the loss
surface the server step sees. Nobody has run the obvious follow-ups, and this
document does not predict their outcome:

- re-bracket S against the FedOpt trajectory's measured norms (the norms are
  recorded above, so the bracket is a cheap calculation, not a sweep);
- or lower the server learning rate under DP so the trajectory stops outrunning
  the clip — FedAvgM's steady-state effective step at β = 0.9 is ~1/(1−β) = 10×;
- either way, re-tune under noise rather than transfer.

Until one of those runs, the honest claim is the narrow one: **no benefit at
this operating point, with a measured reason to think the operating point rather
than the method is what failed.**

Finally, DP noise is drawn unseedably by TFF, so unlike phases A–D these six runs
reproduce in distribution rather than bit-for-bit. The separations here are 13–15
pp against seed ranges of 0.6–3.7 pp, so they survive that comfortably; do not
read the individual figures to three significant digits.

### The composition, and why the refusal is gone

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

The hypothesis the phase was built on was reasonable and turned out to be
beside the point: DP noise is zero-mean, and server momentum averages across
rounds, so momentum should recover signal from noise. It does — that is not
where either arm lost. What neither arm survived was the clip, and no amount of
noise-averaging helps when 82 % of every update is being discarded before the
optimizer ever sees it.

That is worth recording as a methodological lesson rather than only a result.
The composition was checked for *privacy* correctness exhaustively — post-
processing, structural unreachability, ε equality asserted three ways — and the
thing that actually broke was a *utility* coupling between two hyperparameters
that had each been tuned correctly in isolation. Privacy soundness and
mechanism sanity are different audits, and passing the first buys nothing in the
second.
