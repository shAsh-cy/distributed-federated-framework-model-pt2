# FEMNIST: finding a budget at which FedAvg trains

**Goal.** [femnist_cohort.md](femnist_cohort.md) ended on an honest but
unsatisfying note: every cell of the decoupled cohort sweep was
optimiser-limited — a flat line at 5–8 % says nothing about cohort size. This
document finds an operating point at which federated FEMNIST actually trains,
*before* drawing any conclusion about the cohort axis. Cohort fixed at
m = 200, no DP, three seeds per configuration, means with ranges throughout.

## 1. Ruling out the boring explanations first

Three hypotheses that would make a budget search pointless were tested and
eliminated (`tests/test_training_parity.py`, all on both the TF and torch
client paths):

- **Clients might reinitialise instead of training from the received global
  weights.** With lr = 0, both trainers return the received weights
  bit-for-bit — a reinitialising trainer could not. They train from the
  global model.
- **The TF and torch training steps might differ.** Forward-pass parity was
  already proven; now the *training* steps are too: identical weights,
  identical full batch (removing shuffle-order differences), 1 and 3
  SGD+momentum epochs → weights match across frameworks within 5e-4, with a
  guard asserting the step actually moved. Keras folds lr into the momentum
  velocity, torch applies lr at the update — trajectory-equivalent at
  constant lr, now measured rather than assumed.
- **Optimiser defaults might differ.** The torch optimiser is constructed
  with weight_decay = 0 and nesterov = False, Keras's defaults — asserted
  explicitly.

The learning rate itself (0.01, batch 32) is not obviously wrong for
~159-example shards — it is the rate the pooled baseline trains fine with —
but at E = 1 it yields only ~5 gradient steps per client per round. That is
the remaining suspect, and the cheapest knob.

## 2. Local epochs: the stall is budget, nothing else

m = 200, 20 rounds, no DP, 3 seeds:

| E | Mean final | Range | s/round | Wall-clock per run |
|---|---|---|---|---|
| 1 | 0.0796 | 0.0498 | 29.2 | ~10 min |
| 5 | 0.6567 | 0.0226 | 75.0 | ~25 min |
| **10** | **0.7279** | **0.0021** | 133.6 | **~45 min** |

E = 1 replicates the cohort experiment's stall exactly (its 7–8 % flat line
was the same configuration). Multiplying local steps by five, with zero extra
communication, takes the same system from 8 % to 66 %; ten steps reach
**72.8 %**, with the seed range *tightening* to 0.2 pp as the optimisation
becomes better conditioned.

**First configuration within ~15 points of the 85.6 % pooled baseline:
E = 10, R = 20 → 72.8 % (threshold 70.6 %). Wall-clock cost of a single run:
~45 minutes on CPU** (133.6 s/round × 20 rounds, in-process harness,
1,000-writer population).

**No drift signature (the item the E grid was watching for).** Accuracy does
not degrade from E = 5 to E = 10 — it improves by 7 pp and stabilises across
seeds. Client drift on non-IID shards would show the opposite ordering. At
this scale the evidence points at insufficient budget, now partially paid,
not at FedProx.

## 3. Rounds at E = 10: the curve, not the endpoint

<!-- RESULTS:R_CURVE -->

## 4. The cohort axis, re-asked at a budget where FedAvg trains

<!-- RESULTS:M_AT_BUDGET -->

## 5. What was NOT run

R = 200 at E = 10 costs ≈ 8.9 h per seed (≈ 27 h for three) and was not run;
the R = 100 curves in §3 are the basis for judging whether longer training
would change any conclusion. DP is deliberately absent from every run in this
document — reintroducing it belongs after the cohort question has an answer
at a working budget, not before.

## Reproducing

```bash
python scripts/femnist_experiments.py --experiment budget --m 200 --epochs-list 1,5,10 \
    --out docs/_femnist_budget_e.json > e.log 2>&1
python scripts/femnist_experiments.py --experiment budget --m 200 --epochs-list 10 --rounds 100 \
    --out docs/_femnist_r_curve.json > r.log 2>&1
python scripts/femnist_experiments.py --experiment nodp_control --cohorts 5,50,500 \
    --local-epochs 10 --out docs/_femnist_m_at_budget.json > m.log 2>&1
```

Raw per-round data committed as `docs/_femnist_*.json`. The usual warning
applies: do not pipe stdout into a command that waits for EOF.
