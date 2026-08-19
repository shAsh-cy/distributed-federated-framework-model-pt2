# Personalization: a shared backbone, a local head

**Status: implemented, with unit tests; the two measurement phases are queued on
the overnight chain and have not run.** Everything below that is a number is either
arithmetic over the architecture (the communication saving) or a measurement of
the *populations* the experiment will run on (their heterogeneity, their shard
sizes). No accuracy figure is quoted, because none has been produced. §8 states
what will be measured and where it will land; §7 states the prediction *before*
the runs, so the result cannot be read backwards into whatever comes out.

## 1. Why personalization, and why on these splits

A global model is one answer to a question every client asks slightly
differently. Under heterogeneity that answer is a compromise, and the clients it
serves worst are the ones whose data least resembles the population — which is
to say, the clients federated learning exists for. Personalization keeps a
per-client model without giving up the shared statistical strength: the
representation is federated as usual, the classifier head is fitted locally and
never leaves the client.

The version implemented here is **FedRep** (Collins et al., ICML 2021). Each
round a sampled client:

1. receives the shared representation,
2. freezes it and takes `head_epochs` local epochs on its own head,
3. freezes the head and takes the remaining local epochs on the representation,
4. submits **the representation only**. The head stays, and is still there next
   time this client is sampled.

## 2. The two populations, measured

Both splits already exist in this repository, and they are not the same kind of
heterogeneous. This matters more than it might look, and it is measured rather
than assumed:

| | FEMNIST, 1,000 writers (seed 42) | Fashion-MNIST, N=100, Dirichlet α=0.1 (seed 42) |
|---|---|---|
| client boundary | one real writer | synthetic label-skew draw |
| pooled label entropy | 3.664 nats (uniform: 4.127) | 2.303 nats (= uniform, balanced) |
| per-client label entropy, mean | 3.377 nats | 0.829 nats |
| …as a share of pooled | **92.2 %** | **36.0 %** |
| classes present per client (median) | 52 of 62 | 5 of 10 |
| train shard size (min / median / max) | 30 / 159 / 392 | 2 / 411 / 2,732 |
| test shard size (min / median / max) | 4 / 18 / 44 | 0 / 69 / 457 |
| total train / test examples | 196,241 / 22,639 | 60,000 / 10,000 |
| clients with < 10 held-out samples | 6 of 1,000 | 8 of 100 |
| clients with no held-out samples | 0 | 1 |

**FEMNIST's heterogeneity is barely label skew at all.** The median writer
produces 52 of the 62 characters, and per-writer label entropy is 92 % of the
pooled figure. What differs between writers is *style* — a shift in the input
distribution, not in the label prior. **Fashion at α=0.1 is the opposite**: the
labels carry the whole shift, the median client sees five of ten classes, and one
client sees a single class.

That contrast is the reason both phases exist. A locally fitted classifier head
re-weights and re-orients the decision boundary *in representation space*; it
cannot re-learn the representation. So a head should be able to absorb a shifted
label prior almost completely, and should be able to do comparatively little
about a shifted input distribution.

## 3. What is implemented

**The split is declared once, in the architecture spec.** `ArchSpec` gains
`personal_layers`, naming the layers that form the head; everything below is the
backbone. The marker is validated on construction: the named layers must be
weight-bearing and must be the *trailing* run of them, because a "head" taken
from the middle of a network would leave aggregated layers stacked on top of
unaggregated ones, which is not a representation/head decomposition of anything.
Both shipped architectures mark `("logits",)`. A spec with no marker splits into
an all-shared backbone and an empty head — which is exactly FedAvg, so the two
algorithms are one code path over a different marker rather than two code paths.

Everything downstream reads that one marker:

| Layer | What it gains | File |
|---|---|---|
| spec | `personal_mask`, `shared_names`/`personal_names`, `split_weights`/`merge_weights`, per-half parameter counts | `fl/archspec.py` |
| adapters | `to_shared`, `head_of`, `load_shared`, `load_head` — on **both** frameworks, composed from the existing conversions so the transposes still live in one place | `fl/adapters.py` |
| wire | `shared_weights_to_proto`, `proto_to_shared_weights`, `proto_to_personalized_weights` | `fl/serialization.py` |
| state | `HeadStore` — per-client heads keyed by client id, copy-in and copy-out | `fl/personalization.py` |
| harness | `StagedTrainer` (one traced step per variable group), the round loop, per-client evaluation | `scripts/personalization_experiments.py` |

Two decisions in there are worth stating because the alternative fails quietly:

- **The decoder refuses a payload carrying a head tensor** rather than dropping
  it. A peer that sends its head is either broken or leaking, and stripping the
  tensor would hide both — and would hide exactly the parameters personalization
  exists to keep local.
- **`HeadStore` copies on the way in and on the way out.** Local training mutates
  the array it is handed; a store that returned its own array would let one
  client's round rewrite another client's stored head. That is a cross-client
  write no accuracy figure would reveal. The copy costs 32 KB per client on
  `femnist_cnn`.

## 4. Scope: the harness, and what containers would additionally need

**Every recorded personalization number will be an in-process harness number**,
the same scoping the FEMNIST results already carry
([femnist_budget.md](femnist_budget.md)). This is a real limitation and not a
shortcut, so here is precisely what is missing.

FedRep needs one thing the deployed client does not have: **a head that survives
across rounds on the client**. `fl/client.py` holds nothing between rounds except
its shard, and a container that restarts re-registers, reclaims its shard, and
would resume with a head that is either stale, another client's, or freshly
initialised. The third is the dangerous one, because it is invisible: the run
completes, the numbers look plausible, and the personalization was never there.
Wiring it up means durable client-side state keyed to the reclaimed identity,
plus a protocol flag so the server knows to publish and expect backbone-only
payloads. The wire format for that is implemented and tested here; the client
persistence and the protocol flag are not, and no `configs/*.yaml` field enables
personalization — a config knob that silently did nothing on the deployed path
would be worse than its absence.

## 5. The communication saving, measured

Withholding the head removes exactly the head from the payload, and nothing else.
That is a smaller number than "personalization reduces communication" usually
implies, because the head here is a single classifier layer:

| Architecture | total params | head params | head share | payload/transfer | payload saved |
|---|---|---|---|---|---|
| `femnist_cnn` | 231,742 | 7,998 (128×62 + 62) | **3.45 %** | 926,968 B → 894,976 B | 31,992 B |
| `small_cnn` | 225,034 | 1,290 (128×10 + 10) | **0.57 %** | 900,136 B → 894,976 B | 5,160 B |

Both directions, every round, per sampled client. The protobuf-framed figure is
slightly larger than the payload figure — a whole tensor's name, shape and length
prefix go with it — and is recorded per run in the `wire` block of the phase JSON
rather than quoted from arithmetic here.

The honest reading: **on this architecture the saving is a rounding error, and
the two datasets differ by a factor of six.** A 62-class head is 3.45 % of a
231,742-parameter model; the same idea on a model whose head is most of the
parameters would save something worth naming. Quoting one dataset's figure for
the other would be wrong by 6×, which is why both are in the table.

## 6. The evaluation protocol

Personalized accuracy requires per-client test data, and where that data comes
from differs by dataset. The distinction is load-bearing and is carried into the
phase JSON (`per_client_test_data`) so no reader has to infer it:

- **FEMNIST — real.** Each client is one writer, and `test_shards[k]` is that
  writer's own held-out samples, taken verbatim from LEAF's by-writer split. The
  client↔test-data correspondence is upstream's, not this repository's.
- **Fashion-MNIST — synthetic.** A pooled dataset has no client boundaries, so
  the test split is dealt across clients using the *same per-class proportions*
  as the training split (`partition_dirichlet_paired`). Dealing it with a fresh
  Dirichlet draw would give client `k` a training set skewed one way and a test
  set skewed another, and the measurement would then be of that mismatch. The
  train half is bit-identical to the unpaired partitioner at the same seed, so a
  paired run stays comparable with every Dirichlet run already recorded.

Three numbers are reported **per client, on that client's own held-out data**:

| | what it is | which arm produces it |
|---|---|---|
| **global** | the aggregated global model, scored on client `k`'s test shard | FedAvg |
| **fine-tuned** | that same global model with client `k`'s head fitted locally on client `k`'s *training* shard | FedAvg (derived, no extra federated run) |
| **personalized** | the aggregated backbone plus client `k`'s own FedRep head | FedRep |

The middle row is the control that keeps the first comparison honest. Without it,
a FedRep gain could be FedRep's alternating training or it could be the mere
existence of *any* locally fitted head, and those are different claims. It costs
one head-only fit per client at the end of the FedAvg run — minutes, not hours.

### What is held equal between arms

The total local epoch budget, the client learning rate and momentum, the batch
size, the initial global model, the cohort sequence (same seed, same generator,
same call order), and the per-client batch shuffling — seeded from
`(seed, round, client)`, so client 7 in round 3 sees the same batch order in both
arms. Both arms run the *same* loop in `scripts/personalization_experiments.py`;
they differ in which variables the local optimiser may move and which tensors are
submitted.

### Isolation, asserted rather than asserted-in-prose

`tests/test_personalization.py` extends the shard-leakage discipline of
`tests/test_data.py` and `tests/test_femnist.py` to heads and to per-client test
shards:

- Every stored head was fitted on that client's shard and on nothing else — a
  real (tiny) FedRep run is instrumented at `StagedTrainer.fit` and
  `HeadStore.put`, and each stored head is matched by content hash against its
  own client's training data.
- No test sample reaches any training call — content-level, hashing the arrays
  actually handed to the trainer, not the indices used to build them.
- Each client is scored with its own head on its own shard — verified with heads
  that predict a fixed class, so a mispairing lands on a visibly different number
  instead of a slightly worse one.
- A returned head cannot be used to mutate the store, and a stored head cannot be
  mutated through the caller's array.
- On FEMNIST, per-writer test label profiles correlate with that writer's *own*
  training profile measurably better than with another writer's — the premise of
  the whole exercise, checked rather than assumed.

## 7. What the reporting says, and the prediction made before the runs

**The distribution, not the mean.** A method that lifts the median client by a
point and the worst decile by fifteen, and a method that lifts everyone by two,
report the same mean and are not the same result. Every phase records the full
per-client array in its JSON, and summarises with median, quartiles, p10/p90 and
the mean over the worst and best deciles — a *mean over the tail*, not a
percentile point, because one catastrophic client should move the tail statistic
and a percentile would not notice it. Deltas are **paired client by client**, so
"fraction of clients improved" and "fraction worsened" are statements about
individual clients; a method that helps on average while hurting a minority shows
up there and nowhere else.

The headline figure is an ECDF of per-client accuracy, global vs fine-tuned vs
personalized, drawn by `scripts/plot_personalization.py` (pure standard library —
`matplotlib` is not in `requirements.txt` and adding it would mean re-resolving a
dependency set pinned around TFF's exact `typing-extensions==4.5.*`). On an ECDF
the claim is legible directly: a curve shifted right helps everyone, a curve
whose *lower* tail lifts helps the clients the global model serves worst, and a
curve that crosses the baseline helps some clients by hurting others.

### The prediction, stated now

From §2, before any run: **the Fashion α=0.1 arm should show the larger effect,
and FEMNIST may show very little.** A local head adapts the label prior and the
decision boundary in representation space. Fashion at α=0.1 is almost pure label
shift (per-client label entropy 36 % of pooled), which is what a head can absorb.
FEMNIST's shift is writer style at a nearly-pooled label prior (92 % of pooled),
which is a shift in the *representation's input*, and a frozen backbone plus a
retrained head has no mechanism for it.

If FEMNIST comes back flat, that is a result about *which kind of heterogeneity a
head can absorb*, reported with the heterogeneity level in the table above — not
a failed experiment, and not one to bury. Note that FEMNIST is doubly
disadvantaged here and both halves are measured, not guessed: its shift is in the
input distribution rather than the label prior (§2), and its per-writer test
shards are too small for the label signal that does exist to be read cleanly
(§10). A flat FEMNIST result would be consistent with both, and would not
distinguish them. If Fashion also comes back flat, the
claim that personalization is worth its complexity on these splits is simply not
supported, and this document will say so.

## 8. The two phases

Appended to the overnight chain; both resume by re-running the script. Run the
suite first — the personalization tests that drive a real (tiny) FedRep run are
marked `slow` and are the ones that would catch a harness fault before five hours
of compute does:

```bash
pytest          # the whole suite, including the slow personalization runs

scripts/run_personalization_batch.sh --check     # prerequisites only, launches nothing
nohup scripts/run_personalization_batch.sh >> ../fl-personal-launcher.log 2>&1 &
python scripts/plot_personalization.py --phase docs/_personalization_a.json \
    --out docs/personalization_ecdf_femnist.svg
python scripts/plot_personalization.py --phase docs/_personalization_b.json \
    --out docs/personalization_ecdf_fashion.svg
```

| Phase | Population | Budget | Arms | Output | Est. |
|---|---|---|---|---|---|
| **A** | FEMNIST, 1,000 writers | m=200, R=20, E=10 (FedRep: 2 head + 8 backbone) | FedRep, FedAvg (+ fine-tune control), 3 seeds each | `docs/_personalization_a.json` | ~5 h |
| **B** | Fashion-MNIST, N=100, α=0.1 | m=50, R=20, E=2 (FedRep: 1 head + 1 backbone) | same | `docs/_personalization_b.json` | ~45 min |

> **Machine-time provenance.** The suite above was run in `fl-dev-torch` on this
> host on **2026-08-19, 12:22:04–12:26:51 UTC** (17:52:04–17:56:51 +05:30),
> capped at 2 of 6 cores, *while* `fl-compression-batch` was running (it started
> 12:21:10 UTC and held `../.fl-batch.lock` throughout). Any compression cell
> whose round falls in that window shares CPU with this check and should be read
> with that in mind; no other overlap occurred in the first window. A second
> window, **12:29:22–12:32:41 UTC** (17:59:22–18:02:41 +05:30), same cap, re-ran
> `tests/test_femnist.py` alone after the paired-statistic fix in §10. Those two
> windows are the only times this branch has competed with the compression batch
> for CPU.

The launcher, not a bare `python scripts/personalization_batch.py`, because the
host has six cores and every wall-clock figure in this document is quoted per
round. `scripts/run_personalization_batch.sh` waits for any other federated
training container to exit and then takes `../.fl-batch.lock` — the same lock
`scripts/run_compression_batch.sh` takes on `feat/compression` — so the two
batches queue instead of halving each other's throughput and leaving no trace of
it in either JSON. It is safe to start while something else is still running:
it sits and waits. (The two launchers are near-copies because neither branch is
merged yet; when they land they should collapse into one parameterised script,
and until then the lock path and the image pattern are the two lines that must
stay in step.)

Phase A checkpoints each completed run to `_personalization_a_partial.json`;
five hours is long enough that an interruption should not cost the phase. Neither
phase uses differential privacy — the two compose (the head never enters the
aggregate, so it is outside the sensitivity bound entirely), but running both at
once would produce a result nobody could attribute.

The population is loaded **once per phase at seed 42**; the run seed varies model
initialisation, cohort sampling and batch order but not which writers (or which
Dirichlet draw) form the population. That is what makes the per-client arrays
comparable across seeds, so a client's accuracy is averaged over seeds *before*
the distribution is taken.

## 9. Results

**Not yet run.** When phases A and B complete, this section will carry: the three
per-client distributions per phase (median, quartiles, worst and best decile), the
paired deltas with the fraction of clients improved and worsened, the split
between the fine-tuning control and the FedRep effect, and the two ECDFs. The raw
per-client arrays will be in `docs/_personalization_{a,b}.json`.

## 10. What this design will not answer

- **The head-epoch split is untuned.** `head_epochs=2` of 10 on FEMNIST and 1 of
  2 on Fashion are a-priori choices — the head is 3.45 % of the parameters and
  warm-starts from the previous round, so it gets a small slice — recorded as
  constants, in the same spirit as the FedOpt phase-C server learning rates
  transferred rather than re-tuned. A split sweep is the obvious next
  measurement and is not in this budget. Per-round head-fit training accuracy is
  recorded, so the runs will at least say whether the head converged.
- **Fashion runs at E=2, not the E=1 of the recorded Fashion results.** FedRep's
  local update is two stages and both must run; one epoch cannot be split in two.
  Both arms of phase B get E=2, matched to each other — and phase B's FedAvg arm
  is therefore *not* comparable to the recorded E=1 Fashion figures.
- **Fashion's per-client test data is a construction, not a measurement.** §6
  says how it is built and why; a personalization result on Fashion-MNIST is a
  result about that construction.
- **Cold clients are in the population.** At R=20, m=200 over 1,000 writers about
  11.5 writers are expected never to be sampled; their "personalized" accuracy is
  really the untrained initial head. They are reported (`head_updates == 0`,
  `clients_never_sampled`) rather than dropped or silently averaged in. On
  Fashion every client is sampled with probability 1 − 9.5×10⁻⁷, so the issue
  does not arise there.
- **Some clients have very little held-out data, and on FEMNIST that is the
  binding constraint.** FEMNIST's median writer has 18 test samples; six of the
  1,000 have fewer than ten. Fashion at α=0.1 leaves one client with none
  (excluded and counted) and eight with fewer than ten. Every summary is reported
  twice, once over all scorable clients and once restricted to clients with ≥ 10
  held-out samples, so a tail claim can be checked against both.

  How much this costs was measured while testing the loader, and it is worth
  stating plainly. A writer's held-out label profile does correlate with its own
  training profile better than with another writer's — 77 % of writers, paired
  median gap +0.09 — so the premise holds. But the gap is **+0.22 for writers
  with ≥ 30 held-out samples and +0.06 for writers with < 20**, and the second
  group is most of the population. At 18 samples over 62 classes the per-writer
  test profile is largely sampling noise. Per-client *accuracy* on 18 samples is
  still unbiased, but it is quantised into eighteenths and carries binomial noise
  of roughly ±0.11 at p = 0.5; averaging over three seeds shrinks the seed
  component and none of the sampling component, because every seed scores the
  same 18 samples. **The FEMNIST per-client distribution is therefore wide for
  reasons that have nothing to do with personalization**, and the worst-decile
  figure in particular will be dominated by small-shard writers. Read the
  restricted (≥ 10, and preferably the ≥ 30) summaries alongside it. Fashion's
  median client has 69 test samples and does not have this problem.
- **No DP arm, and no comparison against Ditto or pFedMe.** See the Roadmap.

## 11. Related methods, not implemented

On the README Roadmap, deliberately unbuilt:

- **Ditto** (Li et al., ICML 2021) — each client trains a personal model
  regularised toward the global one, so personalization strength is a single
  knob λ rather than an architectural cut, and every parameter is personalizable
  rather than only the head.
- **pFedMe** (T. Dinh et al., NeurIPS 2020) — a Moreau-envelope formulation in
  which the client optimises a personalized model against a proximal term and the
  server aggregates the *outer* models, decoupling personalization from the
  aggregation step.

Both would be measured against the same per-client distributions defined in §6;
neither is in the code.
