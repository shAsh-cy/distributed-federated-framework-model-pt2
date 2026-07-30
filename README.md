# Federated Learning on Fashion-MNIST — TensorFlow Federated + gRPC + client-level DP

A working federated learning system: a gRPC control plane that coordinates real
FedAvg rounds over a non-IID split of Fashion-MNIST, with client-level
differential privacy applied through TensorFlow Federated and epsilon computed by
TensorFlow Privacy's accountant.

Everything reported below was produced by running the code in this repository.
The numbers are in [results/](results/) and reproducible with one command.

---

## Install — read this first

`tensorflow-federated` cannot be installed from PyPI alone, at any version.

It pins `jaxlib==0.4.14` exactly, and **that release has been removed from PyPI**
(the oldest jaxlib now on PyPI is 0.4.18). The failure is worse than an error:
an unpinned `pip install tensorflow-federated` does not fail, it back-tracks past
every real release and installs `tensorflow-federated==0.1.0` — a 2019 placeholder
containing none of the federated API. You end up with a package of the right name
and no working `tff.aggregators`.

`requirements.txt` fixes this with a `--find-links` line pointing at Google's
historical jax index, which still serves jaxlib 0.4.14. **That line is
load-bearing; do not remove it.**

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # includes requirements.txt
```

Platform constraints, both hard:

| Constraint | Value | Why |
|---|---|---|
| Python | `>=3.9,<3.12` — use **3.10** | TFF's own `requires_python` |
| OS | **Linux only** | TFF ships one wheel per release: `manylinux_2_31_x86_64`. There is no Windows or macOS wheel at any version. |

On Windows or macOS, use the Docker path below or WSL2. This is not a
preference — the wheel does not exist.

The exact resolved version matrix, verified on Python 3.10 / linux-amd64:

```
tensorflow==2.14.1          tensorflow-federated==0.87.0    tensorflow-privacy==0.9.0
tensorflow-probability==0.22.1   tensorflow-estimator==2.14.0    keras==2.14.0
jax==0.4.14                 jaxlib==0.4.14                  numpy==1.25.2
protobuf==4.25.9            grpcio==1.74.0                  typing-extensions==4.5.0
```

That last pin is why configuration uses dataclasses rather than pydantic: TFF
requires `typing-extensions==4.5.*`, and pydantic v2 requires `>=4.6.1`. They
cannot coexist.

---

## Results

Three configurations, identical in every respect except the privacy block —
same seed, same model, same partition, same 20 rounds. Produced by
`scripts/run_all_experiments.sh`; raw per-round metrics in [results/](results/).

Fashion-MNIST, 10 clients, Dirichlet non-IID (α = 0.5), C = 0.5 (5 clients
sampled per round), 20 rounds, seed 42. Untrained baseline: **12.25%**.

| configuration | noise `z` | ε at δ=1e-5 | **final accuracy** | dropped client-rounds | wall clock |
|---|---|---|---|---|---|
| no DP | — | ∞ (no guarantee) | **86.93%** | 0 / 100 | 102 s |
| moderate noise | 2.0 | **6.228** | **10.00%** | 85 / 100 | 112 s |
| high noise | 6.0 | **1.639** | **10.00%** | 90 / 100 | 118 s |

Every run transferred exactly 90.0 MB in each direction — 20 rounds × 5 clients
× 900,254 bytes, matching the documented model size.

Without DP the model reaches 86.93% (peak 87.69%), climbing from 69.9% after
round 1. **Both DP configurations collapse to 10.00% — the accuracy of guessing.**
That is not a bug, and the next section explains exactly why.

### Reading the DP results honestly

The DP runs destroy the model. That is the correct outcome for this
configuration, and the reason is arithmetic rather than a bug.

Client-level DP adds Gaussian noise with standard deviation `z·S` to the *sum* of
clipped updates, then divides by the cohort size `m`. The signal — the mean
clipped update — has L2 norm at most `S`. The noise has expected L2 norm
`z·S·√d / m`, where `d` is the parameter count. So the noise-to-signal ratio is

```
    z · √d / m          independent of the clipping norm S
```

With `d = 225,034` (√d ≈ 474) and `m = 5` clients per round:

| configuration | z | noise / signal |
|---|---|---|
| moderate | 2.0 | **190×** |
| high | 6.0 | **569×** |

Measured directly (`||noisy mean delta||` for a unit-signal input): 569.9 against
a predicted 569.3 for the moderate setting — the mechanism is doing exactly what
the theory says. The update is ~190 parts noise to 1 part signal, so the model
diverges within a round or two and accuracy collapses to the 10% of random
guessing.

**The real conclusion:** client-level DP needs `m ≳ z·√d`. For this model and
`z = 2`, that is roughly **950 clients per round** — not 5. This is precisely why
production DP federated learning uses cohorts of thousands. With 10 clients there
is no noise multiplier that buys both a meaningful epsilon and a usable model:
getting the ratio near 1 requires `z ≈ 0.01`, at which epsilon is astronomically
large and the guarantee is vacuous.

Reporting this is more useful than tuning until a chart looks good. Epsilon here
is computed from the mechanism, never chosen, so the guarantee is real — and at
this scale the price of that guarantee is the entire model.

---

## Architecture

```
fl/
  config.py         Typed, validated configuration. One frozen object, loaded from YAML.
  models.py         Keras CNN (225,034 parameters, documented and asserted).
  data.py           Fashion-MNIST loading; IID and Dirichlet non-IID partitioning.
  aggregation.py    FedAvg arithmetic; TFF-backed DP aggregation; epsilon accounting.
  serialization.py  Weight list <-> protobuf, with validation on every decode.
  server.py         gRPC coordinator: sampling, round barrier, evaluation, metrics.
  client.py         gRPC participant: registers, trains its shard, reports.
  proto/
    fl_comm.proto   Versioned wire format. Generated stubs are NOT committed.
configs/            default.yaml, dp_moderate.yaml, dp_high.yaml, docker.yaml
scripts/            run_experiment.py, run_all_experiments.sh, summarise_results.py
tests/              See "Tests" below.
results/            Committed metrics from real runs.
```

### What happens in a round

1. The server samples `ceil(C · N)` of the registered clients. Partial
   participation is what makes this federated rather than merely distributed.
2. It publishes the global weights together with their `model_version`.
3. A barrier opens with a wall-clock deadline.
4. Clients train locally and submit weights, their **sample count**, and the
   **model version they trained from**.
5. At the barrier the server aggregates what arrived, drops and logs the rest,
   evaluates on the held-out test set, and increments `model_version`.

The deadline is enforced, not advisory: a server that blocks on its slowest
participant has no availability story. Drops are logged individually.

Two version numbers travel on the wire and are deliberately distinct:
`protocol_version` (schema, checked once at registration) and `model_version`
(which global model an update was trained from, checked on every update). An
update trained from model *N* is only valid input to the aggregation producing
*N+1*; anything else is rejected with `REJECTED_STALE_MODEL`.

### Data handling

- **The test set never leaves the server.** Clients receive training indices
  only, so a client physically cannot evaluate on held-out data.
- **Non-IID by default.** An IID split makes every client's gradient an unbiased
  estimate of the same global gradient, so FedAvg degenerates into mildly noisy
  centralised SGD and the client-drift problem disappears. `dirichlet_alpha`
  controls the skew; at the default 0.5, shard sizes range 1,816–11,815.
- Both splits are tested for three invariants: shards are pairwise disjoint,
  their union is exactly the 60,000-sample training set, and no held-out test
  image appears in any client shard (checked by hashing pixels, not by comparing
  indices).

---

## Running it

### Locally (Linux / WSL2, Python 3.10)

```bash
python -m fl.server --config configs/default.yaml --metrics-out results/run.json
python -m fl.client --config configs/default.yaml --server 127.0.0.1:8080   # xN
```

Or both at once:

```bash
./run_local.sh configs/default.yaml 10
```

Clients take no `--cid`. A client that registers without an id is assigned the
next free shard by the server, so starting N clients claims N distinct shards
with no per-client configuration. That is what makes `--scale` work.

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build --scale client=5
```

Verified working: 5 rounds to **77.13%** test accuracy, five clients and the
server all exiting 0.

`client` is one scalable service rather than hardcoded `client0`/`client1`.
Replicas need no per-replica configuration — a client that registers without an
id is assigned the next free shard, so five replicas claim five distinct shards
on their own. `configs/docker.yaml` declares `num_clients: 5` to match; scaling
beyond it is refused by the server (`all 5 shards already claimed`) rather than
silently double-counting a shard.

### Reproducing the recorded results

```bash
./scripts/run_all_experiments.sh     # writes results/*.json, prints the table
```

Seeded throughout: Python, NumPy and TensorFlow RNGs, the partition, the initial
model, and the server's client sampling all derive from `config.seed`. Non-DP
accuracy is deterministic up to floating-point summation order. DP runs are
reproducible in configuration but not bit-identical, because the Gaussian noise
is drawn inside TFF's aggregation process and thread scheduling determines
arrival order.

---

## Configuration

One typed, frozen object; nothing else reads environment variables or hard-codes
a hyperparameter. Validation is strict on purpose:

- **Unknown keys are errors.** A silently ignored `noise_multipler` typo would
  otherwise produce a run that claims privacy it does not have.
- **Cross-field checks.** Sampling fewer clients than the quorum requires is
  rejected at load time rather than failing every round at runtime.
- **`privacy.enabled` with `noise_multiplier: 0` is refused** — that combination
  claims a guarantee while providing none.
- **Epsilon is not a config field.** It is computed from the noise multiplier,
  the client sampling rate and the round count. See
  `fl.aggregation.compute_epsilon`.

---

## Differential privacy

The granularity is **client-level** (user-level): the protected unit is one
participant's entire local dataset, so the guarantee concerns whether a given
client took part at all.

This is **not example-level DP**. Example-level DP — what DP-SGD inside a single
trainer provides — protects one training row and says nothing about
participation. Conflating the two overstates the guarantee. Here `l2_clip_norm`
bounds one client's whole round update, which is what makes the client-level
claim true.

Two implementation details that are easy to get wrong, both deliberate:

1. **DP is applied to the delta `w_k − w_global`, never to raw weights.**
   Clipping raw weights to a norm of ~3 would annihilate a trained model.
   Clipping the round's update bounds exactly the quantity a client contributes,
   which is what the sensitivity argument requires.

2. **The DP path is *unweighted*, unlike plain FedAvg.** A DP guarantee needs a
   bound on how much one client can move the released value. Clipping to `S`
   gives sensitivity `S` only if every client is then weighted equally; under
   sample-count weighting a client's influence is `n_k/Σn · S`, which depends on
   its own private shard size, so the sensitivity is no longer a constant the
   accountant can use. TFF encodes this in its type system — 
   `DifferentiallyPrivateFactory` returns an `UnweightedAggregationFactory`.
   **Enabling DP therefore costs sample-count weighting as well as accuracy**,
   which on an unequal non-IID split are two distinct penalties.

Epsilon is computed with the RDP accountant from `dp_accounting` (the accountant
library maintained by the TensorFlow Privacy team), composing `rounds`
Poisson-subsampled Gaussian mechanisms at `delta = 1e-5`. Zero noise returns
`inf`, not a large finite number.

---

## Tests

```bash
pytest --cov=fl --cov-report=term-missing
```

**208 tests, 89% statement coverage**, run on every push by GitHub Actions
alongside `ruff check` and `ruff format --check`.

The suite covers four areas, one file each:

| Area | File | What it checks |
|---|---|---|
| RPC communication | `tests/test_rpc.py` | server starts and serves; client registers; the full 225k-parameter model round-trips through gRPC bit-identically (SHA-256 fingerprint); oversized payloads raise `RESOURCE_EXHAUSTED` and leave the server serving; malformed payloads are rejected cleanly |
| Aggregation logic | `tests/test_aggregation.py` | the hand-computed weighted-average test below; single client; zero reporting clients; mismatched tensor counts and shapes; NaN/Inf |
| Client synchronisation | `tests/test_sync.py` | stale and future `model_version` rejected and *not applied*; barrier holds until the cohort completes or the deadline expires; 12 clients registering simultaneously get 12 distinct shards |
| Fault scenarios | `tests/test_faults.py` | client disconnects mid-round; client times out; client returns NaN; server restarted with clients connected |

Supporting files: `tests/test_config.py`, `tests/test_models.py`,
`tests/test_data.py`, `tests/test_server_rounds.py`, `tests/helpers.py`.

### The aggregation test that matters

FedAvg is weighted by sample count, and the test is built so that a regression to
an unweighted mean cannot pass. Three clients with shard sizes 10, 100 and 1000
holding values 1, 2 and 3:

```
weighted   = (10·1 + 100·2 + 1000·3) / 1110 = 3210/1110 = 2.8919
unweighted = (1 + 2 + 3) / 3                             = 2.0000
```

The 0.89 gap is far outside float tolerance. This matters because with
equal-sized shards the two are *identical* — a broken implementation would pass
unnoticed on a uniform split. The default Dirichlet partition produces shards of
1,816–11,815 samples, so the weighting is load-bearing in practice too.

---

## What this project does and does not claim

**Supported by the code:**

- Real FedAvg, weighted by client sample count, implemented here and tested
  against a hand-computed example.
- A real gRPC control plane carrying real model weights, with sample counts and
  model versions on the wire, and stale updates rejected.
- Genuine non-IID partitioning with a configurable Dirichlet concentration, and
  the skew asserted rather than assumed.
- Server-side evaluation on a test set no client can access.
- Client-level DP applied through TFF's `DifferentiallyPrivateFactory`, with
  epsilon computed by a real accountant.
- Enforced round deadlines, straggler drops, quorum handling, and recovery from
  client disconnects, timeouts, NaN updates and server restarts.

**Not claimed:**

- **Usable accuracy under DP at this scale.** Both DP configurations collapse to
  ~10%. The arithmetic above explains why, and roughly 950 clients per round
  would be needed to change it. The mechanism is correct; the scale is not.
- **Secure aggregation.** The server sees every client's plaintext update. DP
  bounds what the *released model* reveals, not what the server observes.
- **Byzantine robustness.** NaN and malformed updates are rejected, but a
  well-formed malicious update within the clipping norm is aggregated normally.
- **Cross-device scale.** This is a cross-silo design: clients are long-lived,
  registration is in-memory, and no state survives a server restart.
- **Windows or macOS support.** TFF publishes Linux-only wheels.
