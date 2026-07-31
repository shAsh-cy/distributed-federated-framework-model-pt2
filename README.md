# Federated Learning on Fashion-MNIST — TFF + gRPC + client-level DP

Trains a shared image classifier across several containerised clients that never
send their training data anywhere. Coordination runs over a real gRPC protocol;
differential privacy is applied at aggregation through TensorFlow Federated, and
the privacy budget is computed rather than asserted.

Every number below was measured by running this code. The raw per-round metrics
are committed in [`results/`](results/).

---

## The problem this solves

Ten parties each hold a private slice of a labelled image dataset. Individually
none has enough data to train a good classifier; none is willing to hand its raw
data to a central server.

Federated averaging resolves that: the server sends the current model out, each
party trains on its own data locally, and only the resulting **weights** come
back. The server averages them, weighted by how many samples each party trained
on, and repeats.

Two properties make this concrete rather than a slogan, and both are enforced in
code here:

- **Raw data never leaves a client.** Clients receive training indices only; the
  server holds the test set and never ships it out.
- **The parties' data is not identically distributed.** The default split is
  Dirichlet label-skew (α = 0.5), producing shards of 1,816–11,815 samples with
  very different label mixes. An IID split would make every client's gradient an
  unbiased estimate of the same global gradient and reduce federated averaging to
  slightly noisy centralised SGD — the hard part would vanish.

Averaging weights still leaks information about the data that produced them, so
the aggregation step optionally applies **client-level differential privacy**:
each client's contribution is clipped and Gaussian noise is added before the
average is released.

---

## Architecture

```mermaid
flowchart TB
    subgraph clients["Clients — one container each, one private shard each"]
        direction LR
        D1[("Shard 1<br/>raw images")]
        C1["Client 1<br/>local Keras training"]
        DN[("Shard N<br/>raw images")]
        CN["Client N<br/>local Keras training"]
        D1 --> C1
        DN --> CN
    end

    subgraph server["Server — one container"]
        direction TB
        RX["gRPC servicer<br/>register · serve model · accept update<br/>deadline + staleness checks"]
        AGG["FedAvg aggregator<br/>weighted by client sample count<br/>fl/aggregation.py"]
        DP["Client-level DP aggregation<br/>clip to S, add N(0, (z·S)²), mean<br/>TensorFlow Federated"]
        GM["Global model + version"]
        TEST[("Held-out test set")]
        EV["Evaluate every round"]
        RX --> AGG
        AGG --> DP
        DP --> GM
        GM --> EV
        TEST --> EV
    end

    C1 -->|"weights + sample count + model version"| RX
    CN -->|"weights + sample count + model version"| RX
    GM -->|"global weights + version"| C1
    GM -->|"global weights + version"| CN

    D1 --x|"raw images: never transmitted"| server
    DN --x|"raw images: never transmitted"| server
    TEST --x|"test set: never transmitted"| clients
```

Links ending in **✕** are the paths that deliberately do not exist: no client
data reaches the server, and no test data reaches a client. Only model weights,
a sample count and a model version cross the wire.

**One round:** the server samples a fraction `C` of registered clients → publishes
the global weights and their version → opens a barrier with a wall-clock deadline
→ aggregates whatever arrived, dropping and logging the rest → evaluates on the
held-out test set → increments the model version.

The deadline is enforced, not advisory; a server that blocks on its slowest
participant has no availability story. An update trained from model *N* is only
valid input to the aggregation producing *N+1*, so stale submissions are rejected
rather than folded in.

---

## Why both TensorFlow Federated and a custom gRPC layer

They do different jobs, and neither alone covers this project.

**TensorFlow Federated provides the differentially private aggregation.** The DP
path wraps `tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed` into a
real `tff.templates.AggregationProcess` whose state is carried across rounds — TFF
does the clipping and noise, not a hand-rolled approximation of it. Using TFF's
own type system also surfaces a constraint that is easy to get wrong: the factory
returns an `UnweightedAggregationFactory`, because a sensitivity bound only holds
if every client is weighted equally.

**A custom gRPC layer provides the distribution.** TFF's simulation runtime
executes every client inside a single process. That is ideal for validating a
federated *algorithm*, and useless for demonstrating a federated *system*: in one
process there is no registration handshake, no wire format, no round deadline, no
straggler to drop, no stale update to reject, and no bytes-on-the-wire figure to
measure. This repository needed all of those, so the transport is a versioned
`.proto` and a real gRPC server that separate containers connect to over the
network.

**Precisely who does what**, since this is the claim a reviewer should check:

| Component | Provided by |
|---|---|
| FedAvg weighted average | this repo — `fl/aggregation.py` |
| DP clipping, Gaussian noise, aggregation state | TensorFlow Federated |
| ε accounting (RDP, Poisson-subsampled Gaussian) | `dp_accounting`, from TensorFlow Privacy |
| Registration, round barrier, deadlines, staleness, transport | this repo — `fl/proto/`, `fl/server.py`, `fl/client.py` |
| Model, data loading, non-IID partitioning | this repo — `fl/models.py`, `fl/data.py` |

---

## Results

Fashion-MNIST · 10 clients · Dirichlet non-IID (α = 0.5) · C = 0.5 (5 clients
sampled per round) · 20 rounds · seed 42 · 225,034-parameter CNN. The three
configurations differ **only** in their privacy block.

| Configuration | Noise `z` | Clip `S` | δ | **ε (computed)** | **Final accuracy** (round 20) |
|---|---|---|---|---|---|
| No DP | — | — | — | — (no guarantee) | **86.93 %** |
| Moderate noise | 2.0 | 3.0 | 1 × 10⁻⁵ | **6.228** | **10.00 %** |
| High noise | 6.0 | 3.0 | 1 × 10⁻⁵ | **1.639** | **10.00 %** |

Accuracy is reported at round 20 throughout this README. The DP rows use a
clipping norm now known to be badly chosen — see below; they are the numbers this
repo's committed configs actually produce, not the best available at that ε.

Untrained baseline: **12.25 %**. Without DP the model peaks at 87.69 % and ends
at 86.93 %. Every run moved 90,025,400 bytes server→client — exactly 20 rounds ×
5 clients × 900,254 bytes, the serialised size of the model — and 90,029,290
bytes back, the small excess being the sample count and version fields each
update carries. Reproduce with `./scripts/run_all_experiments.sh`.

ε is computed from the noise multiplier, the client sampling rate (q = 0.5) and
the round count — it is never chosen. See `fl.aggregation.compute_epsilon`.

### Why the shipped DP configuration collapses to chance

Two independent causes, both measured.

**The cohort is too small.** DP adds noise of expected L2 norm `z·S·√d / m`. With
d = 225,034 parameters (√d ≈ 474) and m = 5 clients per round, that noise has norm
569 against a typical update of ≈ 1.0. Measured: 569.7 observed versus 569.3
predicted. The first DP round replaces the model with noise.

**The clipping norm is 3× too high.** `S = 3.0` sits *above* the median update norm
(0.984), so it almost never binds — it buys noise (`stddev = S·z`) for no
sensitivity reduction. Lowering it costs nothing in privacy: ε is computed from
`z = σ/S`, which is already normalised by `S`, so ε is **bitwise identical** at
every clipping norm.

Both fixes together, at the **same ε = 6.228**, measured in-process over a
15-cell sweep (`S ∈ {3.0, 1.1, 0.5}` × `m ∈ {5, 20, 50, 100, 200}`) — final
accuracy:

| Final accuracy | m = 5 | m = 20 | m = 50 | m = 100 | m = 200 |
|---|---|---|---|---|---|
| `S` = 3.0 (shipped) | 0.00 % | 10.00 % | 10.00 % | 39.62 % | 46.51 % |
| `S` = 1.1 | 0.00 % | 12.38 % | 65.28 % | 64.34 % | 56.94 % |
| **`S` = 0.5** | 11.23 % | 52.19 % | **73.48 %** | 68.15 % | 57.45 % |
| *No-DP ceiling, same cohort* | *86.93 %* | *83.04 %* | *77.16 %* | *71.54 %* | *57.42 %* |
| **DP / ceiling at `S` = 0.5** | 0.13 | 0.63 | **0.95** | **0.95** | **1.00** |

> **Read the confound before reading the columns.** Holding the sampling rate at
> q = 0.5 requires a population of `N = 2m`, so a larger cohort *necessarily* means
> a smaller shard — 6,000 examples per client at m = 5 down to 150 at m = 200.
> Accuracy moves across these columns for two unrelated reasons: less noise (helps)
> and less data per client (hurts). The **no-DP ceiling row is the control** that
> separates them, and it falls 29 points on its own with no DP involved.

**The last row is the only one that measures the price of privacy** rather than the
price of this experimental design. At m ≥ 50 with a correctly chosen clip,
client-level DP at ε = 6.228 costs **3–5 % of achievable accuracy**; at m = 200 it
costs nothing measurable, and the 57 % there is entirely the thin-shard penalty.

> **Every DP number above is a single run, and DP runs here are not reproducible.**
> TF Privacy draws the Gaussian noise from an unseeded initialiser executed inside
> TFF's executor, which never sees `tf.random.set_seed`; neither library exposes a
> seed on that path. Measured run-to-run spread is **4.7–29.5 accuracy points**
> depending on cohort size. The `S` = 3.0 → 0.5 improvement and the cohort trend are
> several times larger than that and hold; the `S` = 1.1 vs `S` = 0.5 ordering at
> m ≥ 50 is **within noise** and is not a ranking. The no-DP row is exactly
> reproducible. Details and a per-claim breakdown: §10 of the diagnosis.

The best configuration — m = 50, `S` = 0.5, **73.48 %** — was **still improving when
the 20-round budget ran out**: it peaked at round 20, and mean accuracy over rounds
11–20 (70.69 %) was well above rounds 1–10 (48.21 %). It is a lower bound, not a
converged result. The three shipped configurations above are unchanged and were not
re-run; the sweep is a simulation of the same aggregation code, validated against
the recorded non-private run (86.93 %, reproduced exactly).

Full diagnosis — including the ablation that exonerates clipping, the ε-invariance
gate, the fitted signal-decay exponent and a retraction of an earlier claim in that
document — is in **[docs/dp_diagnosis.md](docs/dp_diagnosis.md)**.

---

## Quickstart

### Docker — works from a clean clone on any host with Docker

```bash
git clone <repository-url>
cd federated-learning-starter
docker compose -f docker/docker-compose.yml up --build --scale client=5
```

That builds one image, starts a server and five client containers, and runs 5
rounds to roughly 77 % accuracy (`configs/docker.yaml`). Verified end to end from
a fresh clone; the server writes `/app/results/docker_run.json` and every
container exits 0.

Clients take no `--cid`. A client registering without an id is assigned the next
free shard, so N replicas claim N distinct shards with no per-replica config —
that is what makes `--scale` work. `configs/docker.yaml` declares
`num_clients: 5`; starting more is refused (`all 5 shards already claimed`)
rather than silently double-counting a shard.

### Local — Linux or WSL2, Python 3.10

> **The `--find-links` line in `requirements.txt` is load-bearing.**
> `tensorflow-federated` pins `jaxlib==0.4.14` exactly, and that release has been
> removed from PyPI. Worse than an error: an unpinned
> `pip install tensorflow-federated` does not fail, it back-tracks to the 2019
> placeholder `0.1.0`, which contains none of the federated API. The
> `--find-links` line restores jaxlib from Google's historical index.
>
> **TFF is Linux-only** (`manylinux_2_31_x86_64`, no Windows or macOS wheel at any
> version) and requires Python `>=3.9,<3.12`. Use Docker or WSL2 elsewhere.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # includes requirements.txt

./run_local.sh configs/default.yaml 10       # server + 10 clients, 20 rounds
```

Or run the pieces separately:

```bash
python -m fl.server --config configs/default.yaml --metrics-out results/run.json
python -m fl.client --config configs/default.yaml --server 127.0.0.1:8080   # ×N
```

Re-run the three recorded configurations, then print the comparison table:

```bash
./scripts/run_all_experiments.sh
```

The no-DP run reproduces its recorded 86.93 % exactly. **The two DP runs will not
reproduce their recorded numbers**, for the reason given under *Results*: the
Gaussian noise is drawn inside TFF's executor from an unseeded initialiser, and
neither TFF nor TensorFlow Privacy exposes a seed on that path. Both stay at
chance at this cohort size, which is the finding; the exact figure is a draw.

gRPC stubs are generated on import from `fl/proto/fl_comm.proto`, so there is no
separate build step and the `.proto` stays the single source of truth.

### Tests

```bash
pytest --cov=fl --cov-report=term-missing
```

**208 tests, 89 % statement coverage**, run on every push by GitHub Actions
alongside `ruff check` and `ruff format --check`. Four areas, one file each:
`tests/test_rpc.py` (transport, bit-identical weight round-trip, oversized and
malformed payloads), `tests/test_aggregation.py` (FedAvg arithmetic and
degenerate cohorts), `tests/test_sync.py` (staleness, the round barrier,
concurrent registration), `tests/test_faults.py` (disconnects, timeouts, NaN
updates, server restart).

The central aggregation test uses shard sizes 10, 100 and 1000 holding values 1,
2 and 3, so the weighted average is 3210/1110 = 2.8919 while an unweighted mean
would be exactly 2.0. The 0.89 gap is far outside float tolerance — the test
cannot pass if anyone replaces the weighting with a plain mean, which on
equal-sized shards would otherwise go unnoticed.

### Configuration

One frozen, validated object per run (`fl/config.py`); nothing else reads
environment variables or hard-codes a hyperparameter. Unknown keys are errors, so
a typo'd `noise_multipler` fails loudly instead of silently producing a run that
claims privacy it does not have. `privacy.enabled` with `noise_multiplier: 0` is
rejected outright. ε is not a config field.

---

## Limitations

Stated plainly, because each of these is a real boundary of what was built.

- **Clients are simulated containers on one host, not genuinely separate
  devices.** `docker compose --scale client=5` starts five processes on a single
  machine communicating over a local Docker network. The gRPC transport, the
  registration handshake, the round deadlines and the serialisation costs are all
  real; the *physical* distribution is not. Nothing here has been run across
  actual remote hosts, unreliable links, or heterogeneous hardware.

- **Fashion-MNIST is a benchmark dataset.** It is small, clean, balanced,
  centrally available and already labelled — none of which is true of the private,
  messy, unbalanced data that motivates federated learning in practice. The
  non-IID split is *synthetic*: a Dirichlet partition of a public dataset, not
  naturally occurring per-party heterogeneity. Results here say nothing about how
  the method performs on real federated data.

- **Secure aggregation is not implemented, so the server observes every
  individual client update.** Updates arrive as plaintext weights and the server
  reads each one before averaging. Differential privacy bounds what the *released
  global model* reveals; it does nothing to hide an individual contribution from
  the server itself. A server that wanted to inspect or invert a single client's
  update could do so.

- **There is no adversarial or Byzantine client handling.** Malformed payloads,
  NaN/Inf weights, wrong shapes, stale versions and unregistered senders are all
  rejected — but those are integrity checks, not a threat model. A well-formed
  malicious update within the clipping norm is aggregated like any other. There is
  no update-poisoning detection, no robust aggregation rule (no median, trimmed
  mean or Krum), no client reputation, and no authentication: any process that can
  reach the port can register and contribute.

Also true, and smaller: channels are `insecure_channel` with no TLS; server state
is in memory and does not survive a restart (clients re-register and reclaim their
shards, but the model version resets to 0); only `fashion_mnist` and `small_cnn`
are accepted by config validation.

---

## Roadmap — planned, not built

Nothing in this section is implemented. Each item addresses a limitation stated
above and contradicts no claim made above it.

- **Secure aggregation**, so the server learns only the sum and never an
  individual update — the gap named in Limitations.
- **Robust aggregation rules** (coordinate-wise median, trimmed mean, Krum) and
  update-poisoning detection, to give the system an actual threat model.
- **Client authentication and TLS**, replacing `insecure_channel` and the
  currently open registration endpoint.
- **A re-tuned default clipping norm.** The shipped `l2_clip_norm: 3.0` sits above
  the median update norm and buys noise for nothing; `0.5` reaches 73.48 % at
  m = 50 against 10.00 %, at identical ε. This needs no new code — only a config
  default change and a re-run of the recorded results — and is the single
  highest-value change available. (A configuration item, not a DP item; the
  sweep that establishes it is a simulation, and the committed configs and
  `results/*.json` still reflect the un-tuned value.)
- **Larger cohorts**, which removes the remaining DP penalty at this model size. At
  fixed ε = 6.228 with `S` = 0.5, DP reaches 95 % of its matched non-private
  ceiling at 50–100 clients per round and 100 % at 200. Note the ceiling itself
  falls as the population grows, because this fixed 60,000-example dataset is split
  `N = 2m` ways — so raising the *absolute* number needs more data, not just more
  clients. (A scale item, not a DP item — DP itself is built; see *Results* and
  [docs/dp_diagnosis.md](docs/dp_diagnosis.md).)
- **Adaptive clipping** (quantile-based, as TFF's adaptive factory supports) in
  place of any fixed clipping norm. The measured optimum moves with cohort size —
  the median update norm falls from 0.984 at m = 5 to 0.289 at m = 200 — so no
  single constant is right across configurations, and the sweep above does not
  bracket the optimum at large cohorts.
- **Genuine multi-host deployment**, replacing single-host containers, to
  exercise real network latency, partitions and heterogeneous clients.
- **Persistent server state**, so a restart resumes at the current model version
  instead of resetting to 0.
- **More datasets and architectures**, beyond the single `fashion_mnist` /
  `small_cnn` pair the config currently accepts.

> **Differential privacy is not on this roadmap because it is implemented.**
> Client-level DP via TFF, with ε computed by TensorFlow Privacy's accountant, is
> described under *Results* above and measured at ε = 6.228 and ε = 1.639
> (δ = 1 × 10⁻⁵). Listing it as future work would contradict those results.
