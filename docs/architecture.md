# Architecture: the decisions and the alternatives they beat

Every entry names what was chosen, what was rejected, and why — with the
evidence in this repo rather than by appeal to taste. Written at the v0.2
freeze.

## gRPC inside, HTTP + WebSocket outside

**Chosen:** a versioned protobuf/gRPC protocol for training clients; a
FastAPI HTTP + WebSocket surface (`coordinator/`) for browsers and tooling.
Neither leaks into the other: training clients never speak HTTP, the API
never carries model weights.

**Rejected:** one protocol for both. HTTP-for-everything means base64ing
~900 KB float32 payloads per client per round and hand-rolling deadlines and
streaming semantics gRPC already defines; gRPC-for-everything puts a
protobuf toolchain and a shaky grpc-web proxy between a browser and a
dashboard whose actual needs are "observe, start, stop, replay". The two
consumers want different contracts: clients move binary tensors under
deadline and staleness rules; observers want replayable, self-describing
JSON events (`run_started` carries per-client label histograms precisely so
the browser never computes on raw data). The seam is also a security
boundary: the observability surface can be exposed without exposing the
weight path.

## Hand-written FedAvg, TFF for the DP aggregation only

**Chosen:** FedAvg is ~30 lines of numpy in `fl/aggregation.py`
(sample-count weighting stated and tested); TFF is used exactly where it
carries unique value — `DifferentiallyPrivateFactory` wraps clipping, noise
and cross-round state in a real `AggregationProcess`, and TF Privacy's
accountant computes ε.

**Rejected:** TFF's built-in FedAvg (`build_weighted_fed_avg`). TFF's
training loop assumes its simulation runtime owns the clients — every
client a closure in one process. This repo's point is the opposite: real
registration, a real wire format, real deadlines, real stragglers, clients
in separate containers that can speak either framework. Driving that
through TFF's executor would have meant simulating the very things the repo
exists to make real. The cost of the choice is owning FedAvg's correctness;
the weighted-vs-unweighted signature test (README, Tests) and the
DP-equal-weighting rule exist because of it. The cost of the alternative
would have been the system reducing to a notebook.

## TFF is stalled upstream, and the dependency is three call sites wide

**The upstream state, precisely.** TFF's last PyPI release is 0.87.0 (17
Sep 2024) and its last GitHub *release* is 0.88.0 (26 Sep 2024); a v0.89.0
tag was cut on 31 Oct 2024 and never published to either. Nothing has
shipped since. It is **not** deprecated and **not** archived — the
repository is open, carries no deprecation notice, and commits still land
through Google's Copybara export (most recent at the time of writing:
3 Aug 2026). The accurate word is *stalled*, and it applies to the release
cadence rather than to the project: "dead" is the easier line and the
wrong one.

**What the stall costs here.** 0.87.0 declares `>=3.9,<3.12` and ships one
wheel per release, `manylinux_2_31_x86_64` — no Windows, macOS or aarch64
artifact at any version, which is why non-Linux work goes through Docker
or WSL2. It pins `tensorflow==2.14.1`, and its exact `jaxlib==0.4.14` pin
has since been removed from PyPI, so `requirements.txt` carries a
`--find-links` line into Google's historical index purely to stay
resolvable. The cost a reader can verify fastest is the one already in
*Limitations*: **torch is pinned at 2.0.1 because TFF pins
`typing-extensions==4.5.*`**, and torch ≥ 2.1 requires ≥ 4.8. The same pin
propagates into the API stack — pydantic stays on 1.x, so fastapi stays
below 0.100.

**None of which touches the algorithm layer, which is the point.** This
project never adopted TFF's federated-algorithm stack. FedAvg is
hand-written numpy (`weighted_average` in `fl/aggregation.py`, float64
accumulation); the transport is this repo's own `.proto` and gRPC server;
the wire format is framework-neutral float32 tensors the aggregator
consumes without conversion. On the training path TFF is load-bearing in
exactly three places:

| TFF surface | Used by | Replacing it |
|---|---|---|
| `DifferentiallyPrivateFactory.gaussian_fixed` | `DPFedAvgAggregator` | Clip each delta to `S`, sum, add `N(0, (zS)²)`, divide by `m`. The deltas and `l2_norm` already exist here; what would be taken on is the noise draw and the cross-round state `AggregationProcess` holds. Flower's `DifferentialPrivacyServerSideFixedClipping` is the direct equivalent. |
| `DifferentiallyPrivateFactory.gaussian_adaptive` | `AdaptiveDPFedAvgAggregator` | The real work: a geometric quantile estimator (Andrew et al. 2021) and the noise split it forces. The split is already derived and tested here (`adaptive_noise_breakdown`), so the accounting is understood; the estimator is not written. Flower ships `DifferentialPrivacyServerSideAdaptiveClipping`. |
| `dp_accounting` for ε | `compute_epsilon` | Nothing. `dp-accounting` is a separate PyPI package that TFF pins rather than contains, and is imported directly; dropping TFF leaves `compute_epsilon` untouched. |

One further use sits off the training path: `prepare_femnist_cache`
(`fl/data.py`) calls `tff.simulation.datasets.emnist.load_data` once to
build the FEMNIST cache, and every run afterwards reads the packed
`.npz`. It costs nothing until that cache must be rebuilt.

That the seam is this narrow is not luck. The `Aggregator` protocol takes
`list[ClientUpdate]` and returns `Weights` — plain float32 arrays, nothing
TF-shaped crossing it. Framework-blind aggregation is what leaves the
server unable to tell a TensorFlow client from a PyTorch one (*One
canonical wire format*, above), and a server that cannot see a client's
framework cannot see its aggregator's either. The property bought for the
mixed-framework result is the same one that confines TFF to three call
sites.

**The alternative, if it is ever needed.** Flower is the maintained
option: framework-agnostic by design, releasing roughly monthly (1.26.0 in
Feb 2026 through 1.33.0 in Jul 2026), pure-Python wheels, with DP built in
(fixed and adaptive clipping, server- and client-side) and secure
aggregation (`SecAggWorkflow`, `SecAggPlusWorkflow`) — the latter
something this repo currently implements itself. It requires Python
≥ 3.11, where TFF stops; the two overlap on 3.11 alone.

**No migration is planned.** The pinned stack works, resolves
reproducibly, and its costs are recorded rather than waiting to be
discovered. This entry prices the dependency; it does not schedule its
removal.

## Client-level DP, not example-level

**Chosen:** the protected unit is one client's entire dataset —
`gaussian_fixed` clips each client's whole round update, so the guarantee
concerns whether a participant took part at all.

**Rejected:** example-level DP-SGD inside each trainer. It answers a
different question ("does the model reveal this training row") that
cross-silo federation does not ask; participants here are institutions
whose membership is the secret. Example-level noise also composes per local
step, costing far more accuracy for a guarantee nobody in this threat model
needs, and it guards nothing about the update the server actually sees.
Client-level DP is also what makes the equal-weighting constraint
principled: a sensitivity bound only holds if every client is weighted
equally, which is why the DP aggregator refuses sample-count weighting.

## One canonical wire format, conversions at the edges

**Chosen:** schema V2 — named float32 tensors, explicit shape and dtype,
C-contiguous, in TensorFlow's native layout. PyTorch clients convert at
their own edge (`fl/adapters.py`: conv kernels `permute(2,3,1,0)`, dense
transpose, NCHW→NHWC, the permute-before-flatten trap handled once).

**Rejected:** framework-native serialisation (pickled state_dicts, saved
Keras models) or per-framework branches on the server. Either couples the
server to every client's framework and version — the server would need
torch installed to unpickle a torch client — and makes "the server cannot
tell clients apart" impossible by construction. With one canonical form the
aggregator performs zero conversions, framework appears in exactly one log
line, and the mixed-pool result (0.36 pp from pure-TF, three seeds) is a
property of the system rather than of a compatibility shim. The layout is
TF's own on purpose: the aggregation stack consumes it directly, so exactly
one side pays conversion cost.

## Natural FEMNIST partitioning beside synthetic Dirichlet

**Chosen:** both. Fashion-MNIST with Dirichlet(α) label skew as the
controllable default; LEAF-derived FEMNIST, partitioned by the 3,400 real
writers, as the ground truth for population questions.

**Rejected:** either alone. Dirichlet alone is a simulation of
heterogeneity whose parameters the experimenter controls — the Fashion
cohort sweep's N = 2m confound (bigger cohort ⇒ thinner shards) went
unnoticed until the natural partition existed to expose it, retiring the
fitted k = 0.828 exponent. Natural alone loses the α knob the dashboard's
partition preview and the non-IID ablations need. The pairing is the
repo's best methodological asset: the synthetic partition generates
hypotheses, the natural one arbitrates them (docs/femnist_cohort.md).

## torch-before-TF import order is load-bearing

Importing TensorFlow first and torch second aborts the process —
`std::random_device could not be read`, a core dump, not an exception —
under the pinned coexistence stack (TFF 0.87's exact `typing-extensions`
pin forces torch 2.0.1). The rule is torch first, everywhere; `fl/client`
stays import-light so `main()` controls order, conftest pins it for the
whole test suite, and the v0.2 audit reproduced the abort on a cold
install. **Rejected:** "fix it properly." The collision is inside two
vendors' native initialisation under versions their pins force; the honest
engineering is an enforced convention with the failure documented, not a
patch the repo cannot own. (The audit's one refinement: the abort is
environment-wide and instant, so it cannot ship silently — CI imports
correctly or dies loudly.)

## Secure aggregation and DP are complementary, not alternatives

Pairwise masking hides **individual updates from the server**; DP bounds
**what the aggregate — and the released model — reveals about any one
client**. Masking without DP still leaks through the sum; DP without
masking leaves the server reading plaintext updates. The teaching
implementation (`fl/secure_aggregation.py`, docs/secure_aggregation.md)
demonstrates the protocol: masks cancelling bit-exactly in Z_2^64, Shamir
recovery through both dropout stages, the either-or reveal rule.
**Rejected:** wiring it into the DP path as if composition were a config
flag. The TFF path clips and noises *centrally, after seeing individual
updates*; real composition requires client-side clipping and distributed
noise — a protocol change this repo documents instead of pretending to
have. The Limitations entry stays until the wiring exists.

## The fixed clip stays the default, adaptive clipping notwithstanding

Adaptive clipping is implemented, ε-accounted (σ-additivity: total ε equals
the fixed arm's) and measured across six phases. It stays opt-in because
the measurements say so: warm-started at a tuned clip it *holds* it
(FEMNIST: 68.3 % vs 68.2 %); cold-started from TFF's default it converges
to the median and **never catches the bracketed fixed arm** (54.8 % vs
62.4 % at R = 100, one seed); and where the tuned optimum is a *binding*
clip below the median (Fashion), the median is the wrong target — a lower
target quantile recovers fixed performance, but choosing it required
already knowing the sweep's answer. A method that relocates the tuning
problem does not get to be the default over the tuned answer it relocated
(docs/adaptive_clipping.md, including the retraction trail).

## What the audit found — three bugs unit tests structurally could not catch

The v0.2 audit (docs/audit_v0_2.md) is the most instructive artifact in
this repo for another engineer, because all three of its critical/major
code findings share one shape: **a green test suite proving a property of
the test harness rather than of the system.**

1. **The WebSocket that could never connect (C1).** Starlette's TestClient
   implements the WS handshake in-process — no server WS protocol library
   is ever exercised. `requirements.txt` shipped bare uvicorn, so every
   real upgrade 404'd (`No supported WebSocket library detected`) while
   every WS test — replay, ordering, eviction, all of them — stayed green.
   The tests were correct about the application logic and silent about the
   deployment: the handshake belonged to the harness.

2. **One DP run per process (C2).** TFF's execution context lives in a
   `threading.local`; the only context this repo ever had was installed as
   a side effect of TFF's import, in whichever thread imported it first.
   Tests run single-threaded, so import-thread and aggregation-thread
   coincide and everything passes. The coordinator runs each training run
   on its own thread — the first DP run imports TFF in its thread and
   works; every later DP run gets a fresh thread, an empty context stack,
   and `No default context installed`. No unit test can see this without
   reproducing the runner's exact threading topology, which is the thing
   unit tests abstract away.

3. **The config honoured on one path of three (M3).** `adaptive_clipping`
   was forwarded to the aggregator factory by the gRPC server and dropped
   by the coordinator runner and the one-shot experiment script — which
   defaulted, silently, to fixed clipping while logging the config they
   did not honour. Unit tests exercised the factory (correct) and each
   path's training loop (correct); the defect lived in the *plumbing
   between* them, and every path had a plausible-looking call site. The
   fix is structural, not just a patch: one constructor
   (`aggregator_from_config`) that all three paths must share, so the
   forwarding cannot diverge again.

The common lesson: integration seams — harness-vs-server, thread topology,
config plumbing — need at least one test that crosses the real seam. The
live-mode Playwright e2e (`dashboard/e2e-live/`), the cross-thread DP test,
and the from-Config dispatch test exist now for exactly that reason.
