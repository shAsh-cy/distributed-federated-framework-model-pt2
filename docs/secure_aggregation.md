# Secure aggregation: pairwise additive masking

> **Teaching implementation, not production cryptography.** The module
> ([fl/secure_aggregation.py](../fl/secure_aggregation.py)) opens with the
> full list of elisions; the short version is: share transport is simulated
> in-process, nothing is authenticated, arithmetic is variable-time, and the
> threat model is an honest-but-curious server only. What it teaches
> correctly is the *protocol structure* of Bonawitz et al. (CCS 2017), and
> every claim below is asserted by
> [tests/test_secure_aggregation.py](../tests/test_secure_aggregation.py)
> (30 tests, numpy + stdlib only).

## The protocol

Each pair of clients derives a shared secret (Diffie-Hellman over RFC 3526
group 14) and expands it into a mask; the lower-ordered client adds it, the
higher-ordered subtracts it, so every pairwise mask cancels **exactly** in
the sum. Each client also adds a **self mask**, and Shamir-shares both of
its secrets (self-mask seed and key seed) among all participants at setup.
The server observes only uniformly-masked vectors; the unmasked aggregate is
**bit-identical** to the maskless computation, because updates are quantised
to fixed point and masked in Z_2^64, where cancellation is exact (float
masks cannot cancel bit-exactly; quantisation error is bounded by
2^-24 per element per client). The client's example-count weight rides
*inside* the masked vector, so the server learns only the weight sum —
strictly less than plain FedAvg reveals.

## Dropout, and dropout during recovery

A client that vanishes **before submitting** leaves its pairwise masks
uncancelled in every survivor's submission. Survivors each reveal their
share of the dropped client's *key seed*; at ``threshold`` shares the server
reconstructs it, re-derives the pairwise secrets, verifies the
reconstructed key against the registered public key (aborting on mismatch
rather than subtracting garbage), and cancels the orphaned masks.

A client that vanishes **during recovery** — submitted, then went silent —
is handled by the same threshold: its share *responses* are lost, but the
shares of its own secrets are held by everyone else, so both its self mask
and any recovery it was participating in survive as long as ``threshold``
responders remain. Below threshold the round aborts with an explicit
`InsufficientSharesError`; it never emits a corrupted aggregate.

The rule that makes this safe: for any one client the server collects
either its self-mask shares (it survived) **or** its key-seed shares (it
dropped), never both — the pair would unmask that client's individual
submission. The client object enforces the refusal itself, and a test
exercises it.

## What masking costs

Measured by `communication_cost()`, whose analytic total the tests hold
equal, to the byte, against the server's message log from a live round with
both dropout stages. Routed shares are single-counted (true wire cost is
two hops through the server), so the model *understates* share traffic by
2× — stated rather than hidden.

| Cohort n | Model | Secure round | Plain FedAvg | Ratio |
|---|---|---|---|---|
| 10 | small_cnn (225,034 params) | 18.1 MB | 9.0 MB | 2.005 |
| 10 | femnist_cnn (231,742 params) | 18.6 MB | 9.3 MB | 2.005 |
| 50 | small_cnn | 91.2 MB | 45.0 MB | 2.026 |
| 50 | femnist_cnn | 93.9 MB | 46.3 MB | 2.026 |
| 200 | small_cnn | 378.8 MB | 180.0 MB | 2.104 |
| 200 | femnist_cnn | 389.6 MB | 185.4 MB | 2.101 |

Two effects, clearly separated: the dominant cost at these model sizes is
the **2× inflation of the update itself** (uint64 words versus float32),
which is why every ratio starts at ~2.0; on top of that, key exchange and
share distribution grow as **n²**, visible as the drift from 2.005 at
n = 10 to 2.10 at n = 200. For much smaller models or much larger cohorts
the quadratic term dominates instead — which is exactly the regime the
Bonawitz paper's efficiency work targets.

## On the live training path (no-DP)

The protocol is now wired into the live gRPC training path, not only the
in-process protocol tests. A V3 client
([fl/secure_client.py](../fl/secure_client.py)) announces its masking public key
at registration; each round the server
([fl/secure_server.py](../fl/secure_server.py)) publishes the cohort roster and
global weights, the client trains as usual, Shamir-shares its secrets, and
submits a **masked** update — a vector of uint64 words, never plaintext
`ModelWeights`. The server sums the masked words and cancels the masks; it never
holds an individual update. Dropout rides the existing round deadline: a client
that misses the masked-submission barrier is recovered from survivors' shares,
and a second drop *during* recovery is absorbed by the Shamir threshold — the
same two-stage handling the protocol tests assert, now over real gRPC
([tests/test_secure_grpc.py](../tests/test_secure_grpc.py): 5 clients, 2 rounds,
one induced dropout).

The distributed orchestration is a thin transport over a message-driven core
([fl/secure_round.py](../fl/secure_round.py)) that is byte-for-byte equivalent to
the in-process reference — asserted in
[tests/test_secure_round.py](../tests/test_secure_round.py) — so a live-path bug
is isolated to the wire, not the protocol.

### The exactness claim, on real float32 weights

The protocol tests assert the unmasked sum equals the plain sum *bit-exactly*.
That holds on the live path with **real float32 model weights** for one reason:
updates are quantised to fixed point and masked in Z_2^64, where addition is
associative and masks cancel exactly. Float masks could not — `(a+m)+(b-m)` need
not equal `a+b` in float. So exact cancellation **does** require the fixed-point
quantisation, and the mechanism already implements it;
[tests/test_secure_live.py](../tests/test_secure_live.py) proves it on the real
small_cnn weight shapes (225,034 parameters): the secure aggregate is bit-exact
to the maskless quantised weighted mean.

The price is a bounded quantisation error against float FedAvg, and it is
**measured**, not assumed. Per-element error is at most `m / (2^(F+1) · Σ n_k)`
(`F = 24` fractional bits); on 5–20-client rounds over the real weights the
measured maximum is ≈ **5e-12** and the mean ≈ **1e-12** — three-plus orders of
magnitude below float32's own ~1e-7 resolution, so quantisation is not the
pipeline's error floor. `fl.secure_live.quantization_error` reports the number
beside its analytic bound.

### What the live path costs, measured

`scripts/secagg_overhead.py` measures the overhead the analytic model above
predicts, and isolates the O(m²) term the pairwise scheme is known for:

* **Bytes** track the model exactly (they are the same accounting): a 2.005×
  ratio at m = 10 driven by uint64-vs-float32 update inflation, drifting up as
  key-exchange and share distribution grow as m² (see the table above).
* **Masking compute** is O(m) per client — one self mask plus m−1 pairwise masks
  over a 225k-word vector — so **O(m²) across the cohort**. Measured per-client
  masking time rises ≈ 48 ms → 100 ms → 196 ms for m = 5 → 10 → 20, i.e. the
  cohort total 0.24 s → 1.0 s → 3.9 s, reaching hundreds of seconds by m = 200.
* **Wall-clock per round** secure-vs-plain and the **dropout-recovery cost** (one
  induced mid-round drop) are measured over real gRPC by the same script's
  `walltime` and `dropout` phases.

This quadratic scaling is exactly why **SecAgg+** (sparse neighbour graphs, each
client masking with only O(log n) peers — what Flower ships as its production
secure aggregation) exists: it replaces the O(m²) all-pairs masking with an
O(m log m) graph at a controlled security cost. This teaching implementation does
the full all-pairs scheme, and the overhead measurement is what makes the reason
for SecAgg+ concrete rather than asserted.

## How this composes with the existing DP

**Complementary, not alternatives.** Secure aggregation hides *individual
updates from the server*; differential privacy bounds *what the aggregate —
and the released model — reveals about any one client*. Masking without DP
still leaks through the sum (a cohort of one is its own update); DP without
masking leaves the server reading every plaintext update, which is the
README limitation this work addresses. Together, an honest-but-curious
server sees only a noised aggregate, and the DP guarantee holds against
everyone downstream of it.

One honest wiring caveat, and it is why the live path above is **no-DP**: the
repo's TFF DP path clips and noises **centrally**, after the aggregator has
seen individual updates — the exact thing masking prevents. So masking wires
into the no-DP path only; a secure server carries no DP aggregator and
[fl/secure_server.py](../fl/secure_server.py) refuses a config with
`privacy.enabled` rather than silently ignoring it.

Composing the two for real requires **distributed DP**: moving the clip
client-side and adding the noise distributively — each client contributes a
noise share (or adds local noise) so the *sum* carries the calibrated noise the
server never sees assembled. This is a real protocol change, not a config flag,
and it **changes the accounting** — the sensitivity argument moves to the client
and the noise is contributed in pieces rather than drawn once centrally. Google's
Gboard runs exactly this combination in production (secure aggregation with local
clipping and distributed noise). It is the named Roadmap item; what exists today
is the no-DP secure path plus the documented route to composition, not a pretence
that composition is done.

## What masking does NOT do

It constrains nothing about the aggregate's contents. A malicious client's
update — scaled, poisoned, NaN — aggregates exactly like any other, and
masking makes it *harder to attribute*, since the server no longer sees who
sent what. The Byzantine limitation the README records is therefore
slightly **widened** by secure aggregation, not narrowed: robust
aggregation rules that inspect individual updates (median, trimmed mean,
Krum) are incompatible with masking as-is, and the tension between the two
is a real design fork, not an oversight.

## What production would need

Authenticated key exchange bound to client identity; encrypted share
transport; signatures on round messages; constant-time crypto (or a vetted
library); an active-adversary story for a lying server (who can otherwise
fabricate dropout claims); parameter quantisation agreed per round; and the
distributed-DP wiring above.
