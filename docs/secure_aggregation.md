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

## How this composes with the existing DP

**Complementary, not alternatives.** Secure aggregation hides *individual
updates from the server*; differential privacy bounds *what the aggregate —
and the released model — reveals about any one client*. Masking without DP
still leaks through the sum (a cohort of one is its own update); DP without
masking leaves the server reading every plaintext update, which is the
README limitation this work addresses. Together, an honest-but-curious
server sees only a noised aggregate, and the DP guarantee holds against
everyone downstream of it.

One honest wiring caveat: the repo's TFF DP path clips and noises
**centrally**, after the aggregator has seen individual updates. Composing
it with masking for real requires moving the clip client-side and adding
the noise distributively (each client contributes a noise share so the
*sum* carries the calibrated noise) — a real protocol change, not a config
flag. That is the gap between this module existing and the deployed paths
using it, and the README Limitations entry states it.

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
