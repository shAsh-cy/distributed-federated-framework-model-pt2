# Privacy threats, and what defends against them

Federated learning keeps raw data on the device. That is a real privacy gain and
an incomplete one: the *updates* that leave the device are functions of the data,
and functions of data leak. This note connects the one attack and the two
defences this repo implements — what each protects against, what neither does,
and why production systems run both defences at once.

The three pieces:

| Piece | What it is | What it protects | What it does *not* |
|---|---|---|---|
| **Gradient inversion** (attack) | Reconstruct training images from an update ([scripts/inversion_demo.py](../scripts/inversion_demo.py)) | — | — |
| **Differential privacy** (defence) | Clip + Gaussian noise, ε-accounted ([fl/aggregation.py](../fl/aggregation.py)) | Bounds what the *released model / aggregate* reveals about any one client | Does not, as shipped (central), hide an individual update from the server |
| **Secure aggregation** (defence) | Pairwise masking, the server sees only the sum ([fl/secure_aggregation.py](../fl/secure_aggregation.py), wired in [fl/secure_server.py](../fl/secure_server.py)) | Hides *individual updates* from an honest-but-curious server | Constrains nothing about the aggregate's contents |

## The attack: an update is not opaque

`scripts/inversion_demo.py` reconstructs a client's training images from a single
gradient using the cosine-similarity attack of Geiping et al. (*Inverting
Gradients*, NeurIPS 2020). Given the gradient the client computes on one batch, it
optimises dummy images so their gradient matches — recovering the batch. Against
the small Fashion-MNIST CNN this works cleanly at batch size 1 and degrades as the
batch grows and gradients average (`docs/inversion/batch_size_{1,4,8}.png`).

The point is not that this specific attack is devastating in production — it is
not, and we show why below. The point is that **an update the server can read is
not opaque**: it carries recoverable information about the data that produced it.
That is the threat both defences below address, from opposite directions.

### Honest framing (mandatory, not a footnote)

The clean reconstructions hold under *idealised* conditions: a known, small batch;
a single local step (FedSGD); known labels; and no multi-epoch averaging. Real
FedAvg violates all of these. The demo runs **one attempt against a realistic
E=10 FedAvg update** — ten local epochs of SGD over the whole shard, the actual
thing a client sends — and it **fails** (`docs/inversion/fedavg_e10_failure.png`):
the multi-step delta is a composition over many batches, no longer the single
gradient the attack assumes. That failure is itself the honest result, and it
matches recent literature (e.g. Huang et al. 2021, *Evaluating Gradient Inversion
Attacks and Defenses*), which finds published attacks lean on assumptions
production training breaks. The demo illustrates the **threat model** that
motivates the defences; it does not claim production-realistic leakage. Treat a
figure that reconstructs a recognisable garment as a statement about the threat
model, not a measured production capability.

## Defence 1 — differential privacy bounds what the *output* reveals

Client-level DP (clip each client's update to L2 norm `S`, add Gaussian noise
`N(0, (zS)^2)` to the aggregate, account ε) makes the released model
near-indistinguishable whether or not any one client took part. In the inversion
picture, noise on the update degrades reconstruction: the defence curve
(`docs/inversion/defense_curve.png`) runs the attack at noise multipliers
{0, 0.3, working z = 2.0}, and the reconstruction goes from a recognisable garment
to smeared to mush, each panel labelled with the ε its multiplier buys at the
working config (q = 0.5, R = 20, δ = 1e-5; z = 2.0 → ε ≈ 6.23).

**One honest wrinkle the figure makes explicit.** The repo's DP is *central*: the
TFF aggregator clips and noises **after** summing, so it must see each individual
update, and it noises the *aggregate*, not the individual update. Central DP
therefore does **not** blur what an honest-but-curious *server* sees of one
client's gradient — the server reads it in the clear before any noise is added.
The defence curve models noise added to the update **before it leaves the client**
(local / distributed DP); it is the honest way to show DP degrading inversion,
and it is exactly the wiring that composition with secure aggregation requires
(below). What central DP *does* protect is everyone downstream of the aggregate,
including the released model — which is what bounds membership and reconstruction
from the *output*.

## Defence 2 — secure aggregation hides the *individual update*

Pairwise masking ([docs/secure_aggregation.md](secure_aggregation.md)) makes the
server observe only uniformly-masked vectors whose masks cancel in the sum: it
learns the **aggregate alone**, never an individual update. This directly removes
the attack's input — there is no individual gradient for the server to invert,
because the server never holds one. It is now wired into the live gRPC training
path (no-DP), verified bit-exact on real float32 weights
([docs/secure_aggregation.md](secure_aggregation.md)).

Secure aggregation and DP are **complementary, not alternatives**:

* Masking **without** DP still leaks through the sum — a cohort of one is its own
  update, and even a large aggregate can reveal a client's contribution.
* DP **without** masking leaves the server reading every plaintext update — the
  exact thing inversion attacks, and the reason central DP alone does not defend
  against an honest-but-curious server.

Together, an honest-but-curious server sees only a **noised aggregate**, and the
DP guarantee holds against everyone downstream of it.

## What NEITHER defends against: a malicious update

Both defences are about *confidentiality* — what an observer learns. Neither
constrains what a participant *contributes*. A well-formed malicious update —
scaled, poisoned, back-doored — aggregates exactly like any honest one. This is
the **Byzantine limitation** the README records, and secure aggregation
*widens* it: masking makes a poisoned update **harder to attribute**, since the
server no longer sees who sent what, and it is **incompatible** with robust
aggregation rules that inspect individual updates (coordinate-wise median,
trimmed mean, Krum) — those need the very per-client visibility masking removes.
Closing this needs a third, orthogonal line of defence (robust aggregation,
update-poisoning detection, client authentication), which is on the Roadmap and
not built here. Stating it plainly: **this system defends the confidentiality of
honest participants, not the integrity of the model against dishonest ones.**

## Why production runs both

A deployment that takes privacy seriously wants both guarantees at once: the
server should learn nothing about an individual update (secure aggregation) *and*
the released model should reveal little about any participant (DP). Google's
Gboard next-word-prediction system runs exactly this combination — secure
aggregation with **local** (client-side) clipping and distributed DP noise.

Composing them here is a real protocol change, not a config flag, precisely
because of the central-DP wrinkle above: the shipped DP path clips and noises
centrally, after seeing individual updates, which masking prevents. True
composition needs **distributed DP** — client-side clipping and locally-added (or
share-split) noise so the *sum* carries the calibrated noise the server never
sees assembled — and that changes the accounting (the noise is contributed in
pieces, and the sensitivity argument moves to the client). That is the named
Roadmap item; secure aggregation is wired for the no-DP path today, and the
distributed-DP composition is the next step, documented in
[docs/architecture.md](architecture.md) rather than pretended complete.

## Pointers

* Attack and the honest E=10 failure: [scripts/inversion_demo.py](../scripts/inversion_demo.py), grids in [docs/inversion/](inversion/)
* DP mechanism and ε accounting: [fl/aggregation.py](../fl/aggregation.py), [docs/dp_diagnosis.md](dp_diagnosis.md)
* Secure aggregation protocol and the live wiring: [docs/secure_aggregation.md](secure_aggregation.md), [fl/secure_server.py](../fl/secure_server.py)
* Why they do not compose as-is, and what would fix it: [docs/architecture.md](architecture.md)
