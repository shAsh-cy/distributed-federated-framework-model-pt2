"""FedAvg aggregation, with and without client-level differential privacy.

Two aggregators live here and they weight clients differently on purpose.

**FedAvgAggregator** implements textbook FedAvg: a weighted average of client
weights by local sample count,

    w <- sum_k (n_k * w_k) / sum_k n_k

**DPFedAvgAggregator** implements client-level DP and is deliberately *uniform*,

    w <- w_global + (1 / m) * ( sum_k clip(w_k - w_global, S) + N(0, (z*S)^2 I) )

The switch from weighted to uniform is not an oversight, and it is the subtlety
most easily got wrong. A DP guarantee needs a bound on how much one client can
move the released value. Clipping each client's contribution to L2 norm ``S``
gives sensitivity ``S`` -- but only if every client is then weighted equally. Under
sample-count weighting, one client's influence is ``n_k / sum(n) * S``, which
depends on the private data (its own shard size), so the sensitivity is no longer
a constant the accountant can use and the reported epsilon would be a fiction.
TFF encodes this in its type system: ``DifferentiallyPrivateFactory`` returns an
``UnweightedAggregationFactory``, not a weighted one.

The second subtlety is *what* gets clipped. DP is applied to the **delta**
``w_k - w_global``, never to raw weights. Clipping raw weights to a norm of ~1
would annihilate a trained model; clipping the round's update bounds exactly the
quantity a client contributes, which is what the sensitivity argument needs.

Consequence, stated plainly: enabling DP costs you sample-count weighting as well
as accuracy. On a non-IID split with unequal shards those are two distinct
penalties, and both show up in the recorded results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

Weights = list[np.ndarray]


class AggregationError(ValueError):
    """Raised when a set of client updates cannot be aggregated."""


@dataclass(frozen=True)
class ClientUpdate:
    """One client's contribution to one round.

    Attributes:
        client_id: Identity of the reporting client.
        weights: Model weights after local training, in ``get_weights()`` order.
        num_examples: Size of the client's local training set. This is the
            FedAvg weighting term ``n_k``; the server never infers it, because
            only the client knows its own shard.
        model_version: Global model version these weights were trained from.
            Staleness is enforced by the server, not here.
    """

    client_id: str
    weights: Weights
    num_examples: int
    model_version: int = 0


def validate_updates(updates: list[ClientUpdate]) -> None:
    """Reject update sets that cannot be meaningfully averaged.

    Checks, in order: non-empty; positive sample counts; consistent weight-list
    arity; consistent per-tensor shapes; all values finite.

    A NaN in one client's weights would silently poison the global model -- the
    average of anything with NaN is NaN -- so it is rejected here rather than
    discovered several rounds later as a model that predicts nothing.
    """
    if not updates:
        raise AggregationError("no client updates to aggregate")

    for u in updates:
        if u.num_examples <= 0:
            raise AggregationError(
                f"client {u.client_id!r} reported num_examples={u.num_examples}; must be positive"
            )

    arity = len(updates[0].weights)
    if arity == 0:
        raise AggregationError("client updates contain no tensors")

    reference_shapes = [np.asarray(w).shape for w in updates[0].weights]
    for u in updates:
        if len(u.weights) != arity:
            raise AggregationError(
                f"client {u.client_id!r} sent {len(u.weights)} tensors, "
                f"expected {arity} (client {updates[0].client_id!r})"
            )
        for i, w in enumerate(u.weights):
            shape = np.asarray(w).shape
            if shape != reference_shapes[i]:
                raise AggregationError(
                    f"client {u.client_id!r} tensor {i} has shape {shape}, "
                    f"expected {reference_shapes[i]}"
                )
            if not np.all(np.isfinite(np.asarray(w))):
                raise AggregationError(
                    f"client {u.client_id!r} tensor {i} contains non-finite values "
                    "(NaN or Inf); refusing to poison the global model"
                )


def weighted_average(updates: list[ClientUpdate]) -> Weights:
    """FedAvg: average client weights in proportion to local sample count.

        w = sum_k (n_k * w_k) / sum_k n_k

    Accumulation is done in float64 regardless of input dtype. With 60,000
    samples spread over unequal shards the weighting terms differ by orders of
    magnitude, and float32 accumulation loses low-order bits of the smaller
    contributors.
    """
    validate_updates(updates)
    total = sum(u.num_examples for u in updates)
    arity = len(updates[0].weights)

    out: Weights = []
    for i in range(arity):
        acc = np.zeros(np.asarray(updates[0].weights[i]).shape, dtype=np.float64)
        for u in updates:
            acc += np.asarray(u.weights[i], dtype=np.float64) * u.num_examples
        out.append((acc / total).astype(np.float32))
    return out


def uniform_average(updates: list[ClientUpdate]) -> Weights:
    """Unweighted mean over clients: every client counts once, regardless of n_k.

    This is what the DP path uses -- equal weighting is what makes the
    sensitivity bound hold. Kept as a named function so the difference from
    :func:`weighted_average` is explicit and directly testable, rather than
    buried behind a flag.
    """
    validate_updates(updates)
    arity = len(updates[0].weights)
    m = len(updates)
    out: Weights = []
    for i in range(arity):
        acc = np.zeros(np.asarray(updates[0].weights[i]).shape, dtype=np.float64)
        for u in updates:
            acc += np.asarray(u.weights[i], dtype=np.float64)
        out.append((acc / m).astype(np.float32))
    return out


def l2_norm(weights: Weights) -> float:
    """Global L2 norm of a weight list, treated as one flat vector."""
    return float(
        math.sqrt(sum(float(np.sum(np.square(np.asarray(w, np.float64)))) for w in weights))
    )


def subtract(a: Weights, b: Weights) -> Weights:
    """Element-wise ``a - b``.

    ``strict=True`` is deliberate: silently truncating to the shorter list would
    drop trailing layers from a model delta and leave the result looking valid.
    """
    return [
        np.asarray(x, np.float32) - np.asarray(y, np.float32) for x, y in zip(a, b, strict=True)
    ]


def add(a: Weights, b: Weights) -> Weights:
    """Element-wise ``a + b``. See :func:`subtract` on ``strict=True``."""
    return [
        np.asarray(x, np.float32) + np.asarray(y, np.float32) for x, y in zip(a, b, strict=True)
    ]


class Aggregator(Protocol):
    """Everything the server needs from an aggregation strategy."""

    #: Human-readable name recorded in the metrics file.
    name: str

    def aggregate(self, updates: list[ClientUpdate], global_weights: Weights) -> Weights:
        """Combine client updates into the next global model."""
        ...


class FedAvgAggregator:
    """Sample-count-weighted FedAvg. No privacy guarantee."""

    name = "fedavg"

    def aggregate(self, updates: list[ClientUpdate], global_weights: Weights) -> Weights:
        del global_weights  # Weighted averaging of absolute weights needs no anchor.
        return weighted_average(updates)


class DPFedAvgAggregator:
    """Client-level differentially private FedAvg, backed by TFF.

    Wraps ``tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed``, which
    clips each client's contribution to L2 norm ``l2_clip_norm``, sums, adds
    Gaussian noise with standard deviation ``noise_multiplier * l2_clip_norm``,
    and divides by ``clients_per_round``.

    The granularity is **client-level** (user-level): the protected unit is one
    participant's entire local dataset, so the guarantee concerns whether a given
    client took part at all. This is *not* example-level DP -- example-level DP,
    as produced by DP-SGD inside a single trainer, protects one training row and
    says nothing about participation. The clipping norm here bounds one client's
    whole round update, which is what makes the client-level claim true.

    TFF is used for the aggregation itself, not merely imported: the noise is
    drawn and the clipping applied inside a real ``tff.templates.AggregationProcess``
    whose state is carried across rounds.
    """

    name = "dp-fedavg"

    def __init__(
        self,
        noise_multiplier: float,
        l2_clip_norm: float,
        clients_per_round: int,
    ) -> None:
        if noise_multiplier <= 0:
            raise ValueError(
                f"noise_multiplier must be > 0 for a DP aggregator, got {noise_multiplier}"
            )
        if l2_clip_norm <= 0:
            raise ValueError(f"l2_clip_norm must be > 0, got {l2_clip_norm}")
        if clients_per_round < 1:
            raise ValueError(f"clients_per_round must be >= 1, got {clients_per_round}")

        self.noise_multiplier = float(noise_multiplier)
        self.l2_clip_norm = float(l2_clip_norm)
        self.clients_per_round = int(clients_per_round)
        self._process = None
        self._state = None
        self._value_type = None

    def _ensure_process(self, template: Weights) -> None:
        """Build the TFF aggregation process lazily, once the weight shapes are known."""
        import tensorflow_federated as tff

        value_type = tff.to_type(
            [tff.TensorType(np.float32, np.asarray(w).shape) for w in template]
        )
        if self._process is not None and value_type == self._value_type:
            return

        factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
            noise_multiplier=self.noise_multiplier,
            clients_per_round=float(self.clients_per_round),
            clip=self.l2_clip_norm,
        )
        self._value_type = value_type
        self._process = factory.create(value_type)
        self._state = self._process.initialize()

    def aggregate(self, updates: list[ClientUpdate], global_weights: Weights) -> Weights:
        """Clip, noise and average the client *deltas*, then apply to the global model.

        Note that ``num_examples`` is intentionally unused: see the module
        docstring for why sample-count weighting is incompatible with the
        sensitivity bound this aggregator's epsilon relies on.
        """
        validate_updates(updates)

        deltas = [subtract(u.weights, global_weights) for u in updates]
        self._ensure_process(global_weights)

        output = self._process.next(self._state, deltas)
        self._state = output.state
        mean_delta = [np.asarray(t, dtype=np.float32) for t in output.result]
        return add(global_weights, mean_delta)


def compute_epsilon(
    noise_multiplier: float,
    sampling_rate: float,
    rounds: int,
    delta: float = 1e-5,
) -> float:
    """Compute the DP epsilon actually achieved, using TF Privacy's accountant.

    Epsilon is a *consequence* of the mechanism, never a knob. It is derived here
    from the three things that determine it:

    * ``noise_multiplier`` -- Gaussian sigma as a multiple of the clipping norm.
    * ``sampling_rate`` -- ``q``, the fraction of the client population sampled
      each round. Client-level, so the population is the client count, not the
      example count.
    * ``rounds`` -- how many times the mechanism is composed.

    Uses ``dp_accounting`` (the accountant library maintained by the TensorFlow
    Privacy team and pinned by TFF) with the RDP accountant, composing ``rounds``
    Poisson-subsampled Gaussian mechanisms.

    Returns ``inf`` when ``noise_multiplier`` is 0: no noise is no privacy, and
    returning a large-but-finite number there would be a lie.
    """
    if rounds < 0:
        raise ValueError(f"rounds must be >= 0, got {rounds}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate}")
    if noise_multiplier <= 0:
        return float("inf")
    if rounds == 0:
        return 0.0

    from dp_accounting import dp_event
    from dp_accounting import rdp as rdp_module

    accountant = rdp_module.RdpAccountant()
    event = dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(sampling_rate, dp_event.GaussianDpEvent(noise_multiplier)),
        rounds,
    )
    accountant.compose(event)
    return float(accountant.get_epsilon(delta))


def make_aggregator(
    *,
    dp_enabled: bool,
    noise_multiplier: float,
    l2_clip_norm: float,
    clients_per_round: int,
) -> Aggregator:
    """Build the aggregator a config asks for."""
    if not dp_enabled:
        return FedAvgAggregator()
    return DPFedAvgAggregator(
        noise_multiplier=noise_multiplier,
        l2_clip_norm=l2_clip_norm,
        clients_per_round=clients_per_round,
    )
