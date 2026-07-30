"""FedAvg aggregation.

Implements textbook FedAvg: a weighted average of client weights by local
sample count,

    w <- sum_k (n_k * w_k) / sum_k n_k

The weighting term ``n_k`` is reported by each client, never inferred by the
server, because only a client knows how large its own shard is. On the default
non-IID split shard sizes range from roughly 1,800 to 11,800, so the weighting
is numerically meaningful rather than a no-op.
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
                f"client {u.client_id!r} reported num_examples={u.num_examples}; "
                "must be positive"
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

    Not what FedAvg uses. It exists so that the difference from
    :func:`weighted_average` is explicit and directly testable -- the two agree
    exactly when shards are equal-sized, which is precisely the case in which a
    broken weighted average would go unnoticed.
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
    return float(math.sqrt(sum(float(np.sum(np.square(np.asarray(w, np.float64)))) for w in weights)))


def subtract(a: Weights, b: Weights) -> Weights:
    """Element-wise ``a - b``."""
    return [np.asarray(x, np.float32) - np.asarray(y, np.float32) for x, y in zip(a, b)]


def add(a: Weights, b: Weights) -> Weights:
    """Element-wise ``a + b``."""
    return [np.asarray(x, np.float32) + np.asarray(y, np.float32) for x, y in zip(a, b)]


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


def make_aggregator(*, clients_per_round: int) -> Aggregator:
    """Build the aggregator a config asks for."""
    del clients_per_round
    return FedAvgAggregator()
