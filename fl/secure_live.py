"""Secure aggregation over model weight LISTS — the bridge to the training path.

:mod:`fl.secure_aggregation` masks and sums flat vectors. A federated update is
a *list* of tensors (a CNN's kernels and biases), and the training path weights
each client's contribution by its example count. This module is the thin,
framework-free adapter between the two: it flattens a weight list into the one
vector the protocol masks, runs a secure round, and unflattens the aggregate
back into the model's tensor shapes.

It exists so that everything the live gRPC server needs from secure aggregation
— and the whole of THE EXACTNESS CLAIM — is expressed in numpy alone and can be
tested without gRPC, TensorFlow, or a network. The gRPC layer
(:mod:`fl.secure_server`) moves the same masked words over the wire; the
arithmetic here is what makes the aggregate correct, and it is identical to what
that layer computes.

THE EXACTNESS CLAIM, precisely
------------------------------
Plain FedAvg computes ``sum_k n_k w_k / sum_k n_k`` in float. Secure aggregation
cannot: float masks do not cancel bit-exactly, because float addition is not
associative — ``(a + m) + (b - m)`` need not equal ``a + b``. So the protocol
quantises to fixed point and masks in Z_2^64, where addition IS associative and
masks cancel exactly. The consequence, made concrete by :func:`secure_average`
and measured by :func:`quantization_error`:

* The secure aggregate is **bit-identical** to the same weighted sum computed
  maskless in the fixed-point domain (:func:`quantized_weighted_average`). This
  is exact — asserted, not approximate.
* Against the float64 weighted average it differs by the **quantisation error**
  only: at worst ``m / 2^(F+1)`` per element in the summed domain, i.e. bounded
  by ``m / (2^(F+1) * sum_k n_k)`` per element of the mean, where ``F`` is
  ``FRACTIONAL_BITS``. For a 10-client round over ~60k examples at ``F = 24``
  that ceiling is ~5e-12 — far below float32's own ~1e-7 resolution, so the
  quantisation is not the dominant error source in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .secure_aggregation import (
    FRACTIONAL_BITS,
    fixed_point_decode,
    fixed_point_encode,
    run_secure_round,
)

Weights = list[np.ndarray]


@dataclass(frozen=True)
class WeightUpdate:
    """One client's contribution to a secure round: its full post-training
    weight list and the example count that weights it in FedAvg."""

    client_id: str
    weights: Weights
    num_examples: int


def flatten_weights(weights: Weights) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Concatenate a weight list into one 1-D vector, keeping the shapes needed
    to invert it. float64 so the flatten itself loses nothing before quantisation."""
    shapes = [np.asarray(w).shape for w in weights]
    if not shapes:
        raise ValueError("cannot flatten an empty weight list")
    flat = np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w in weights])
    return flat, shapes


def unflatten_weights(flat: np.ndarray, shapes: list[tuple[int, ...]]) -> Weights:
    """Invert :func:`flatten_weights`, casting back to float32 (the wire dtype)."""
    out: Weights = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape)) if shape else 1
        chunk = flat[offset : offset + size]
        if chunk.size != size:
            raise ValueError(
                f"flat vector exhausted: needed {size} for shape {shape}, got {chunk.size}"
            )
        out.append(chunk.reshape(shape).astype(np.float32))
        offset += size
    if offset != flat.size:
        raise ValueError(f"flat vector has {flat.size} elements, shapes account for {offset}")
    return out


def _check_consistent(updates: list[WeightUpdate]) -> list[tuple[int, ...]]:
    if not updates:
        raise ValueError("no updates to aggregate")
    reference = [np.asarray(w).shape for w in updates[0].weights]
    for u in updates:
        shapes = [np.asarray(w).shape for w in u.weights]
        if shapes != reference:
            raise ValueError(
                f"client {u.client_id!r} weight shapes {shapes} disagree with {reference}"
            )
        if u.num_examples <= 0:
            raise ValueError(f"client {u.client_id!r} num_examples must be positive")
    return reference


def _quantized_flat_mean(updates: list[WeightUpdate]) -> np.ndarray:
    """The maskless ground-truth mean as a float64 flat vector (no float32 cast):
    encode ``weights * n`` in fixed point, sum in Z_2^64, decode, divide by the
    example total. The secure aggregate must equal this bit-for-bit."""
    length = sum(int(np.asarray(w).size) for w in updates[0].weights) + 1
    total = np.zeros(length, dtype=np.uint64)
    for u in updates:
        flat, _ = flatten_weights(u.weights)
        payload = np.concatenate([flat * u.num_examples, [float(u.num_examples)]])
        total += fixed_point_encode(payload)
    decoded = fixed_point_decode(total)
    return decoded[:-1] / decoded[-1]


def _float_flat_mean(updates: list[WeightUpdate]) -> np.ndarray:
    """Plain float64 weighted FedAvg as a flat vector, for the error baseline."""
    total_n = sum(u.num_examples for u in updates)
    acc = np.zeros(sum(int(np.asarray(w).size) for w in updates[0].weights), dtype=np.float64)
    for u in updates:
        flat, _ = flatten_weights(u.weights)
        acc += flat * u.num_examples
    return acc / total_n


def _secure_flat_mean(
    updates: list[WeightUpdate],
    threshold: int,
    drop_before_submit: set[str] | frozenset[str] = frozenset(),
    drop_during_recovery: set[str] | frozenset[str] = frozenset(),
) -> tuple[np.ndarray, dict]:
    """Run the protocol and return the aggregate as a float64 flat vector."""
    flat_updates = [
        (u.client_id, flatten_weights(u.weights)[0], float(u.num_examples)) for u in updates
    ]
    return run_secure_round(
        flat_updates,
        threshold=threshold,
        drop_before_submit=drop_before_submit,
        drop_during_recovery=drop_during_recovery,
    )


def quantized_weighted_average(updates: list[WeightUpdate]) -> Weights:
    """The maskless ground truth as a weight list (see :func:`_quantized_flat_mean`)."""
    shapes = _check_consistent(updates)
    return unflatten_weights(_quantized_flat_mean(updates), shapes)


def float_weighted_average(updates: list[WeightUpdate]) -> Weights:
    """Plain float64 FedAvg as a weight list, for measuring what quantisation costs."""
    shapes = _check_consistent(updates)
    return unflatten_weights(_float_flat_mean(updates), shapes)


def secure_average(
    updates: list[WeightUpdate],
    threshold: int,
    drop_before_submit: set[str] | frozenset[str] = frozenset(),
    drop_during_recovery: set[str] | frozenset[str] = frozenset(),
) -> tuple[Weights, dict]:
    """Run one secure round over weight-list updates and return the aggregated
    weight list plus the protocol report.

    This is the reference the gRPC live path reproduces: each client masks its
    flattened ``weights * num_examples`` (plus the weight itself), the server
    sums the masked words and cancels the masks, and the result is the
    sample-count-weighted mean — the FedAvg no-DP aggregate, computed without the
    server ever seeing an individual update.
    """
    shapes = _check_consistent(updates)
    mean_flat, report = _secure_flat_mean(
        updates, threshold, drop_before_submit, drop_during_recovery
    )
    return unflatten_weights(mean_flat, shapes), report


def secure_equals_quantized(updates: list[WeightUpdate], threshold: int) -> bool:
    """True iff the secure aggregate is BIT-IDENTICAL to the maskless quantised
    mean — the exactness claim, as a bool, compared in float64 before any float32
    storage cast could blur it."""
    _check_consistent(updates)
    secure_flat, _ = _secure_flat_mean(updates, threshold)
    return bool(np.array_equal(secure_flat, _quantized_flat_mean(updates)))


def quantization_error(updates: list[WeightUpdate]) -> dict:
    """Measure the error fixed-point quantisation introduces into a clean round,
    against the float64 weighted mean, and report it beside its analytic bound.

    The comparison is done in float64 on the flat vectors, deliberately BEFORE
    the float32 storage cast: a float32 ULP (~1e-7) would otherwise swamp the
    ~1e-12 quantisation signal and make the measurement meaningless. Returns
    max/mean absolute error per weight element and the ``m / (2^(F+1) *
    sum_k n_k)`` ceiling, so a caller can assert the measurement sits under the
    bound and quote the real number rather than the worst case.
    """
    shapes = _check_consistent(updates)
    threshold = len(updates)  # clean round: everyone responds
    secure_flat, _ = _secure_flat_mean(updates, threshold)
    reference_flat = _float_flat_mean(updates)

    errors = np.abs(secure_flat - reference_flat)
    total_n = sum(u.num_examples for u in updates)
    num_elements = int(sum(int(np.prod(s)) if s else 1 for s in shapes))
    bound = len(updates) / (2.0 ** (FRACTIONAL_BITS + 1) * total_n)
    return {
        "num_clients": len(updates),
        "num_elements": num_elements,
        "fractional_bits": FRACTIONAL_BITS,
        "max_abs_error": float(errors.max()),
        "mean_abs_error": float(errors.mean()),
        "analytic_bound_per_element": float(bound),
        "within_bound": bool(errors.max() <= bound),
    }
