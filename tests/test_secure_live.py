"""Secure aggregation over weight LISTS, and THE EXACTNESS CLAIM on real weights.

numpy + stdlib only, like tests/test_secure_aggregation.py — no gRPC, no TF — so
the exactness property is proven on the same host that cannot run the training
stack. The shapes here are the real small_cnn's (225,034 parameters); the values
are random float32, standing in for a trained update.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.secure_live import (
    WeightUpdate,
    flatten_weights,
    float_weighted_average,
    quantization_error,
    quantized_weighted_average,
    secure_average,
    secure_equals_quantized,
    unflatten_weights,
)

# The small_cnn weight-list shapes, in get_weights() order (fl/models.py):
# conv1 k/b, conv2 k/b, dense1 k/b, dense2 k/b. 225,034 parameters.
SMALL_CNN_SHAPES: list[tuple[int, ...]] = [
    (3, 3, 1, 32),
    (32,),
    (3, 3, 32, 64),
    (64,),
    (1600, 128),
    (128,),
    (128, 10),
    (10,),
]


def _random_weights(rng: np.random.Generator, scale: float = 0.05) -> list[np.ndarray]:
    """A weight list shaped like the small_cnn, values in a trained-update range."""
    return [rng.normal(scale=scale, size=shape).astype(np.float32) for shape in SMALL_CNN_SHAPES]


def _updates(n: int, seed: int = 0) -> list[WeightUpdate]:
    rng = np.random.default_rng(seed)
    return [
        WeightUpdate(
            f"client-{i}", _random_weights(rng), num_examples=int(rng.integers(3000, 7000))
        )
        for i in range(n)
    ]


class TestFlatten:
    def test_round_trips_shapes_and_values(self):
        weights = _random_weights(np.random.default_rng(1))
        flat, shapes = flatten_weights(weights)
        assert shapes == SMALL_CNN_SHAPES
        assert flat.size == 225_034
        restored = unflatten_weights(flat, shapes)
        for a, b in zip(weights, restored, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="empty weight list"):
            flatten_weights([])


class TestExactnessOnRealWeights:
    """THE EXACTNESS CLAIM: the secure aggregate is bit-identical to the maskless
    quantised weighted mean, on real-shaped float32 weight lists."""

    def test_secure_sum_is_bit_exact_to_the_quantised_mean(self):
        updates = _updates(5)
        secure, report = secure_average(updates, threshold=3)
        quantised = quantized_weighted_average(updates)
        for s, q in zip(secure, quantised, strict=True):
            # float32 storage is identical because both are the same float64 value
            # cast the same way; the real bit-exactness is asserted below in f64.
            np.testing.assert_array_equal(s, q)
        assert report["dropped"] == []
        assert report["weight_sum"] == pytest.approx(sum(u.num_examples for u in updates))

    def test_exactness_holds_in_float64_before_any_cast(self):
        """The claim compared where it actually lives: float64, before the float32
        storage cast that unflatten applies."""
        updates = _updates(6, seed=2)
        assert secure_equals_quantized(updates, threshold=4)

    def test_dropout_recovers_the_survivor_mean_exactly(self):
        updates = _updates(5, seed=3)
        secure, report = secure_average(updates, threshold=3, drop_before_submit={"client-2"})
        survivors = [u for u in updates if u.client_id != "client-2"]
        quantised = quantized_weighted_average(survivors)
        for s, q in zip(secure, quantised, strict=True):
            np.testing.assert_array_equal(s, q)
        assert report["dropped"] == ["client-2"]

    def test_second_dropout_during_recovery_still_exact(self):
        updates = _updates(6, seed=4)
        secure, report = secure_average(
            updates,
            threshold=3,
            drop_before_submit={"client-0"},
            drop_during_recovery={"client-1"},
        )
        submitted = [u for u in updates if u.client_id != "client-0"]
        quantised = quantized_weighted_average(submitted)
        for s, q in zip(secure, quantised, strict=True):
            np.testing.assert_array_equal(s, q)
        assert report["dropped"] == ["client-0"]
        assert "client-1" in report["survivors"]
        assert "client-1" not in report["responders"]


class TestQuantizationError:
    """What exactness costs against float FedAvg — bounded, measured, reported."""

    def test_error_is_under_the_analytic_bound(self):
        updates = _updates(10, seed=5)
        result = quantization_error(updates)
        assert result["num_elements"] == 225_034
        assert result["within_bound"]
        assert result["max_abs_error"] <= result["analytic_bound_per_element"]

    def test_error_is_negligible_beside_float32_resolution(self):
        """The headline for the write-up: the quantisation error is far below
        float32's own ~1e-7 resolution, so it is not the pipeline's error floor."""
        updates = _updates(10, seed=6)
        result = quantization_error(updates)
        assert result["max_abs_error"] < 1e-6
        assert result["mean_abs_error"] < result["max_abs_error"]

    def test_secure_average_matches_float_fedavg_to_quantisation(self):
        updates = _updates(8, seed=7)
        secure, _ = secure_average(updates, threshold=8)
        reference = float_weighted_average(updates)
        for s, r in zip(secure, reference, strict=True):
            np.testing.assert_allclose(s, r, atol=1e-5)
