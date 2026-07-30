"""Tests for FedAvg aggregation arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from fl.aggregation import (
    AggregationError,
    ClientUpdate,
    FedAvgAggregator,
    uniform_average,
    validate_updates,
    weighted_average,
)


def _update(cid: str, value: float, n: int, tensors: int = 1) -> ClientUpdate:
    return ClientUpdate(
        client_id=cid,
        weights=[np.full((2, 2), value, dtype=np.float32) for _ in range(tensors)],
        num_examples=n,
    )


# ---------------------------------------------------------------------------
# The weighted-average test the whole design rests on
# ---------------------------------------------------------------------------


def test_fedavg_matches_hand_computed_three_client_example():
    """Hand-computed FedAvg over shard sizes 10, 100 and 1000.

        w = (10*1 + 100*2 + 1000*3) / (10 + 100 + 1000)
          = (10 + 200 + 3000) / 1110
          = 3210 / 1110
          = 2.891891891...

    The unweighted mean of the same three values is exactly 2.0. The two differ
    by ~0.89, far outside any floating-point tolerance, so this test fails
    immediately if the implementation is changed to an unweighted mean. That is
    its purpose: with equal-sized shards the two are identical and a broken
    implementation would pass unnoticed.
    """
    updates = [_update("a", 1.0, 10), _update("b", 2.0, 100), _update("c", 3.0, 1000)]

    result = weighted_average(updates)

    expected = 3210.0 / 1110.0
    assert result[0].shape == (2, 2)
    np.testing.assert_allclose(result[0], np.full((2, 2), expected), rtol=1e-6)

    unweighted = 2.0
    assert abs(expected - unweighted) > 0.85
    assert not np.allclose(result[0], np.full((2, 2), unweighted), atol=0.5)


def test_fedavg_is_not_an_unweighted_mean():
    """Explicit guard, phrased as the failure mode it is defending against."""
    updates = [_update("a", 1.0, 10), _update("b", 2.0, 100), _update("c", 3.0, 1000)]
    weighted = weighted_average(updates)
    unweighted = uniform_average(updates)
    assert not np.allclose(weighted[0], unweighted[0], atol=1e-3)
    np.testing.assert_allclose(unweighted[0], np.full((2, 2), 2.0), rtol=1e-6)


def test_fedavg_weighting_holds_across_every_tensor_in_the_list():
    updates = [
        ClientUpdate("a", [np.float32([1.0]), np.float32([[10.0, 10.0]])], 10),
        ClientUpdate("b", [np.float32([2.0]), np.float32([[20.0, 20.0]])], 100),
        ClientUpdate("c", [np.float32([3.0]), np.float32([[30.0, 30.0]])], 1000),
    ]
    result = weighted_average(updates)
    np.testing.assert_allclose(result[0], [3210.0 / 1110.0], rtol=1e-6)
    np.testing.assert_allclose(result[1], [[32100.0 / 1110.0] * 2], rtol=1e-6)


def test_dominant_client_pulls_the_average_towards_itself():
    """A 1000-sample client must dominate two tiny ones."""
    updates = [_update("a", 0.0, 1), _update("b", 0.0, 1), _update("c", 1.0, 1000)]
    result = weighted_average(updates)
    np.testing.assert_allclose(result[0], np.full((2, 2), 1000.0 / 1002.0), rtol=1e-6)
    assert result[0].mean() > 0.99


def test_equal_sample_counts_reduce_to_the_plain_mean():
    updates = [_update("a", 1.0, 50), _update("b", 2.0, 50), _update("c", 3.0, 50)]
    np.testing.assert_allclose(weighted_average(updates)[0], np.full((2, 2), 2.0), rtol=1e-6)


# ---------------------------------------------------------------------------
# Degenerate client sets
# ---------------------------------------------------------------------------


def test_single_client_round_returns_that_client_unchanged():
    updates = [_update("solo", 7.5, 42)]
    np.testing.assert_allclose(weighted_average(updates)[0], np.full((2, 2), 7.5), rtol=0)


def test_zero_reporting_clients_is_an_error():
    with pytest.raises(AggregationError, match="no client updates"):
        weighted_average([])


def test_mismatched_tensor_shapes_rejected():
    a = ClientUpdate("a", [np.zeros((2, 2), np.float32)], 10)
    b = ClientUpdate("b", [np.zeros((3, 3), np.float32)], 10)
    with pytest.raises(AggregationError, match="has shape"):
        weighted_average([a, b])


def test_mismatched_tensor_counts_rejected():
    a = ClientUpdate("a", [np.zeros(2, np.float32), np.zeros(2, np.float32)], 10)
    b = ClientUpdate("b", [np.zeros(2, np.float32)], 10)
    with pytest.raises(AggregationError, match="sent 1 tensors, expected 2"):
        weighted_average([a, b])


def test_nan_weights_rejected():
    good = _update("good", 1.0, 10)
    bad = ClientUpdate("bad", [np.full((2, 2), np.nan, np.float32)], 10)
    with pytest.raises(AggregationError, match="non-finite"):
        weighted_average([good, bad])


def test_inf_weights_rejected():
    good = _update("good", 1.0, 10)
    bad = ClientUpdate("bad", [np.full((2, 2), np.inf, np.float32)], 10)
    with pytest.raises(AggregationError, match="non-finite"):
        weighted_average([good, bad])


@pytest.mark.parametrize("n", [0, -1])
def test_non_positive_sample_count_rejected(n):
    with pytest.raises(AggregationError, match="must be positive"):
        weighted_average([_update("a", 1.0, n)])


def test_empty_tensor_list_rejected():
    with pytest.raises(AggregationError, match="no tensors"):
        validate_updates([ClientUpdate("a", [], 10)])


def test_float64_accumulation_survives_extreme_weight_disparity():
    """Small contributors must not vanish into float32 rounding."""
    updates = [_update("tiny", 1.0, 1), _update("huge", 0.0, 10_000_000)]
    result = weighted_average(updates)
    assert result[0][0, 0] > 0.0


def test_fedavg_aggregator_matches_weighted_average():
    updates = [_update("a", 1.0, 10), _update("b", 2.0, 100), _update("c", 3.0, 1000)]
    agg = FedAvgAggregator()
    np.testing.assert_allclose(
        agg.aggregate(updates, [np.zeros((2, 2), np.float32)])[0],
        weighted_average(updates)[0],
        rtol=0,
    )
    assert agg.name == "fedavg"
