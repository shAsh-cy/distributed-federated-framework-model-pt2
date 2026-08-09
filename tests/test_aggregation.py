"""Tests for FedAvg arithmetic and differentially private aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from fl.aggregation import (
    AggregationError,
    ClientUpdate,
    DPFedAvgAggregator,
    FedAvgAggregator,
    compute_epsilon,
    l2_norm,
    make_aggregator,
    subtract,
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


# ---------------------------------------------------------------------------
# Differential privacy
# ---------------------------------------------------------------------------


def test_dp_aggregator_requires_positive_noise():
    with pytest.raises(ValueError, match="noise_multiplier must be > 0"):
        DPFedAvgAggregator(noise_multiplier=0.0, l2_clip_norm=1.0, clients_per_round=3)


def test_dp_aggregator_rejects_bad_clip_and_cohort():
    with pytest.raises(ValueError, match="l2_clip_norm must be > 0"):
        DPFedAvgAggregator(noise_multiplier=1.0, l2_clip_norm=0.0, clients_per_round=3)
    with pytest.raises(ValueError, match="clients_per_round must be >= 1"):
        DPFedAvgAggregator(noise_multiplier=1.0, l2_clip_norm=1.0, clients_per_round=0)


@pytest.mark.slow
def test_dp_aggregator_ignores_sample_counts():
    """Client-level DP must weight every client equally.

    Sample-count weighting would make one client's influence depend on its own
    (private) shard size, so the sensitivity bound -- and therefore the reported
    epsilon -- would not hold. With negligible noise and a huge clipping norm the
    DP path must reproduce the *uniform* mean, not the weighted one.
    """
    global_w = [np.zeros((2, 2), np.float32)]
    updates = [_update("a", 1.0, 10), _update("b", 2.0, 100), _update("c", 3.0, 1000)]

    agg = DPFedAvgAggregator(noise_multiplier=1e-9, l2_clip_norm=1e6, clients_per_round=3)
    result = agg.aggregate(updates, global_w)

    np.testing.assert_allclose(result[0], np.full((2, 2), 2.0), atol=1e-3)
    weighted = 3210.0 / 1110.0
    assert not np.allclose(result[0], np.full((2, 2), weighted), atol=0.1)


@pytest.mark.slow
def test_dp_clipping_bounds_each_client_contribution():
    """A client with an enormous update must not move the model more than the clip allows."""
    global_w = [np.zeros((10,), np.float32)]
    honest = ClientUpdate("honest", [np.full((10,), 0.01, np.float32)], 100)
    huge = ClientUpdate("huge", [np.full((10,), 1000.0, np.float32)], 100)

    agg = DPFedAvgAggregator(noise_multiplier=1e-9, l2_clip_norm=1.0, clients_per_round=2)
    result = agg.aggregate([honest, huge], global_w)

    # Each delta is clipped to L2 <= 1, so the mean delta has L2 <= 1.
    assert l2_norm(subtract(result, global_w)) <= 1.0 + 1e-3


@pytest.mark.slow
def test_dp_noise_actually_perturbs_the_result():
    """With real noise the output must differ from the noiseless mean."""
    global_w = [np.zeros((50,), np.float32)]
    updates = [ClientUpdate(f"c{i}", [np.full((50,), 0.1, np.float32)], 100) for i in range(4)]
    agg = DPFedAvgAggregator(noise_multiplier=1.0, l2_clip_norm=1.0, clients_per_round=4)
    noisy = agg.aggregate(updates, global_w)
    assert not np.allclose(noisy[0], np.full((50,), 0.1), atol=1e-4)


@pytest.mark.slow
def test_dp_aggregator_carries_state_across_rounds():
    global_w = [np.zeros((4,), np.float32)]
    updates = [ClientUpdate(f"c{i}", [np.full((4,), 0.5, np.float32)], 10) for i in range(3)]
    agg = DPFedAvgAggregator(noise_multiplier=0.5, l2_clip_norm=1.0, clients_per_round=3)
    first = agg.aggregate(updates, global_w)
    second = agg.aggregate(updates, global_w)
    assert agg._state is not None
    # Fresh noise every round, so two identical inputs give different outputs.
    assert not np.allclose(first[0], second[0], atol=1e-9)


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------


def test_zero_noise_gives_infinite_epsilon():
    """No noise is no privacy; a large finite number here would be a lie."""
    assert compute_epsilon(0.0, 0.5, 20) == float("inf")


def test_zero_rounds_gives_zero_epsilon():
    assert compute_epsilon(1.0, 0.5, 0) == 0.0


def test_epsilon_matches_known_accountant_value():
    """Regression pin against dp_accounting's RDP accountant."""
    eps = compute_epsilon(noise_multiplier=1.0, sampling_rate=0.5, rounds=20, delta=1e-5)
    assert eps == pytest.approx(16.55, abs=0.05)


def test_epsilon_grows_with_rounds():
    """Every round composes more privacy loss."""
    values = [compute_epsilon(1.0, 0.5, r) for r in (1, 5, 20, 50)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_epsilon_shrinks_as_noise_grows():
    """More noise must buy a tighter guarantee."""
    values = [compute_epsilon(z, 0.5, 20) for z in (0.5, 1.0, 2.0, 4.0)]
    assert values == sorted(values, reverse=True)


def test_epsilon_shrinks_as_sampling_rate_falls():
    """Privacy amplification by subsampling."""
    values = [compute_epsilon(1.0, q, 20) for q in (1.0, 0.5, 0.1)]
    assert values == sorted(values, reverse=True)


def test_epsilon_is_larger_at_smaller_delta():
    assert compute_epsilon(1.0, 0.5, 20, delta=1e-6) > compute_epsilon(1.0, 0.5, 20, delta=1e-4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rounds": -1}, "rounds must be >= 0"),
        ({"delta": 0.0}, "delta must be in"),
        ({"delta": 1.0}, "delta must be in"),
        ({"sampling_rate": 0.0}, "sampling_rate must be in"),
        ({"sampling_rate": 1.5}, "sampling_rate must be in"),
    ],
)
def test_epsilon_argument_validation(kwargs, message):
    base = {"noise_multiplier": 1.0, "sampling_rate": 0.5, "rounds": 10, "delta": 1e-5}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        compute_epsilon(**base)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_aggregator_returns_plain_fedavg_when_dp_disabled():
    agg = make_aggregator(
        dp_enabled=False, noise_multiplier=0.0, l2_clip_norm=1.0, clients_per_round=5
    )
    assert isinstance(agg, FedAvgAggregator)
    assert agg.name == "fedavg"


def test_make_aggregator_returns_dp_when_enabled():
    agg = make_aggregator(
        dp_enabled=True, noise_multiplier=1.0, l2_clip_norm=1.0, clients_per_round=5
    )
    assert isinstance(agg, DPFedAvgAggregator)
    assert agg.name == "dp-fedavg"


def test_fedavg_aggregator_matches_weighted_average():
    updates = [_update("a", 1.0, 10), _update("b", 2.0, 100), _update("c", 3.0, 1000)]
    agg = FedAvgAggregator()
    np.testing.assert_allclose(
        agg.aggregate(updates, [np.zeros((2, 2), np.float32)])[0],
        weighted_average(updates)[0],
        rtol=0,
    )


# -- noise calibration -------------------------------------------------------


class TestCalibrateNoiseMultiplier:
    def test_round_trips_the_recorded_moderate_setting(self):
        """z=2.0 at q=0.5, 20 rounds, delta=1e-5 is the recorded eps=6.228 run.

        Calibrating back from that epsilon must recover z ~= 2.0 -- a
        hand-checkable anchor tying the inverse to a known (z, eps) pair.
        """
        from fl.aggregation import calibrate_noise_multiplier, compute_epsilon

        target = compute_epsilon(2.0, 0.5, 20, 1e-5)
        z = calibrate_noise_multiplier(target, 0.5, 20, 1e-5)
        assert abs(z - 2.0) < 0.01
        # And the mechanism it names really achieves the budget it was asked for.
        assert abs(compute_epsilon(z, 0.5, 20, 1e-5) - target) <= 1e-4 * target

    def test_lower_sampling_rate_needs_less_noise_at_same_budget(self):
        """Privacy amplification: q=0.01 should need far smaller z than q=0.5."""
        from fl.aggregation import calibrate_noise_multiplier

        z_small_q = calibrate_noise_multiplier(6.228, 0.01, 20, 1e-5)
        z_big_q = calibrate_noise_multiplier(6.228, 0.5, 20, 1e-5)
        assert z_small_q < z_big_q

    def test_tighter_budget_needs_more_noise(self):
        from fl.aggregation import calibrate_noise_multiplier

        assert calibrate_noise_multiplier(1.0, 0.5, 20) > calibrate_noise_multiplier(6.0, 0.5, 20)

    def test_unreachable_target_raises(self):
        from fl.aggregation import calibrate_noise_multiplier

        with pytest.raises(ValueError, match="outside the reachable range"):
            calibrate_noise_multiplier(1e12, 0.5, 20)

    def test_nonpositive_target_rejected(self):
        from fl.aggregation import calibrate_noise_multiplier

        with pytest.raises(ValueError, match="must be > 0"):
            calibrate_noise_multiplier(0.0, 0.5, 20)


@pytest.mark.slow
class TestDPAcrossThreads:
    def test_second_dp_aggregation_in_a_fresh_thread_succeeds(self):
        """Audit finding C2 (docs/audit_v0_2.md): TFF's context stack is
        thread-local, so before the per-thread context guard the process's
        second DP aggregation on a fresh thread died with "No default context
        installed". The coordinator runs every training run on its own
        thread, so this is exactly one-DP-run-per-process without the fix."""
        import threading

        results: list[Exception | None] = []

        def run_one() -> None:
            try:
                agg = DPFedAvgAggregator(
                    noise_multiplier=1.0, l2_clip_norm=1.0, clients_per_round=2
                )
                template = [np.zeros((2, 2), dtype=np.float32)]
                agg.aggregate(
                    [
                        ClientUpdate("a", [np.ones((2, 2), dtype=np.float32)], 1),
                        ClientUpdate("b", [np.ones((2, 2), dtype=np.float32)], 1),
                    ],
                    template,
                )
                results.append(None)
            except Exception as exc:  # noqa: BLE001 - the exception IS the assertion
                results.append(exc)

        for _ in range(2):
            thread = threading.Thread(target=run_one)
            thread.start()
            thread.join(timeout=120)

        assert results == [None, None], results
