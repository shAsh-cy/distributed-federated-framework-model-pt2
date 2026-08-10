"""The FedOpt server optimizers: identity, hand-computed Adam, state, Yogi.

The four claims a server-optimizer implementation must prove before any
training run is allowed to use it:

1. FedAvg-as-identity reproduces the existing FedAvg output EXACTLY -- not
   approximately -- so FedOpt is a strict generalisation, not a parallel path
   that drifts.
2. FedAdam matches a two-round hand-computed example, so the no-bias-correction
   variant of Algorithm 2 (Reddi et al. 2021) is what is actually implemented.
3. State persists across rounds and resets between runs.
4. Yogi's second moment differs from Adam's in the documented way, on both
   sides of the sign.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fl.aggregation import (
    ClientUpdate,
    FedAvgAggregator,
    FedOptAggregator,
    make_aggregator,
)
from fl.server_optimizer import (
    AdamServerOptimizer,
    SGDServerOptimizer,
    make_server_optimizer,
)


def _update(cid: str, value: float, n: int) -> ClientUpdate:
    return ClientUpdate(
        client_id=cid,
        weights=[
            np.full((2, 3), value, dtype=np.float32),
            np.full((4,), -value, dtype=np.float32),
        ],
        num_examples=n,
    )


def _scalar_delta(value: float) -> list[np.ndarray]:
    return [np.array([value], dtype=np.float64)]


# ---------------------------------------------------------------------------
# 1. FedAvg as the identity case
# ---------------------------------------------------------------------------


class TestFedAvgIdentity:
    def test_identity_server_optimizer_reproduces_fedavg_bit_exactly(self):
        """SGD(lr=1, momentum=0) wrapped in FedOpt == FedAvgAggregator, bitwise.

        np.array_equal, not allclose: the identity case is the compatibility
        contract for every existing non-DP result in the repo, and "close" is
        not the same model as "the same model".
        """
        rng = np.random.default_rng(7)
        updates = [
            ClientUpdate(
                f"c{i}",
                [
                    rng.normal(size=(5, 4)).astype(np.float32),
                    rng.normal(size=(9,)).astype(np.float32),
                ],
                num_examples=int(n),
            )
            for i, n in enumerate([10, 100, 1000, 37])
        ]
        global_weights = [
            rng.normal(size=(5, 4)).astype(np.float32),
            rng.normal(size=(9,)).astype(np.float32),
        ]

        plain = FedAvgAggregator().aggregate(updates, global_weights)
        identity = FedOptAggregator(SGDServerOptimizer(learning_rate=1.0, momentum=0.0))
        via_fedopt = identity.aggregate(updates, global_weights)

        for a, b in zip(plain, via_fedopt, strict=True):
            assert a.dtype == b.dtype == np.float32
            assert np.array_equal(a, b)

    def test_identity_holds_across_multiple_rounds(self):
        """No state may leak into the identity path: round 3 must still match."""
        identity = FedOptAggregator(SGDServerOptimizer(learning_rate=1.0, momentum=0.0))
        plain = FedAvgAggregator()
        global_weights = [np.zeros((2, 3), np.float32), np.zeros((4,), np.float32)]
        for value in (1.0, -2.5, 0.125):
            updates = [_update("a", value, 10), _update("b", 2 * value, 30)]
            expected = plain.aggregate(updates, global_weights)
            got = identity.aggregate(updates, global_weights)
            for a, b in zip(expected, got, strict=True):
                assert np.array_equal(a, b)
            global_weights = expected

    def test_make_aggregator_default_path_is_the_plain_fedavg_class(self):
        """The default config short-circuits to FedAvgAggregator itself --
        existing runs do not even pass through the FedOpt code."""
        agg = make_aggregator(
            dp_enabled=False, noise_multiplier=0.0, l2_clip_norm=1.0, clients_per_round=5
        )
        assert isinstance(agg, FedAvgAggregator)

    def test_make_aggregator_damped_fedavg_goes_through_fedopt(self):
        agg = make_aggregator(
            dp_enabled=False,
            noise_multiplier=0.0,
            l2_clip_norm=1.0,
            clients_per_round=5,
            server_learning_rate=0.5,
        )
        assert isinstance(agg, FedOptAggregator)
        assert agg.name == "fedavg"


# ---------------------------------------------------------------------------
# 2. FedAdam against a two-round hand-computed example
# ---------------------------------------------------------------------------


class TestFedAdamHandComputed:
    """lr=1, beta1=0.9, beta2=0.99, tau=0.1; v initialises to tau^2 = 0.01.

    Round 1, delta = 1.0:
        m1 = 0.9*0    + 0.1*1.0    = 0.1
        v1 = 0.99*0.01 + 0.01*1.0  = 0.0099 + 0.01 = 0.0199
        step1 = 0.1 / (sqrt(0.0199) + 0.1)

    Round 2, delta = -0.5:
        m2 = 0.9*0.1    + 0.1*(-0.5)  = 0.09 - 0.05      = 0.04
        v2 = 0.99*0.0199 + 0.01*0.25  = 0.019701 + 0.0025 = 0.022201
        sqrt(v2) = 0.149 EXACTLY (149^2 = 22201), so
        step2 = 0.04 / 0.249
    """

    def _two_steps(self) -> tuple[float, float]:
        opt = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1)
        step1 = opt.step(_scalar_delta(1.0))[0][0]
        step2 = opt.step(_scalar_delta(-0.5))[0][0]
        return float(step1), float(step2)

    def test_round_one_matches_hand_computation(self):
        step1, _ = self._two_steps()
        assert step1 == pytest.approx(0.1 / (math.sqrt(0.0199) + 0.1), rel=1e-12)

    def test_round_two_matches_hand_computation(self):
        _, step2 = self._two_steps()
        assert step2 == pytest.approx(0.04 / 0.249, rel=1e-12)

    def test_no_bias_correction_is_what_makes_round_one_small(self):
        """Bias-corrected Adam would take a step of magnitude ~lr on round one
        (m-hat = delta, v-hat = delta^2). The paper's variant takes ~0.41 here.
        This test fails if someone 'fixes' the missing correction."""
        step1, _ = self._two_steps()
        # Corrected: m-hat = 0.1/(1-0.9) = 1.0, v-hat = 0.01/(1-0.99) = 1.0
        # (taking v0 = 0 as textbook Adam does), step = 1/(sqrt(1)+0.1).
        bias_corrected = 1.0 / (math.sqrt(1.0) + 0.1)  # ~0.909
        assert abs(step1 - bias_corrected) > 0.4

    def test_fedadam_through_the_aggregator_applies_the_same_step(self):
        """End to end: one client whose update IS the delta (global = 0),
        aggregated through FedOptAggregator, lands exactly on step1."""
        opt = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1)
        agg = FedOptAggregator(opt)
        global_weights = [np.zeros((1,), np.float32)]
        updates = [ClientUpdate("a", [np.array([1.0], np.float32)], num_examples=5)]
        out = agg.aggregate(updates, global_weights)
        assert out[0][0] == pytest.approx(0.1 / (math.sqrt(0.0199) + 0.1), rel=1e-6)


# ---------------------------------------------------------------------------
# 3. State: persists across rounds, resets between runs
# ---------------------------------------------------------------------------


class TestServerOptimizerState:
    def test_momentum_accumulates_across_rounds(self):
        """FedAvgM, beta=0.9, lr=1, constant delta 1.0:
        v walks 1.0, 1.9, 2.71 -- the geometric series toward 1/(1-beta)=10."""
        opt = SGDServerOptimizer(learning_rate=1.0, momentum=0.9)
        steps = [float(opt.step(_scalar_delta(1.0))[0][0]) for _ in range(3)]
        assert steps == pytest.approx([1.0, 1.9, 2.71], rel=1e-12)

    def test_reset_restores_fresh_construction_behaviour(self):
        opt = SGDServerOptimizer(learning_rate=1.0, momentum=0.9)
        first = float(opt.step(_scalar_delta(1.0))[0][0])
        assert float(opt.step(_scalar_delta(1.0))[0][0]) != first
        opt.reset()
        assert float(opt.step(_scalar_delta(1.0))[0][0]) == first

    def test_adam_state_persists_and_resets(self):
        opt = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1)
        first = float(opt.step(_scalar_delta(1.0))[0][0])
        second = float(opt.step(_scalar_delta(1.0))[0][0])
        assert first != second
        opt.reset()
        assert float(opt.step(_scalar_delta(1.0))[0][0]) == first

    def test_two_aggregators_do_not_share_state(self):
        """Each run constructs its own aggregator; one run's momentum must
        never bleed into another's. Distinct instances, distinct trajectories."""

        def one_round(agg: FedOptAggregator) -> float:
            updates = [ClientUpdate("a", [np.array([1.0], np.float32)], num_examples=5)]
            return float(agg.aggregate(updates, [np.zeros((1,), np.float32)])[0][0])

        warmed = FedOptAggregator(SGDServerOptimizer(learning_rate=1.0, momentum=0.9))
        one_round(warmed)
        warmed_second = one_round(warmed)

        fresh = FedOptAggregator(SGDServerOptimizer(learning_rate=1.0, momentum=0.9))
        fresh_first = one_round(fresh)

        assert warmed_second == pytest.approx(1.9, rel=1e-6)
        assert fresh_first == pytest.approx(1.0, rel=1e-6)

    def test_aggregator_reset_forwards_to_the_optimizer(self):
        opt = SGDServerOptimizer(learning_rate=1.0, momentum=0.9)
        agg = FedOptAggregator(opt)
        updates = [ClientUpdate("a", [np.array([1.0], np.float32)], num_examples=5)]
        agg.aggregate(updates, [np.zeros((1,), np.float32)])
        agg.reset()
        assert opt._velocity is None


# ---------------------------------------------------------------------------
# 4. Yogi vs Adam: the documented second-moment difference
# ---------------------------------------------------------------------------


class TestYogiVersusAdam:
    """Adam interpolates v geometrically toward delta^2; Yogi moves additively
    by exactly (1-beta2)*delta^2 in the direction of the gap. Same m, same
    step rule, different v -- checked on both sides of sign(v - delta^2)."""

    def test_first_moments_are_identical(self):
        adam = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1)
        yogi = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1, yogi=True)
        adam.step(_scalar_delta(1.0))
        yogi.step(_scalar_delta(1.0))
        np.testing.assert_array_equal(adam._m[0], yogi._m[0])

    def test_second_moment_when_delta_squared_exceeds_v(self):
        """v0 = tau^2 = 0.01, delta = 1.0, so delta^2 = 1.0 > v:
        Yogi: v = 0.01 + 0.01*1.0 = 0.02  (additive, +(1-beta2)*delta^2)
        Adam: v = 0.99*0.01 + 0.01*1.0 = 0.0199."""
        adam = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1)
        yogi = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1, yogi=True)
        adam.step(_scalar_delta(1.0))
        yogi.step(_scalar_delta(1.0))
        assert float(yogi._v[0][0]) == pytest.approx(0.02, rel=1e-12)
        assert float(adam._v[0][0]) == pytest.approx(0.0199, rel=1e-12)

    def test_second_moment_when_delta_squared_is_below_v(self):
        """Continue each optimizer down its own trajectory with delta = 0.1.

        Yogi (from its v = 0.02, delta^2 = 0.01 < v, so sign is +1):
            v = 0.02 - 0.01*0.01 = 0.0199 -- v DECREASES, by the fixed
            increment (1-beta2)*delta^2, not by a fraction of the gap.
        Adam (from its v = 0.0199):
            v = 0.99*0.0199 + 0.01*0.01 = 0.019701 + 0.0001 = 0.019801.
        """
        yogi = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1, yogi=True)
        yogi.step(_scalar_delta(1.0))  # v -> 0.02
        yogi.step(_scalar_delta(0.1))  # v -> 0.02 - 0.01*0.01 = 0.0199
        assert float(yogi._v[0][0]) == pytest.approx(0.0199, rel=1e-12)

        adam = AdamServerOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.99, tau=0.1)
        adam.step(_scalar_delta(1.0))  # v -> 0.0199
        adam.step(_scalar_delta(0.1))  # v -> 0.99*0.0199 + 0.0001 = 0.019801
        assert float(adam._v[0][0]) == pytest.approx(0.019801, rel=1e-12)
        assert float(adam._v[0][0]) != pytest.approx(float(yogi._v[0][0]), rel=1e-6)

    def test_yogi_and_adam_produce_different_steps_after_a_burst(self):
        """After a burst of large deltas Adam's v inflates faster
        multiplicatively, so its steps shrink faster than Yogi's."""
        adam = AdamServerOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, tau=1e-3)
        yogi = AdamServerOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, tau=1e-3, yogi=True)
        for _ in range(5):
            a = adam.step(_scalar_delta(10.0))
            y = yogi.step(_scalar_delta(10.0))
        assert float(a[0][0]) != pytest.approx(float(y[0][0]), rel=1e-9)


# ---------------------------------------------------------------------------
# Factory dispatch and refusals
# ---------------------------------------------------------------------------


class TestFactories:
    def test_make_server_optimizer_dispatch(self):
        assert make_server_optimizer("fedavg").name == "fedavg"
        assert make_server_optimizer("fedavg").momentum == 0.0
        assert make_server_optimizer("fedavgm", momentum=0.9).name == "fedavgm"
        assert make_server_optimizer("fedadam", learning_rate=0.1).name == "fedadam"
        yogi = make_server_optimizer("fedyogi", learning_rate=0.1)
        assert yogi.name == "fedyogi"
        assert yogi.yogi is True

    def test_make_server_optimizer_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="unknown server optimizer"):
            make_server_optimizer("adam")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"learning_rate": -1.0}, "learning_rate"),
        ],
    )
    def test_sgd_rejects_bad_hyperparameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SGDServerOptimizer(**kwargs)

    def test_sgd_rejects_momentum_of_one(self):
        with pytest.raises(ValueError, match="momentum"):
            SGDServerOptimizer(learning_rate=1.0, momentum=1.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"learning_rate": 0.1, "beta1": 1.0}, "beta1"),
            ({"learning_rate": 0.1, "beta2": -0.1}, "beta2"),
            ({"learning_rate": 0.1, "tau": 0.0}, "tau"),
        ],
    )
    def test_adam_rejects_bad_hyperparameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            AdamServerOptimizer(**kwargs)

    def test_make_aggregator_builds_named_fedopt_aggregators(self):
        for name in ("fedavgm", "fedadam", "fedyogi"):
            agg = make_aggregator(
                dp_enabled=False,
                noise_multiplier=0.0,
                l2_clip_norm=1.0,
                clients_per_round=5,
                server_optimizer=name,
                server_learning_rate=0.1,
            )
            assert isinstance(agg, FedOptAggregator)
            assert agg.name == name

    def test_make_aggregator_refuses_fedopt_with_dp(self):
        """DP composes with the UNIFORM mean and a fixed sensitivity bound;
        the FedOpt path composes with the weighted mean. Combining them is
        refused loudly rather than silently producing unaccounted claims."""
        with pytest.raises(ValueError, match="not supported together"):
            make_aggregator(
                dp_enabled=True,
                noise_multiplier=1.0,
                l2_clip_norm=1.0,
                clients_per_round=5,
                server_optimizer="fedadam",
            )

    def test_make_aggregator_refuses_damped_fedavg_with_dp(self):
        with pytest.raises(ValueError, match="not supported together"):
            make_aggregator(
                dp_enabled=True,
                noise_multiplier=1.0,
                l2_clip_norm=1.0,
                clients_per_round=5,
                server_learning_rate=0.5,
            )
