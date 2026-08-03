"""Adaptive clipping: config, accounting breakdown, and adaptation behaviour.

Unit tests only, per the branch constraint — no training runs. The TFF
adaptive query's noise is drawn in its executor (unseedable, like the fixed
DP path), so behavioural tests use a near-zero count stddev... which the
budget-split arithmetic forbids for real privacy; the tests therefore use a
tiny noise multiplier with a small-but-valid count stddev and assert
tolerances, not exact values.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.aggregation import (
    AdaptiveDPFedAvgAggregator,
    ClientUpdate,
    DPFedAvgAggregator,
    FedAvgAggregator,
    adaptive_noise_breakdown,
    compute_epsilon,
    make_aggregator,
)
from fl.config import Config, ConfigError

pytestmark = pytest.mark.slow

TEMPLATE = [np.zeros(4, dtype=np.float32)]


def _updates_with_norms(norms: list[float]) -> list[ClientUpdate]:
    """One update per norm: a delta of exactly that L2 norm from zero weights."""
    return [
        ClientUpdate(f"c{i}", [np.full(4, n / 2.0, dtype=np.float32)], 100)
        for i, n in enumerate(norms)
    ]  # ||full(4, n/2)|| = sqrt(4 * n^2/4) = n


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_fixed_path_remains_the_default(self):
        cfg = Config.from_dict(
            {"privacy": {"enabled": True, "noise_multiplier": 2.0, "l2_clip_norm": 0.5}}
        )
        assert cfg.privacy.adaptive_clipping is False

    @pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_quantiles_rejected(self, quantile):
        with pytest.raises(ConfigError, match="adaptive_target_quantile"):
            Config.from_dict(
                {"privacy": {"adaptive_clipping": True, "adaptive_target_quantile": quantile}}
            )

    def test_invalid_learning_rate_rejected(self):
        with pytest.raises(ConfigError, match="adaptive_learning_rate"):
            Config.from_dict({"privacy": {"adaptive_learning_rate": 0.0}})

    def test_invalid_count_stddev_rejected(self):
        with pytest.raises(ConfigError, match="adaptive_clipped_count_stddev"):
            Config.from_dict({"privacy": {"adaptive_clipped_count_stddev": -1.0}})

    def test_valid_adaptive_config_accepted(self):
        cfg = Config.from_dict(
            {
                "privacy": {
                    "enabled": True,
                    "noise_multiplier": 2.0,
                    "l2_clip_norm": 0.1,
                    "adaptive_clipping": True,
                    "adaptive_target_quantile": 0.5,
                    "adaptive_learning_rate": 0.2,
                }
            }
        )
        assert cfg.privacy.adaptive_clipping is True


# ---------------------------------------------------------------------------
# Accounting breakdown
# ---------------------------------------------------------------------------


class TestNoiseBreakdown:
    def test_hand_checked_split_matches_tf_privacy(self):
        """z=2.0, m=50, default count stddev m/20=2.5.

        Hand computation: count multiplier = 2*2.5 = 5.0 (clipped-bit
        sensitivity 1/2); value multiplier = (2^-2 - 5^-2)^(-1/2)
        = (0.25 - 0.04)^(-1/2) = 0.21^(-1/2) = 2.182178902...
        TF Privacy's own state (probed empirically on TFF 0.87) carries
        exactly 2.182179 for this case; the identity z^-2 = z_v^-2 + z_b^-2
        is Andrew et al.'s sigma-additivity.
        """
        breakdown = adaptive_noise_breakdown(2.0, 50)
        assert breakdown["clipped_count_stddev"] == 2.5
        assert breakdown["count_noise_multiplier"] == 5.0
        assert breakdown["value_noise_multiplier"] == pytest.approx(2.1821789, abs=1e-6)
        # Sigma-additivity closes exactly:
        z_v = breakdown["value_noise_multiplier"]
        z_b = breakdown["count_noise_multiplier"]
        assert z_v**-2 + z_b**-2 == pytest.approx(2.0**-2, rel=1e-12)

    def test_total_epsilon_is_the_nominal_mechanisms(self):
        """The joint release composes back to a Gaussian at the nominal z, so
        total epsilon equals compute_epsilon at the nominal multiplier — the
        recorded eps=6.228 anchor holds unchanged under adaptive clipping."""
        anchor = compute_epsilon(2.0, 0.5, 20, 1e-5)
        assert anchor == pytest.approx(6.2284173, abs=1e-6)
        breakdown = adaptive_noise_breakdown(2.0, 50)
        recombined = (
            breakdown["value_noise_multiplier"] ** -2 + breakdown["count_noise_multiplier"] ** -2
        ) ** -0.5
        assert compute_epsilon(recombined, 0.5, 20, 1e-5) == pytest.approx(anchor, rel=1e-9)

    def test_component_epsilons_are_reportable_separately(self):
        """dp_accounting CAN separate the parts: each component is a Gaussian
        event it prices individually. The parts are necessarily looser than
        the sigma-additive total (naive composition over-counts), which is
        why the total is reported at the nominal z, not by composing these."""
        breakdown = adaptive_noise_breakdown(2.0, 50)
        eps_value = compute_epsilon(breakdown["value_noise_multiplier"], 0.5, 20, 1e-5)
        eps_count = compute_epsilon(breakdown["count_noise_multiplier"], 0.5, 20, 1e-5)
        eps_total = compute_epsilon(2.0, 0.5, 20, 1e-5)
        assert eps_value < eps_total  # each part alone is cheaper than the whole
        assert eps_count < eps_value  # the quantile estimate is the smaller spend
        assert eps_value + eps_count > eps_total  # naive addition over-counts

    def test_overspending_count_budget_raises(self):
        with pytest.raises(ValueError, match="entire budget"):
            adaptive_noise_breakdown(2.0, 50, clipped_count_stddev=0.9)  # 2*0.9 <= 2.0

    def test_aggregator_constructor_validates_the_split(self):
        with pytest.raises(ValueError, match="entire budget"):
            AdaptiveDPFedAvgAggregator(
                noise_multiplier=2.0,
                initial_l2_clip_norm=0.1,
                clients_per_round=50,
                clipped_count_stddev=0.5,
            )


# ---------------------------------------------------------------------------
# Adaptation behaviour
# ---------------------------------------------------------------------------


def _run_rounds(agg: AdaptiveDPFedAvgAggregator, norms: list[float], rounds: int) -> None:
    for _ in range(rounds):
        agg.aggregate(_updates_with_norms(norms), [np.zeros(4, dtype=np.float32)])


class TestAdaptation:
    def test_clip_tracks_the_median_of_a_known_distribution(self):
        """Norms fixed at linspace(0.5, 1.5): median 1.0. Starting from a clip
        of 0.1, forty rounds of geometric adaptation must bring the clip into
        the neighbourhood of the median."""
        agg = AdaptiveDPFedAvgAggregator(
            noise_multiplier=0.01,  # tiny value noise; adaptation is the subject
            initial_l2_clip_norm=0.1,
            clients_per_round=10,
            target_quantile=0.5,
            learning_rate=0.2,
            clipped_count_stddev=0.05,  # small but valid: 2*0.05 > 0.01
        )
        norms = list(np.linspace(0.5, 1.5, 10))
        _run_rounds(agg, norms, 40)
        assert 0.7 < agg.current_clip < 1.4, agg.current_clip
        assert len(agg.clip_history) == 40

    def test_adaptation_is_stable_under_a_step_change(self):
        """Norm magnitude steps 1.0 -> 5.0: the clip must follow to the new
        median and then settle rather than oscillate."""
        agg = AdaptiveDPFedAvgAggregator(
            noise_multiplier=0.01,
            initial_l2_clip_norm=0.1,
            clients_per_round=10,
            clipped_count_stddev=0.05,
        )
        _run_rounds(agg, list(np.linspace(0.5, 1.5, 10)), 40)
        before = agg.current_clip
        _run_rounds(agg, list(np.linspace(2.5, 7.5, 10)), 50)
        after = agg.current_clip
        assert after > before * 2, "clip did not follow the step change"
        assert 3.5 < after < 7.0, after
        # Stability: over the last ten rounds the clip stays in a tight band.
        tail = agg.clip_history[-10:]
        assert max(tail) / min(tail) < 1.25, tail

    def test_clip_trajectory_is_exposed_for_plotting(self):
        agg = AdaptiveDPFedAvgAggregator(
            noise_multiplier=0.01,
            initial_l2_clip_norm=0.5,
            clients_per_round=5,
            clipped_count_stddev=0.05,
        )
        _run_rounds(agg, [1.0] * 5, 3)
        assert len(agg.clip_history) == 3
        assert agg.clip_history[-1] == agg.current_clip


# ---------------------------------------------------------------------------
# The fixed path is untouched
# ---------------------------------------------------------------------------


class TestFixedPathUnchanged:
    def test_default_dispatch_is_the_fixed_aggregator(self):
        agg = make_aggregator(
            dp_enabled=True, noise_multiplier=2.0, l2_clip_norm=0.5, clients_per_round=5
        )
        assert isinstance(agg, DPFedAvgAggregator)
        assert agg.name == "dp-fedavg"

    def test_adaptive_dispatch_is_opt_in(self):
        agg = make_aggregator(
            dp_enabled=True,
            noise_multiplier=2.0,
            l2_clip_norm=0.1,
            clients_per_round=50,
            adaptive_clipping=True,
        )
        assert isinstance(agg, AdaptiveDPFedAvgAggregator)
        assert agg.initial_l2_clip_norm == 0.1

    def test_no_dp_dispatch_unchanged(self):
        assert isinstance(
            make_aggregator(
                dp_enabled=False, noise_multiplier=0.0, l2_clip_norm=1.0, clients_per_round=5
            ),
            FedAvgAggregator,
        )

    def test_fixed_aggregator_has_no_adaptation_state(self):
        agg = DPFedAvgAggregator(noise_multiplier=2.0, l2_clip_norm=0.5, clients_per_round=5)
        assert not hasattr(agg, "clip_history")
        assert not hasattr(agg, "current_clip")
