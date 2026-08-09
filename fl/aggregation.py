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

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

LOGGER = logging.getLogger("fl.aggregation")

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


def _ensure_tff_context() -> None:
    """Install a TFF execution context for the CURRENT thread if it has none.

    TFF's context stack is ``threading.local``; the default context installed
    at import time exists only in the importing thread. Any DP aggregation on
    a fresh thread — the coordinator runs each training run on one — would
    otherwise die with ``RuntimeError: No default context installed`` on the
    process's second DP run (audit finding C2, docs/audit_v0_2.md).
    """
    import tensorflow_federated as tff
    from tensorflow_federated.python.core.impl.context_stack import runtime_error_context

    current = tff.framework.get_context_stack().current
    if isinstance(current, runtime_error_context.RuntimeErrorContext):
        tff.backends.native.set_sync_local_cpp_execution_context()


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

        _ensure_tff_context()
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


class AdaptiveDPFedAvgAggregator:
    """Client-level DP FedAvg with quantile-based adaptive clipping.

    Wraps ``tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive``
    (Andrew et al. 2021): the clipping norm starts at ``initial_l2_clip_norm``
    and geometrically tracks the ``target_quantile`` of the actual client
    update norms, adapting by at most ``exp(learning_rate)`` per round.

    Privacy accounting: the quantile estimate itself consumes budget. TF
    Privacy splits the nominal noise multiplier ``z`` into an inflated value
    noise ``z_v`` on the sum and a Gaussian on the clipped count with stddev
    ``sigma_b`` (sensitivity 1/2), such that ``z^-2 = z_v^-2 + (2*sigma_b)^-2``
    -- the joint release is exactly a Gaussian mechanism at the nominal ``z``,
    so :func:`compute_epsilon` over the nominal multiplier remains the correct
    total. :func:`adaptive_noise_breakdown` reports the split, and the test
    suite anchors it to a hand-computed case.

    The adapted clip is exposed per round: ``current_clip`` after each
    ``aggregate`` call, and the full trajectory in ``clip_history`` -- so the
    adaptation can be plotted against the measured median update norm.
    """

    name = "adaptive-dp-fedavg"

    def __init__(
        self,
        noise_multiplier: float,
        initial_l2_clip_norm: float,
        clients_per_round: int,
        target_quantile: float = 0.5,
        learning_rate: float = 0.2,
        clipped_count_stddev: float | None = None,
    ) -> None:
        if noise_multiplier <= 0:
            raise ValueError(
                f"noise_multiplier must be > 0 for a DP aggregator, got {noise_multiplier}"
            )
        if initial_l2_clip_norm <= 0:
            raise ValueError(f"initial_l2_clip_norm must be > 0, got {initial_l2_clip_norm}")
        if clients_per_round < 1:
            raise ValueError(f"clients_per_round must be >= 1, got {clients_per_round}")
        if not 0.0 < target_quantile < 1.0:
            raise ValueError(f"target_quantile must be in (0, 1), got {target_quantile}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        # Validates the split is achievable (raises otherwise):
        adaptive_noise_breakdown(noise_multiplier, clients_per_round, clipped_count_stddev)

        self.noise_multiplier = float(noise_multiplier)
        self.initial_l2_clip_norm = float(initial_l2_clip_norm)
        self.clients_per_round = int(clients_per_round)
        self.target_quantile = float(target_quantile)
        self.learning_rate = float(learning_rate)
        self.clipped_count_stddev = clipped_count_stddev
        #: Clip in force for the NEXT round (starts at the initial estimate).
        self.current_clip: float = float(initial_l2_clip_norm)
        #: Adapted clip after each aggregate() call, in order.
        self.clip_history: list[float] = []
        self._process = None
        self._state = None
        self._value_type = None

    def _ensure_process(self, template: Weights) -> None:
        import tensorflow_federated as tff

        _ensure_tff_context()
        value_type = tff.to_type(
            [tff.TensorType(np.float32, np.asarray(w).shape) for w in template]
        )
        if self._process is not None and value_type == self._value_type:
            return
        kwargs = {}
        if self.clipped_count_stddev is not None:
            kwargs["clipped_count_stddev"] = float(self.clipped_count_stddev)
        factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
            noise_multiplier=self.noise_multiplier,
            clients_per_round=float(self.clients_per_round),
            initial_l2_norm_clip=self.initial_l2_clip_norm,
            target_unclipped_quantile=self.target_quantile,
            learning_rate=self.learning_rate,
            **kwargs,
        )
        self._value_type = value_type
        self._process = factory.create(value_type)
        self._state = self._process.initialize()

    def aggregate(self, updates: list[ClientUpdate], global_weights: Weights) -> Weights:
        """Clip adaptively, noise, average the deltas; record the new clip."""
        validate_updates(updates)
        deltas = [subtract(u.weights, global_weights) for u in updates]
        self._ensure_process(global_weights)

        output = self._process.next(self._state, deltas)
        self._state = output.state
        # measurements['dp_query_metrics']['clip'] is the clip the quantile
        # estimator derived from THIS round, in force for the next one.
        metrics = output.measurements.get("dp_query_metrics", {})
        clip = metrics.get("clip")
        if clip is not None:
            self.current_clip = float(clip)
            self.clip_history.append(self.current_clip)
            LOGGER.info("adaptive clip after round %d: %.6f", len(self.clip_history), clip)
        mean_delta = [np.asarray(t, dtype=np.float32) for t in output.result]
        return add(global_weights, mean_delta)


def adaptive_noise_breakdown(
    noise_multiplier: float,
    clients_per_round: int,
    clipped_count_stddev: float | None = None,
) -> dict:
    """The privacy-budget split behind adaptive clipping, made explicit.

    TF Privacy's ``QuantileAdaptiveClipSumQuery`` spends part of the nominal
    budget on the quantile estimate: the clipped-count bit (sensitivity 1/2
    per client) is noised with stddev ``sigma_b`` -- an effective Gaussian
    multiplier of ``2 * sigma_b`` -- and the value noise is inflated to
    ``z_v = (z^-2 - (2 sigma_b)^-2)^(-1/2)`` so the two compose back to
    exactly the nominal ``z`` by Gaussian sigma-additivity
    (``z^-2 = z_v^-2 + (2 sigma_b)^-2``, Andrew et al. 2021). dp_accounting
    sees this structure directly: TFF's aggregator state carries a
    ComposedDpEvent of the two GaussianDpEvents, verified empirically.

    Consequence for reporting: total epsilon for an adaptive run is
    :func:`compute_epsilon` at the NOMINAL multiplier -- unchanged from the
    fixed-clip path -- and this function reports where that budget goes.

    Raises:
        ValueError: when ``2 * sigma_b <= z``, which would demand negative
            value-noise variance; the quantile estimate would be consuming
            more than the whole budget.
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be > 0, got {noise_multiplier}")
    sigma_b = (
        float(clipped_count_stddev)
        if clipped_count_stddev is not None
        else clients_per_round / 20.0
    )
    count_multiplier = 2.0 * sigma_b  # sensitivity of the clipped bit is 1/2
    if count_multiplier <= noise_multiplier:
        raise ValueError(
            f"clipped_count_stddev={sigma_b} spends the entire budget on the quantile "
            f"estimate: need 2*stddev > noise_multiplier ({noise_multiplier}). Raise the "
            "count stddev (slower adaptation) or lower the nominal noise."
        )
    value_multiplier = (noise_multiplier**-2 - count_multiplier**-2) ** -0.5
    return {
        "nominal_noise_multiplier": float(noise_multiplier),
        "value_noise_multiplier": float(value_multiplier),
        "count_noise_multiplier": float(count_multiplier),
        "clipped_count_stddev": float(sigma_b),
    }


def calibrate_noise_multiplier(
    target_epsilon: float,
    sampling_rate: float,
    rounds: int,
    delta: float = 1e-5,
    rel_tol: float = 1e-4,
    z_low: float = 1e-3,
    z_high: float = 200.0,
) -> float:
    """Find the noise multiplier that achieves ``target_epsilon``, by bisection.

    The inverse of :func:`compute_epsilon` in its ``noise_multiplier`` argument,
    which is strictly decreasing (more noise, less epsilon). Needed whenever a
    sweep holds epsilon fixed while the sampling rate varies: with a fixed
    population N, raising the cohort m raises q = m/N, which weakens privacy
    amplification by subsampling and demands a different z for the same budget.

    Epsilon remains a computed quantity, never a knob: this function chooses the
    *mechanism* (z) and the reported epsilon is still whatever the accountant
    says that mechanism achieves. Callers should recompute and report
    ``compute_epsilon(z, ...)`` for the returned z rather than quoting the
    target.

    Raises:
        ValueError: if the target lies outside what ``[z_low, z_high]`` can
            reach at these parameters.
    """
    if target_epsilon <= 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")

    def eps(z: float) -> float:
        return compute_epsilon(z, sampling_rate, rounds, delta)

    eps_at_high, eps_at_low = eps(z_high), eps(z_low)
    if not eps_at_high <= target_epsilon <= eps_at_low:
        raise ValueError(
            f"target_epsilon {target_epsilon} is outside the reachable range "
            f"[{eps_at_high:.4g}, {eps_at_low:.4g}] for z in [{z_low}, {z_high}] "
            f"at q={sampling_rate}, rounds={rounds}, delta={delta}"
        )

    lo, hi = z_low, z_high  # eps(lo) >= target >= eps(hi)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        value = eps(mid)
        if abs(value - target_epsilon) <= rel_tol * target_epsilon:
            return mid
        if value > target_epsilon:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def make_aggregator(
    *,
    dp_enabled: bool,
    noise_multiplier: float,
    l2_clip_norm: float,
    clients_per_round: int,
    adaptive_clipping: bool = False,
    adaptive_target_quantile: float = 0.5,
    adaptive_learning_rate: float = 0.2,
    adaptive_clipped_count_stddev: float | None = None,
) -> Aggregator:
    """Build the aggregator a config asks for. Fixed clipping is the default;
    the adaptive path is opt-in and l2_clip_norm becomes its initial estimate."""
    if not dp_enabled:
        return FedAvgAggregator()
    if adaptive_clipping:
        return AdaptiveDPFedAvgAggregator(
            noise_multiplier=noise_multiplier,
            initial_l2_clip_norm=l2_clip_norm,
            clients_per_round=clients_per_round,
            target_quantile=adaptive_target_quantile,
            learning_rate=adaptive_learning_rate,
            clipped_count_stddev=adaptive_clipped_count_stddev,
        )
    return DPFedAvgAggregator(
        noise_multiplier=noise_multiplier,
        l2_clip_norm=l2_clip_norm,
        clients_per_round=clients_per_round,
    )
