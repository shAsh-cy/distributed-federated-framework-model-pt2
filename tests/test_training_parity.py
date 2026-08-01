"""Training-step parity between the TF and torch client paths.

The adapter suite proves the *forward* passes agree; nothing there proves the
*training* steps do. Keras and torch SGD differ in where the learning rate
enters the momentum recursion (Keras folds lr into the velocity, torch applies
it at the update), which is trajectory-equivalent only at constant lr, and
torch's SGD carries a weight_decay argument Keras's does not. These tests pin
the actual behaviour:

* both trainers start from the received global weights (no reinitialisation);
* one-and-multi-step SGD+momentum updates match across frameworks on an
  identical full batch, within float32 tolerance;
* the torch optimiser is constructed without weight decay or Nesterov, the
  Keras defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.adapters import TFAdapter
from fl.archspec import SMALL_CNN_SPEC, build_tf
from fl.client import _TFTrainer, _TorchTrainer

pytestmark = pytest.mark.slow

RNG = np.random.default_rng(17)


def _batch(n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    x = RNG.random((n, 28, 28, 1)).astype(np.float32)
    y = RNG.integers(0, 10, size=n).astype(np.int64)
    return x, y


def _global_weights(seed: int = 3):
    return TFAdapter(SMALL_CNN_SPEC).to_canonical(build_tf(SMALL_CNN_SPEC, seed=seed))


class TestTrainsFromReceivedWeights:
    """With lr = 0 a trainer that starts from the received weights returns them
    bit-for-bit; a trainer that reinitialised could not."""

    def test_tf_trainer_starts_from_global_weights(self):
        weights = _global_weights()
        x, y = _batch()
        out, _loss, _acc = _TFTrainer("small_cnn", 0.0, 0.9).fit(
            [w.copy() for w in weights], x, y, epochs=2, batch_size=8
        )
        for a, b in zip(weights, out, strict=True):
            assert np.array_equal(a, b)

    def test_torch_trainer_starts_from_global_weights(self):
        weights = _global_weights()
        x, y = _batch()
        out, _loss, _acc = _TorchTrainer("small_cnn", 0.0, 0.9).fit(
            [w.copy() for w in weights], x, y, epochs=2, batch_size=8
        )
        for a, b in zip(weights, out, strict=True):
            assert np.array_equal(a, b)


class TestSgdStepParity:
    """Identical weights, identical full batch, identical lr/momentum ->
    matching weights after k steps in both frameworks.

    Full batch (batch_size = n) removes shuffle-order differences, so any
    disagreement is optimiser arithmetic, not data ordering.
    """

    @pytest.mark.parametrize("epochs", [1, 3])
    def test_updates_match(self, epochs):
        weights = _global_weights()
        x, y = _batch(32)

        tf_out, _, _ = _TFTrainer("small_cnn", 0.05, 0.9).fit(
            [w.copy() for w in weights], x, y, epochs=epochs, batch_size=32
        )
        torch_out, _, _ = _TorchTrainer("small_cnn", 0.05, 0.9).fit(
            [w.copy() for w in weights], x, y, epochs=epochs, batch_size=32
        )

        # The step must be real before tolerances mean anything.
        moved = sum(float(np.linalg.norm(a - b)) for a, b in zip(tf_out, weights, strict=True))
        assert moved > 1e-2, "lr=0.05 step did not move the TF weights; test is vacuous"

        for name, a, b in zip(SMALL_CNN_SPEC.canonical_names(), tf_out, torch_out, strict=True):
            np.testing.assert_allclose(
                a, b, atol=5e-4, rtol=5e-4, err_msg=f"SGD trajectories diverged at {name}"
            )

    def test_torch_optimiser_matches_keras_defaults(self):
        trainer = _TorchTrainer("small_cnn", 0.01, 0.9)
        group = trainer._opt.param_groups[0]
        assert group["weight_decay"] == 0, "Keras SGD has no weight decay; torch must not either"
        assert group["nesterov"] is False, "Keras SGD defaults nesterov=False"
        assert group["momentum"] == pytest.approx(0.9)
        assert group["lr"] == pytest.approx(0.01)
