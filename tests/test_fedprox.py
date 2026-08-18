"""FedProx: the proximal term does what the math says, on both frameworks.

Two properties pin the implementation:

* **The term vanishes at the anchor.** The proximal gradient is
  ``mu * (w - w_global)``, which is exactly zero at the round's starting
  weights -- so the FIRST local step must be identical to plain FedAvg's,
  for any mu. An implementation that clips, scales or otherwise mangles the
  first step fails this.
* **The term binds away from the anchor.** With a large mu, local training
  must stay near the global model: the drift norm shrinks by orders of
  magnitude versus mu=0 on the same data from the same start.

The wire test (test_server_rounds.py) covers mu travelling in
GetGlobalModelResponse; config validation lives in test_config.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.aggregation import l2_norm, subtract


def _toy_data(seed: int = 0, n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 28, 28, 1)).astype(np.float32)
    y = rng.integers(0, 10, size=n).astype(np.int32)
    return x, y


def _start_weights() -> list[np.ndarray]:
    from fl.models import build_model

    return build_model("small_cnn", seed=11).get_weights()


# ---------------------------------------------------------------------------
# TensorFlow
# ---------------------------------------------------------------------------


class TestTFProximal:
    def test_first_step_from_anchor_matches_a_plain_gradient_step(self):
        """One full-batch step: prox trainer output == hand-driven plain SGD.

        The proximal gradient mu*(w - anchor) is exactly zero at the anchor,
        so even a large mu must not change step one.
        """
        import tensorflow as tf

        from fl.fedprox import TFProximalTrainer
        from fl.models import build_model, compile_for_training

        x, y = _toy_data()
        start = _start_weights()

        prox_model = compile_for_training(build_model("small_cnn"), 0.05, 0.0)
        prox_model.set_weights(start)
        TFProximalTrainer(prox_model, mu=10.0).fit(x, y, epochs=1, batch_size=len(y))

        plain_model = compile_for_training(build_model("small_cnn"), 0.05, 0.0)
        plain_model.set_weights(start)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            loss = loss_fn(y, plain_model(x, training=True))
        grads = tape.gradient(loss, plain_model.trainable_variables)
        plain_model.optimizer.apply_gradients(
            zip(grads, plain_model.trainable_variables, strict=True)
        )

        for a, b in zip(prox_model.get_weights(), plain_model.get_weights(), strict=True):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_large_mu_pins_local_training_to_the_anchor(self):
        """Five epochs of drift, mu=0 versus mu=10: the proximal arm must
        stay far closer to the global model.

        mu=10 with lr=0.05 keeps the proximal dynamics stable (the spring's
        per-step contraction is lr*mu = 0.5 < 2); a much larger mu would
        overshoot the anchor and diverge -- which is a property of SGD on a
        stiff quadratic, not a FedProx implementation bug.
        """
        from fl.client import _TFTrainer

        x, y = _toy_data()
        start = _start_weights()

        free_w, _, _ = _TFTrainer("small_cnn", 0.05, 0.9).fit(start, x, y, epochs=5, batch_size=16)
        pinned_w, _, _ = _TFTrainer("small_cnn", 0.05, 0.9).fit(
            start, x, y, epochs=5, batch_size=16, proximal_mu=10.0
        )

        free_drift = l2_norm(subtract(free_w, start))
        pinned_drift = l2_norm(subtract(pinned_w, start))
        assert np.isfinite(pinned_drift)
        assert pinned_drift < free_drift / 5.0
        assert pinned_drift > 0.0  # taxed, not forbidden

    def test_mu_zero_uses_the_untouched_keras_path(self):
        from fl.client import _TFTrainer

        trainer = _TFTrainer("small_cnn", 0.05, 0.9)
        x, y = _toy_data()
        trainer.fit(_start_weights(), x, y, epochs=1, batch_size=32)
        assert trainer._prox is None

    def test_prox_trainer_rejects_bad_construction(self):
        from fl.fedprox import TFProximalTrainer
        from fl.models import build_model, compile_for_training

        model = compile_for_training(build_model("small_cnn"), 0.05, 0.0)
        with pytest.raises(ValueError, match="mu must be > 0"):
            TFProximalTrainer(model, mu=0.0)
        with pytest.raises(ValueError, match="compiled"):
            TFProximalTrainer(build_model("small_cnn"), mu=0.1)

    def test_prox_trainer_reuses_one_traced_step_across_fits(self):
        """The tf.function must not retrace per fit call or per batch size:
        at m=200 clients x 20 rounds, retracing dominates run time. Two fits
        with different batch splits share one concrete trace."""
        from fl.fedprox import TFProximalTrainer
        from fl.models import build_model, compile_for_training

        x, y = _toy_data(n=48)
        model = compile_for_training(build_model("small_cnn"), 0.05, 0.0)
        model.set_weights(_start_weights())
        prox = TFProximalTrainer(model, mu=1.0)
        prox.fit(x, y, epochs=1, batch_size=32)  # batches of 32 and 16
        # Keras may trace twice on the very first call (variable-creating
        # initialisation trace); what must NOT happen is growth after that.
        after_first = prox._step.experimental_get_tracing_count()
        prox.fit(x, y, epochs=1, batch_size=20)  # batches of 20 and 8
        prox.fit(x, y, epochs=3, batch_size=7)
        assert prox._step.experimental_get_tracing_count() == after_first


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------


class TestTorchProximal:
    def test_first_step_from_anchor_matches_plain_fedavg_step(self):
        """epochs=1, one full batch: mu>0 and mu=0 take the identical step."""
        import torch

        from fl.client import _TorchTrainer

        x, y = _toy_data()
        start = _start_weights()

        torch.manual_seed(3)
        plain_w, _, _ = _TorchTrainer("small_cnn", 0.05, 0.0).fit(
            start, x, y, epochs=1, batch_size=len(y)
        )
        torch.manual_seed(3)
        prox_w, _, _ = _TorchTrainer("small_cnn", 0.05, 0.0).fit(
            start, x, y, epochs=1, batch_size=len(y), proximal_mu=10.0
        )
        for a, b in zip(plain_w, prox_w, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_large_mu_pins_local_training_to_the_anchor(self):
        import torch

        from fl.client import _TorchTrainer

        x, y = _toy_data()
        start = _start_weights()

        torch.manual_seed(3)
        free_w, _, _ = _TorchTrainer("small_cnn", 0.05, 0.9).fit(
            start, x, y, epochs=5, batch_size=16
        )
        torch.manual_seed(3)
        pinned_w, _, _ = _TorchTrainer("small_cnn", 0.05, 0.9).fit(
            start, x, y, epochs=5, batch_size=16, proximal_mu=10.0
        )

        free_drift = l2_norm(subtract(free_w, start))
        pinned_drift = l2_norm(subtract(pinned_w, start))
        assert np.isfinite(pinned_drift)
        assert pinned_drift < free_drift / 5.0
        assert pinned_drift > 0.0
