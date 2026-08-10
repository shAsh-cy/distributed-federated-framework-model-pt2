"""FedProx local training (Li et al., MLSys 2020).

FedProx changes exactly one thing about FedAvg, and it is client-side: the
local objective gains a proximal term anchored at the round's global model,

    h_k(w) = F_k(w) + (mu/2) * ||w - w_global||^2

whose gradient contribution is ``mu * (w - w_global)`` -- a spring pulling
local training back toward where it started. The term vanishes at the anchor,
so the FIRST local step of a round is identical to FedAvg's; drift is taxed,
not forbidden. Convention note: the coefficient is the paper's ``mu/2`` form
(gradient ``mu * (w - w_global)``); a bare ``mu * ||.||^2`` objective is the
same family with mu rescaled by 2.

Aggregation is untouched: FedProx clients are FedAvg clients that trained a
different local objective, which is why this lives beside the client
trainers rather than in :mod:`fl.aggregation`.

Performance shape: one :class:`TFProximalTrainer` is built per (model, mu)
and reused across every client and round -- the train step is a single
``tf.function`` traced once (batch dimension left free), with the anchor held
in ``tf.Variable`` slots re-assigned per ``fit`` call. A naive per-round
tape loop retraces per call and is unusably slow at m=200 clients.
"""

from __future__ import annotations

import numpy as np

Weights = list[np.ndarray]


class TFProximalTrainer:
    """A proximal training loop over a compiled Keras model.

    Contract mirrors ``model.fit``: the caller sets the round's global
    weights on the model first; ``fit`` snapshots them as the anchor, trains
    in place with the model's own optimizer, and returns final-epoch mean
    loss and accuracy. The reported loss is the full local objective
    (data term plus proximal term), matching Keras's convention of reporting
    the loss actually optimised.
    """

    def __init__(self, model: object, mu: float) -> None:
        import tensorflow as tf

        if mu <= 0:
            raise ValueError(f"mu must be > 0 for a proximal trainer, got {mu}")
        if getattr(model, "optimizer", None) is None:
            raise ValueError("model must be compiled (an optimizer is required)")

        self._model = model
        self.mu = float(mu)
        self._anchors = [
            tf.Variable(v, trainable=False, name=f"fedprox_anchor_{i}")
            for i, v in enumerate(model.trainable_variables)
        ]
        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        input_spec = tf.TensorSpec([None, *model.input_shape[1:]], tf.float32)
        label_spec = tf.TensorSpec([None], tf.int32)
        self._step = tf.function(self._train_step, input_signature=[input_spec, label_spec])

    def _train_step(self, x: object, y: object) -> object:
        import tensorflow as tf

        model = self._model
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            data_loss = self._loss_fn(y, logits)
            proximal = tf.add_n(
                [
                    tf.reduce_sum(tf.square(v - a))
                    for v, a in zip(model.trainable_variables, self._anchors, strict=True)
                ]
            )
            loss = data_loss + 0.5 * self.mu * proximal
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
        self._accuracy.update_state(y, logits)
        return loss

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch_size: int
    ) -> tuple[float, float]:
        """Train in place from the model's CURRENT weights, which become the anchor.

        Returns ``(loss, accuracy)`` over the final epoch, Keras-style.
        """
        for anchor, variable in zip(self._anchors, self._model.trainable_variables, strict=True):
            anchor.assign(variable)

        xs = np.asarray(x, dtype=np.float32)
        ys = np.asarray(y, dtype=np.int32)
        n = len(ys)
        epoch_losses: list[float] = []
        for _ in range(max(1, epochs)):
            self._accuracy.reset_state()
            epoch_losses.clear()
            permutation = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch = permutation[start : start + batch_size]
                loss = self._step(xs[batch], ys[batch])
                epoch_losses.append(float(loss))
        return float(np.mean(epoch_losses)), float(self._accuracy.result())
