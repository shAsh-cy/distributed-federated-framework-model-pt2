"""Keras model definitions.

Only one architecture is shipped: a small CNN sized so that a full federated run
completes on CPU in minutes rather than hours. Federated learning experiments are
bottlenecked by rounds x clients, not by single-model capacity, so a larger
backbone would buy accuracy at the cost of making the system untestable.

Parameter count (see :data:`SMALL_CNN_PARAMS`, asserted in tests):

===========================  ==============================  ===========
Layer                        Shape                           Parameters
===========================  ==============================  ===========
Conv2D(32, 3x3)              (3*3*1)*32 + 32                       320
Conv2D(64, 3x3)              (3*3*32)*64 + 64                   18,496
Dense(128)                   1600*128 + 128                    204,928
Dense(10)                    128*10 + 10                         1,290
---------------------------  ------------------------------  -----------
**Total**                                                     **225,034**
===========================  ==============================  ===========

Spatial reduction: 28x28 -> conv(valid) 26x26 -> pool 13x13 -> conv(valid) 11x11
-> pool 5x5 -> flatten 5*5*64 = 1600.

At float32 that is 225,034 * 4 = 900,136 bytes, so one model transfer is ~0.86 MiB
in each direction per client per round. That figure is what the server's
bytes-transferred metric should be sanity-checked against.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

INPUT_SHAPE: tuple[int, int, int] = (28, 28, 1)
NUM_CLASSES: int = 10
FEMNIST_CLASSES: int = 62

#: Exact trainable parameter count of :func:`build_small_cnn`.
SMALL_CNN_PARAMS: int = 225_034

#: Exact trainable parameter count of :func:`build_femnist_cnn`. Identical
#: backbone to ``small_cnn``; only the logits layer widens (128*62 + 62 = 7,998
#: parameters instead of 1,290), so 225,034 - 1,290 + 7,998 = 231,742.
FEMNIST_CNN_PARAMS: int = 231_742


def build_small_cnn(seed: int | None = None) -> tf.keras.Model:
    """Build the Fashion-MNIST CNN (10 classes).

    Constructed from the framework-neutral spec in :mod:`fl.archspec` — the
    same spec the PyTorch twin is built from — with layer order, names and
    initialisers identical to the original hand-written Sequential, so a
    given seed still produces bit-identical initial weights.
    """
    from .archspec import SMALL_CNN_SPEC, build_tf

    return build_tf(SMALL_CNN_SPEC, seed)


def build_femnist_cnn(seed: int | None = None) -> tf.keras.Model:
    """Build the FEMNIST CNN (62 classes: 10 digits, 26+26 letters).

    Deliberately the same backbone as ``small_cnn`` rather than the much larger
    LEAF reference CNN (~6.6M parameters): DP noise scales with sqrt(d), CPU
    round time scales with d, and keeping d within 3% of the Fashion-MNIST model
    makes noise magnitudes directly comparable across the two datasets.
    Constructed from :data:`fl.archspec.FEMNIST_CNN_SPEC`.
    """
    from .archspec import FEMNIST_CNN_SPEC, build_tf

    return build_tf(FEMNIST_CNN_SPEC, seed)


_BUILDERS = {"small_cnn": build_small_cnn, "femnist_cnn": build_femnist_cnn}


def build_model(name: str = "small_cnn", seed: int | None = None) -> tf.keras.Model:
    """Build a model by config name (``small_cnn`` or ``femnist_cnn``)."""
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {sorted(_BUILDERS)}") from None
    return builder(seed=seed)


def count_parameters(model: tf.keras.Model) -> int:
    """Total trainable parameter count."""
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def weights_nbytes(weights: list[np.ndarray]) -> int:
    """Serialised size in bytes of a weight list, ignoring protobuf framing."""
    return int(sum(np.asarray(w).nbytes for w in weights))


def compile_for_training(
    model: tf.keras.Model, learning_rate: float, momentum: float
) -> tf.keras.Model:
    """Compile with the client-side optimiser (plain SGD with momentum)."""
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def compile_for_evaluation(model: tf.keras.Model) -> tf.keras.Model:
    """Compile for server-side evaluation only; the optimiser is never used."""
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
