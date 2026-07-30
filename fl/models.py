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

#: Exact trainable parameter count of :func:`build_small_cnn`.
SMALL_CNN_PARAMS: int = 225_034


def build_small_cnn(seed: int | None = None) -> tf.keras.Model:
    """Build the Fashion-MNIST CNN.

    Args:
        seed: If given, every weight initialiser is seeded from it, so two calls
            with the same seed produce bit-identical initial weights. The server
            relies on this to construct the initial global model deterministically.

    Returns:
        An uncompiled Keras model. Compilation is the caller's job because the
        server (evaluation only) and the clients (training) want different
        optimisers.
    """
    init = (
        (lambda: tf.keras.initializers.GlorotUniform(seed=seed))
        if seed is not None
        else (lambda: "glorot_uniform")
    )
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(
                32, 3, activation="relu", kernel_initializer=init(), name="conv1"
            ),
            tf.keras.layers.MaxPooling2D(2, name="pool1"),
            tf.keras.layers.Conv2D(
                64, 3, activation="relu", kernel_initializer=init(), name="conv2"
            ),
            tf.keras.layers.MaxPooling2D(2, name="pool2"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init(), name="dense1"),
            tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=init(), name="logits"),
        ],
        name="small_cnn",
    )


_BUILDERS = {"small_cnn": build_small_cnn}


def build_model(name: str = "small_cnn", seed: int | None = None) -> tf.keras.Model:
    """Build a model by config name."""
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
