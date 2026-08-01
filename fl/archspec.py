"""Framework-neutral architecture specification.

One spec object describes an architecture; :func:`build_tf` and
:func:`build_torch` construct the TensorFlow and PyTorch realisations of it.
Neither model is ever defined by hand in framework code — the spec is the
single source of truth, which is what makes the cross-framework equality
assertions in the tests meaningful rather than coincidental.

Canonical weight order and shapes follow the wire contract documented in
``fl/proto/fl_comm.proto``: Conv2D kernels ``(h, w, in, out)``, Dense kernels
``(in, out)``, BatchNorm as gamma/beta/moving_mean/moving_variance. The
PyTorch model's forward pass flattens convolutional feature maps in
``(height, width, channel)`` order — by permuting NCHW activations to NHWC
before flattening — so that a dense-after-flatten kernel means the same thing
in both frameworks. Without that permutation the two models would have
identical parameter counts, identical shapes after conversion, and completely
different functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_VALID_ACTIVATIONS = (None, "relu")


@dataclass(frozen=True)
class Conv2D:
    """2D convolution, VALID padding, stride 1."""

    filters: int
    kernel: int
    name: str
    activation: str | None = "relu"


@dataclass(frozen=True)
class MaxPool2D:
    """Non-overlapping max pool (kernel = stride = ``pool``), VALID padding."""

    name: str
    pool: int = 2


@dataclass(frozen=True)
class BatchNorm:
    """Channel-wise batch normalisation over the preceding conv's channels.

    ``epsilon`` is part of the layer's *function*, not an implementation
    detail: Keras defaults to 1e-3 and PyTorch to 1e-5, and leaving each
    framework its own default makes two models with identical weights compute
    measurably different outputs. The spec fixes one value for both.
    """

    name: str
    epsilon: float = 1e-3


@dataclass(frozen=True)
class Flatten:
    """Flatten a feature map in (height, width, channel) order."""

    name: str


@dataclass(frozen=True)
class Dense:
    """Fully connected layer."""

    units: int
    name: str
    activation: str | None = None


LayerSpec = Conv2D | MaxPool2D | BatchNorm | Flatten | Dense


@dataclass(frozen=True)
class ArchSpec:
    """A complete architecture: input shape (H, W, C) plus an ordered layer list."""

    name: str
    input_shape: tuple[int, int, int]
    layers: tuple[LayerSpec, ...]

    def __post_init__(self) -> None:
        for layer in self.layers:
            act = getattr(layer, "activation", None)
            if act not in _VALID_ACTIVATIONS:
                raise ValueError(f"unsupported activation {act!r} on layer {layer.name!r}")
        # Walking the shapes validates layer compatibility eagerly.
        self.shape_walk()

    def shape_walk(self) -> list[tuple[int, ...]]:
        """Activation shape after every layer, starting from ``input_shape``.

        Feature maps are tracked as (H, W, C); flattened activations as (F,).
        """
        shapes: list[tuple[int, ...]] = []
        current: tuple[int, ...] = self.input_shape
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                h, w, _c = current
                current = (h - layer.kernel + 1, w - layer.kernel + 1, layer.filters)
            elif isinstance(layer, MaxPool2D):
                h, w, c = current
                current = (h // layer.pool, w // layer.pool, c)
            elif isinstance(layer, BatchNorm):
                if len(current) != 3:
                    raise ValueError(f"BatchNorm {layer.name!r} must follow a feature map")
            elif isinstance(layer, Flatten):
                current = (int(np.prod(current)),)
            elif isinstance(layer, Dense):
                if len(current) != 1:
                    raise ValueError(f"Dense {layer.name!r} must follow a flatten or dense")
                current = (layer.units,)
            shapes.append(current)
        return shapes

    def canonical_names(self) -> list[str]:
        """Wire tensor names, in canonical (= keras ``get_weights``) order."""
        names: list[str] = []
        for layer in self.layers:
            if isinstance(layer, Conv2D | Dense):
                names += [f"{layer.name}/kernel", f"{layer.name}/bias"]
            elif isinstance(layer, BatchNorm):
                names += [
                    f"{layer.name}/gamma",
                    f"{layer.name}/beta",
                    f"{layer.name}/moving_mean",
                    f"{layer.name}/moving_variance",
                ]
        return names

    def canonical_shapes(self) -> list[tuple[int, ...]]:
        """Canonical shape of every weight tensor, matching :meth:`canonical_names`."""
        shapes: list[tuple[int, ...]] = []
        current: tuple[int, ...] = self.input_shape
        for layer, out_shape in zip(self.layers, self.shape_walk(), strict=True):
            if isinstance(layer, Conv2D):
                shapes += [
                    (layer.kernel, layer.kernel, current[-1], layer.filters),
                    (layer.filters,),
                ]
            elif isinstance(layer, BatchNorm):
                shapes += [(current[-1],)] * 4
            elif isinstance(layer, Dense):
                shapes += [(current[0], layer.units), (layer.units,)]
            current = out_shape
        return shapes

    def parameter_count(self) -> int:
        """Total scalar parameters (trainable + BatchNorm moving statistics)."""
        return int(sum(np.prod(s) for s in self.canonical_shapes()))


# ---------------------------------------------------------------------------
# The shipped architectures, defined once
# ---------------------------------------------------------------------------


def _cnn_spec(name: str, num_classes: int) -> ArchSpec:
    return ArchSpec(
        name=name,
        input_shape=(28, 28, 1),
        layers=(
            Conv2D(32, 3, "conv1"),
            MaxPool2D("pool1"),
            Conv2D(64, 3, "conv2"),
            MaxPool2D("pool2"),
            Flatten("flatten"),
            Dense(128, "dense1", activation="relu"),
            Dense(num_classes, "logits"),
        ),
    )


SMALL_CNN_SPEC = _cnn_spec("small_cnn", 10)
FEMNIST_CNN_SPEC = _cnn_spec("femnist_cnn", 62)

SPECS: dict[str, ArchSpec] = {"small_cnn": SMALL_CNN_SPEC, "femnist_cnn": FEMNIST_CNN_SPEC}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_tf(spec: ArchSpec, seed: int | None = None):
    """Construct the TensorFlow/Keras realisation of ``spec``."""
    import tensorflow as tf

    init = (
        (lambda: tf.keras.initializers.GlorotUniform(seed=seed))
        if seed is not None
        else (lambda: "glorot_uniform")
    )
    layers: list = [tf.keras.layers.Input(shape=spec.input_shape)]
    for layer in spec.layers:
        if isinstance(layer, Conv2D):
            layers.append(
                tf.keras.layers.Conv2D(
                    layer.filters,
                    layer.kernel,
                    activation=layer.activation,
                    kernel_initializer=init(),
                    name=layer.name,
                )
            )
        elif isinstance(layer, MaxPool2D):
            layers.append(tf.keras.layers.MaxPooling2D(layer.pool, name=layer.name))
        elif isinstance(layer, BatchNorm):
            layers.append(
                tf.keras.layers.BatchNormalization(epsilon=layer.epsilon, name=layer.name)
            )
        elif isinstance(layer, Flatten):
            layers.append(tf.keras.layers.Flatten(name=layer.name))
        elif isinstance(layer, Dense):
            layers.append(
                tf.keras.layers.Dense(
                    layer.units,
                    activation=layer.activation,
                    kernel_initializer=init(),
                    name=layer.name,
                )
            )
    return tf.keras.Sequential(layers, name=spec.name)


def build_torch(spec: ArchSpec):
    """Construct the PyTorch realisation of ``spec``.

    The module's ``forward`` takes NCHW float32 tensors (PyTorch's native
    activation layout) and flattens feature maps in NHWC element order — see
    the module docstring for why that permutation is load-bearing.
    """
    import torch
    from torch import nn

    class SpecNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            mods: dict[str, nn.Module] = {}
            channels = spec.input_shape[-1]
            features = None
            for layer, shape in zip(spec.layers, spec.shape_walk(), strict=True):
                if isinstance(layer, Conv2D):
                    mods[layer.name] = nn.Conv2d(channels, layer.filters, layer.kernel)
                    channels = layer.filters
                elif isinstance(layer, MaxPool2D):
                    mods[layer.name] = nn.MaxPool2d(layer.pool)
                elif isinstance(layer, BatchNorm):
                    mods[layer.name] = nn.BatchNorm2d(channels, eps=layer.epsilon)
                elif isinstance(layer, Flatten):
                    features = shape[0]
                elif isinstance(layer, Dense):
                    assert features is not None, "Dense before Flatten is rejected by the spec"
                    mods[layer.name] = nn.Linear(features, layer.units)
                    features = layer.units
            self.blocks = nn.ModuleDict(mods)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.spec.layers:
                if isinstance(layer, Conv2D):
                    x = self.blocks[layer.name](x)
                    if layer.activation == "relu":
                        x = torch.relu(x)
                elif isinstance(layer, MaxPool2D | BatchNorm):
                    x = self.blocks[layer.name](x)
                elif isinstance(layer, Flatten):
                    # NCHW -> NHWC before flattening, so the element order
                    # matches the canonical dense kernel's input indexing.
                    x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
                elif isinstance(layer, Dense):
                    x = self.blocks[layer.name](x)
                    if layer.activation == "relu":
                        x = torch.relu(x)
            return x

    return SpecNet()
