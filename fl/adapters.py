"""Framework adapters: native model weights <-> canonical wire form.

Every conversion between a framework's native tensor layout and the canonical
layout documented in ``fl/proto/fl_comm.proto`` lives here and nowhere else.
The server never converts anything (canonical form is what it aggregates); a
client converts exactly twice per round, at its own edge: canonical -> native
before local training, native -> canonical before submission.

The conversions that actually bite, and their direction out of PyTorch:

=============  ==========================  =============================
Tensor         PyTorch native              Canonical (= TF native)
=============  ==========================  =============================
Conv2D kernel  (out, in, h, w)             (h, w, in, out): permute(2,3,1,0)
Dense kernel   (out, in)                   (in, out): transpose
BatchNorm      weight / bias /             gamma / beta /
               running_mean / running_var  moving_mean / moving_variance
=============  ==========================  =============================

Biases are shape-identical in both frameworks. TensorFlow's adapter is the
identity on layout — that is the point of choosing TF's layout as canonical
(see the proto), not an accident.

All conversions are pure axis permutations and renames: float32 values are
preserved bit-for-bit, which the round-trip tests assert with exact equality
rather than tolerance.
"""

from __future__ import annotations

import numpy as np

from .archspec import ArchSpec, BatchNorm, Conv2D, Dense

Canonical = list[np.ndarray]
"""Weight list in canonical order and layout (== keras ``get_weights()``)."""


class AdapterError(ValueError):
    """Raised when weights do not match the spec the adapter was built for."""


class TFAdapter:
    """Keras <-> canonical. Layout-identity by design; validates shapes."""

    framework = "tensorflow"

    def __init__(self, spec: ArchSpec) -> None:
        self.spec = spec
        self._shapes = spec.canonical_shapes()

    def _check(self, weights: Canonical) -> None:
        got = [tuple(w.shape) for w in weights]
        if got != self._shapes:
            raise AdapterError(
                f"weights do not match spec {self.spec.name!r}: got {got}, expected {self._shapes}"
            )

    def to_canonical(self, model) -> Canonical:
        """Extract canonical weights from a Keras model built from this spec."""
        weights = [np.asarray(w, dtype=np.float32) for w in model.get_weights()]
        self._check(weights)
        return weights

    def from_canonical(self, model, weights: Canonical) -> None:
        """Load canonical weights into a Keras model built from this spec."""
        self._check(weights)
        model.set_weights(weights)


class TorchAdapter:
    """PyTorch <-> canonical. Owns every transpose and rename; nothing leaks."""

    framework = "torch"

    def __init__(self, spec: ArchSpec) -> None:
        self.spec = spec
        self._shapes = spec.canonical_shapes()

    def to_canonical(self, net) -> Canonical:
        """Extract canonical weights from a torch module built from this spec."""
        state = {**dict(net.named_parameters()), **dict(net.named_buffers())}
        out: Canonical = []
        for layer in self.spec.layers:
            if isinstance(layer, Conv2D):
                kernel = state[f"blocks.{layer.name}.weight"]
                bias = state[f"blocks.{layer.name}.bias"]
                # (out, in, h, w) -> (h, w, in, out)
                out.append(
                    np.ascontiguousarray(
                        kernel.detach().numpy().transpose(2, 3, 1, 0), dtype=np.float32
                    )
                )
                out.append(np.asarray(bias.detach().numpy(), dtype=np.float32))
            elif isinstance(layer, Dense):
                kernel = state[f"blocks.{layer.name}.weight"]
                bias = state[f"blocks.{layer.name}.bias"]
                # (out, in) -> (in, out)
                out.append(np.ascontiguousarray(kernel.detach().numpy().T, dtype=np.float32))
                out.append(np.asarray(bias.detach().numpy(), dtype=np.float32))
            elif isinstance(layer, BatchNorm):
                prefix = f"blocks.{layer.name}"
                # weight->gamma, bias->beta, running_mean->moving_mean,
                # running_var->moving_variance, in canonical order.
                for key in ("weight", "bias", "running_mean", "running_var"):
                    out.append(
                        np.asarray(state[f"{prefix}.{key}"].detach().numpy(), dtype=np.float32)
                    )
        got = [tuple(w.shape) for w in out]
        if got != self._shapes:
            raise AdapterError(
                f"torch weights do not match spec {self.spec.name!r}: got {got}, "
                f"expected {self._shapes}"
            )
        return out

    def from_canonical(self, net, weights: Canonical) -> None:
        """Load canonical weights into a torch module built from this spec."""
        import torch

        got = [tuple(w.shape) for w in weights]
        if got != self._shapes:
            raise AdapterError(
                f"canonical weights do not match spec {self.spec.name!r}: got {got}, "
                f"expected {self._shapes}"
            )
        it = iter(weights)
        with torch.no_grad():
            for layer in self.spec.layers:
                block = getattr(net.blocks, layer.name, None) if hasattr(net, "blocks") else None
                if isinstance(layer, Conv2D):
                    kernel, bias = next(it), next(it)
                    # (h, w, in, out) -> (out, in, h, w)
                    block.weight.copy_(torch.from_numpy(kernel.transpose(3, 2, 0, 1).copy()))
                    block.bias.copy_(torch.from_numpy(bias))
                elif isinstance(layer, Dense):
                    kernel, bias = next(it), next(it)
                    block.weight.copy_(torch.from_numpy(kernel.T.copy()))
                    block.bias.copy_(torch.from_numpy(bias))
                elif isinstance(layer, BatchNorm):
                    gamma, beta, mean, var = next(it), next(it), next(it), next(it)
                    block.weight.copy_(torch.from_numpy(gamma))
                    block.bias.copy_(torch.from_numpy(beta))
                    block.running_mean.copy_(torch.from_numpy(mean))
                    block.running_var.copy_(torch.from_numpy(var))


def make_adapter(framework: str, spec: ArchSpec):
    """Adapter registry keyed by the client-facing framework name."""
    if framework == "tensorflow":
        return TFAdapter(spec)
    if framework == "torch":
        return TorchAdapter(spec)
    raise ValueError(f"unknown framework {framework!r}; expected 'tensorflow' or 'torch'")
