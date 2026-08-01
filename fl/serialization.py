"""Conversion between Keras weight lists and the ``ModelWeights`` wire message.

Weights travel as raw little-endian float32 buffers with an explicit shape rather
than as ``repeated float``. For the 225,034-parameter model that is 900,136 bytes
of payload; ``repeated float`` would pay per-element varint framing on top.

Every decode is validated. A buffer whose length disagrees with its declared
shape is rejected rather than reshaped into whatever happens to fit, because a
truncated transfer that silently becomes a differently-shaped tensor would be
aggregated into the global model without complaint.
"""

from __future__ import annotations

import numpy as np

from .proto import fl_comm_pb2

Weights = list[np.ndarray]

#: Wire dtype. Fixed rather than negotiated: Keras models here are float32
#: throughout, and allowing per-tensor dtypes would add a conversion matrix for
#: no benefit.
WIRE_DTYPE = np.dtype("<f4")


class SerializationError(ValueError):
    """Raised when a weight payload cannot be decoded."""


def weights_to_proto(weights: Weights, names: list[str] | None = None) -> fl_comm_pb2.ModelWeights:
    """Encode a Keras weight list as a ``ModelWeights`` message."""
    if names is not None and len(names) != len(weights):
        raise SerializationError(f"got {len(names)} names for {len(weights)} tensors")
    msg = fl_comm_pb2.ModelWeights()
    for i, w in enumerate(weights):
        arr = np.ascontiguousarray(np.asarray(w, dtype=WIRE_DTYPE))
        tensor = msg.tensors.add()
        tensor.name = names[i] if names is not None else f"t{i}"
        tensor.shape.extend(int(d) for d in arr.shape)
        tensor.data = arr.tobytes()
        tensor.dtype = fl_comm_pb2.TENSOR_DTYPE_FLOAT32
    return msg


def proto_to_weights(msg: fl_comm_pb2.ModelWeights) -> Weights:
    """Decode a ``ModelWeights`` message, validating every tensor."""
    out: Weights = []
    for i, tensor in enumerate(msg.tensors):
        if tensor.dtype != fl_comm_pb2.TENSOR_DTYPE_FLOAT32:
            raise SerializationError(
                f"tensor {i} ({tensor.name!r}) declares unsupported dtype {tensor.dtype}; "
                "only TENSOR_DTYPE_FLOAT32 is supported and payloads are rejected rather "
                "than coerced"
            )
        shape = tuple(int(d) for d in tensor.shape)
        if any(d < 0 for d in shape):
            raise SerializationError(f"tensor {i} ({tensor.name!r}) has negative dimension {shape}")

        expected_elements = int(np.prod(shape)) if shape else 1
        expected_bytes = expected_elements * WIRE_DTYPE.itemsize
        if len(tensor.data) != expected_bytes:
            raise SerializationError(
                f"tensor {i} ({tensor.name!r}) declares shape {shape} "
                f"({expected_bytes} bytes) but carries {len(tensor.data)} bytes"
            )
        out.append(np.frombuffer(tensor.data, dtype=WIRE_DTYPE).reshape(shape).copy())
    return out


def proto_nbytes(msg: fl_comm_pb2.ModelWeights) -> int:
    """Serialised size of a weight message, including protobuf framing.

    This is what the server reports as bytes transferred; it is the real
    on-the-wire cost, not the sum of the tensor payloads.
    """
    return msg.ByteSize()


def weights_fingerprint(weights: Weights) -> bytes:
    """Order-sensitive digest of a weight list, used to prove round-trip fidelity."""
    import hashlib

    h = hashlib.sha256()
    for w in weights:
        arr = np.ascontiguousarray(np.asarray(w, dtype=WIRE_DTYPE))
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.digest()
