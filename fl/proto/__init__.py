"""Wire format for the federated control plane.

``fl_comm.proto`` is the single source of truth. The generated ``*_pb2.py`` and
``*_pb2_grpc.py`` stubs are **not** committed -- committed stubs drift from their
schema, and reviewers cannot tell which of the two is authoritative.

Instead :func:`ensure_generated` regenerates them on demand, and is called at
import time below. It is a no-op once the stubs exist and are newer than the
``.proto``, so it costs one ``stat`` call in the steady state.
"""

from __future__ import annotations

from ._generate import ensure_generated

ensure_generated()

from . import fl_comm_pb2, fl_comm_pb2_grpc  # noqa: E402  (must follow generation)

PROTOCOL_VERSION = fl_comm_pb2.PROTOCOL_VERSION_V1

__all__ = ["fl_comm_pb2", "fl_comm_pb2_grpc", "ensure_generated", "PROTOCOL_VERSION"]
