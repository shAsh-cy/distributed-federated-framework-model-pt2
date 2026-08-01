"""Coordinator API: the HTTP/WebSocket observability and control surface.

Architectural rule, stated once and enforced by structure: **gRPC is the
internal client-to-aggregator protocol; HTTP and WebSocket are the external
surface.** Training clients speak protobuf to the aggregator and never touch
this package; browsers and tooling speak JSON to this package and never touch
the training protocol. The API consumes events the training loop pushes — it
never controls the loop's timing (see :mod:`coordinator.runner`).
"""

__version__ = "0.1.0"
