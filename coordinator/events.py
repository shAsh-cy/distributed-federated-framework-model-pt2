"""Versioned event schema for run observability.

The replay guarantee, which every other design choice serves: **full run state
is reconstructible from the event stream alone.** A frontend replays events
from sequence zero; it never polls for state, and there is no state endpoint
whose answer cannot be derived from the stream. Consequences:

* ``run_started`` carries everything static — the config, the client
  population, and per-client label histograms (the dashboard renders data
  heterogeneity per node and cannot compute it client-side; the raw data
  never leaves the clients).
* Every event carries the run id, a server-assigned contiguous sequence
  number (0-based, assigned at persistence, gap-free per run), and the schema
  version, so a consumer can detect both missed events and schema drift.
* Events are immutable facts, never rewritten. A crashed run gains a
  ``run_failed`` event; nothing is edited retroactively.

Pydantic v1 syntax throughout: TFF pins typing-extensions==4.5.*, which
excludes pydantic v2 (see requirements.txt).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

#: Bumped whenever an emitted event's meaning or fields change incompatibly.
#: Consumers must reject versions they do not know rather than guess.
SCHEMA_VERSION: int = 1


class _EventBase(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    #: Contiguous per-run sequence number, assigned by the store at persist
    #: time. -1 means "not yet persisted"; no consumer ever sees -1.
    seq: int = -1
    #: Unix timestamp, seconds. Assigned by the emitter.
    ts: float

    class Config:
        extra = "forbid"


class ClientInfo(BaseModel):
    client_id: str
    num_examples: int
    #: Per-class example counts on this client's shard, length = num_classes.
    label_histogram: list[int]

    class Config:
        extra = "forbid"


class RunStarted(_EventBase):
    type: Literal["run_started"] = "run_started"
    config: dict
    num_classes: int
    #: The full client population with per-client label histograms.
    clients: list[ClientInfo]


class RoundStarted(_EventBase):
    type: Literal["round_started"] = "round_started"
    round: int
    model_version: int


class ClientSampled(_EventBase):
    type: Literal["client_sampled"] = "client_sampled"
    round: int
    client_id: str
    #: Self-reported training framework; observability only, like the wire field.
    framework: str | None = None


class ClientReported(_EventBase):
    type: Literal["client_reported"] = "client_reported"
    round: int
    client_id: str
    num_examples: int
    local_accuracy: float | None = None
    local_loss: float | None = None
    wall_clock_seconds: float | None = None
    bytes: int | None = None


class ClientDropped(_EventBase):
    type: Literal["client_dropped"] = "client_dropped"
    round: int
    client_id: str
    reason: Literal["deadline", "disconnect", "stale_version", "invalid_payload", "stopped"]


class RoundAggregated(_EventBase):
    type: Literal["round_aggregated"] = "round_aggregated"
    round: int
    model_version: int
    global_accuracy: float | None = None
    global_loss: float | None = None
    bytes_sent: int | None = None
    bytes_received: int | None = None
    #: Cumulative epsilon after this round; None when DP is disabled.
    cumulative_epsilon: float | None = None
    #: The two fields that make a privacy panel meaningful rather than
    #: decorative: what the updates actually looked like against the clip.
    median_update_norm: float | None = None
    clipped_fraction: float | None = None


class RunCompleted(_EventBase):
    type: Literal["run_completed"] = "run_completed"
    final_accuracy: float | None = None
    final_loss: float | None = None
    rounds_completed: int
    #: True when the run ended via a stop request rather than exhausting its
    #: configured rounds.
    stopped_early: bool = False


class RunFailed(_EventBase):
    type: Literal["run_failed"] = "run_failed"
    error: str
    rounds_completed: int


Event = (
    RunStarted
    | RoundStarted
    | ClientSampled
    | ClientReported
    | ClientDropped
    | RoundAggregated
    | RunCompleted
    | RunFailed
)

_EVENT_TYPES: dict[str, type] = {
    "run_started": RunStarted,
    "round_started": RoundStarted,
    "client_sampled": ClientSampled,
    "client_reported": ClientReported,
    "client_dropped": ClientDropped,
    "round_aggregated": RoundAggregated,
    "run_completed": RunCompleted,
    "run_failed": RunFailed,
}

#: Event types that end a run. After one of these, the stream is complete.
TERMINAL_TYPES: tuple[str, ...] = ("run_completed", "run_failed")


class UnknownEventError(ValueError):
    """Raised when a payload names an event type or schema version we do not speak."""


def parse_event(payload: dict) -> Event:
    """Parse a stored/received payload into its typed event, strictly.

    Unknown types and unknown schema versions are errors, not passes: a
    consumer that silently skips what it does not understand will render a
    dashboard that is confidently wrong.
    """
    etype = payload.get("type")
    cls = _EVENT_TYPES.get(etype)
    if cls is None:
        raise UnknownEventError(f"unknown event type {etype!r}")
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UnknownEventError(
            f"unsupported schema_version {version!r}; this consumer speaks {SCHEMA_VERSION}"
        )
    return cls(**payload)


__all__ = [
    "SCHEMA_VERSION",
    "TERMINAL_TYPES",
    "ClientDropped",
    "ClientInfo",
    "ClientReported",
    "ClientSampled",
    "Event",
    "RoundAggregated",
    "RoundStarted",
    "RunCompleted",
    "RunFailed",
    "RunStarted",
    "UnknownEventError",
    "parse_event",
]
