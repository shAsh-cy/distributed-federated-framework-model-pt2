"""Event store and live pub/sub hub.

Ordering and durability contract, in one place:

* ``append`` assigns the next contiguous per-run sequence number, persists the
  event, and only then broadcasts it to live subscribers. A late-connecting
  client that replays from the database and then follows the live stream can
  therefore never observe a gap: anything it missed live is already on disk.
* Appends are serialised per store by a lock; the unique (run_id, seq)
  constraint in the schema turns any future concurrency bug into a loud
  IntegrityError instead of silent event loss.
* Subscribers get a bounded queue (default 1024 events). A consumer that
  falls further behind than that is disconnected rather than allowed to grow
  the queue without bound — backpressure by eviction. This is safe precisely
  because of the replay contract: the evicted client reconnects with
  ``since=<last seq it processed>`` and loses nothing.

The hub bridges the runner's worker thread to asyncio WebSocket consumers via
``loop.call_soon_threadsafe``; the training loop never blocks on a slow
browser (it does not even know one exists).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from .db import EventRow, Run, make_session_factory
from .events import TERMINAL_TYPES, Event


@dataclass
class Subscription:
    run_id: str
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1024))
    #: Set when the hub evicted this subscriber for falling behind.
    evicted: bool = False


class EventStore:
    """Persistence plus fan-out. One instance per process."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._sessions = make_session_factory(engine)
        self._lock = threading.Lock()
        self._next_seq: dict[str, int] = {}
        self._subs: dict[str, list[Subscription]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- wiring --------------------------------------------------------------

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once from the ASGI startup hook; the hub needs the loop to
        hand events from the runner thread to async consumers."""
        self._loop = loop

    # -- runs ----------------------------------------------------------------

    def create_run(
        self,
        config: dict,
        source: str = "live",
        label: str = "",
        group_key: str | None = None,
        is_aggregate: bool = False,
        seed: int | None = None,
        status: str = "pending",
    ) -> str:
        run_id = str(uuid.uuid4())
        with self._sessions() as s:
            s.add(
                Run(
                    id=run_id,
                    created_at=time.time(),
                    status=status,
                    config_json=json.dumps(config),
                    source=source,
                    label=label,
                    group_key=group_key,
                    is_aggregate=is_aggregate,
                    seed=seed,
                )
            )
            s.commit()
        return run_id

    def set_status(self, run_id: str, status: str, final_metrics: dict | None = None) -> None:
        with self._sessions() as s:
            run = s.get(Run, run_id)
            if run is None:
                raise KeyError(run_id)
            run.status = status
            if final_metrics is not None:
                run.final_metrics_json = json.dumps(final_metrics)
            s.commit()

    def get_run(self, run_id: str) -> Run | None:
        with self._sessions() as s:
            run = s.get(Run, run_id)
            if run is not None:
                s.expunge(run)
            return run

    def list_runs(self) -> list[Run]:
        with self._sessions() as s:
            runs = s.execute(select(Run).order_by(Run.created_at.desc())).scalars().all()
            for r in runs:
                s.expunge(r)
            return runs

    # -- events --------------------------------------------------------------

    def append(self, event: Event) -> Event:
        """Assign seq, persist, then broadcast. Returns the sequenced event."""
        with self._lock:
            seq = self._next_seq.get(event.run_id)
            if seq is None:
                seq = self._load_next_seq(event.run_id)
            event = event.copy(update={"seq": seq})
            payload = event.dict()
            with self._sessions() as s:
                s.add(
                    EventRow(
                        run_id=event.run_id,
                        seq=seq,
                        type=event.type,
                        ts=event.ts,
                        payload_json=json.dumps(payload),
                    )
                )
                s.commit()
            self._next_seq[event.run_id] = seq + 1
        self._broadcast(event.run_id, payload)
        return event

    def _load_next_seq(self, run_id: str) -> int:
        with self._sessions() as s:
            row = s.execute(
                select(EventRow.seq)
                .where(EventRow.run_id == run_id)
                .order_by(EventRow.seq.desc())
                .limit(1)
            ).scalar()
            return 0 if row is None else int(row) + 1

    def events_since(self, run_id: str, since: int = 0) -> list[dict]:
        """All persisted events with seq >= since, in order."""
        with self._sessions() as s:
            rows = (
                s.execute(
                    select(EventRow)
                    .where(EventRow.run_id == run_id, EventRow.seq >= since)
                    .order_by(EventRow.seq)
                )
                .scalars()
                .all()
            )
            return [r.payload() for r in rows]

    def is_terminal(self, run_id: str) -> bool:
        """True when the run's stream already ended in a terminal event."""
        with self._sessions() as s:
            last_type = s.execute(
                select(EventRow.type)
                .where(EventRow.run_id == run_id)
                .order_by(EventRow.seq.desc())
                .limit(1)
            ).scalar()
            return last_type in TERMINAL_TYPES

    # -- pub/sub -------------------------------------------------------------

    def subscribe(self, run_id: str) -> Subscription:
        sub = Subscription(run_id=run_id)
        with self._lock:
            self._subs.setdefault(run_id, []).append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            subs = self._subs.get(sub.run_id, [])
            if sub in subs:
                subs.remove(sub)

    def _broadcast(self, run_id: str, payload: dict) -> None:
        loop = self._loop
        if loop is None:
            return  # no async consumers wired (importer, tests without WS)
        with self._lock:
            subs = list(self._subs.get(run_id, []))
        for sub in subs:
            loop.call_soon_threadsafe(self._deliver, sub, payload)

    def _deliver(self, sub: Subscription, payload: dict) -> None:
        """Runs on the event loop. Bounded queue; evict rather than balloon."""
        try:
            sub.queue.put_nowait(payload)
        except asyncio.QueueFull:
            sub.evicted = True
            self.unsubscribe(sub)
            # Wake the consumer so its read loop notices the eviction instead
            # of blocking forever on a queue that will never be fed again.
            try:
                sub.queue.get_nowait()
                sub.queue.put_nowait(payload)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                pass
