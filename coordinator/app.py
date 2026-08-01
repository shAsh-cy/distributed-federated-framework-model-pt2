"""FastAPI application: REST endpoints plus the WebSocket event stream.

Protocol split (the architectural rule this service exists to embody):
gRPC + protobuf is the internal protocol between training clients and the
aggregator — binary weights, deadlines, staleness. HTTP + WebSocket + JSON is
the external surface for browsers and tooling — observe, start, stop, replay.
Neither leaks into the other: training clients never speak HTTP, and this
service never carries model weights.

WebSocket contract (`/runs/{id}/events?since=N`):

1. On connect, every persisted event with ``seq >= since`` is sent, in order.
2. The stream then continues live, still strictly ordered by ``seq``.
3. Reconnection is replay: send the last processed seq + 1 as ``since`` and
   the stream resumes without gaps or duplicates.
4. A consumer that falls > queue-bound behind is evicted (see store); it
   reconnects via (3) and has lost nothing.
5. After a terminal event (run_completed / run_failed) the server closes the
   socket; a connection to an already-terminal run replays and closes.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from . import capabilities
from .db import Run, create_all, make_engine
from .events import SCHEMA_VERSION, TERMINAL_TYPES
from .runner import Runner
from .store import EventStore


class RunSummary(BaseModel):
    id: str
    created_at: float
    status: str
    source: str
    label: str
    group_key: str | None = None
    is_aggregate: bool = False
    seed: int | None = None


class RunDetail(RunSummary):
    config: dict
    final_metrics: dict | None = None
    num_events: int


class StartRunRequest(BaseModel):
    config: dict


class StartRunResponse(BaseModel):
    run_id: str


class StopRunResponse(BaseModel):
    run_id: str
    stopping: bool
    detail: str


def _summary(run: Run) -> RunSummary:
    return RunSummary(
        id=run.id,
        created_at=run.created_at,
        status=run.status,
        source=run.source,
        label=run.label,
        group_key=run.group_key,
        is_aggregate=bool(run.is_aggregate),
        seed=run.seed,
    )


def create_app(store: EventStore | None = None, runner: Runner | None = None) -> FastAPI:
    """App factory. Tests inject a store over :memory: and a stub executor."""
    if store is None:
        engine = make_engine()
        create_all(engine)
        store = EventStore(engine)
    if runner is None:
        runner = Runner(store)

    app = FastAPI(
        title="Federated Learning Coordinator",
        description=__doc__,
        version=f"schema-v{SCHEMA_VERSION}",
    )
    app.state.store = store
    app.state.runner = runner

    @app.on_event("startup")
    async def _attach_loop() -> None:
        store.attach_loop(asyncio.get_running_loop())

    def get_store() -> EventStore:
        return app.state.store

    def get_runner() -> Runner:
        return app.state.runner

    # -- capabilities -------------------------------------------------------

    @app.get("/datasets")
    def list_datasets() -> list[dict]:
        return capabilities.datasets()

    @app.get("/algorithms")
    def list_algorithms() -> list[dict]:
        return capabilities.algorithms()

    @app.get("/architectures")
    def list_architectures() -> list[dict]:
        return capabilities.architectures()

    # -- runs ---------------------------------------------------------------

    @app.post("/runs", response_model=StartRunResponse, status_code=201)
    def start_run(body: StartRunRequest, r: Runner = Depends(get_runner)) -> StartRunResponse:
        from fl.config import ConfigError

        try:
            run_id = r.start(body.config)
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return StartRunResponse(run_id=run_id)

    @app.get("/runs", response_model=list[RunSummary])
    def list_runs(s: EventStore = Depends(get_store)) -> list[RunSummary]:
        return [_summary(r) for r in s.list_runs()]

    @app.get("/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str, s: EventStore = Depends(get_store)) -> RunDetail:
        run = s.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        events = s.events_since(run_id)
        return RunDetail(
            **_summary(run).dict(),
            config=run.config(),
            final_metrics=run.final_metrics(),
            num_events=len(events),
        )

    @app.post("/runs/{run_id}/stop", response_model=StopRunResponse)
    def stop_run(
        run_id: str, s: EventStore = Depends(get_store), r: Runner = Depends(get_runner)
    ) -> StopRunResponse:
        run = s.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        accepted = r.stop(run_id)
        if not accepted:
            return StopRunResponse(
                run_id=run_id,
                stopping=False,
                detail=f"run is not live (status={run.status}); nothing to stop",
            )
        return StopRunResponse(
            run_id=run_id,
            stopping=True,
            detail=(
                "stop requested; the run finishes its current client fit and ends "
                "gracefully (between rounds: cleanly; mid-round: remaining cohort "
                "dropped with reason=stopped, partial round not aggregated)"
            ),
        )

    # -- live stream --------------------------------------------------------

    @app.websocket("/runs/{run_id}/events")
    async def run_events(ws: WebSocket, run_id: str, since: int = 0) -> None:
        await ws.accept()
        if store.get_run(run_id) is None:
            await ws.close(code=4404, reason=f"unknown run {run_id}")
            return

        # Subscribe FIRST, then replay from the database: anything that lands
        # between the replay query and the live loop is caught by the queue and
        # deduplicated by seq. This ordering is what makes the no-gap claim true.
        sub = store.subscribe(run_id)
        try:
            replayed = store.events_since(run_id, since=since)
            last_seq = since - 1
            for payload in replayed:
                await ws.send_text(json.dumps(payload))
                last_seq = payload["seq"]
            if replayed and replayed[-1]["type"] in TERMINAL_TYPES:
                await ws.close(code=1000, reason="run already terminal")
                return
            if store.is_terminal(run_id):
                # Terminal event exists but sits before `since`; nothing more
                # will ever arrive. Close rather than hold the socket open.
                await ws.close(code=1000, reason="run already terminal")
                return

            while True:
                payload = await sub.queue.get()
                if sub.evicted:
                    await ws.close(code=4429, reason="consumer too slow; reconnect with ?since=")
                    return
                if payload["seq"] <= last_seq:
                    continue  # duplicate from the subscribe-then-replay overlap
                await ws.send_text(json.dumps(payload))
                last_seq = payload["seq"]
                if payload["type"] in TERMINAL_TYPES:
                    await ws.close(code=1000, reason="run finished")
                    return
        except WebSocketDisconnect:
            pass
        finally:
            store.unsubscribe(sub)

    return app
