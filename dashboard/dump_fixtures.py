"""Dump real coordinator API responses as dashboard fixtures.

Run from the repo root inside the API-enabled image:

    python dashboard/dump_fixtures.py

Imports the committed history (97 runs) into an in-memory store, boots the
real FastAPI app, and records actual endpoint responses plus real event
streams for representative runs. These files are what MSW serves in tests and
mock mode — recorded reality, not invented shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from fastapi.testclient import TestClient  # noqa: E402

from coordinator.app import create_app  # noqa: E402
from coordinator.db import create_all, make_engine  # noqa: E402
from coordinator.importer import import_history  # noqa: E402
from coordinator.store import EventStore  # noqa: E402

OUT = Path("dashboard/fixtures")


def main() -> int:
    engine = make_engine(":memory:")
    create_all(engine)
    store = EventStore(engine)
    report = import_history(store, root=".")
    print("imported:", report["imported_runs"])

    OUT.mkdir(parents=True, exist_ok=True)
    app = create_app(store=store)
    with TestClient(app) as client:
        for name in ("datasets", "algorithms", "architectures", "runs"):
            payload = client.get(f"/{name}").json()
            (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
            print(name, "->", len(payload))

        runs = client.get("/runs").json()
        picks: dict[str, str] = {}
        for r in runs:
            if r["label"] == "grpc/no_dp":
                picks["run_grpc_nodp"] = r["id"]
            if r["label"] == "grpc/dp_moderate":
                picks["run_grpc_dp"] = r["id"]
            if r["is_aggregate"] and "run_summary_aggregate" not in picks:
                picks["run_summary_aggregate"] = r["id"]
            if "sweep/m=50/" in r["label"] and "seed=42" in r["label"]:
                picks["run_femnist_seed"] = r["id"]
        for key, rid in picks.items():
            detail = client.get(f"/runs/{rid}").json()
            (OUT / f"{key}.json").write_text(json.dumps(detail, indent=1), encoding="utf-8")
            events = store.events_since(rid)
            (OUT / f"{key}_events.json").write_text(
                json.dumps(events, indent=1), encoding="utf-8"
            )
            print(key, rid[:8], "events:", len(events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
