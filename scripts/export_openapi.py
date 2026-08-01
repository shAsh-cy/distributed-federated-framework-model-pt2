"""Export the coordinator's OpenAPI schema to docs/openapi.json.

The committed schema is the contract the frontend generates its client from;
a test asserts the committed file matches the live app, so schema drift fails
CI instead of surfacing as a broken dashboard.

    python scripts/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coordinator.app import create_app  # noqa: E402
from coordinator.db import create_all, make_engine  # noqa: E402
from coordinator.store import EventStore  # noqa: E402

OUT = Path("docs/openapi.json")


def build_schema() -> dict:
    engine = make_engine(":memory:")
    create_all(engine)
    app = create_app(store=EventStore(engine))
    return app.openapi()


def main() -> int:
    OUT.write_text(json.dumps(build_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
