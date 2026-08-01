"""SQLAlchemy models and engine setup for run/event persistence.

SQLite on purpose: one file, survives restarts, needs no service, and the
write rate (one row per event, tens per round) is far below anything SQLite
finds interesting. Alembic owns the schema in deployment
(``alembic upgrade head``); tests may create tables directly from this
metadata, and a dedicated test asserts the two produce the same schema so
the migration cannot drift from the models.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

#: Default database location, under the gitignored data/ directory.
DEFAULT_DB_PATH = Path("data/coordinator.sqlite")

RUN_STATUSES = ("pending", "running", "completed", "failed", "stopped")


class Run(Base):
    __tablename__ = "runs"

    id = Column(String(36), primary_key=True)
    created_at = Column(Float, nullable=False)
    status = Column(String(16), nullable=False, default="pending")
    #: JSON of the fl.config Config dict this run was started from.
    config_json = Column(Text, nullable=False)
    #: "live" for runs executed by this coordinator; "imported" for history
    #: loaded from the repo's committed result files.
    source = Column(String(16), nullable=False, default="live")
    #: Human label; for imported runs, the originating file/cell.
    label = Column(String(255), nullable=False, default="")
    #: Optional grouping key (e.g. "femnist_sweep/m=50") tying multi-seed
    #: sibling runs and their summary row together.
    group_key = Column(String(255), nullable=True)
    #: True on synthetic rows that carry a multi-seed mean/range summary
    #: rather than a single execution. Ranges are preserved, never collapsed.
    is_aggregate = Column(Boolean, nullable=False, default=False)
    seed = Column(Integer, nullable=True)
    #: JSON: final metrics. For aggregate rows: {"mean_final": .., "range_final": ..,
    #: "final_per_seed": [..], ...}. For single runs: {"final_accuracy": .., ...}.
    final_metrics_json = Column(Text, nullable=True)

    __table_args__ = ()

    def config(self) -> dict:
        return json.loads(self.config_json)

    def final_metrics(self) -> dict | None:
        return json.loads(self.final_metrics_json) if self.final_metrics_json else None


class EventRow(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)
    #: Contiguous per-run sequence number starting at 0. The unique
    #: constraint makes gap/duplicate bugs loud.
    seq = Column(Integer, nullable=False)
    type = Column(String(32), nullable=False)
    ts = Column(Float, nullable=False)
    payload_json = Column(Text, nullable=False)

    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_events_run_seq"),)

    def payload(self) -> dict:
        return json.loads(self.payload_json)


def make_engine(db_path: str | Path | None = None) -> Engine:
    """Engine for a file-backed SQLite database (":memory:" for tests).

    check_same_thread=False throughout: the runner thread writes while the API
    thread reads; sessions are serialised by the store's lock. In-memory
    databases additionally need a StaticPool — SQLite gives every *connection*
    its own private :memory: database, so a pooled engine would hand the
    runner thread a fresh empty schema.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if str(db_path) == ":memory:":
        from sqlalchemy.pool import StaticPool

        return create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True
    )


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, future=True)


def create_all(engine: Engine) -> None:
    """Create tables from metadata. Tests and first-run convenience; Alembic
    is the source of truth for deployments and is asserted equivalent."""
    Base.metadata.create_all(engine)
