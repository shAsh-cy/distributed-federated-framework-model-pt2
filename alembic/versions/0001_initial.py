"""Initial schema: runs and events.

Revision ID: 0001
Revises:
Create Date: 2026-08-01
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("group_key", sa.String(length=255), nullable=True),
        sa.Column("is_aggregate", sa.Boolean(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("final_metrics_json", sa.Text(), nullable=True),
    )
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.UniqueConstraint("run_id", "seq", name="uq_events_run_seq"),
    )
    op.create_index("ix_events_run_id", "events", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_events_run_id", table_name="events")
    op.drop_table("events")
    op.drop_table("runs")
