"""Drop raw_gbfs_payloads.discovery_payload: the discovery response is a
near-static list of feed URLs, so storing it in full on every run was the
same redundant-blob problem station_information had, just smaller.
discovery_url already records which feed was hit.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-18

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("raw_gbfs_payloads", "discovery_payload")


def downgrade() -> None:
    op.add_column(
        "raw_gbfs_payloads",
        sa.Column("discovery_payload", sa.JSON(), nullable=True),
    )
