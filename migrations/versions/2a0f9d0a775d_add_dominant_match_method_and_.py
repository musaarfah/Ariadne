"""add dominant match method and resolution reason

Revision ID: 2a0f9d0a775d
Revises: 8959d9ed298a
Create Date: 2026-07-26 16:09:52.482207

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '2a0f9d0a775d'
down_revision: str | None = '8959d9ed298a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Alembic does not diff enum values, so this is hand-written. Postgres 12+ permits
    # ALTER TYPE ... ADD VALUE inside a transaction as long as the value is not used in the
    # same transaction, which it is not.
    op.execute("ALTER TYPE match_method ADD VALUE IF NOT EXISTS 'DOMINANT' AFTER 'TRIGRAM'")
    op.add_column("resolutions", sa.Column("reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("resolutions", "reason")
    # Postgres cannot drop a single enum value. Reversing means rebuilding the type, which
    # is not worth it for a value nothing depends on; the column drop is the reversible part.
