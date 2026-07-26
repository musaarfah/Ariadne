"""television resolution outcome, drop films.media_type

Revision ID: 4190d441327d
Revises: 2a0f9d0a775d
Create Date: 2026-07-26 16:20:30.731089

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '4190d441327d'
down_revision: str | None = '2a0f9d0a775d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Hand-written: alembic does not diff enum values.
    op.execute("ALTER TYPE match_method ADD VALUE IF NOT EXISTS 'TELEVISION' BEFORE 'UNRESOLVED'")

    # films.media_type is removed rather than left unused. TMDB movie ids and TV ids are
    # separate namespaces that can collide on the same integer, and films.tmdb_id is the
    # primary key, so television can never be stored here safely. Television is recorded on
    # resolutions instead, where the count stays queryable without risking a wrong join.
    op.drop_column("films", "media_type")
    op.execute("DROP TYPE IF EXISTS media_type")


def downgrade() -> None:
    op.execute("CREATE TYPE media_type AS ENUM ('MOVIE', 'TV')")
    op.add_column(
        "films",
        sa.Column(
            "media_type",
            sa.Enum("MOVIE", "TV", name="media_type"),
            nullable=False,
            server_default="MOVIE",
        ),
    )
    # Postgres cannot drop a single enum value, so TELEVISION persists on the type.
