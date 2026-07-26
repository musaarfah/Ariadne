import uuid
from decimal import Decimal

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.orm import Session

from ariadne.db.models import Film, MatchMethod, Rating, Resolution, Upload, UploadStatus

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "films",
    "people",
    "credits",
    "resolutions",
    "uploads",
    "ratings",
    "diary_entries",
    "likes",
    "analysis_runs",
    "crew_effects",
    "recommendations",
}


def test_migrations_create_every_table(test_engine: Engine):
    tables = set(inspect(test_engine).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_pg_trgm_is_installed(test_engine: Engine):
    # The resolver's fuzzy fallback depends on it.
    with test_engine.connect() as conn:
        installed = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
        ).scalar_one()
    assert installed


def test_trigram_index_exists(test_engine: Engine):
    with test_engine.connect() as conn:
        indexes = conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'films'")
        ).scalars()
    assert "ix_films_normalized_title_trgm" in set(indexes)


def test_similarity_query_runs(test_engine: Engine):
    # Proves the operator class is usable, which is the whole reason for the extension.
    with test_engine.connect() as conn:
        score = conn.execute(
            text("SELECT similarity('gangs of wasseypur part 1', 'gangs of wasseypur part i')")
        ).scalar_one()
    assert 0.0 < score < 1.0


def test_upload_and_rating_round_trip(session: Session):
    upload = Upload(token="test-token", status=UploadStatus.PENDING)
    session.add(upload)
    session.flush()

    session.add(
        Rating(
            upload_id=upload.id,
            letterboxd_uri="https://boxd.it/2bg8",
            name="The Godfather",
            year=1972,
            rating=Decimal("4.5"),
        )
    )
    session.flush()

    stored = session.execute(select(Rating).where(Rating.upload_id == upload.id)).scalar_one()
    assert stored.rating == Decimal("4.5")
    assert isinstance(upload.id, uuid.UUID)


def test_rating_precision_holds_half_stars(session: Session):
    upload = Upload(token="precision-token")
    session.add(upload)
    session.flush()

    session.add(
        Rating(
            upload_id=upload.id,
            letterboxd_uri="uri",
            name="A Film",
            year=2001,
            rating=Decimal("0.5"),
        )
    )
    session.flush()
    session.expire_all()

    stored = session.execute(select(Rating).where(Rating.upload_id == upload.id)).scalar_one()
    assert stored.rating == Decimal("0.5")


def test_deleting_an_upload_cascades_to_its_ratings(session: Session):
    upload = Upload(token="cascade-token")
    session.add(upload)
    session.flush()
    session.add(
        Rating(
            upload_id=upload.id,
            letterboxd_uri="uri",
            name="A Film",
            year=2001,
            rating=Decimal("3.0"),
        )
    )
    session.flush()

    session.delete(upload)
    session.flush()

    remaining = session.execute(select(Rating).where(Rating.upload_id == upload.id)).all()
    assert remaining == []


def test_television_is_recorded_on_resolutions_not_films(session: Session):
    """Films holds only films.

    TMDB movie ids and TV ids are separate namespaces that can collide on the same integer,
    and films.tmdb_id is the primary key, so television cannot be stored there safely. It is
    recorded as a resolution outcome instead, which keeps the exclusion count queryable.
    """
    assert not hasattr(Film, "media_type")

    session.add(
        Resolution(
            letterboxd_uri="https://boxd.it/obi",
            name="Obi-Wan Kenobi",
            year=2022,
            tmdb_id=None,
            match_method=MatchMethod.TELEVISION,
            reason="matched a television series",
        )
    )
    session.flush()

    stored = session.get(Resolution, "https://boxd.it/obi")
    assert stored is not None
    assert stored.match_method is MatchMethod.TELEVISION
    assert stored.tmdb_id is None
