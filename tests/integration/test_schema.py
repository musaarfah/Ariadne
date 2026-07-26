import uuid
from decimal import Decimal

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.orm import Session

from ariadne.db.models import Film, MediaType, Rating, Upload, UploadStatus

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "films",
    "people",
    "credits",
    "resolutions",
    "uploads",
    "ratings",
    "diary_entries",
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
        Rating(upload_id=upload.id, letterboxd_uri="https://boxd.it/2bg8", rating=Decimal("4.5"))
    )
    session.flush()

    stored = session.execute(select(Rating).where(Rating.upload_id == upload.id)).scalar_one()
    assert stored.rating == Decimal("4.5")
    assert isinstance(upload.id, uuid.UUID)


def test_rating_precision_holds_half_stars(session: Session):
    upload = Upload(token="precision-token")
    session.add(upload)
    session.flush()

    session.add(Rating(upload_id=upload.id, letterboxd_uri="uri", rating=Decimal("0.5")))
    session.flush()
    session.expire_all()

    stored = session.execute(select(Rating).where(Rating.upload_id == upload.id)).scalar_one()
    assert stored.rating == Decimal("0.5")


def test_deleting_an_upload_cascades_to_its_ratings(session: Session):
    upload = Upload(token="cascade-token")
    session.add(upload)
    session.flush()
    session.add(Rating(upload_id=upload.id, letterboxd_uri="uri", rating=Decimal("3.0")))
    session.flush()

    session.delete(upload)
    session.flush()

    remaining = session.execute(select(Rating).where(Rating.upload_id == upload.id)).all()
    assert remaining == []


def test_tv_is_representable(session: Session):
    # Letterboxd exports contain television; it must be recordable so it can be excluded
    # with a reported count rather than silently dropped.
    session.add(
        Film(
            tmdb_id=1,
            title="Obi-Wan Kenobi",
            normalized_title="obi wan kenobi",
            year=2022,
            media_type=MediaType.TV,
        )
    )
    session.flush()

    stored = session.get(Film, 1)
    assert stored is not None
    assert stored.media_type is MediaType.TV
