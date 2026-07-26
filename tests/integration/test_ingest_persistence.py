from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ariadne.core.ingest.export import parse_export
from ariadne.core.ingest.persist import new_token, persist_export
from ariadne.db.models import DiaryEntry, Like, Rating, Upload, UploadStatus
from tests.unit.test_export_parser import write_export_dir

pytestmark = pytest.mark.integration


def test_export_persists_with_correct_counts(tmp_path: Path, session: Session):
    parsed = parse_export(write_export_dir(tmp_path))
    upload = persist_export(session, parsed)

    assert upload.status is UploadStatus.PENDING
    assert upload.film_count == 4

    counts = {
        Rating: session.scalar(
            select(func.count()).select_from(Rating).where(Rating.upload_id == upload.id)
        ),
        DiaryEntry: session.scalar(
            select(func.count()).select_from(DiaryEntry).where(DiaryEntry.upload_id == upload.id)
        ),
        Like: session.scalar(
            select(func.count()).select_from(Like).where(Like.upload_id == upload.id)
        ),
    }
    assert counts == {Rating: 4, DiaryEntry: 3, Like: 1}


def test_no_pii_reaches_the_database(tmp_path: Path, session: Session):
    """The export contains an email, name, location and bio. None of it is persistable."""
    parsed = parse_export(write_export_dir(tmp_path))
    upload = persist_export(session, parsed)
    session.flush()

    stored = [
        *session.scalars(select(Rating).where(Rating.upload_id == upload.id)),
        *session.scalars(select(DiaryEntry).where(DiaryEntry.upload_id == upload.id)),
        *session.scalars(select(Like).where(Like.upload_id == upload.id)),
        upload,
    ]
    blob = " ".join(
        str(getattr(row, column.name)) for row in stored for column in row.__table__.columns
    )

    for secret in ("secret@example.com", "Real Name", "Somewhere", "A bio", "someone"):
        assert secret not in blob


def test_diary_is_stored_under_the_join_key(tmp_path: Path, session: Session):
    parsed = parse_export(write_export_dir(tmp_path))
    upload = persist_export(session, parsed)

    keys = set(
        session.scalars(select(DiaryEntry.film_key).where(DiaryEntry.upload_id == upload.id))
    )
    assert "whiplash::2014" in keys


def test_tokens_are_unique_and_unguessable():
    tokens = {new_token() for _ in range(500)}
    assert len(tokens) == 500
    assert all(len(token) >= 40 for token in tokens)


def test_deleting_an_upload_removes_every_child_row(tmp_path: Path, session: Session):
    parsed = parse_export(write_export_dir(tmp_path))
    upload = persist_export(session, parsed)
    upload_id = upload.id

    session.delete(upload)
    session.flush()

    for model in (Rating, DiaryEntry, Like):
        remaining = session.scalar(
            select(func.count()).select_from(model).where(model.upload_id == upload_id)
        )
        assert remaining == 0


def test_two_uploads_do_not_mix(tmp_path: Path, session: Session):
    parsed = parse_export(write_export_dir(tmp_path))
    first = persist_export(session, parsed)
    second = persist_export(session, parsed)

    assert first.token != second.token
    for upload in (first, second):
        count = session.scalar(
            select(func.count()).select_from(Rating).where(Rating.upload_id == upload.id)
        )
        assert count == 4

    assert session.scalar(select(func.count()).select_from(Upload)) == 2
