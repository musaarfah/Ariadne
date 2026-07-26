import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ariadne.core.catalog.credits import (
    films_with_credits,
    ingest_credits_for_upload,
    store_credits,
)
from ariadne.core.catalog.roles import CINEMATOGRAPHER, COMPOSER, DIRECTOR, EDITOR
from ariadne.core.catalog.store import upsert_film
from ariadne.core.evaluation.coverage import build_coverage
from ariadne.core.ingest.export import parse_export
from ariadne.core.ingest.persist import persist_export
from ariadne.db.models import Credit, Film, Person, Resolution
from ariadne.db.models import MatchMethod as MM
from tests.unit.test_export_parser import write_export_dir

pytestmark = pytest.mark.integration

FIXTURES = Path("fixtures/tmdb")


def godfather_payload() -> dict[str, Any]:
    """The recorded detail and credits fixtures, shaped as append_to_response returns them."""
    movie = json.loads((FIXTURES / "movie_238.json").read_text(encoding="utf-8"))
    credits = json.loads((FIXTURES / "credits_238.json").read_text(encoding="utf-8"))
    return {**movie, "credits": credits}


def test_store_credits_writes_people_and_credits(session: Session):
    payload = godfather_payload()
    upsert_film(session, payload)

    written, people = store_credits(session, 238, payload)

    assert written > 50
    assert people > 50
    assert session.scalar(select(func.count()).select_from(Credit)) == written
    assert session.scalar(select(func.count()).select_from(Person)) == people


def test_credits_are_idempotent(session: Session):
    payload = godfather_payload()
    upsert_film(session, payload)

    first, _ = store_credits(session, 238, payload)
    store_credits(session, 238, payload)

    assert session.scalar(select(func.count()).select_from(Credit)) == first


def test_every_department_is_kept_not_just_modelled_roles(session: Session):
    payload = godfather_payload()
    upsert_film(session, payload)
    store_credits(session, 238, payload)

    departments = set(session.scalars(select(Credit.department).distinct()))
    # Role scope is a config choice, so the raw breadth has to survive ingest.
    assert {"Directing", "Editing", "Camera", "Sound", "Art", "Writing"} <= departments
    assert len(departments) > 6


def test_detail_payload_fills_in_country(session: Session):
    """Search results omit origin_country, which is why credits fetch detail (F29)."""
    upsert_film(session, {"id": 238, "title": "The Godfather", "release_date": "1972-03-14"})
    assert session.get(Film, 238).country is None

    upsert_film(session, godfather_payload())
    assert session.get(Film, 238).country == "US"


def test_films_with_credits_reports_what_can_be_skipped(session: Session):
    payload = godfather_payload()
    upsert_film(session, payload)
    assert films_with_credits(session) == set()

    store_credits(session, 238, payload)
    assert films_with_credits(session) == {238}


# --- coverage --------------------------------------------------------------------------


class FakeCreditsClient:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.calls = 0
        self.request_count = 0
        self.retry_count = 0

    def get_movie(self, tmdb_id: int, append: str | None = None) -> dict[str, Any]:
        self.calls += 1
        self.request_count += 1
        return {**self._payload, "id": tmdb_id}


def _upload_with_one_resolved_film(session: Session, tmp_path) -> Any:
    parsed = parse_export(write_export_dir(tmp_path))
    upload = persist_export(session, parsed)
    upsert_film(session, godfather_payload())
    session.add(
        Resolution(
            letterboxd_uri=parsed.ratings[0].letterboxd_uri,
            name=parsed.ratings[0].name,
            year=parsed.ratings[0].year,
            tmdb_id=238,
            match_method=MM.EXACT,
            confidence=1.0,
            reason="test",
        )
    )
    session.flush()
    return upload


def test_ingest_skips_films_that_already_have_credits(session: Session, tmp_path):
    upload = _upload_with_one_resolved_film(session, tmp_path)
    payload = godfather_payload()
    store_credits(session, 238, payload)

    client = FakeCreditsClient(payload)
    stats = ingest_credits_for_upload(session, client, upload.id)  # type: ignore[arg-type]

    assert stats.skipped_cached == 1
    assert client.calls == 0


def test_refresh_refetches(session: Session, tmp_path):
    upload = _upload_with_one_resolved_film(session, tmp_path)
    payload = godfather_payload()
    store_credits(session, 238, payload)

    client = FakeCreditsClient(payload)
    stats = ingest_credits_for_upload(session, client, upload.id, refresh=True)  # type: ignore[arg-type]

    assert stats.fetched == 1
    assert client.calls == 1


def test_coverage_finds_the_modelled_roles(session: Session, tmp_path):
    upload = _upload_with_one_resolved_film(session, tmp_path)
    store_credits(session, 238, godfather_payload())

    report = build_coverage(session, upload.id)

    assert report.films == 1
    assert report.films_with_any_credit == 1
    for role in (DIRECTOR, EDITOR, CINEMATOGRAPHER, COMPOSER):
        assert report.by_role[role] == 1, role
    assert report.role_rate(DIRECTOR) == 1.0
    assert report.median_crew > 50


def test_coverage_counts_films_with_more_than_one_person_in_a_role(session: Session, tmp_path):
    """The Godfather credits two editors, which the model must handle rather than assume away."""
    upload = _upload_with_one_resolved_film(session, tmp_path)
    store_credits(session, 238, godfather_payload())

    report = build_coverage(session, upload.id)
    assert report.multi_credit_films[EDITOR] == 1


def test_coverage_groups_by_decade_and_region(session: Session, tmp_path):
    upload = _upload_with_one_resolved_film(session, tmp_path)
    store_credits(session, 238, godfather_payload())

    report = build_coverage(session, upload.id)
    assert report.decade_films[1970] == 1
    assert report.region_films["US"] == 1
