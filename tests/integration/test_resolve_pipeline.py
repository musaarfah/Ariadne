"""The resolution pipeline: cache, local catalog, API, and television detection.

The TMDB client is faked throughout, so these exercise the orchestration and the caching
economics rather than TMDB itself.
"""

from typing import Any

import pytest
from sqlalchemy.orm import Session

from ariadne.core.catalog.pipeline import ResolveStats, resolve_one, resolve_upload
from ariadne.core.catalog.store import local_candidates, upsert_film
from ariadne.core.ingest.export import parse_export
from ariadne.core.ingest.persist import persist_export
from ariadne.db.models import Film, MatchMethod, Resolution
from tests.unit.test_export_parser import write_export_dir

pytestmark = pytest.mark.integration

GODFATHER: dict[str, Any] = {
    "id": 238,
    "title": "The Godfather",
    "original_title": "The Godfather",
    "release_date": "1972-03-14",
    "vote_count": 23211,
    "vote_average": 8.687,
    "origin_country": ["US"],
}

WASSEYPUR: dict[str, Any] = {
    "id": 117691,
    "title": "Gangs of Wasseypur - Part 1",
    "original_title": "गैंग्स ऑफ़ वास्सेपुर",
    "release_date": "2012-06-21",
    "vote_count": 300,
    "vote_average": 8.0,
    "origin_country": ["IN"],
}


class FakeClient:
    """Stands in for TmdbClient, counting the calls it receives."""

    def __init__(
        self,
        movie_results: list[dict[str, Any]] | None = None,
        tv_results: list[dict[str, Any]] | None = None,
    ):
        self._movies = movie_results if movie_results is not None else []
        self._tv = tv_results if tv_results is not None else []
        self.movie_searches = 0
        self.tv_searches = 0
        self.request_count = 0
        self.retry_count = 0

    def search_movies(self, title: str, year: int | None = None) -> list[dict[str, Any]]:
        self.movie_searches += 1
        self.request_count += 1
        return self._movies

    def search_tv(self, title: str, year: int | None = None) -> list[dict[str, Any]]:
        self.tv_searches += 1
        self.request_count += 1
        return self._tv


# --- the API path ----------------------------------------------------------------------


def test_api_hit_resolves_and_caches_the_film(session: Session):
    client = FakeClient([GODFATHER])
    stats = ResolveStats()

    resolution = resolve_one(
        session,
        client,
        "https://boxd.it/g",
        "The Godfather",
        1972,
        stats,  # type: ignore[arg-type]
    )

    assert resolution.tmdb_id == 238
    assert resolution.match_method is MatchMethod.EXACT
    assert stats.from_api == 1
    # The film is now in the catalog, which is what makes the next account cheap.
    assert session.get(Film, 238) is not None


def test_resolution_stores_the_raw_payload(session: Session):
    client = FakeClient([GODFATHER])
    resolve_one(
        session,
        client,
        "https://boxd.it/g",
        "The Godfather",
        1972,
        ResolveStats(),  # type: ignore[arg-type]
    )
    film = session.get(Film, 238)
    assert film is not None
    assert film.raw["vote_average"] == pytest.approx(8.687)
    assert film.normalized_title == "the godfather"


def test_normalized_title_is_stored_folded(session: Session):
    client = FakeClient([WASSEYPUR])
    resolve_one(
        session,
        client,  # type: ignore[arg-type]
        "https://boxd.it/w",
        "Gangs of Wasseypur – Part 1",
        2012,
        ResolveStats(),
    )
    film = session.get(Film, 117691)
    assert film is not None
    assert film.normalized_title == "gangs of wasseypur - part 1"


# --- caching ---------------------------------------------------------------------------


def test_second_lookup_of_the_same_uri_makes_no_request(session: Session):
    client = FakeClient([GODFATHER])
    stats = ResolveStats()

    resolve_one(session, client, "https://boxd.it/g", "The Godfather", 1972, stats)  # type: ignore[arg-type]
    resolve_one(session, client, "https://boxd.it/g", "The Godfather", 1972, stats)  # type: ignore[arg-type]

    assert client.movie_searches == 1
    assert stats.from_cache == 1
    assert stats.resolved == 2


def test_a_different_uri_for_a_known_film_resolves_locally(session: Session):
    """The Stage 2 economics: another account's copy of a known film costs no API call."""
    upsert_film(session, GODFATHER)
    client = FakeClient([])
    stats = ResolveStats()

    resolution = resolve_one(
        session,
        client,
        "https://boxd.it/other",
        "The Godfather",
        1972,
        stats,  # type: ignore[arg-type]
    )

    assert resolution.tmdb_id == 238
    assert stats.from_local == 1
    assert client.movie_searches == 0


def test_local_lookup_tolerates_the_dash_difference(session: Session):
    upsert_film(session, WASSEYPUR)
    candidates = local_candidates(session, "Gangs of Wasseypur – Part 1", 2012)
    assert [c["id"] for c in candidates] == [117691]


def test_local_lookup_respects_the_year_window(session: Session):
    upsert_film(session, GODFATHER)
    assert local_candidates(session, "The Godfather", 1972)
    assert local_candidates(session, "The Godfather", 1973)
    # Two years out is a rejection, not a near miss.
    assert local_candidates(session, "The Godfather", 1975) == []


def test_local_lookup_needs_a_year(session: Session):
    upsert_film(session, GODFATHER)
    assert local_candidates(session, "The Godfather", None) == []


# --- television ------------------------------------------------------------------------


def test_television_is_recorded_as_such_not_as_a_generic_failure(session: Session):
    client = FakeClient(
        movie_results=[
            {
                "id": 1015606,
                "title": "Obi-Wan Kenobi: A Jedi's Return",
                "release_date": "2022-09-08",
                "vote_count": 1019,
            }
        ],
        tv_results=[{"id": 83867, "name": "Obi-Wan Kenobi", "first_air_date": "2022-05-27"}],
    )
    stats = ResolveStats()

    resolution = resolve_one(
        session,
        client,
        "https://boxd.it/obi",
        "Obi-Wan Kenobi",
        2022,
        stats,  # type: ignore[arg-type]
    )

    assert resolution.tmdb_id is None
    assert resolution.match_method is MatchMethod.TELEVISION
    assert stats.television == 1
    assert stats.failed == 0
    assert client.tv_searches == 1


def test_television_ids_never_enter_the_films_table(session: Session):
    """TMDB movie and TV ids are separate namespaces and could collide on tmdb_id."""
    client = FakeClient(
        movie_results=[],
        tv_results=[{"id": 238, "name": "Some Series", "first_air_date": "2022-01-01"}],
    )
    resolve_one(
        session,
        client,
        "https://boxd.it/s",
        "Some Series",
        2022,
        ResolveStats(),  # type: ignore[arg-type]
    )
    assert session.get(Film, 238) is None


def test_a_plain_failure_is_not_labelled_television(session: Session):
    client = FakeClient(movie_results=[], tv_results=[])
    stats = ResolveStats()

    resolution = resolve_one(
        session,
        client,
        "https://boxd.it/x",
        "Nonexistent Film",
        1999,
        stats,  # type: ignore[arg-type]
    )

    assert resolution.match_method is MatchMethod.UNRESOLVED
    assert stats.failed == 1
    assert stats.television == 0


def test_tv_search_is_skipped_when_a_film_resolves(session: Session):
    client = FakeClient([GODFATHER], tv_results=[])
    resolve_one(
        session,
        client,
        "https://boxd.it/g",
        "The Godfather",
        1972,
        ResolveStats(),  # type: ignore[arg-type]
    )
    assert client.tv_searches == 0


# --- failures are rows, not gaps -------------------------------------------------------


def test_failures_are_persisted_so_the_denominator_exists(session: Session):
    client = FakeClient(movie_results=[], tv_results=[])
    resolve_one(
        session,
        client,
        "https://boxd.it/x",
        "Nonexistent Film",
        1999,
        ResolveStats(),  # type: ignore[arg-type]
    )

    stored = session.get(Resolution, "https://boxd.it/x")
    assert stored is not None
    assert stored.tmdb_id is None
    assert stored.reason


# --- whole upload ----------------------------------------------------------------------


def test_resolve_upload_walks_every_rating(session: Session, tmp_path):
    parsed = parse_export(write_export_dir(tmp_path))
    upload = persist_export(session, parsed)

    client = FakeClient(movie_results=[], tv_results=[])
    stats = resolve_upload(session, client, upload.id)  # type: ignore[arg-type]

    assert stats.total == 4
    assert stats.from_api == 4


def test_resolve_upload_honours_the_limit(session: Session, tmp_path):
    parsed = parse_export(write_export_dir(tmp_path))
    upload = persist_export(session, parsed)

    client = FakeClient(movie_results=[], tv_results=[])
    stats = resolve_upload(session, client, upload.id, limit=2)  # type: ignore[arg-type]

    assert stats.total == 2
    assert client.movie_searches == 2


def test_offline_mode_makes_no_requests(session: Session, tmp_path):
    parsed = parse_export(write_export_dir(tmp_path))
    upload = persist_export(session, parsed)

    stats = resolve_upload(session, None, upload.id)

    assert stats.total == 4
    assert stats.from_api == 0
    assert stats.resolved == 0


def test_resolution_rate_is_reported(session: Session):
    client = FakeClient([GODFATHER])
    stats = ResolveStats()
    resolve_one(session, client, "https://boxd.it/a", "The Godfather", 1972, stats)  # type: ignore[arg-type]

    failing = FakeClient(movie_results=[], tv_results=[])
    resolve_one(session, failing, "https://boxd.it/b", "Nonexistent", 1999, stats)  # type: ignore[arg-type]

    assert stats.total == 2
    assert stats.resolution_rate == pytest.approx(0.5)
