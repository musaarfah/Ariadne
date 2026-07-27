"""Reading and writing the shared catalog.

The catalog is identical for every upload, so a film resolved once is resolved for everyone.
That is what makes the 15th recruited account cheap, and it is where the trigram index earns
its place: matching against films we already hold avoids an API call entirely.
"""

from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ariadne.core.catalog.normalize import normalize_title
from ariadne.core.catalog.resolve import CONSIDER_SIMILARITY, ResolutionOutcome
from ariadne.db.models import Film, Person, Resolution

# Local candidates are cheap, but a runaway query on a large catalog is not.
LOCAL_CANDIDATE_LIMIT = 20


def release_year(payload: dict[str, Any]) -> int | None:
    head = (payload.get("release_date") or "")[:4]
    return int(head) if head.isdigit() else None


def country_of(payload: dict[str, Any]) -> str | None:
    countries = payload.get("origin_country")
    if isinstance(countries, list) and countries:
        first = countries[0]
        if isinstance(first, str) and len(first) == 2:
            return first
    return None


def upsert_film(session: Session, payload: dict[str, Any]) -> Film:
    """Insert or refresh one film from a TMDB payload.

    Accepts either a search result or a full detail response. Search results carry fewer
    fields, so an existing row's values are only overwritten when the new payload actually
    has something to say — a detail fetch must not be downgraded by a later search hit.
    """
    tmdb_id = payload["id"]
    title = payload.get("title") or payload.get("original_title") or ""

    values: dict[str, Any] = {
        "tmdb_id": tmdb_id,
        "title": title,
        "original_title": payload.get("original_title"),
        "normalized_title": normalize_title(title),
        "year": release_year(payload),
        "vote_average": payload.get("vote_average"),
        "vote_count": payload.get("vote_count"),
        "country": country_of(payload),
        "raw": payload,
    }

    statement = insert(Film).values(**values)
    updates: dict[str, Any] = {
        column: statement.excluded[column]
        for column in ("title", "original_title", "normalized_title")
    }
    # `raw` is the only field a later payload can silently destroy rather than merely fail to
    # improve: the COALESCE below protects scalars, but a slim search result is a perfectly
    # non-null JSON object that would replace a full detail response and take its credits with it.
    # Keep the stored payload unless the incoming one is at least as complete.
    updates["raw"] = text(
        "CASE WHEN jsonb_exists(EXCLUDED.raw, 'credits') "
        "OR NOT jsonb_exists(films.raw, 'credits') "
        "THEN EXCLUDED.raw ELSE films.raw END"
    )
    # COALESCE so a sparse search payload cannot null out fields a detail fetch supplied.
    for column in ("year", "vote_average", "vote_count", "country"):
        updates[column] = text(f"COALESCE(EXCLUDED.{column}, films.{column})")

    session.execute(statement.on_conflict_do_update(index_elements=[Film.tmdb_id], set_=updates))
    session.flush()

    film = session.get(Film, tmdb_id)
    assert film is not None  # noqa: S101 - just written above
    return film


def upsert_person(session: Session, tmdb_id: int, name: str) -> Person:
    statement = insert(Person).values(tmdb_id=tmdb_id, name=name)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[Person.tmdb_id], set_={"name": statement.excluded.name}
        )
    )
    session.flush()
    person = session.get(Person, tmdb_id)
    assert person is not None  # noqa: S101 - just written above
    return person


def cached_resolution(session: Session, letterboxd_uri: str) -> Resolution | None:
    return session.get(Resolution, letterboxd_uri)


def record_resolution(
    session: Session,
    letterboxd_uri: str,
    name: str,
    year: int | None,
    outcome: ResolutionOutcome,
) -> Resolution:
    """Persist an outcome, including failures.

    Failures are rows with a NULL tmdb_id rather than absent rows, because the resolution-rate
    metric needs a denominator and the audit needs to see what was refused and why.
    """
    values = {
        "letterboxd_uri": letterboxd_uri,
        "name": name,
        "year": year,
        "tmdb_id": outcome.tmdb_id,
        "match_method": outcome.method,
        "confidence": outcome.confidence,
        "reason": outcome.reason,
    }
    statement = insert(Resolution).values(**values)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[Resolution.letterboxd_uri],
            set_={
                key: statement.excluded[key]
                for key in ("name", "year", "tmdb_id", "match_method", "confidence", "reason")
            },
        )
    )
    session.flush()
    resolution = session.get(Resolution, letterboxd_uri)
    assert resolution is not None  # noqa: S101 - just written above
    return resolution


def local_candidates(session: Session, title: str, year: int | None) -> list[dict[str, Any]]:
    """Films we already hold that could be this title, shaped like TMDB search results.

    Returned as payload dicts so the caller can score local and remote candidates through
    exactly the same code path.

    Restricted to an **exact** year, unlike the API path. TMDB's search already filters by
    year, so a near-year candidate can only reach the resolver from here — and it did: with the
    2014 Whiplash already cached, a lookup for Whiplash (2013) found it locally, matched the
    title exactly, and resolved the short film to the feature. The local path is an
    optimisation, so it stays conservative and leaves year disagreements to the API.

    The `%` operator is what uses the GIN trigram index; `similarity() >= x` would not. Its
    cutoff comes from pg_trgm.similarity_threshold, set here per transaction so the query does
    not depend on server configuration.

    set_config rather than SET LOCAL because SET does not accept bind parameters, and building
    the statement by string interpolation is not worth it for a value we already have.
    """
    if year is None:
        return []

    normalized = normalize_title(title)
    session.execute(
        text("SELECT set_config('pg_trgm.similarity_threshold', :threshold, true)").bindparams(
            threshold=str(CONSIDER_SIMILARITY)
        )
    )

    rows = session.execute(
        select(Film)
        .where(
            Film.year == year,
            Film.normalized_title.op("%")(normalized),
        )
        .order_by(text("similarity(films.normalized_title, :normalized) DESC"))
        .params(normalized=normalized)
        .limit(LOCAL_CANDIDATE_LIMIT)
    ).scalars()

    return [
        {
            "id": film.tmdb_id,
            "title": film.title,
            "original_title": film.original_title,
            "release_date": f"{film.year}-01-01" if film.year else None,
            "vote_count": film.vote_count or 0,
            "vote_average": film.vote_average,
        }
        for film in rows
    ]
