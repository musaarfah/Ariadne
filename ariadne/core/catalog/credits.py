"""Fetching and storing crew credits.

One request per film: detail with credits appended. That halves the call count against fetching
them separately, and detail is needed regardless because search results omit origin_country,
which the region coverage metric depends on.

Every department is stored, not just the roles the model uses (D26). One call returns the whole
list, so filtering here would save nothing and would mean refetching every film after any change
of mind about role scope.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ariadne.core.catalog.store import upsert_film
from ariadne.core.catalog.tmdb import TmdbClient, TmdbError, TmdbNotFound
from ariadne.db.models import Credit, Person, Rating, Resolution


@dataclass
class CreditsStats:
    films: int = 0
    fetched: int = 0
    skipped_cached: int = 0
    missing: int = 0
    errors: int = 0
    credits_stored: int = 0
    people_stored: int = 0
    crew_sizes: list[int] = field(default_factory=list)
    departments: Counter[str] = field(default_factory=Counter)

    @property
    def median_crew(self) -> float:
        if not self.crew_sizes:
            return 0.0
        ordered = sorted(self.crew_sizes)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2


def store_credits(session: Session, tmdb_id: int, payload: dict[str, Any]) -> tuple[int, int]:
    """Persist a film's crew. Returns (credits written, people written)."""
    crew = payload.get("credits", {}).get("crew", [])
    if not isinstance(crew, list):
        return 0, 0

    people: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    for member in crew:
        person_id = member.get("id")
        name = member.get("name")
        department = member.get("department")
        job = member.get("job")
        if not isinstance(person_id, int) or not isinstance(name, str):
            continue
        if not isinstance(department, str) or not isinstance(job, str):
            continue

        people[person_id] = name

        # TMDB occasionally repeats a credit within one response, which would otherwise trip
        # the unique constraint mid-batch.
        key = (tmdb_id, person_id, job)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "film_id": tmdb_id,
                "person_id": person_id,
                "department": department,
                "job": job,
            }
        )

    if people:
        session.execute(
            insert(Person)
            .values([{"tmdb_id": pid, "name": name} for pid, name in people.items()])
            .on_conflict_do_nothing(index_elements=["tmdb_id"]),
        )
    if rows:
        session.execute(insert(Credit).values(rows).on_conflict_do_nothing(constraint="uq_credit"))
    session.flush()
    return len(rows), len(people)


def ingest_film_credits(
    session: Session, client: TmdbClient, tmdb_id: int, stats: CreditsStats
) -> None:
    payload = client.get_movie(tmdb_id, append="credits")

    # The detail payload is richer than the search result the film was created from, so this
    # also fills in country and refreshes vote counts.
    upsert_film(session, payload)

    credits_written, people_written = store_credits(session, tmdb_id, payload)
    credits_written += store_cast(session, tmdb_id, payload)
    stats.fetched += 1
    stats.credits_stored += credits_written
    stats.people_stored += people_written

    crew = payload.get("credits", {}).get("crew", [])
    stats.crew_sizes.append(len(crew))
    for member in crew:
        department = member.get("department")
        if isinstance(department, str):
            stats.departments[department] += 1


def resolved_film_ids(session: Session, upload_id: object) -> list[int]:
    return [
        tmdb_id
        for tmdb_id in session.scalars(
            select(Resolution.tmdb_id)
            .join(Rating, Rating.letterboxd_uri == Resolution.letterboxd_uri)
            .where(Rating.upload_id == upload_id, Resolution.tmdb_id.is_not(None))
            .distinct()
            .order_by(Resolution.tmdb_id)
        )
        if tmdb_id is not None
    ]


def films_with_credits(session: Session) -> set[int]:
    return set(session.scalars(select(Credit.film_id).distinct()))


def ingest_credits_for_upload(
    session: Session,
    client: TmdbClient,
    upload_id: object,
    limit: int | None = None,
    on_progress: Any = None,
    refresh: bool = False,
) -> CreditsStats:
    """Fetch credits for every resolved film on an upload.

    Films already holding credits are skipped unless `refresh` is set, so an interrupted run
    resumes rather than restarting.
    """
    stats = CreditsStats()
    film_ids = resolved_film_ids(session, upload_id)
    if limit is not None:
        film_ids = film_ids[:limit]

    already = set() if refresh else films_with_credits(session)

    for index, tmdb_id in enumerate(film_ids, start=1):
        stats.films += 1
        if tmdb_id in already:
            stats.skipped_cached += 1
        else:
            try:
                ingest_film_credits(session, client, tmdb_id, stats)
            except TmdbNotFound:
                stats.missing += 1
            except TmdbError:
                stats.errors += 1
        if on_progress is not None:
            on_progress(index, len(film_ids), stats)

    return stats


def credit_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Credit)) or 0


# --- cast -------------------------------------------------------------------------------

# Only the top-billed are stored. TMDB lists ~81 cast per film and a fortieth-billed extra has no
# bearing on whether someone liked the film, so storing them would add noise and rows in equal
# measure. Ten is generous: it reaches well past the leads into recognisable supporting parts.
CAST_BILLING_LIMIT = 10

# Synthesised, because TMDB cast entries carry `character` and `order` rather than a department and
# job. Using the same credits table keeps one code path for every kind of credit.
CAST_DEPARTMENT = "Acting"
CAST_JOB = "Actor"


def store_cast(session: Session, tmdb_id: int, payload: dict[str, Any]) -> int:
    """Persist a film's top-billed cast. Returns rows written.

    Billing order is deliberately not preserved as a separate role. Splitting leads from supporting
    parts would give a person two effect estimates on half the evidence each, and sparsity is
    already the binding constraint. The order remains in `films.raw` if weighting is wanted later.
    """
    cast = payload.get("credits", {}).get("cast", [])
    if not isinstance(cast, list):
        return 0

    people: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    for member in cast:
        person_id = member.get("id")
        name = member.get("name")
        order = member.get("order")
        if not isinstance(person_id, int) or not isinstance(name, str):
            continue
        if not isinstance(order, int) or order >= CAST_BILLING_LIMIT:
            continue
        if person_id in seen:
            continue

        seen.add(person_id)
        people[person_id] = name
        rows.append(
            {
                "film_id": tmdb_id,
                "person_id": person_id,
                "department": CAST_DEPARTMENT,
                "job": CAST_JOB,
            }
        )

    if people:
        session.execute(
            insert(Person)
            .values([{"tmdb_id": pid, "name": name} for pid, name in people.items()])
            .on_conflict_do_nothing(index_elements=["tmdb_id"]),
        )
    if rows:
        session.execute(insert(Credit).values(rows).on_conflict_do_nothing(constraint="uq_credit"))
    session.flush()
    return len(rows)


def backfill_cast(session: Session, on_progress: Any = None) -> tuple[int, int]:
    """Store cast for every film whose raw payload already carries it.

    Costs nothing: `get_movie(append="credits")` returned cast all along and the whole payload was
    written to `films.raw`. Only `store_credits` ignored it, so roughly 100,000 credits have been
    sitting in the database unused (D98).

    Returns (films processed, credits written).
    """
    from ariadne.db.models import Film

    film_ids = list(session.scalars(select(Film.tmdb_id).order_by(Film.tmdb_id)))
    written = 0
    processed = 0

    for index, tmdb_id in enumerate(film_ids, start=1):
        film = session.get(Film, tmdb_id)
        if film is None or not film.raw:
            continue
        if "credits" not in film.raw:
            continue
        written += store_cast(session, tmdb_id, film.raw)
        processed += 1
        if on_progress is not None:
            on_progress(index, len(film_ids), processed, written)

    return processed, written
