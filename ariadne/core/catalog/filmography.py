"""Fetching what else the people in a library have worked on.

Recommendations come from adjacency: films by the crew already in the user's graph. That needs each
person's filmography, which the library itself cannot supply — it only holds credits for films the
user has seen.

One request per person, bounded by a minimum film count. Traversal is not restricted to the people
whose effects are estimable (D65): a cinematographer with three films is far too thin to measure but
perfectly good to walk through, and 12-film people alone would reach almost nothing.

The person-credits payload carries genre *ids* rather than names, so a genre map is fetched once and
the names are written into the stored film's raw JSON. Without that the rich expectation model would
see every recommended film as genreless.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ariadne.core.catalog.roles import BELOW_THE_LINE, canonical_role
from ariadne.core.catalog.store import upsert_film
from ariadne.core.catalog.tmdb import TmdbClient, TmdbError, TmdbNotFound
from ariadne.db.models import Credit, Person, Rating, Resolution

# A person needs at least this many films in the library before their filmography is worth a
# request. Three keeps the fetch to roughly 1,400 calls while still reaching well beyond the 138
# people who are estimable.
MIN_FILMS_TO_TRAVERSE = 3

# Films with fewer votes than this are not fetched as candidates. Below it TMDB's vote_average is
# too noisy to place a film on the user's scale at all, and the expectation model would be guessing.
MIN_CANDIDATE_VOTES = 20


@dataclass
class FilmographyStats:
    people: int = 0
    fetched: int = 0
    missing: int = 0
    errors: int = 0
    films_seen: int = 0
    films_stored: int = 0
    credits_stored: int = 0
    skipped_low_votes: int = 0
    by_role: Counter[str] = field(default_factory=Counter)


def fetch_genre_map(client: TmdbClient) -> dict[int, str]:
    payload = client._get("/genre/movie/list", {})
    genres = payload.get("genres", [])
    return {
        g["id"]: g["name"]
        for g in genres
        if isinstance(g, dict) and isinstance(g.get("id"), int) and isinstance(g.get("name"), str)
    }


def people_worth_traversing(
    session: Session, upload_id: object, min_films: int = MIN_FILMS_TO_TRAVERSE
) -> list[int]:
    """People credited in a modelled role on at least `min_films` of the user's rated films."""
    rated = set(
        session.scalars(
            select(Resolution.tmdb_id)
            .join(Rating, Rating.letterboxd_uri == Resolution.letterboxd_uri)
            .where(Rating.upload_id == upload_id, Resolution.tmdb_id.is_not(None))
        )
    )
    counts: Counter[int] = Counter()
    for film_id, department, job, person_id in session.execute(
        select(Credit.film_id, Credit.department, Credit.job, Credit.person_id)
    ).all():
        if film_id in rated and canonical_role(department, job) in BELOW_THE_LINE:
            counts[person_id] += 1
    return sorted(person_id for person_id, n in counts.items() if n >= min_films)


def _as_film_payload(entry: dict[str, Any], genre_map: dict[int, str]) -> dict[str, Any]:
    """Reshape a person-credit entry into the film payload upsert_film expects."""
    genre_ids = entry.get("genre_ids") or []
    return {
        "id": entry["id"],
        "title": entry.get("title"),
        "original_title": entry.get("original_title"),
        "release_date": entry.get("release_date"),
        "vote_average": entry.get("vote_average"),
        "vote_count": entry.get("vote_count"),
        "genres": [
            {"name": genre_map[g]} for g in genre_ids if isinstance(g, int) and g in genre_map
        ],
    }


def ingest_filmographies(
    session: Session,
    client: TmdbClient,
    people: list[int],
    genre_map: dict[int, str],
    on_progress: Any = None,
) -> FilmographyStats:
    stats = FilmographyStats(people=len(people))

    # Before any credit is written: a credit's person foreign key has to resolve. These people all
    # came from existing library credits so they should be present, but relying on that would make
    # the order load-bearing for no reason.
    _ensure_people(session, people)

    known_films = set(session.scalars(select(Credit.film_id).distinct()))

    for index, person_id in enumerate(people, start=1):
        try:
            payload = client._get(f"/person/{person_id}/movie_credits", {})
        except TmdbNotFound:
            stats.missing += 1
            continue
        except TmdbError:
            stats.errors += 1
            continue

        stats.fetched += 1
        rows: list[dict[str, Any]] = []

        for entry in payload.get("crew", []):
            film_id = entry.get("id")
            department = entry.get("department")
            job = entry.get("job")
            if not isinstance(film_id, int) or not isinstance(job, str):
                continue
            if not isinstance(department, str):
                continue

            role = canonical_role(department, job)
            if role not in BELOW_THE_LINE:
                continue

            stats.films_seen += 1
            if (entry.get("vote_count") or 0) < MIN_CANDIDATE_VOTES:
                stats.skipped_low_votes += 1
                continue

            if film_id not in known_films:
                upsert_film(session, _as_film_payload(entry, genre_map))
                known_films.add(film_id)
                stats.films_stored += 1

            rows.append(
                {
                    "film_id": film_id,
                    "person_id": person_id,
                    "department": department,
                    "job": job,
                }
            )
            stats.by_role[role] += 1

        if rows:
            session.execute(
                insert(Credit).values(rows).on_conflict_do_nothing(constraint="uq_credit")
            )
            session.flush()
            stats.credits_stored += len(rows)

        if on_progress is not None:
            on_progress(index, len(people), stats)

    return stats


def _ensure_people(session: Session, people: list[int]) -> None:
    existing = set(session.scalars(select(Person.tmdb_id).where(Person.tmdb_id.in_(people))))
    missing = [p for p in people if p not in existing]
    if not missing:
        return

    session.execute(
        insert(Person)
        .values([{"tmdb_id": p, "name": f"person {p}"} for p in missing])
        .on_conflict_do_nothing(index_elements=["tmdb_id"])
    )
    session.flush()
