"""Loading a user's rated films with everything a predictor might use.

One query, materialised once, so the harness can be run hundreds of times without touching the
database again. Iteration speed is the point of building the harness before the model.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ariadne.core.catalog.roles import canonical_role
from ariadne.db.models import Credit, Film, Rating, Resolution


@dataclass(frozen=True)
class RatedFilm:
    letterboxd_uri: str
    tmdb_id: int
    title: str
    rating: float
    logged_date: date | None
    year: int | None
    vote_average: float | None
    vote_count: int | None
    country: str | None
    genres: tuple[str, ...]

    # role -> the people credited in it. Empty when the role is uncredited for this film, which
    # is common enough that predictors must handle it rather than assume presence.
    crew: dict[str, tuple[int, ...]]

    def people_in(self, role: str) -> tuple[int, ...]:
        return self.crew.get(role, ())


def _genres_from_raw(raw: dict[str, Any] | None) -> tuple[str, ...]:
    if not raw:
        return ()
    genres = raw.get("genres")
    if not isinstance(genres, list):
        return ()
    return tuple(
        g["name"] for g in genres if isinstance(g, dict) and isinstance(g.get("name"), str)
    )


def load_dataset(session: Session, upload_id: object) -> list[RatedFilm]:
    """Every rated film that resolved, with metadata and modelled crew attached."""
    rows = session.execute(
        select(
            Rating.letterboxd_uri,
            Rating.rating,
            Rating.logged_date,
            Film.tmdb_id,
            Film.title,
            Film.year,
            Film.vote_average,
            Film.vote_count,
            Film.country,
            Film.raw,
        )
        .join(Resolution, Resolution.letterboxd_uri == Rating.letterboxd_uri)
        .join(Film, Film.tmdb_id == Resolution.tmdb_id)
        .where(Rating.upload_id == upload_id)
        .order_by(Rating.letterboxd_uri)
    ).all()

    film_ids = {row[3] for row in rows}
    crew_by_film: dict[int, dict[str, list[int]]] = {}
    for film_id, department, job, person_id in session.execute(
        select(Credit.film_id, Credit.department, Credit.job, Credit.person_id).where(
            Credit.film_id.in_(film_ids)
        )
    ).all():
        role = canonical_role(department, job)
        if role is None:
            continue
        crew_by_film.setdefault(film_id, {}).setdefault(role, []).append(person_id)

    dataset: list[RatedFilm] = []
    for uri, rating, logged_date, tmdb_id, title, year, va, vc, country, raw in rows:
        crew = {
            role: tuple(sorted(set(people)))
            for role, people in crew_by_film.get(tmdb_id, {}).items()
        }
        dataset.append(
            RatedFilm(
                letterboxd_uri=uri,
                tmdb_id=tmdb_id,
                title=title,
                rating=float(rating),
                logged_date=logged_date,
                year=year,
                vote_average=float(va) if va is not None else None,
                vote_count=vc,
                country=country,
                genres=_genres_from_raw(raw),
                crew=crew,
            )
        )
    return dataset


def person_names(session: Session, person_ids: set[int]) -> dict[int, str]:
    from ariadne.db.models import Person

    if not person_ids:
        return {}
    rows = session.execute(
        select(Person.tmdb_id, Person.name).where(Person.tmdb_id.in_(person_ids))
    ).all()
    return {tmdb_id: name for tmdb_id, name in rows}
