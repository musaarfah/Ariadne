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
from ariadne.core.ingest.export import film_key
from ariadne.db.models import Credit, DiaryEntry, Film, Rating, Resolution

# Value of one rewatch, in stars. Deliberately small and bounded: a rewatch is evidence of
# preference, not four times the preference. Cap and bonus are both tested through the harness
# rather than asserted.
REWATCH_BONUS = 0.25
MAX_REWATCH_CREDIT = 3


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

    # Rewatches recorded in the diary, and whether the diary covers this film at all.
    #
    # The two fields are separate on purpose. The diary begins 2023-11-26 and covers 511 of 1,345
    # films, so a film with no diary record has an *unknown* rewatch count, not a count of zero.
    # Treating absence as zero would invent a signal for 62% of the library (F7).
    rewatches: int = 0
    in_diary: bool = False

    def people_in(self, role: str) -> tuple[int, ...]:
        return self.crew.get(role, ())

    @property
    def preference(self) -> float:
        """Rating plus a bounded bonus for rewatching, where the diary knows.

        A second, independent view of the same underlying preference. It matters because the
        rating scale runs out of resolution exactly where the gate metric operates: 222 films tie
        at 5.0, and rewatch splits that mass into 88 returned-to and 77 watched-once (F4).
        """
        if not self.in_diary:
            return self.rating
        return self.rating + REWATCH_BONUS * min(self.rewatches, MAX_REWATCH_CREDIT)


def genres_from_raw(raw: dict[str, Any] | None) -> tuple[str, ...]:
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
            Rating.name,
            Rating.year,
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

    # Diary joins on name+year, because its URIs are entry links rather than film links (F6).
    diary_rewatches: dict[str, int] = {}
    diary_seen: set[str] = set()
    for key, is_rewatch in session.execute(
        select(DiaryEntry.film_key, DiaryEntry.is_rewatch).where(DiaryEntry.upload_id == upload_id)
    ).all():
        diary_seen.add(key)
        if is_rewatch:
            diary_rewatches[key] = diary_rewatches.get(key, 0) + 1

    film_ids = {row[5] for row in rows}
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
    for (
        uri,
        rating,
        logged_date,
        lb_name,
        lb_year,
        tmdb_id,
        title,
        year,
        va,
        vc,
        country,
        raw,
    ) in rows:
        crew = {
            role: tuple(sorted(set(people)))
            for role, people in crew_by_film.get(tmdb_id, {}).items()
        }
        # Keyed on Letterboxd's own title and year, which is what the diary rows were stored
        # under. Using TMDB's instead silently lost 40 films to title differences — the en dash
        # in "Gangs of Wasseypur – Part 1" and the 1975/1976 disagreement on "Salo" among them.
        key = film_key(lb_name, lb_year)
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
                genres=genres_from_raw(raw),
                crew=crew,
                rewatches=diary_rewatches.get(key, 0),
                in_diary=key in diary_seen,
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
