"""Level 1 metrics: is the plumbing correct?

Everything downstream inherits resolution error, so these are reported before any modelling
happens. Two resolution rates are computed deliberately. Counting television entries as
failures understates the resolver, since they are not films; excluding them without saying so
overstates it. Both numbers, always.
"""

from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ariadne.db.models import DiaryEntry, Film, MatchMethod, Rating, Resolution, Upload

# Methods that mean "the resolver was not certain from title and year alone". Reported as the
# fuzzy-fallback rate, and the pool the audit oversamples.
FUZZY_METHODS = (MatchMethod.TRIGRAM, MatchMethod.DOMINANT)


@dataclass
class Level1Report:
    entries: int = 0
    resolved: int = 0
    television: int = 0
    unresolved: int = 0

    methods: Counter[str] = field(default_factory=Counter)
    confidences: Counter[str] = field(default_factory=Counter)

    films_in_catalog: int = 0
    films_with_votes: int = 0
    films_with_country: int = 0
    by_decade: Counter[int] = field(default_factory=Counter)

    diary_entries: int = 0
    diary_films: int = 0
    diary_films_matched: int = 0

    # Two Letterboxd entries resolving to one TMDB film. Almost always an error, and the check
    # that found all 8 sibling collapses in the first full run.
    collisions: list[tuple[int, str, list[str]]] = field(default_factory=list)

    @property
    def film_entries(self) -> int:
        """Entries that are actually films, so excluding identified television."""
        return self.entries - self.television

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.entries if self.entries else 0.0

    @property
    def resolution_rate_excluding_television(self) -> float:
        return self.resolved / self.film_entries if self.film_entries else 0.0

    @property
    def fuzzy_fallback(self) -> int:
        return sum(self.methods[method.value] for method in FUZZY_METHODS)

    @property
    def fuzzy_fallback_rate(self) -> float:
        return self.fuzzy_fallback / self.resolved if self.resolved else 0.0

    @property
    def diary_join_rate(self) -> float:
        return self.diary_films_matched / self.diary_films if self.diary_films else 0.0


def build_report(session: Session, upload_id: object) -> Level1Report:
    report = Level1Report()

    rows = session.execute(
        select(Resolution.match_method, Resolution.confidence, Resolution.tmdb_id)
        .join(Rating, Rating.letterboxd_uri == Resolution.letterboxd_uri)
        .where(Rating.upload_id == upload_id)
    ).all()

    for method, confidence, tmdb_id in rows:
        report.entries += 1
        report.methods[method.value] += 1

        if method is MatchMethod.TELEVISION:
            report.television += 1
        elif tmdb_id is None:
            report.unresolved += 1
        else:
            report.resolved += 1
            key = "1.00" if confidence is None or confidence >= 1.0 else f"{confidence:.2f}"
            report.confidences[key] += 1

    report.films_in_catalog = session.scalar(select(func.count()).select_from(Film)) or 0
    report.films_with_votes = (
        session.scalar(select(func.count()).select_from(Film).where(Film.vote_count.is_not(None)))
        or 0
    )
    report.films_with_country = (
        session.scalar(select(func.count()).select_from(Film).where(Film.country.is_not(None))) or 0
    )

    for (year,) in session.execute(select(Film.year).where(Film.year.is_not(None))).all():
        report.by_decade[(year // 10) * 10] += 1

    report.diary_entries = (
        session.scalar(
            select(func.count()).select_from(DiaryEntry).where(DiaryEntry.upload_id == upload_id)
        )
        or 0
    )
    diary_keys = set(
        session.scalars(select(DiaryEntry.film_key).where(DiaryEntry.upload_id == upload_id)).all()
    )
    report.diary_films = len(diary_keys)

    # The diary join is by name+year (F6), so it is re-derived here from ratings rather than
    # trusted from ingest. A silent divergence between the two is exactly what this catches.
    from ariadne.core.ingest.export import film_key

    rating_keys = {
        film_key(name, year)
        for name, year in session.execute(
            select(Rating.name, Rating.year).where(Rating.upload_id == upload_id)
        ).all()
    }
    report.diary_films_matched = len(diary_keys & rating_keys)
    report.collisions = _find_collisions(session, upload_id)

    return report


def _find_collisions(session: Session, upload_id: object) -> list[tuple[int, str, list[str]]]:
    """TMDB ids claimed by more than one Letterboxd entry on this upload."""
    duplicated = session.execute(
        select(Resolution.tmdb_id)
        .join(Rating, Rating.letterboxd_uri == Resolution.letterboxd_uri)
        .where(Rating.upload_id == upload_id, Resolution.tmdb_id.is_not(None))
        .group_by(Resolution.tmdb_id)
        .having(func.count() > 1)
    ).scalars()

    collisions: list[tuple[int, str, list[str]]] = []
    for tmdb_id in duplicated:
        if tmdb_id is None:
            continue
        film = session.get(Film, tmdb_id)
        claimants = list(
            session.execute(
                select(Rating.name, Rating.year)
                .join(Resolution, Resolution.letterboxd_uri == Rating.letterboxd_uri)
                .where(Rating.upload_id == upload_id, Resolution.tmdb_id == tmdb_id)
                .order_by(Rating.name)
            ).all()
        )
        collisions.append(
            (
                tmdb_id,
                f"{film.title} ({film.year})" if film else "?",
                [f"{name} ({year})" for name, year in claimants],
            )
        )
    return collisions


def upload_by_token(session: Session, token: str) -> Upload | None:
    return session.execute(select(Upload).where(Upload.token == token)).scalar_one_or_none()
