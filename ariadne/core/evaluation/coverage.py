"""Crew coverage: does TMDB actually credit the people the model needs?

The most consequential Level 1 number. `Ariadne.MD` §6 estimates 60–75% below-the-line coverage
from an assumption, not a measurement. If a role lands near 40%, that role's results are computed
on a biased subsample and have to say so; if it lands near 85%, the product is in better shape
than planned.

Broken out by decade and region because F5 predicts coverage is worst for pre-1980 and
non-Anglophone films — which is also where a shortfall matters least for this library, since
83% of it is post-2000.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ariadne.core.catalog.roles import ALL_ROLES, canonical_role
from ariadne.db.models import Credit, Film, Rating, Resolution

# Regions with fewer films than this are pooled, since a percentage over three films is noise.
MIN_REGION_FILMS = 15


@dataclass
class CoverageReport:
    films: int = 0
    films_with_any_credit: int = 0

    # role -> number of films with at least one person in that role
    by_role: Counter[str] = field(default_factory=Counter)

    # (decade, role) -> count, and decade -> film count
    by_decade: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    decade_films: Counter[int] = field(default_factory=Counter)

    by_region: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    region_films: Counter[str] = field(default_factory=Counter)

    people: int = 0
    credits: int = 0
    crew_sizes: list[int] = field(default_factory=list)

    # How often each modelled role has more than one credited person, which the model has to
    # handle rather than assume away.
    multi_credit_films: Counter[str] = field(default_factory=Counter)

    def role_rate(self, role: str) -> float:
        return self.by_role[role] / self.films if self.films else 0.0

    @property
    def median_crew(self) -> float:
        if not self.crew_sizes:
            return 0.0
        ordered = sorted(self.crew_sizes)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2


def build_coverage(session: Session, upload_id: object) -> CoverageReport:
    report = CoverageReport()

    films = session.execute(
        select(Film.tmdb_id, Film.year, Film.country)
        .join(Resolution, Resolution.tmdb_id == Film.tmdb_id)
        .join(Rating, Rating.letterboxd_uri == Resolution.letterboxd_uri)
        .where(Rating.upload_id == upload_id)
        .distinct()
    ).all()

    credits_by_film: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    for film_id, department, job, person_id in session.execute(
        select(Credit.film_id, Credit.department, Credit.job, Credit.person_id)
    ).all():
        credits_by_film[film_id].append((department, job, person_id))

    all_people: set[int] = set()

    for tmdb_id, year, country in films:
        report.films += 1
        decade = (year // 10) * 10 if year else 0
        region = country or "??"
        report.decade_films[decade] += 1
        report.region_films[region] += 1

        crew = credits_by_film.get(tmdb_id, [])
        report.credits += len(crew)
        report.crew_sizes.append(len(crew))
        if crew:
            report.films_with_any_credit += 1

        per_role: dict[str, set[int]] = defaultdict(set)
        for department, job, person_id in crew:
            all_people.add(person_id)
            role = canonical_role(department, job)
            if role is not None:
                per_role[role].add(person_id)

        for role, people in per_role.items():
            report.by_role[role] += 1
            report.by_decade[decade][role] += 1
            report.by_region[region][role] += 1
            if len(people) > 1:
                report.multi_credit_films[role] += 1

    report.people = len(all_people)
    return report


def pooled_regions(report: CoverageReport) -> tuple[list[str], int]:
    """Regions worth reporting individually, plus how many films fall into the pooled rest."""
    named = [r for r, n in report.region_films.items() if n >= MIN_REGION_FILMS]
    pooled = sum(n for r, n in report.region_films.items() if n < MIN_REGION_FILMS)
    return sorted(named, key=lambda r: -report.region_films[r]), pooled


def role_order() -> tuple[str, ...]:
    return ALL_ROLES
