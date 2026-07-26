"""What a library knows about its owner.

Everything here is **tier 1**: counts and direct observations, carrying no uncertainty because there
is no estimate involved. "You have watched 41 Akshay Kumar films" is not a claim about taste that
could be wrong — it is a fact about the export.

This distinction is the product's whole defence against drifting into astrology. Four hypotheses
about crew effects were tested and rejected (F70, F72, F73), so the *predictive* content of
who-made-what is close to nothing for this user. The *observational* content is abundant, and it
needs no model at all.

Tier 2 lives elsewhere — the decomposition, the crew effects — and always carries an interval.
Tier 3 is an explicit refusal with the cost of an answer attached.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ariadne.core.evaluation.dataset import RatedFilm

# A film needs this many TMDB votes before its residual means anything. Without it, films nobody has
# rated get a vote_average of 0, an enormous artefactual residual, and a place at the top of "your
# boldest opinions" — which is exactly what the first prototype did (D101).
MIN_VOTES_FOR_RESIDUAL = 50

# The ceiling of the rating scale. Films here are unordered by rating alone, which is what makes
# rewatch informative (F4).
TOP_RATING = 5.0

# How many entries each section returns.
SECTION_SIZE = 8


@dataclass(frozen=True)
class Loyalty:
    """Someone whose work the user has seen repeatedly. A count, not an effect."""

    person_id: int
    role: str
    films: int


@dataclass(frozen=True)
class Disagreement:
    """A film rated far from what its context predicts. Requires a vote floor."""

    film: RatedFilm
    residual: float

    @property
    def direction(self) -> str:
        return "above" if self.residual > 0 else "below"


@dataclass(frozen=True)
class RevealedPreference:
    """Ratings say one thing; returning to a film says another."""

    rewatched: list[RatedFilm]
    top_rated_never_revisited: int
    top_rated_in_diary: int
    top_rated_total: int

    @property
    def revisit_rate(self) -> float:
        """Of top-rated films the diary covers, the share actually returned to."""
        if not self.top_rated_in_diary:
            return 0.0
        return 1 - (self.top_rated_never_revisited / self.top_rated_in_diary)


@dataclass(frozen=True)
class BlindSpot:
    """A slice rated well above the user's average and barely watched."""

    label: str
    films: int
    mean_rating: float
    library_mean: float

    @property
    def lift(self) -> float:
        return self.mean_rating - self.library_mean


@dataclass(frozen=True)
class RatingStyle:
    whole_star_share: float
    levels_used: int
    levels_available: int
    mean: float
    sd: float
    top_rating_share: float

    @property
    def is_decisive(self) -> bool:
        """Whole stars most of the time: deciding rather than calibrating."""
        return self.whole_star_share >= 0.6


@dataclass
class Portrait:
    films: int
    loyalties: list[Loyalty] = field(default_factory=list)
    above: list[Disagreement] = field(default_factory=list)
    below: list[Disagreement] = field(default_factory=list)
    revealed: RevealedPreference | None = None
    blind_spots: list[BlindSpot] = field(default_factory=list)
    style: RatingStyle | None = None
    excluded_low_votes: int = 0


def _residuals(films: list[RatedFilm], expectation: object) -> list[tuple[RatedFilm, float]]:
    values = expectation.residuals(films)  # type: ignore[attr-defined]
    return list(zip(films, (float(v) for v in values), strict=True))


def loyalties(
    films: list[RatedFilm], roles: tuple[str, ...], limit: int = SECTION_SIZE
) -> list[Loyalty]:
    """People the user has seen most, by role. Pure counts."""
    counts: Counter[tuple[str, int]] = Counter()
    for film in films:
        for role in roles:
            for person_id in film.people_in(role):
                counts[(role, person_id)] += 1
    return [
        Loyalty(person_id=person_id, role=role, films=n)
        for (role, person_id), n in counts.most_common(limit)
    ]


def disagreements(
    films: list[RatedFilm],
    expectation: object,
    limit: int = SECTION_SIZE,
    min_votes: int = MIN_VOTES_FOR_RESIDUAL,
) -> tuple[list[Disagreement], list[Disagreement], int]:
    """Films rated furthest from expectation, both directions, and how many were excluded.

    The vote floor is not optional. Films with no votes carry a vote_average of 0, so the residual
    is an artefact of missing data rather than an opinion, and they dominate the positive tail
    (D101).
    """
    eligible = [
        (f, r) for f, r in _residuals(films, expectation) if (f.vote_count or 0) >= min_votes
    ]
    excluded = len(films) - len(eligible)
    ordered = sorted(eligible, key=lambda pair: -pair[1])

    above = [Disagreement(film=f, residual=r) for f, r in ordered[:limit] if r > 0]
    below = [Disagreement(film=f, residual=r) for f, r in reversed(ordered[-limit:]) if r < 0]
    return above, below, excluded


# A "films that define your taste" section was built here and removed. Ranked by absolute residual
# it returned the negative tail verbatim — 16 of the 20 largest residuals are negative, because the
# library mean of 3.43 sits high on a bounded scale and leaves 2.93 stars of room below a typical
# prediction against 1.57 above. Normalising by that available headroom fixes the bias and creates a
# worse one: every 5.0 rating uses 100% of its headroom, so the ranking ties. Both tails are already
# reported, labelled and comparable, by `disagreements` (F74, D106).


def revealed_preference(films: list[RatedFilm]) -> RevealedPreference:
    """Rewatching as a second, independent statement of preference.

    Only films the diary covers can be counted either way: the diary begins part-way through the
    library, so absence of a rewatch record is unknown rather than zero (F7).
    """
    rewatched = sorted(
        (f for f in films if f.rewatches > 0), key=lambda f: (-f.rewatches, -f.rating)
    )
    top = [f for f in films if f.rating >= TOP_RATING]
    top_in_diary = [f for f in top if f.in_diary]
    never = [f for f in top_in_diary if f.rewatches == 0]

    return RevealedPreference(
        rewatched=rewatched[:SECTION_SIZE],
        top_rated_never_revisited=len(never),
        top_rated_in_diary=len(top_in_diary),
        top_rated_total=len(top),
    )


def blind_spots(
    films: list[RatedFilm], min_films: int = 5, max_films: int = 60, limit: int = SECTION_SIZE
) -> list[BlindSpot]:
    """Slices rated above the library average but thinly watched.

    Bounded on both sides: below `min_films` an average is noise, and above `max_films` it is not a
    blind spot but a habit.
    """
    library_mean = float(np.mean([f.rating for f in films])) if films else 0.0
    groups: dict[str, list[float]] = defaultdict(list)

    for film in films:
        if film.year:
            groups[f"decade {(film.year // 10) * 10}s"].append(film.rating)
        if film.country:
            groups[f"country {film.country}"].append(film.rating)
        for genre in film.genres:
            groups[f"genre {genre}"].append(film.rating)

    found = [
        BlindSpot(
            label=label,
            films=len(ratings),
            mean_rating=float(np.mean(ratings)),
            library_mean=library_mean,
        )
        for label, ratings in groups.items()
        if min_films <= len(ratings) <= max_films and float(np.mean(ratings)) > library_mean
    ]
    return sorted(found, key=lambda spot: -spot.lift)[:limit]


def rating_style(films: list[RatedFilm]) -> RatingStyle:
    ratings = np.array([f.rating for f in films], dtype=float)
    if len(ratings) == 0:
        return RatingStyle(0.0, 0, 10, 0.0, 0.0, 0.0)
    return RatingStyle(
        whole_star_share=float(np.mean(ratings == np.round(ratings))),
        levels_used=len(set(ratings.tolist())),
        levels_available=10,
        mean=float(ratings.mean()),
        sd=float(ratings.std(ddof=1)) if len(ratings) > 1 else 0.0,
        top_rating_share=float(np.mean(ratings >= TOP_RATING)),
    )


def build_portrait(films: list[RatedFilm], expectation: object, roles: tuple[str, ...]) -> Portrait:
    above, below, excluded = disagreements(films, expectation)
    return Portrait(
        films=len(films),
        loyalties=loyalties(films, roles),
        above=above,
        below=below,
        revealed=revealed_preference(films),
        blind_spots=blind_spots(films),
        style=rating_style(films),
        excluded_low_votes=excluded,
    )
