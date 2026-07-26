"""Recommending unseen films by walking the crew graph.

A candidate is any film in the catalog the user has not rated, credited to at least one person from
their library. Scoring reuses the fitted crew model unchanged, so a recommendation is the same
arithmetic as a prediction — no separate ranking heuristic to disagree with the model.

Two Level 3 metrics guard against being accurate and useless:

**Novelty.** A recommender that suggests The Godfather to someone with 1,300 films is precisely
right and worth nothing. Novelty is the candidate's vote count as a percentile of the user's own
library, and low is the target.

**Non-obviousness.** The original spec measured whether a recommendation's strongest edge was
something other than the director, but the crew model excludes directors entirely, so that would be
100% by construction and prove nothing. The honest version asks whether the film's *director* is
someone the user already watches — because "another film by a director you love" is what the feature
is supposed to improve on.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ariadne.core.catalog.roles import BELOW_THE_LINE, DIRECTOR, canonical_role
from ariadne.core.evaluation.dataset import RatedFilm, genres_from_raw
from ariadne.core.taste.crew import CrewModel
from ariadne.db.models import Credit, Film

# Candidates need at least this many votes for the expectation model to place them at all.
MIN_CANDIDATE_VOTES = 20

# Most recommendations any single person may account for. Without a cap the strongest effect takes
# every slot, which is both a worse list and an unmeasurable one.
MAX_PER_PERSON = 2


@dataclass
class Candidate:
    film: RatedFilm
    score: float
    expectation: float
    reason_person: int | None
    reason_role: str | None
    reason_effect: float
    novelty_percentile: float
    known_people: int

    # None until the director is known. Candidates arrive from below-the-line filmographies, so
    # nothing about their direction is stored and the question cannot be answered without a
    # separate lookup. None means unknown, which is not the same as "unfamiliar" — reading it that
    # way made non-obviousness report 100% while recommending Kubrick films to a Kubrick watcher.
    director_is_familiar: bool | None = None

    @property
    def crew_adjustment(self) -> float:
        """The part of the score that is about this user's crew, rather than the film's acclaim.

        Ranking by total predicted rating is dominated by the expectation model: it spans roughly
        one to five stars while crew effects span about a third of a star either way. Sorting on it
        reproduces "well-regarded films" with a faint crew flavour, which is the popularity baseline
        wearing a costume. The adjustment is the part that is actually a recommendation.
        """
        return self.score - self.expectation

    @property
    def is_non_obvious(self) -> bool | None:
        return None if self.director_is_familiar is None else not self.director_is_familiar


@dataclass
class RecommendationReport:
    candidates: list[Candidate] = field(default_factory=list)
    catalog_films: int = 0
    scoreable: int = 0
    already_rated: int = 0

    def top(self, n: int, by: str = "crew", per_person: int = MAX_PER_PERSON) -> list[Candidate]:
        """Best candidates, ranked by crew contribution, capped per reason-person.

        The cap is not cosmetic. Without it the single strongest person takes every slot: the
        unconstrained ranking returned eight Peter Jackson films, because his writer effect is
        the largest in the library and his filmography is long. Worse, he directs his own films, so
        every recommendation was "another film by a director you already watch" — precisely what
        this feature exists to improve on.
        """
        key = (lambda c: -c.score) if by == "score" else (lambda c: -c.crew_adjustment)
        chosen: list[Candidate] = []
        used: Counter[int] = Counter()
        for candidate in sorted(self.candidates, key=key):
            person = candidate.reason_person or 0
            if used[person] >= per_person:
                continue
            used[person] += 1
            chosen.append(candidate)
            if len(chosen) >= n:
                break
        return chosen

    def novelty(self, n: int, by: str = "crew") -> float:
        chosen = self.top(n, by=by)
        return float(np.mean([c.novelty_percentile for c in chosen])) if chosen else 0.0

    def non_obviousness(self, n: int, by: str = "crew") -> float | None:
        """Share of recommendations whose director the user has never rated.

        None when no director is known for any of them, rather than a number that would read as
        100%.
        """
        known = [c for c in self.top(n, by=by) if c.is_non_obvious is not None]
        if not known:
            return None
        return float(np.mean([bool(c.is_non_obvious) for c in known]))

    @property
    def coverage(self) -> float:
        """Share of the catalog the model can score at all."""
        return self.scoreable / self.catalog_films if self.catalog_films else 0.0


def _percentile_ranker(values: list[int]) -> Callable[[int], float]:
    """Where a vote count falls within the user's own library, as a share."""
    ordered = np.sort(np.array(values, dtype=float))

    def rank(value: int) -> float:
        if len(ordered) == 0:
            return 0.0
        return float(np.searchsorted(ordered, value, side="right") / len(ordered))

    return rank


def build_recommendations(
    session: Session, upload_id: object, model: CrewModel, rated: list[RatedFilm]
) -> RecommendationReport:
    rated_ids = {film.tmdb_id for film in rated}
    familiar_directors = {person for film in rated for person in film.people_in(DIRECTOR)}
    ranker = _percentile_ranker([film.vote_count or 0 for film in rated])

    crew_by_film: dict[int, dict[str, list[int]]] = {}
    for film_id, department, job, person_id in session.execute(
        select(Credit.film_id, Credit.department, Credit.job, Credit.person_id)
    ).all():
        role = canonical_role(department, job)
        if role is None:
            continue
        crew_by_film.setdefault(film_id, {}).setdefault(role, []).append(person_id)

    report = RecommendationReport()
    candidates: list[Candidate] = []

    for tmdb_id, title, year, vote_average, vote_count, country, raw in session.execute(
        select(
            Film.tmdb_id,
            Film.title,
            Film.year,
            Film.vote_average,
            Film.vote_count,
            Film.country,
            Film.raw,
        )
    ).all():
        report.catalog_films += 1
        if tmdb_id in rated_ids:
            report.already_rated += 1
            continue
        if (vote_count or 0) < MIN_CANDIDATE_VOTES or vote_average is None:
            continue

        crew = {
            role: tuple(sorted(set(people)))
            for role, people in crew_by_film.get(tmdb_id, {}).items()
        }
        # Rating is unused by predict; a candidate has none by definition.
        candidate_film = RatedFilm(
            letterboxd_uri="",
            tmdb_id=tmdb_id,
            title=title,
            rating=0.0,
            logged_date=None,
            year=year,
            vote_average=float(vote_average),
            vote_count=vote_count,
            country=country,
            genres=genres_from_raw(raw),
            crew=crew,
        )

        known = [
            (role, person_id, model.effect_for(role, person_id))
            for role in BELOW_THE_LINE
            for person_id in candidate_film.people_in(role)
            if model.effect_for(role, person_id) is not None
        ]
        if not known:
            continue

        report.scoreable += 1
        strongest = max(known, key=lambda item: abs(item[2] or 0.0))
        directors = candidate_film.people_in(DIRECTOR)

        candidates.append(
            Candidate(
                film=candidate_film,
                score=float(model.predict([candidate_film])[0]),
                expectation=float(model.expectation_for([candidate_film])[0]),
                reason_person=strongest[1],
                reason_role=strongest[0],
                reason_effect=float(strongest[2] or 0.0),
                novelty_percentile=ranker(vote_count or 0),
                known_people=len(known),
                director_is_familiar=(
                    any(person in familiar_directors for person in directors) if directors else None
                ),
            )
        )

    report.candidates = candidates
    return report


@dataclass
class Disagreement:
    film: RatedFilm
    crew_score: float
    director_score: float

    @property
    def gap(self) -> float:
        return self.crew_score - self.director_score


def find_disagreements(
    films: list[RatedFilm], crew: CrewModel, director_model: object
) -> list[Disagreement]:
    """Films where crew-based and director-based taste point in different directions.

    The differentiator from `Ariadne.MD` §5. Where the two models agree there is nothing to say;
    where they diverge, the crew is carrying information the director does not.
    """
    crew_scores = crew.predict(films)
    director_scores = director_model.predict(films)  # type: ignore[attr-defined]
    return sorted(
        (
            Disagreement(film=film, crew_score=float(c), director_score=float(d))
            for film, c, d in zip(films, crew_scores, director_scores, strict=True)
        ),
        key=lambda item: -abs(item.gap),
    )


def resolve_directors(
    candidates: list[Candidate],
    client: object,
    familiar_directors: set[int],
) -> int:
    """Look up who directed each candidate, so non-obviousness can be measured at all.

    Candidates come from below-the-line filmographies, so their direction is unknown until asked.
    One request each, which is why it is done for a shortlist rather than the whole catalog.
    """
    resolved = 0
    for candidate in candidates:
        if candidate.director_is_familiar is not None:
            continue
        try:
            payload = client.get_movie(candidate.film.tmdb_id, append="credits")  # type: ignore[attr-defined]
        except Exception:
            continue
        directors = {
            member.get("id")
            for member in payload.get("credits", {}).get("crew", [])
            if member.get("job") == "Director" and isinstance(member.get("id"), int)
        }
        if not directors:
            continue
        candidate.director_is_familiar = bool(directors & familiar_directors)
        resolved += 1
    return resolved
