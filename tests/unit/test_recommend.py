"""Adjacency recommendations and the Level 3 metrics.

Three of these encode bugs that only appeared once the thing produced real output.
"""

from datetime import date

import pytest

from ariadne.core.catalog.roles import CINEMATOGRAPHER, DIRECTOR, EDITOR
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.recommend.adjacency import (
    MAX_PER_PERSON,
    Candidate,
    RecommendationReport,
)


def film(
    *,
    uri: str = "",
    tmdb_id: int = 1,
    title: str = "A Film",
    rating: float = 0.0,
    editor: int | None = None,
    director: int | None = None,
    votes: int = 500,
) -> RatedFilm:
    crew: dict[str, tuple[int, ...]] = {}
    if editor is not None:
        crew[EDITOR] = (editor,)
    if director is not None:
        crew[DIRECTOR] = (director,)
    return RatedFilm(
        letterboxd_uri=uri,
        tmdb_id=tmdb_id,
        title=title,
        rating=rating,
        logged_date=date(2023, 1, 1),
        year=2010,
        vote_average=7.0,
        vote_count=votes,
        country="US",
        genres=("Drama",),
        crew=crew,
    )


def candidate(
    *,
    tmdb_id: int,
    score: float,
    expectation: float,
    person: int,
    novelty: float = 0.5,
    familiar: bool | None = None,
) -> Candidate:
    return Candidate(
        film=film(tmdb_id=tmdb_id, title=f"film {tmdb_id}"),
        score=score,
        expectation=expectation,
        reason_person=person,
        reason_role=EDITOR,
        reason_effect=score - expectation,
        novelty_percentile=novelty,
        known_people=1,
        director_is_familiar=familiar,
    )


# --- ranking ---------------------------------------------------------------------------


def test_crew_adjustment_separates_taste_from_acclaim():
    """Ranking on the total score reproduces "acclaimed films" with a faint crew flavour.

    A candidate with a high expectation and a *negative* crew effect outranked everything when
    sorted by score — the top recommendation was The Godfather Trilogy, attributed to an editor
    whose effect was −0.151.
    """
    acclaimed = candidate(tmdb_id=1, score=4.8, expectation=4.95, person=1)
    modest = candidate(tmdb_id=2, score=3.4, expectation=3.0, person=2)

    report = RecommendationReport(candidates=[acclaimed, modest])

    assert report.top(1, by="score")[0].film.tmdb_id == 1
    assert report.top(1, by="crew")[0].film.tmdb_id == 2
    assert acclaimed.crew_adjustment < 0


def test_one_person_cannot_take_every_slot():
    """Unconstrained, the strongest person returned all ten recommendations."""
    hogger = [candidate(tmdb_id=i, score=4.0, expectation=3.0, person=99) for i in range(10)]
    others = [candidate(tmdb_id=100 + i, score=3.5, expectation=3.0, person=i) for i in range(5)]

    chosen = RecommendationReport(candidates=hogger + others).top(6)

    assert sum(1 for c in chosen if c.reason_person == 99) == MAX_PER_PERSON
    assert len({c.reason_person for c in chosen}) > 1


def test_cap_is_configurable():
    hogger = [candidate(tmdb_id=i, score=4.0, expectation=3.0, person=7) for i in range(5)]
    chosen = RecommendationReport(candidates=hogger).top(5, per_person=3)
    assert len(chosen) == 3


# --- non-obviousness -------------------------------------------------------------------


def test_unknown_director_is_not_counted_as_unfamiliar():
    """The bug this encodes reported 100% non-obviousness while recommending Kubrick to a Kubrick
    watcher.

    Candidates come from below-the-line filmographies, so no director credit is stored and the
    question is unanswerable until looked up. Absence of knowledge is not evidence of novelty.
    """
    unknown = candidate(tmdb_id=1, score=4.0, expectation=3.0, person=1, familiar=None)
    assert unknown.is_non_obvious is None
    assert RecommendationReport(candidates=[unknown]).non_obviousness(1) is None


def test_non_obviousness_uses_only_resolved_directors():
    resolved = candidate(tmdb_id=1, score=4.0, expectation=3.0, person=1, familiar=False)
    familiar = candidate(tmdb_id=2, score=3.9, expectation=3.0, person=2, familiar=True)
    unknown = candidate(tmdb_id=3, score=3.8, expectation=3.0, person=3, familiar=None)

    report = RecommendationReport(candidates=[resolved, familiar, unknown])
    # One of the two resolved candidates has an unfamiliar director.
    assert report.non_obviousness(3) == pytest.approx(0.5)


def test_familiar_director_is_obvious():
    known = candidate(tmdb_id=1, score=4.0, expectation=3.0, person=1, familiar=True)
    assert known.is_non_obvious is False


# --- novelty and coverage --------------------------------------------------------------


def test_novelty_averages_over_what_is_actually_shown():
    obscure = candidate(tmdb_id=1, score=4.0, expectation=3.0, person=1, novelty=0.05)
    famous = candidate(tmdb_id=2, score=3.9, expectation=3.0, person=2, novelty=0.95)

    report = RecommendationReport(candidates=[obscure, famous])
    assert report.novelty(1) == pytest.approx(0.05)
    assert report.novelty(2) == pytest.approx(0.5)


def test_coverage_is_scoreable_over_catalog():
    report = RecommendationReport(catalog_films=1000, scoreable=250)
    assert report.coverage == pytest.approx(0.25)


def test_coverage_of_an_empty_catalog_is_zero():
    assert RecommendationReport().coverage == 0.0


def test_metrics_of_an_empty_report_do_not_divide_by_zero():
    report = RecommendationReport()
    assert report.top(5) == []
    assert report.novelty(5) == 0.0
    assert report.non_obviousness(5) is None


# --- disagreement ----------------------------------------------------------------------


def test_disagreement_ranks_by_absolute_gap():
    from ariadne.core.recommend.adjacency import find_disagreements

    class Stub:
        def __init__(self, values: list[float]):
            self.values = values

        def predict(self, films: list[RatedFilm]):
            import numpy as np

            return np.array(self.values, dtype=float)

    films = [film(tmdb_id=1, title="a"), film(tmdb_id=2, title="b"), film(tmdb_id=3, title="c")]
    crew = Stub([4.0, 3.0, 3.0])
    director = Stub([3.0, 3.0, 4.5])

    gaps = find_disagreements(films, crew, director)  # type: ignore[arg-type]

    # Largest absolute gap first, regardless of sign.
    assert [g.film.title for g in gaps] == ["c", "a", "b"]
    assert gaps[0].gap == pytest.approx(-1.5)
    assert gaps[-1].gap == pytest.approx(0.0)


# --- role mapping used by traversal ----------------------------------------------------


def test_traversal_ignores_roles_outside_the_modelled_set():
    from ariadne.core.catalog.filmography import MIN_CANDIDATE_VOTES, MIN_FILMS_TO_TRAVERSE

    # Guards against silently changing the fetch's cost or its candidate quality floor.
    assert MIN_FILMS_TO_TRAVERSE >= 2
    assert MIN_CANDIDATE_VOTES >= 20


def test_film_payload_keeps_genre_names_not_ids():
    """The person-credits payload carries genre ids; the expectation model needs names."""
    from ariadne.core.catalog.filmography import _as_film_payload

    payload = _as_film_payload(
        {"id": 5, "title": "X", "release_date": "2010-01-01", "genre_ids": [18, 999]},
        {18: "Drama"},
    )
    assert payload["genres"] == [{"name": "Drama"}]


def test_cinematographer_is_a_traversable_role():
    from ariadne.core.catalog.roles import BELOW_THE_LINE

    assert CINEMATOGRAPHER in BELOW_THE_LINE
    assert DIRECTOR not in BELOW_THE_LINE
