"""The decomposition, and the pieces it needed.

Two of these are regression guards for mistakes already made once: a base score reported in a
different metric from the differences measured against it, and a genre rung fitted on an expectation
that already contained genre.
"""

from datetime import date

import numpy as np
import pytest

from ariadne.core.catalog.roles import DIRECTOR
from ariadne.core.evaluation.baselines import Context, GenreOnly, _PersonEffects
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.evaluation.metrics import (
    GATE_K,
    GATE_THRESHOLD,
    compare,
    precision_at_k,
    variance_explained,
)
from ariadne.core.evaluation.splits import temporal_split
from ariadne.core.recommend.decomposition import Layer, decompose_split
from ariadne.core.taste.expectation import fit_rich_expectation

RNG = np.random.default_rng(20260727)


def film(
    index: int,
    rating: float,
    *,
    logged: date | None = None,
    year: int = 2000,
    country: str = "US",
    genres: tuple[str, ...] = ("Drama",),
    vote_average: float = 6.0,
    crew: dict[str, tuple[int, ...]] | None = None,
) -> RatedFilm:
    return RatedFilm(
        letterboxd_uri=f"https://boxd.it/{index}",
        tmdb_id=index,
        title=f"Film {index}",
        rating=rating,
        logged_date=logged,
        year=year,
        vote_average=vote_average,
        vote_count=500,
        country=country,
        genres=genres,
        crew=crew or {},
    )


# --- variance explained: hand-checkable ------------------------------------------------


def test_perfect_predictions_explain_everything():
    actual = np.array([1.0, 3.0, 5.0])

    assert variance_explained(actual.copy(), actual) == pytest.approx(1.0)


def test_predicting_the_mean_explains_nothing():
    actual = np.array([1.0, 3.0, 5.0])
    predicted = np.full(3, actual.mean())

    assert variance_explained(predicted, actual) == pytest.approx(0.0)


def test_a_predictor_worse_than_the_mean_is_reported_negative():
    """Not clipped at zero. A layer can genuinely make things worse, and saying so is the point."""
    actual = np.array([1.0, 3.0, 5.0])
    predicted = np.array([5.0, 3.0, 1.0])

    assert variance_explained(predicted, actual) < 0.0


def test_variance_explained_is_zero_when_every_rating_is_identical():
    """No variation to explain, and the denominator would be zero."""
    actual = np.full(10, 4.0)

    assert variance_explained(np.full(10, 3.0), actual) == 0.0


def test_variance_explained_needs_at_least_two_films():
    assert variance_explained(np.array([3.0]), np.array([4.0])) == 0.0


# --- the generalised paired bootstrap --------------------------------------------------


def test_compare_measures_the_metric_it_is_given():
    """The observed difference must be the difference in the supplied metric, not in precision."""
    actual = np.array([1.0, 2.0, 4.0, 5.0] * 30, dtype=float)
    good = actual.copy()
    bad = np.full(len(actual), actual.mean())

    result = compare("good", good, "mean", bad, actual, resamples=50, metric=variance_explained)

    assert result.observed_diff == pytest.approx(
        variance_explained(good, actual) - variance_explained(bad, actual)
    )
    assert result.prob_a_better == 1.0


def test_compare_without_a_metric_still_measures_gate_precision():
    """Regression guard: the gate comparisons in the writeup call this with no metric argument."""
    actual = np.array([5.0] * 50 + [2.0] * 50, dtype=float)
    good = actual.copy()
    bad = -actual

    result = compare("good", good, "bad", bad, actual, resamples=50)

    assert result.observed_diff == pytest.approx(
        precision_at_k(good, actual, GATE_K, GATE_THRESHOLD)
        - precision_at_k(bad, actual, GATE_K, GATE_THRESHOLD)
    )


# --- feature ablation ------------------------------------------------------------------


def test_dropping_a_feature_block_removes_it_from_the_model():
    train = [
        film(i, 3.0 + (i % 5) * 0.5, year=1950 + i, country="JP" if i % 2 else "US")
        for i in range(40)
    ]

    full = fit_rich_expectation(train)
    consensus = fit_rich_expectation(train, features=())

    assert full.countries and full.decades and full.genres
    assert consensus.countries == () and consensus.decades == () and consensus.genres == ()


def test_selecting_one_feature_block_excludes_the_others():
    train = [film(i, 3.0 + (i % 5) * 0.5, country="JP" if i % 2 else "US") for i in range(40)]

    genre_only = fit_rich_expectation(train, features=("genre",))

    assert genre_only.genres == ("Drama",)
    assert genre_only.countries == ()
    assert genre_only.decades == ()


def test_the_consensus_model_still_predicts_from_votes():
    """`features=()` must leave a working model, not an empty one that falls back to the mean."""
    train = [film(i, 2.0 + (i % 7) * 0.5, vote_average=4.0 + (i % 7) * 0.5) for i in range(40)]
    consensus = fit_rich_expectation(train, features=())

    predictions = consensus.predict(train)

    assert predictions.std() > 0.0


# --- the genre rung was degenerate -----------------------------------------------------


def test_the_genre_rung_is_not_a_copy_of_the_context_rung():
    """D84 noted `genre_only` scored identically to `context`, because it fitted genre effects on
    residuals from a model that already used genre. It is now built on the popularity expectation,
    so the rung measures something (D108)."""
    train = [
        film(
            i,
            2.0 + (i % 7) * 0.5,
            genres=("Horror",) if i % 3 else ("Drama",),
            country="JP" if i % 2 else "US",
            year=1960 + i,
        )
        for i in range(60)
    ]
    genre, context = GenreOnly(), Context()
    genre.fit(train)
    context.fit(train)

    assert not np.allclose(genre.predict(train), context.predict(train))


# --- person effects over a supplied expectation ----------------------------------------


def test_person_effects_accept_an_expectation_factory():
    """The decomposition fits director effects over an ablated context model, so the factory form
    has to work as well as the two named ones — and the supplied factory has to actually be used.

    The library gives Japanese films a two-star lift that consensus cannot see, so a model with the
    country block predicts differently from one without it. Two ways this test can pass without
    measuring anything, both hit on the way to writing it: a constant vote_average makes both models
    fall back to the mean, and directors assigned on `i % 4` beside a country assigned on `i % 2`
    give each director exactly one country, so director effects proxy country and both models fit
    perfectly.
    """
    train = [
        film(
            i,
            2.0 + (2.0 if i % 2 else 0.0),
            country="JP" if i % 2 else "US",
            vote_average=5.0 + (i % 5) * 0.3,
            crew={DIRECTOR: (i % 3,)},
        )
        for i in range(40)
    ]

    with_country = _PersonEffects("a", "", DIRECTOR, "rich")
    without_country = _PersonEffects(
        "b", "", DIRECTOR, lambda rows: fit_rich_expectation(rows, features=())
    )
    with_country.fit(train)
    without_country.fit(train)

    assert with_country.predict(train).std() > 0.0
    assert not np.allclose(with_country.predict(train), without_country.predict(train))


# --- overlap arithmetic ----------------------------------------------------------------


def _layer(label: str, diff: float) -> Layer:
    actual = np.array([1.0, 2.0, 4.0, 5.0] * 10, dtype=float)
    shifted = actual + diff
    comparison = compare(label, shifted, "base", actual, actual, resamples=20)
    return Layer(label=label, detail="", explained=comparison, ranking=comparison)


def test_overlap_is_the_gap_between_the_marginals_and_the_total():
    """The layers share information, so measuring them one at a time double-counts. The display
    shows the gap rather than presenting the marginals as if they partitioned anything."""
    from ariadne.core.recommend.decomposition import Decomposition

    report = Decomposition(split="x", note="", test_n=10, base_explained=0.2, base_gate=0.5)
    report.layers = [_layer("a", 0.1), _layer("b", 0.2)]
    report.combined = _layer("all", 0.25)

    assert report.sum_of_layers == pytest.approx(
        sum(layer.explained.observed_diff for layer in report.layers)
    )
    assert report.overlap == pytest.approx(
        report.sum_of_layers - report.combined.explained.observed_diff
    )


def test_overlap_is_zero_when_nothing_was_measured_together():
    from ariadne.core.recommend.decomposition import Decomposition

    assert (
        Decomposition(split="x", note="", test_n=0, base_explained=0.0, base_gate=0.0).overlap
        == 0.0
    )


# --- the assembled decomposition -------------------------------------------------------


def _library() -> list[RatedFilm]:
    """A library with a real director effect, spread across the temporal cut."""
    films = []
    for i in range(300):
        favourite = i % 5 == 0
        films.append(
            film(
                i,
                min(5.0, 2.5 + (1.5 if favourite else 0.0) + RNG.normal(0, 0.4)),
                logged=date(2023, 1, 1) if i % 2 else date(2024, 6, 1),
                year=1970 + (i % 5) * 10,
                country="JP" if i % 3 else "US",
                genres=("Horror",) if i % 4 else ("Drama",),
                vote_average=5.0 + (i % 5) * 0.4,
                crew={DIRECTOR: (1 if favourite else 2 + i % 7,)},
            )
        )
    return films


def test_the_base_score_is_reported_in_the_gate_configuration():
    """Regression guard. `base_gate` was computed with precision_at_k's defaults — P@20 at >= 4.0 —
    while every layer difference beside it was a difference in P@100 at >= 4.5. The base read 0.800
    against layers moving a number whose actual value was 0.680 (D109)."""
    split = temporal_split(_library())
    report = decompose_split(split, resamples=20)

    actual = np.array([f.rating for f in split.test], dtype=float)
    consensus = fit_rich_expectation(split.train, features=())
    expected = precision_at_k(consensus.predict(split.test), actual, GATE_K, GATE_THRESHOLD)

    assert report.base_gate == pytest.approx(expected)
    assert report.base_gate != pytest.approx(precision_at_k(consensus.predict(split.test), actual))


def test_the_decomposition_finds_a_director_effect_that_is_really_there():
    """The synthetic library gives every fifth film the same director and a +1.5-star lift, so the
    director layer must clear zero. A decomposition that cannot recover a planted effect cannot be
    trusted when it reports one is absent."""
    report = decompose_split(temporal_split(_library()), resamples=200)
    director = next(layer for layer in report.layers if layer.label == "who directed it")

    assert director.explained.observed_diff > 0.0
    assert director.helps


def test_every_layer_is_measured_against_the_same_base():
    report = decompose_split(temporal_split(_library()), resamples=20)

    assert {layer.explained.name_b for layer in report.layers} == {"base"}
    assert len(report.layers) == 6
    assert report.combined is not None
