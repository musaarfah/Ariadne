"""The evaluation harness: splits, metrics, expectation, shrinkage, baselines.

Built and tested before any real model exists, so that no model change is ever scored by
intuition.
"""

import json
from datetime import date

import numpy as np
import pytest

from ariadne.constants import TEMPORAL_SPLIT_DATE
from ariadne.core.catalog.roles import DIRECTOR
from ariadne.core.evaluation.baselines import (
    DirectorOnly,
    GenreOnly,
    GlobalMean,
    Popularity,
    ladder,
)
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.evaluation.harness import evaluate, run_payload
from ariadne.core.evaluation.metrics import (
    GATE_K,
    GATE_THRESHOLD,
    precision_at_k,
    precision_grid,
    score,
)
from ariadne.core.evaluation.splits import (
    describe,
    random_split,
    shuffled_ratings,
    temporal_split,
)
from ariadne.core.taste.expectation import fit_expectation
from ariadne.core.taste.shrinkage import estimate_k, shrink


def make_film(
    rating: float,
    *,
    uri: str = "u",
    logged: date | None = None,
    vote_average: float | None = 7.0,
    genres: tuple[str, ...] = ("Drama",),
    director: int | None = None,
) -> RatedFilm:
    return RatedFilm(
        letterboxd_uri=uri,
        tmdb_id=abs(hash(uri)) % 10_000_000,
        title=uri,
        rating=rating,
        logged_date=logged,
        year=2010,
        vote_average=vote_average,
        vote_count=1000,
        country="US",
        genres=genres,
        crew={DIRECTOR: (director,)} if director is not None else {},
    )


# --- metrics ---------------------------------------------------------------------------


def test_constant_predictions_do_not_produce_nan():
    """The standard deviation of 100 identical floats is 1.3e-15, not zero.

    An exact `std() == 0` guard therefore never fired, spearman came back NaN, and the run was
    unwritable because JSON has no NaN. The guard now compares min to max.
    """
    predicted = np.full(100, 3.342)
    actual = np.random.default_rng(0).choice([1.0, 2.0, 3.0, 4.0, 5.0], 100)

    metrics = score(predicted, actual)

    assert metrics.spearman == 0.0
    assert "NaN" not in json.dumps(metrics.as_dict())


def test_every_metric_survives_json():
    metrics = score(np.array([1.0, 2.0, 3.0]), np.array([5.0, 4.0, 3.0]))
    assert json.loads(json.dumps(metrics.as_dict()))["n"] == 3


def test_precision_at_k_counts_only_the_top_k():
    predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    actual = np.array([5.0, 5.0, 1.0, 1.0, 1.0])
    assert precision_at_k(predicted, actual, k=2) == 1.0
    assert precision_at_k(predicted, actual, k=4) == 0.5


def test_precision_at_k_handles_k_larger_than_the_set():
    predicted = np.array([2.0, 1.0])
    actual = np.array([5.0, 1.0])
    assert precision_at_k(predicted, actual, k=100) == 0.5


def test_base_rate_and_lift_are_reported_together():
    """P@k alone is uninterpretable: 0.70 is excellent at a 0.30 base rate and poor at 0.70."""
    actual = np.array([5.0] * 5 + [1.0] * 5)
    metrics = score(np.arange(10, dtype=float), actual)
    assert metrics.base_rate == 0.5
    assert metrics.lift == pytest.approx(metrics.precision_at_k - metrics.base_rate)


def test_gate_metric_is_reported_alongside_the_product_metric():
    rng = np.random.default_rng(3)
    metrics = score(rng.random(400) * 5, rng.choice([1.0, 3.0, 4.5, 5.0], 400))
    payload = metrics.as_dict()
    assert payload["gate_k"] == GATE_K
    assert payload["gate_threshold"] == GATE_THRESHOLD


def test_centred_mae_removes_the_block_shift():
    """Predictions a constant above the truth are perfectly ranked but wrong on absolute scale."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = actual + 0.5
    metrics = score(predicted, actual)
    assert metrics.mae == pytest.approx(0.5)
    assert metrics.mae_centred == pytest.approx(0.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        score(np.array([1.0]), np.array([1.0, 2.0]))


def test_precision_grid_covers_thresholds_and_ks():
    rng = np.random.default_rng(4)
    grid = precision_grid(rng.random(300) * 5, rng.choice([1.0, 4.0, 5.0], 300))
    assert set(grid) == {4.0, 4.5, 5.0}
    assert 100 in grid[4.5]


# --- splits ----------------------------------------------------------------------------


def test_temporal_split_cuts_on_the_documented_date():
    before = make_film(4.0, uri="a", logged=date(2023, 6, 1))
    after = make_film(5.0, uri="b", logged=date(2024, 6, 1))
    split = temporal_split([before, after])

    assert split.train == [before]
    assert split.test == [after]
    assert TEMPORAL_SPLIT_DATE.isoformat() in split.name


def test_films_without_a_log_date_are_dropped_and_counted():
    split = temporal_split([make_film(4.0, uri="a", logged=None)])
    assert split.train == []
    assert split.dropped == 1


def test_drift_reports_the_shift_between_blocks():
    train = [make_film(3.0, uri=f"t{i}", logged=date(2023, 1, 1)) for i in range(10)]
    test = [make_film(5.0, uri=f"s{i}", logged=date(2024, 1, 1)) for i in range(10)]
    drift = temporal_split(train + test).drift()

    assert drift.mean_shift == pytest.approx(2.0)
    assert drift.test.top_rating_share == 1.0
    assert drift.train.top_rating_share == 0.0


def test_random_split_is_reproducible():
    films = [make_film(float(i % 5) + 1, uri=f"f{i}") for i in range(100)]
    first = random_split(films)
    second = random_split(films)
    assert [f.letterboxd_uri for f in first.test] == [f.letterboxd_uri for f in second.test]


def test_describe_handles_a_single_film():
    stats = describe([make_film(4.0)])
    assert stats.n == 1
    assert stats.sd == 0.0


def test_shuffling_preserves_the_ratings_but_moves_them():
    films = [make_film(float(i % 5) + 1, uri=f"f{i}") for i in range(50)]
    shuffled = shuffled_ratings(films)

    assert sorted(f.rating for f in shuffled) == sorted(f.rating for f in films)
    assert [f.letterboxd_uri for f in shuffled] == [f.letterboxd_uri for f in films]
    assert [f.rating for f in shuffled] != [f.rating for f in films]


# --- expectation -----------------------------------------------------------------------


def test_expectation_learns_the_scale_rather_than_assuming_half():
    """TMDB is 0-10 and the user is 0.5-5.0, but the relationship is not simply half."""
    train = [
        make_film(1.0, uri="a", vote_average=2.0),
        make_film(2.0, uri="b", vote_average=4.0),
        make_film(3.0, uri="c", vote_average=6.0),
        make_film(4.0, uri="d", vote_average=8.0),
    ]
    expectation = fit_expectation(train)
    assert expectation.slope == pytest.approx(0.5, abs=1e-6)
    assert expectation.predict_one(make_film(0, vote_average=10.0)) == pytest.approx(5.0, abs=1e-6)


def test_expectation_falls_back_when_vote_average_is_missing():
    train = [make_film(4.0, uri="a", vote_average=None), make_film(2.0, uri="b", vote_average=None)]
    expectation = fit_expectation(train)
    assert expectation.predict_one(make_film(0, vote_average=None)) == pytest.approx(3.0)
    assert expectation.fitted_on == 0


def test_expectation_survives_a_constant_vote_average():
    train = [make_film(4.0, uri="a", vote_average=7.0), make_film(2.0, uri="b", vote_average=7.0)]
    expectation = fit_expectation(train)
    assert expectation.slope == 0.0


def test_residuals_are_rating_minus_expectation():
    train = [make_film(float(i), uri=f"f{i}", vote_average=float(i) * 2) for i in range(1, 5)]
    expectation = fit_expectation(train)
    assert np.allclose(expectation.residuals(train), 0.0, atol=1e-9)


# --- shrinkage -------------------------------------------------------------------------


def test_small_groups_are_shrunk_harder_than_large_ones():
    """Same mean, different evidence. The two-film group must move further toward zero.

    Both groups need genuine within-group spread for this to mean anything: with identical
    values there is no noise to shrink away, and no shrinkage is the correct answer.
    """
    groups = {1: [2.0, 0.0], 2: [1.5, 0.5] * 20}
    effects = shrink(groups)

    assert effects[1].raw_mean == pytest.approx(effects[2].raw_mean)
    assert effects[1].n < effects[2].n
    assert effects[1].shrunk < effects[2].shrunk


def test_identical_observations_are_not_shrunk():
    """No within-group spread means no noise, so the group mean is known exactly."""
    effects = shrink({1: [1.0, 1.0], 2: [1.0] * 40})
    assert effects[1].shrunk == pytest.approx(1.0)
    assert effects[2].shrunk == pytest.approx(1.0)


def test_shrinkage_never_exceeds_the_raw_mean():
    effects = shrink({1: [2.0] * 5, 2: [-2.0] * 5})
    for effect in effects.values():
        assert abs(effect.shrunk) <= abs(effect.raw_mean) + 1e-12


def test_no_real_spread_means_no_effects():
    """If every group looks the same, the differences are noise and must shrink away."""
    rng = np.random.default_rng(7)
    groups = {i: list(rng.normal(0, 1, 20)) for i in range(30)}
    effects = shrink(groups)
    assert max(abs(e.shrunk) for e in effects.values()) < 0.35


def test_genuine_spread_survives_shrinkage():
    groups = {i: [2.0] * 20 if i % 2 else [-2.0] * 20 for i in range(20)}
    effects = shrink(groups)
    assert max(abs(e.shrunk) for e in effects.values()) > 1.5


def test_estimate_k_is_larger_when_the_signal_is_weaker():
    rng = np.random.default_rng(9)
    noisy = {i: list(rng.normal(0, 1, 5)) for i in range(20)}
    clear = {i: [3.0] * 5 if i % 2 else [-3.0] * 5 for i in range(20)}
    assert estimate_k(noisy) > estimate_k(clear)


def test_shrinkage_of_nothing_is_nothing():
    assert shrink({}) == {}


# --- baselines -------------------------------------------------------------------------


def test_global_mean_predicts_one_number():
    predictor = GlobalMean()
    predictor.fit([make_film(2.0, uri="a"), make_film(4.0, uri="b")])
    assert list(predictor.predict([make_film(0.0)])) == [3.0]


def test_popularity_tracks_vote_average():
    train = [make_film(float(i), uri=f"f{i}", vote_average=float(i) * 2) for i in range(1, 6)]
    predictor = Popularity()
    predictor.fit(train)
    predictions = predictor.predict(train)
    assert predictions[0] < predictions[-1]


def test_predictors_refuse_to_predict_before_fitting():
    with pytest.raises(RuntimeError):
        Popularity().predict([make_film(1.0)])
    with pytest.raises(RuntimeError):
        DirectorOnly().predict([make_film(1.0)])


def test_director_baseline_learns_a_preference():
    loved, hated = 1, 2
    train = [make_film(5.0, uri=f"l{i}", director=loved) for i in range(10)]
    train += [make_film(1.0, uri=f"h{i}", director=hated) for i in range(10)]

    predictor = DirectorOnly()
    predictor.fit(train)
    predictions = predictor.predict(
        [make_film(0.0, uri="x", director=loved), make_film(0.0, uri="y", director=hated)]
    )
    assert predictions[0] > predictions[1]


def test_unknown_director_gets_no_adjustment():
    train = [make_film(5.0, uri=f"l{i}", director=1) for i in range(10)]
    predictor = DirectorOnly()
    predictor.fit(train)

    unseen = make_film(0.0, uri="new", director=999)
    no_crew = make_film(0.0, uri="none", director=None)
    assert predictor.predict([unseen])[0] == pytest.approx(predictor.predict([no_crew])[0])


def test_genre_baseline_learns_a_preference():
    train = [make_film(5.0, uri=f"a{i}", genres=("Horror",)) for i in range(10)]
    train += [make_film(1.0, uri=f"b{i}", genres=("Comedy",)) for i in range(10)]

    predictor = GenreOnly()
    predictor.fit(train)
    predictions = predictor.predict(
        [make_film(0.0, uri="x", genres=("Horror",)), make_film(0.0, uri="y", genres=("Comedy",))]
    )
    assert predictions[0] > predictions[1]


def test_ladder_is_ordered_floor_first():
    names = [p.name for p in ladder()]
    assert names == ["global_mean", "popularity", "genre_only", "director_only"]


# --- harness ---------------------------------------------------------------------------


def _dataset() -> list[RatedFilm]:
    films = []
    for i in range(120):
        director = 1 if i % 2 else 2
        rating = 5.0 if director == 1 else 2.0
        logged = date(2023, 1, 1) if i < 80 else date(2024, 6, 1)
        films.append(
            make_film(rating, uri=f"f{i}", logged=logged, vote_average=6.0, director=director)
        )
    return films


def test_evaluate_scores_every_predictor_on_both_splits():
    results = evaluate(_dataset())
    assert [r.split.split("@")[0] for r in results] == ["temporal", "random"]
    for split in results:
        assert [r.name for r in split.results] == [p.name for p in ladder()]


def test_evaluate_finds_the_planted_director_effect():
    results = evaluate(_dataset())
    temporal = results[0]
    director = temporal.by_name("director_only")
    floor = temporal.by_name("global_mean")
    assert director is not None and floor is not None
    assert director.metrics.spearman > floor.metrics.spearman


def test_run_payload_is_json_serialisable():
    payload = run_payload(evaluate(_dataset()), films=120)
    encoded = json.dumps(payload)
    assert "NaN" not in encoded
    assert json.loads(encoded)["films"] == 120


def test_shuffled_ratings_destroy_the_effect():
    """The Level 4 negative control the 1.7 model must pass, exercised on a baseline here."""
    real = evaluate(_dataset())[0].by_name("director_only")
    shuffled = evaluate(shuffled_ratings(_dataset()))[0].by_name("director_only")
    assert real is not None and shuffled is not None
    assert abs(shuffled.metrics.spearman) < abs(real.metrics.spearman)
