"""Track 1 crew effects and the negative controls."""

from datetime import date

import numpy as np
import pytest

from ariadne.core.catalog.roles import CINEMATOGRAPHER, DIRECTOR, EDITOR
from ariadne.core.evaluation.controls import (
    MAX_COUNT_RATIO,
    MAX_EFFECT_RATIO,
    build_null,
    observed_noise_sd,
    quantise,
    shuffle_test,
    sweep,
)
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.taste.crew import MIN_FILMS_TO_REPORT, CrewModel


def film(
    rating: float,
    *,
    uri: str,
    editor: int | None = None,
    dp: int | None = None,
    director: int | None = None,
    vote_average: float = 7.0,
) -> RatedFilm:
    crew: dict[str, tuple[int, ...]] = {}
    if editor is not None:
        crew[EDITOR] = (editor,)
    if dp is not None:
        crew[CINEMATOGRAPHER] = (dp,)
    if director is not None:
        crew[DIRECTOR] = (director,)
    return RatedFilm(
        letterboxd_uri=uri,
        tmdb_id=abs(hash(uri)) % 10_000_000,
        title=uri,
        rating=rating,
        logged_date=date(2023, 1, 1),
        year=2010,
        vote_average=vote_average,
        vote_count=1000,
        country="US",
        genres=("Drama",),
        crew=crew,
    )


def library(loved_editor: int = 1, n_loved: int = 14, n_other: int = 200) -> list[RatedFilm]:
    """A library where one editor's films are consistently rated above consensus.

    The loved editor's ratings vary rather than all being 5.0. With identical values there is no
    within-group spread, so nothing needs shrinking and shrinkage correctly does nothing — which
    makes the fixture unable to exercise the behaviour it is meant to test.
    """
    films = [
        film(4.5 if i % 3 else 5.0, uri=f"loved{i}", editor=loved_editor) for i in range(n_loved)
    ]
    films += [film(3.0 + (i % 3) * 0.5, uri=f"other{i}", editor=100 + i) for i in range(n_other)]
    return films


# --- fitting ---------------------------------------------------------------------------


def test_a_real_preference_is_found():
    model = CrewModel(roles=(EDITOR,))
    model.fit(library())

    top = model.reportable_effects(EDITOR)[0]
    assert top.person_id == 1
    assert top.effect > 0


def test_effects_are_shrunk_toward_zero():
    model = CrewModel(roles=(EDITOR,))
    model.fit(library())
    top = model.reportable_effects(EDITOR)[0]
    assert abs(top.effect) < abs(top.raw_mean)


def test_no_within_group_spread_means_no_shrinkage():
    """A group whose observations are identical has a mean known exactly, so it is not shrunk."""
    films = [film(5.0, uri=f"a{i}", editor=1) for i in range(14)]
    films += [film(3.0, uri=f"b{i}", editor=100 + i) for i in range(100)]

    model = CrewModel(roles=(EDITOR,))
    model.fit(films)
    top = next(e for e in model.effects(EDITOR) if e.person_id == 1)
    assert top.effect == pytest.approx(top.raw_mean)


def test_people_below_the_threshold_are_withheld_not_dropped():
    """They stay fitted so the graph can be traversed through them (D65), just not reported."""
    model = CrewModel(roles=(EDITOR,))
    model.fit(library())

    fitted = model.effects(EDITOR)
    reportable = model.reportable_effects(EDITOR)
    assert len(fitted) > len(reportable)
    assert all(e.n_films >= MIN_FILMS_TO_REPORT for e in reportable)


def test_predict_requires_fit():
    with pytest.raises(RuntimeError):
        CrewModel().predict([film(4.0, uri="x")])


def test_prediction_moves_with_the_effect():
    model = CrewModel(roles=(EDITOR,))
    model.fit(library())

    loved = film(0.0, uri="new-loved", editor=1)
    unknown = film(0.0, uri="new-unknown", editor=9999)
    assert model.predict([loved])[0] > model.predict([unknown])[0]


def test_roles_are_averaged_not_summed():
    """Two liked collaborators are not evidence of twice the effect."""
    films = [film(5.0, uri=f"a{i}", editor=1, dp=2) for i in range(14)]
    films += [film(3.0, uri=f"b{i}", editor=100 + i, dp=200 + i) for i in range(120)]

    both = CrewModel(roles=(EDITOR, CINEMATOGRAPHER))
    both.fit(films)
    editor_only = CrewModel(roles=(EDITOR,))
    editor_only.fit(films)

    target = film(0.0, uri="t", editor=1, dp=2)
    combined = both.predict([target])[0]
    single = editor_only.predict([target])[0]
    assert combined == pytest.approx(single, abs=0.05)


def test_director_is_excluded_by_default():
    """The thesis is that below-the-line crew beats the director, so it cannot include them."""
    assert DIRECTOR not in CrewModel().roles


# --- inseparability --------------------------------------------------------------------


def test_a_person_tied_to_one_director_is_flagged():
    films = [film(5.0, uri=f"a{i}", dp=7, director=42) for i in range(13)]
    films += [film(3.0, uri=f"b{i}", dp=100 + i, director=200 + i) for i in range(120)]

    model = CrewModel(roles=(CINEMATOGRAPHER,))
    model.fit(films)
    effect = next(e for e in model.effects(CINEMATOGRAPHER) if e.person_id == 7)

    assert effect.inseparable_from == 42
    assert effect.director_overlap == 1.0


def test_a_person_across_many_directors_is_not_flagged():
    films = [film(5.0, uri=f"a{i}", dp=7, director=i) for i in range(13)]
    films += [film(3.0, uri=f"b{i}", dp=100 + i, director=500 + i) for i in range(120)]

    model = CrewModel(roles=(CINEMATOGRAPHER,))
    model.fit(films)
    effect = next(e for e in model.effects(CINEMATOGRAPHER) if e.person_id == 7)

    assert effect.inseparable_from is None
    assert effect.director_overlap < 0.2


def test_near_total_overlap_is_reported_even_when_not_absolute():
    """Nine of ten films sharing a director is worth surfacing, not hiding behind an exact test."""
    films = [film(5.0, uri=f"a{i}", dp=7, director=42) for i in range(12)]
    films += [film(5.0, uri="odd", dp=7, director=99)]
    films += [film(3.0, uri=f"b{i}", dp=100 + i, director=200 + i) for i in range(120)]

    model = CrewModel(roles=(CINEMATOGRAPHER,))
    model.fit(films)
    effect = next(e for e in model.effects(CINEMATOGRAPHER) if e.person_id == 7)

    assert effect.inseparable_from is None
    assert effect.director_overlap == pytest.approx(12 / 13)


def test_one_film_is_not_a_pattern():
    films = [film(5.0, uri="only", dp=7, director=42)]
    films += [film(3.0, uri=f"b{i}", dp=100 + i, director=200 + i) for i in range(20)]

    model = CrewModel(roles=(CINEMATOGRAPHER,))
    model.fit(films)
    effect = next(e for e in model.effects(CINEMATOGRAPHER) if e.person_id == 7)
    assert effect.inseparable_from is None


# --- the shuffle test ------------------------------------------------------------------


def test_shuffling_destroys_a_planted_effect():
    result = shuffle_test(library())
    assert result.collapsed
    assert result.max_ratio <= MAX_EFFECT_RATIO
    assert result.count_ratio <= MAX_COUNT_RATIO


def test_the_pass_criterion_rejects_a_near_miss():
    """The first version only asked whether shuffled < real, and passed 1.111 against 1.147."""
    from ariadne.core.evaluation.controls import ShuffleResult

    marginal = ShuffleResult(
        real_max_effect=1.147,
        shuffled_max_effect=1.111,
        real_reportable=106,
        shuffled_reportable=106,
        real_above_threshold=44,
        shuffled_above_threshold=37,
        threshold=0.25,
    )
    assert not marginal.collapsed


def test_noise_alone_produces_no_reportable_signal():
    rng = np.random.default_rng(11)
    films = [
        film(float(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])), uri=f"f{i}", editor=i % 30)
        for i in range(400)
    ]
    model = CrewModel(roles=(EDITOR,))
    model.fit(films)
    assert model.max_absolute_effect() < 0.35


# --- synthetic sweep -------------------------------------------------------------------


def test_quantise_matches_the_letterboxd_scale():
    values = quantise(np.array([-1.0, 0.6, 3.3, 4.8, 9.0]))
    assert values.tolist() == [0.5, 0.5, 3.5, 5.0, 5.0]
    assert all(v * 2 == int(v * 2) for v in values)


def test_noise_sd_reflects_the_data():
    tight = [film(3.0, uri=f"t{i}") for i in range(50)]
    assert observed_noise_sd(tight) < 0.1


def test_sweep_recovers_large_effects_more_often_than_small_ones():
    films = [film(3.0, uri=f"f{i}", editor=i % 40) for i in range(240)]
    result = sweep(films, EDITOR, effect_sizes=(0.25, 2.0), film_counts=(6,), trials=6)

    small = next(c for c in result.cells if c.effect_size == 0.25)
    large = next(c for c in result.cells if c.effect_size == 2.0)
    assert large.rate >= small.rate


def test_sweep_skips_film_counts_with_no_candidates():
    films = [film(3.0, uri=f"f{i}", editor=1) for i in range(5)]
    result = sweep(films, EDITOR, effect_sizes=(1.0,), film_counts=(99,), trials=2)
    assert result.cells == []


def test_floor_is_none_when_nothing_is_detected():
    films = [film(3.0, uri=f"f{i}", editor=i % 40) for i in range(240)]
    result = sweep(films, EDITOR, effect_sizes=(0.01,), film_counts=(6,), trials=4)
    assert result.floor_for(0.01) is None


# --- permutation null ------------------------------------------------------------------


def test_null_is_built_from_shuffled_data_only():
    films = library()
    null = build_null(films, EDITOR, permutations=15, min_films=8)

    model = CrewModel(roles=(EDITOR,))
    model.fit(films)
    observed = abs(model.reportable_effects(EDITOR)[0].effect)

    assert null.critical_value < observed


def test_p_value_can_never_be_zero():
    """An observed effect is never reported as impossible under noise."""
    null = build_null(library(), EDITOR, permutations=10, min_films=8)
    assert null.p_value(999.0) > 0


def test_a_noise_sized_effect_is_not_significant():
    null = build_null(library(), EDITOR, permutations=20, min_films=8)
    assert null.p_value(0.0) == pytest.approx(1.0)


def test_null_uses_the_maximum_per_permutation():
    """Testing hundreds of people means the null has to be over the best noise achieved."""
    null = build_null(library(), EDITOR, permutations=12, min_films=8)
    assert len(null.max_effects) == 12
    assert null.percentile(95) >= null.percentile(50)


# --- across-role combination -----------------------------------------------------------


def test_across_strategies_all_produce_a_prediction():
    films = library()
    target = film(0.0, uri="t", editor=1, dp=2)

    seen = {}
    for across in ("mean", "max", "sum", "sum_scaled"):
        model = CrewModel(roles=(EDITOR, CINEMATOGRAPHER), across=across)
        model.fit(films)
        seen[across] = float(model.predict([target])[0])

    assert len(seen) == 4
    assert all(np.isfinite(v) for v in seen.values())


def test_sum_is_at_least_as_large_as_mean_when_effects_agree():
    """With every contributing role pointing the same way, summing cannot be smaller."""
    films = [film(5.0 if i % 3 else 4.5, uri=f"a{i}", editor=1, dp=2) for i in range(14)]
    films += [
        film(3.0 + (i % 3) * 0.5, uri=f"b{i}", editor=100 + i, dp=200 + i) for i in range(150)
    ]
    target = film(0.0, uri="t", editor=1, dp=2)

    mean_model = CrewModel(roles=(EDITOR, CINEMATOGRAPHER), across="mean")
    mean_model.fit(films)
    sum_model = CrewModel(roles=(EDITOR, CINEMATOGRAPHER), across="sum")
    sum_model.fit(films)

    mean_adj = mean_model.predict([target])[0] - mean_model.expectation_for([target])[0]
    sum_adj = sum_model.predict([target])[0] - sum_model.expectation_for([target])[0]
    assert sum_adj >= mean_adj - 1e-9


def test_scaled_sum_sits_between_mean_and_sum():
    films = [film(5.0 if i % 3 else 4.5, uri=f"a{i}", editor=1, dp=2) for i in range(14)]
    films += [
        film(3.0 + (i % 3) * 0.5, uri=f"b{i}", editor=100 + i, dp=200 + i) for i in range(150)
    ]
    target = film(0.0, uri="t", editor=1, dp=2)

    def adjustment(across: str) -> float:
        model = CrewModel(roles=(EDITOR, CINEMATOGRAPHER), across=across)
        model.fit(films)
        return float(model.predict([target])[0] - model.expectation_for([target])[0])

    assert adjustment("mean") <= adjustment("sum_scaled") + 1e-9
    assert adjustment("sum_scaled") <= adjustment("sum") + 1e-9


def test_unknown_across_strategy_falls_back_to_mean():
    films = library()
    target = film(0.0, uri="t", editor=1)

    fallback = CrewModel(roles=(EDITOR,), across="nonsense")
    fallback.fit(films)
    default = CrewModel(roles=(EDITOR,), across="mean")
    default.fit(films)

    assert fallback.predict([target])[0] == pytest.approx(default.predict([target])[0])
