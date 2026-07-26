"""The tier-1 portrait.

These features carry no uncertainty, which makes them easy to get quietly wrong in a different way:
a count over the wrong denominator reads exactly as confident as a correct one. The tests here are
mostly about denominators and exclusions.
"""

import pytest

from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.recommend.portrait import (
    MIN_VOTES_FOR_RESIDUAL,
    blind_spots,
    build_portrait,
    disagreements,
    loyalties,
    rating_style,
    revealed_preference,
)


def film(
    tmdb_id: int,
    rating: float,
    *,
    title: str = "",
    vote_average: float | None = 6.0,
    vote_count: int | None = 500,
    year: int | None = 2000,
    country: str | None = "US",
    genres: tuple[str, ...] = ("Drama",),
    crew: dict[str, tuple[int, ...]] | None = None,
    rewatches: int = 0,
    in_diary: bool = False,
) -> RatedFilm:
    return RatedFilm(
        letterboxd_uri=f"https://boxd.it/{tmdb_id}",
        tmdb_id=tmdb_id,
        title=title or f"Film {tmdb_id}",
        rating=rating,
        logged_date=None,
        year=year,
        vote_average=vote_average,
        vote_count=vote_count,
        country=country,
        genres=genres,
        crew=crew or {},
        rewatches=rewatches,
        in_diary=in_diary,
    )


class FixedExpectation:
    """Predicts the same value for every film, so residuals are hand-checkable."""

    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def residuals(self, films: list[RatedFilm]) -> list[float]:
        return [f.rating - self.prediction for f in films]


# --- loyalties: counts, and only over the roles asked for -------------------------------


def test_loyalties_counts_films_per_person_and_role():
    films = [
        film(1, 4.0, crew={"actor": (10, 11), "director": (20,)}),
        film(2, 3.0, crew={"actor": (10,), "director": (21,)}),
        film(3, 5.0, crew={"actor": (10,)}),
    ]
    found = loyalties(films, ("actor", "director"))

    assert found[0].person_id == 10
    assert found[0].role == "actor"
    assert found[0].films == 3


def test_loyalties_ignores_roles_outside_the_requested_scope():
    """The research scope excludes actors (D97), the product scope includes them. The caller
    decides, and a role absent from `roles` must not leak into the counts."""
    films = [film(i, 4.0, crew={"actor": (10,), "director": (20,)}) for i in range(5)]

    assert [loyalty.role for loyalty in loyalties(films, ("director",))] == ["director"]


def test_same_person_in_two_roles_is_counted_separately():
    """A writer-director is two credits, not one. Collapsing them would double-count the person and
    make the count answer a different question than the label claims."""
    films = [film(i, 4.0, crew={"writer": (10,), "director": (10,)}) for i in range(4)]
    found = {loyalty.role: loyalty.films for loyalty in loyalties(films, ("writer", "director"))}

    assert found == {"writer": 4, "director": 4}


# --- disagreements: the vote floor is the whole point ----------------------------------


def test_low_vote_films_are_excluded_and_counted():
    """D101. A film nobody has rated carries a vote_average of 0, which produced an enormous
    artefactual residual and put unrated films at the top of "your boldest opinions"."""
    films = [
        film(1, 5.0, vote_average=0.0, vote_count=0),
        film(2, 5.0, vote_count=MIN_VOTES_FOR_RESIDUAL),
        film(3, 1.0, vote_count=MIN_VOTES_FOR_RESIDUAL),
    ]
    above, below, excluded = disagreements(films, FixedExpectation(3.0))

    assert excluded == 1
    assert [item.film.tmdb_id for item in above] == [2]
    assert [item.film.tmdb_id for item in below] == [3]


def test_missing_vote_count_is_treated_as_too_few():
    """`None` means TMDB told us nothing, which is strictly less information than zero votes."""
    films = [film(1, 5.0, vote_count=None), film(2, 5.0, vote_count=500)]
    above, _, excluded = disagreements(films, FixedExpectation(3.0))

    assert excluded == 1
    assert [item.film.tmdb_id for item in above] == [2]


def test_disagreements_split_by_direction_and_never_include_zero():
    films = [film(1, 4.0), film(2, 3.0), film(3, 2.0)]
    above, below, _ = disagreements(films, FixedExpectation(3.0))

    assert [item.film.tmdb_id for item in above] == [1]
    assert [item.film.tmdb_id for item in below] == [3]
    assert all(item.direction == "above" for item in above)
    assert all(item.direction == "below" for item in below)


def test_disagreements_are_ordered_by_distance_from_expectation():
    films = [film(1, 3.5), film(2, 5.0), film(3, 4.0)]
    above, _, _ = disagreements(films, FixedExpectation(3.0))

    assert [item.film.tmdb_id for item in above] == [2, 3, 1]


def test_a_film_never_appears_in_both_tails():
    """The two lists are built from opposite ends of one ordering, so with few enough films the
    slices can overlap unless the sign check holds."""
    films = [film(1, 5.0), film(2, 1.0)]
    above, below, _ = disagreements(films, FixedExpectation(3.0), limit=8)

    assert {item.film.tmdb_id for item in above}.isdisjoint({item.film.tmdb_id for item in below})


# --- revealed preference: the diary is the denominator ---------------------------------


def test_top_rated_never_revisited_counts_only_films_the_diary_covers():
    """F7. The diary starts part-way through the library, so a film with no diary record has an
    unknown rewatch count. Counting it as never-revisited would invent the finding."""
    films = [
        film(1, 5.0, in_diary=True, rewatches=0),
        film(2, 5.0, in_diary=True, rewatches=2),
        film(3, 5.0, in_diary=False, rewatches=0),
        film(4, 3.0, in_diary=True, rewatches=0),
    ]
    revealed = revealed_preference(films)

    assert revealed.top_rated_total == 3
    assert revealed.top_rated_in_diary == 2
    assert revealed.top_rated_never_revisited == 1
    assert revealed.revisit_rate == pytest.approx(0.5)


def test_revisit_rate_is_zero_when_the_diary_covers_no_top_rated_film():
    films = [film(1, 5.0, in_diary=False)]

    assert revealed_preference(films).revisit_rate == 0.0


def test_rewatched_films_are_ordered_by_rewatch_count():
    films = [
        film(1, 3.0, rewatches=1, in_diary=True),
        film(2, 5.0, rewatches=4, in_diary=True),
        film(3, 5.0, rewatches=2, in_diary=True),
    ]

    assert [f.tmdb_id for f in revealed_preference(films).rewatched] == [2, 3, 1]


# --- blind spots: bounded on both sides ------------------------------------------------


def test_blind_spots_require_a_minimum_number_of_films():
    """Below the floor a mean is noise, and a two-film 5.0 average would top every list."""
    films = [film(i, 3.0, genres=("Drama",)) for i in range(20)]
    films += [film(100 + i, 5.0, genres=("Noir",)) for i in range(2)]

    assert not [spot for spot in blind_spots(films, min_films=5) if spot.label == "genre Noir"]


def test_blind_spots_exclude_slices_that_are_really_habits():
    """A slice the user watches constantly is not a blind spot, however well they rate it."""
    films = [film(i, 3.0, genres=("Drama",)) for i in range(20)]
    films += [film(100 + i, 5.0, genres=("Noir",)) for i in range(30)]

    labels = [spot.label for spot in blind_spots(films, min_films=5, max_films=10)]

    assert "genre Noir" not in labels


def test_blind_spots_only_report_slices_above_the_library_mean():
    films = [film(i, 4.0, genres=("Drama",)) for i in range(20)]
    films += [film(100 + i, 1.0, genres=("Horror",)) for i in range(6)]

    assert all(spot.lift > 0 for spot in blind_spots(films, min_films=5))


def test_blind_spot_labels_say_what_kind_of_slice_they_are():
    """Decades, countries and genres share one ranked list, so a bare "JP" next to a bare "1950s"
    gave no clue what was being compared."""
    films = [film(i, 5.0, year=1955, country="JP", genres=("Western",)) for i in range(6)]
    films += [film(100 + i, 1.0, year=2020, country="US", genres=("Drama",)) for i in range(20)]

    labels = {spot.label for spot in blind_spots(films, min_films=5)}

    assert labels == {"decade 1950s", "country JP", "genre Western"}


# --- rating style ----------------------------------------------------------------------


def test_rating_style_measures_whole_stars_and_the_ceiling():
    films = [film(1, 5.0), film(2, 4.0), film(3, 3.0), film(4, 3.5)]
    style = rating_style(films)

    assert style.whole_star_share == pytest.approx(0.75)
    assert style.is_decisive
    assert style.levels_used == 4
    assert style.top_rating_share == pytest.approx(0.25)


def test_rating_style_of_a_calibrating_user_is_not_decisive():
    films = [film(i, 3.5 if i % 2 else 4.0) for i in range(10)]

    assert not rating_style(films).is_decisive


def test_rating_style_on_a_single_film_has_no_spread():
    """`ddof=1` on one observation divides by zero, and NaN into JSONB is rejected by Postgres."""
    assert rating_style([film(1, 4.0)]).sd == 0.0


def test_rating_style_of_an_empty_library_does_not_divide_by_zero():
    assert rating_style([]).mean == 0.0


# --- the assembled portrait ------------------------------------------------------------


def test_build_portrait_reports_the_full_film_count_not_the_eligible_one():
    """The headline count is the library; the vote floor applies only to the residual sections."""
    films = [film(1, 5.0, vote_count=0), film(2, 4.0), film(3, 2.0)]
    report = build_portrait(films, FixedExpectation(3.0), ("director",))

    assert report.films == 3
    assert report.excluded_low_votes == 1


def test_build_portrait_on_an_empty_library_returns_empty_sections():
    report = build_portrait([], FixedExpectation(3.0), ("director",))

    assert report.films == 0
    assert report.loyalties == []
    assert report.above == []
    assert report.blind_spots == []
