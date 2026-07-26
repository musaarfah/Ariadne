"""Assertions about the recorded TMDB responses.

These are not tests of our code so much as tests of our assumptions about TMDB. They exist
because each one caught, or could catch, a resolver design error.
"""

import pytest

from ariadne.core.catalog.fixtures import SEED_SEARCHES, load_fixture, search_fixture_name


def results_for(title: str, year: int | None):
    return load_fixture(search_fixture_name(title, year))["results"]


def test_every_seed_search_was_recorded():
    for seed in SEED_SEARCHES:
        assert load_fixture(search_fixture_name(seed.title, seed.year)) is not None


def test_fixture_names_are_ascii():
    # Filenames must be identical across platforms and unicode normalisation forms.
    for seed in SEED_SEARCHES:
        assert search_fixture_name(seed.title, seed.year).isascii()


def test_whiplash_short_and_feature_are_different_films():
    """The reason year cannot be a mere tiebreaker."""
    short = results_for("Whiplash", 2013)
    feature = results_for("Whiplash", 2014)

    assert short[0]["id"] == 367412
    assert feature[0]["id"] == 244786
    assert short[0]["id"] != feature[0]["id"]
    # The short is far less voted-on, so popularity alone would pick the wrong one for 2013.
    assert short[0]["vote_count"] < feature[0]["vote_count"]


def test_nosferatu_is_disambiguated_by_year():
    assert results_for("Nosferatu", 1922)[0]["id"] == 653
    assert results_for("Nosferatu", 2024)[0]["id"] == 426063


@pytest.mark.parametrize(
    ("title", "year", "expected_id"),
    [
        ("Alien³", 1992, 8077),
        ("Bāhubali: The Beginning", 2015, 256040),
        ("Léon: The Professional", 1994, 101),
        ("WALL·E", 2008, 10681),
        ("The Godfather", 1972, 238),
    ],
)
def test_unicode_titles_match_tmdb_exactly(title: str, year: int, expected_id: int):
    """TMDB stores the same codepoints Letterboxd exports, so these need no folding."""
    results = results_for(title, year)
    assert any(r["id"] == expected_id and r["title"] == title for r in results)


def test_en_dash_does_not_match_exactly():
    """Letterboxd writes an en dash where TMDB writes a hyphen. Folding is required."""
    title = "Gangs of Wasseypur – Part 1"
    results = results_for(title, 2012)

    assert not any(r["title"] == title for r in results)
    assert results[0]["id"] == 117691
    assert results[0]["title"] == "Gangs of Wasseypur - Part 1"


def test_salo_year_disagrees_between_letterboxd_and_tmdb():
    """Letterboxd says 1975, TMDB's release_date says 1976.

    Strict year equality would reject the correct film, so the resolver needs tolerance —
    which has to coexist with Whiplash, where a one-year gap separates two distinct films.
    """
    results = results_for("Salò, or the 120 Days of Sodom", 1975)
    correct = next(r for r in results if r["id"] == 5336)
    assert correct["release_date"].startswith("1976")


def test_television_has_no_correct_movie_match():
    """Obi-Wan Kenobi is a miniseries. Movie search offers only a documentary about it.

    A resolver that trusts the top title-similarity hit would attach that documentary's
    crew to the user's taste profile.
    """
    results = results_for("Obi-Wan Kenobi", 2022)
    assert all(r["title"] != "Obi-Wan Kenobi" for r in results)
    assert results[0]["title"] == "Obi-Wan Kenobi: A Jedi's Return"


def test_recorded_credits_include_below_the_line_roles():
    credits = load_fixture("credits_238.json")
    jobs = {member["job"] for member in credits["crew"]}
    assert {"Director", "Editor", "Director of Photography", "Original Music Composer"} <= jobs
