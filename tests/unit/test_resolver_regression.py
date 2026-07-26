"""The resolver regression set.

Every title in the reference export that is non-ASCII or collides with another film of the
same name, plus a control and one television entry. Runs offline against recorded fixtures.

The expected ids below were verified by hand against each candidate's title, release date and
overview — not copied from whatever the resolver happened to return. Two of them exist because
the resolver got them wrong first:

- Salò resolved to "Backstage on the Set of Salò, or the 120 Days of Sodom" because that
  documentary carries Letterboxd's 1975 date while the film itself is dated 1976. Exact title
  now outranks exact year.
- Obi-Wan Kenobi resolved to "Obi-Wan Kenobi: A Jedi's Return", a documentary, at 0.45
  similarity. Fuzzy matches now need 0.75.
"""

import pytest

from ariadne.core.catalog.fixtures import SEED_SEARCHES, load_fixture, search_fixture_name
from ariadne.core.catalog.resolve import resolve_from_results
from ariadne.db.models import MatchMethod

# (title, year, expected tmdb_id or None, expected method)
EXPECTED: tuple[tuple[str, int, int | None, MatchMethod], ...] = (
    # Same-title collisions, separated by year alone.
    ("Whiplash", 2013, 367412, MatchMethod.EXACT),
    ("Whiplash", 2014, 244786, MatchMethod.EXACT),
    ("Nosferatu", 1922, 653, MatchMethod.EXACT),
    ("Nosferatu", 2024, 426063, MatchMethod.EXACT),
    ("Joker", 2012, 129507, MatchMethod.EXACT),
    ("Joker", 2019, 475557, MatchMethod.EXACT),
    ("Aladdin", 2019, 420817, MatchMethod.EXACT),
    ("Frozen", 2013, 109445, MatchMethod.EXACT),
    ("Beauty and the Beast", 1991, 10020, MatchMethod.EXACT),
    ("Les Misérables", 1998, 4415, MatchMethod.EXACT),
    ("Les Misérables", 2012, 82695, MatchMethod.EXACT),
    # Duplicated title AND year on TMDB; the real film dominates by vote count.
    ("Aladdin", 1992, 812, MatchMethod.DOMINANT),
    ("Frozen", 2010, 44363, MatchMethod.DOMINANT),
    ("Beauty and the Beast", 2017, 321612, MatchMethod.DOMINANT),
    # En dash on Letterboxd, hyphen on TMDB.
    ("Mission: Impossible – Rogue Nation", 2015, 177677, MatchMethod.EXACT),
    ("Mission: Impossible – Ghost Protocol", 2011, 56292, MatchMethod.EXACT),
    ("Mission: Impossible – Fallout", 2018, 353081, MatchMethod.EXACT),
    ("Mission: Impossible – The Final Reckoning", 2025, 575265, MatchMethod.EXACT),
    ("John Wick: Chapter 3 – Parabellum", 2019, 458156, MatchMethod.EXACT),
    ("Gangs of Wasseypur – Part 1", 2012, 117691, MatchMethod.EXACT),
    ("Gangs of Wasseypur – Part 2", 2012, 126400, MatchMethod.EXACT),
    ("Golmaal – Fun Unlimited", 2006, 19670, MatchMethod.EXACT),
    ("Star Wars: Episode I – The Phantom Menace", 1999, 1893, MatchMethod.EXACT),
    # These two also contain an invisible no-break space after the dash.
    ("Star Wars: Episode II – Attack of the Clones", 2002, 1894, MatchMethod.EXACT),
    ("Star Wars: Episode III – Revenge of the Sith", 2005, 1895, MatchMethod.EXACT),
    # Accents and other non-ASCII, which TMDB stores identically.
    ("Léon: The Professional", 1994, 101, MatchMethod.EXACT),
    ("The Double Life of Véronique", 1991, 1600, MatchMethod.EXACT),
    ("Bāhubali: The Beginning", 2015, 256040, MatchMethod.EXACT),
    ("Bāhubali 2: The Conclusion", 2017, 350312, MatchMethod.EXACT),
    ("Üç Harfliler: Fal", 2025, 1432726, MatchMethod.EXACT),
    ("Alien³", 1992, 8077, MatchMethod.EXACT),
    ("WALL·E", 2008, 10681, MatchMethod.EXACT),
    # Letterboxd says 1975, TMDB says 1976. Exact title must beat the documentary's exact year.
    ("Salò, or the 120 Days of Sodom", 1975, 5336, MatchMethod.TRIGRAM),
    # TMDB renamed this one, so the title is genuinely fuzzy but well above threshold.
    ("Mission: Impossible – Dead Reckoning", 2023, 575264, MatchMethod.TRIGRAM),
    # Control.
    ("The Godfather", 1972, 238, MatchMethod.EXACT),
    # Television. There is no correct movie, so the only correct answer is to refuse.
    ("Obi-Wan Kenobi", 2022, None, MatchMethod.UNRESOLVED),
)


def resolve_seed(title: str, year: int):
    fixture = load_fixture(search_fixture_name(title, year))
    return resolve_from_results(fixture["results"], title, year)


@pytest.mark.parametrize(("title", "year", "expected_id", "expected_method"), EXPECTED)
def test_regression_case(title: str, year: int, expected_id: int | None, expected_method):
    outcome = resolve_seed(title, year)
    assert outcome.tmdb_id == expected_id, outcome.reason
    assert outcome.method is expected_method, outcome.reason


def test_the_regression_set_covers_every_seed():
    """Adding a seed without an expected answer must fail rather than pass silently."""
    seeds = {(s.title, s.year) for s in SEED_SEARCHES}
    expected = {(title, year) for title, year, _, _ in EXPECTED}
    assert seeds == expected


def test_television_is_refused_rather_than_mismatched():
    outcome = resolve_seed("Obi-Wan Kenobi", 2022)
    assert not outcome.resolved
    assert "below" in outcome.reason


def test_salo_prefers_the_film_over_the_making_of():
    outcome = resolve_seed("Salò, or the 120 Days of Sodom", 1975)
    assert outcome.tmdb_id == 5336
    assert outcome.candidate is not None
    assert outcome.candidate.year == 1976
    assert outcome.candidate.year_gap == 1


def test_dominance_records_its_reasoning():
    outcome = resolve_seed("Aladdin", 1992)
    assert outcome.method is MatchMethod.DOMINANT
    assert "dominate" in outcome.reason


def test_every_resolution_carries_a_reason():
    for title, year, _, _ in EXPECTED:
        assert resolve_seed(title, year).reason
