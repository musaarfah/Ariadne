"""Our trigram similarity must equal Postgres's, or thresholds mean different things.

Resolution scores TMDB search results in Python and, once the catalog is populated, scores our
own films in Postgres via the GIN index. A threshold calibrated on one path would be wrong on
the other if the two implementations disagreed even slightly.

Two properties of pg_trgm shape how this can be tested:

- similarity() returns `real`, not double precision, so agreement can only be asserted to
  float4's ~1.2e-07 relative precision. A tighter tolerance is unsatisfiable, not a failure.
- show_trgm() hashes any trigram containing a multibyte character into a 3-byte value shown as
  hex, so trigram *sets* can only be compared for ASCII input. For multibyte input the
  similarity *value* is still comparable, because the hashing is consistent on both sides.
"""

import pytest
from sqlalchemy import Engine, text

from ariadne.core.catalog.normalize import loose_title, normalize_title
from ariadne.core.catalog.similarity import similarity, trigrams

pytestmark = pytest.mark.integration

# float4 carries about 7 significant digits.
FLOAT4_TOLERANCE = 1e-6

PAIRS = [
    ("gangs of wasseypur - part 1", "gangs of wasseypur - part 2"),
    ("leon: the professional", "leon"),
    ("the godfather", "godfather"),
    ("alien3", "alien"),
    ("salò, or the 120 days of sodom", "salò o le 120 giornate di sodoma"),
    ("wall-e", "burn-e"),
    ("nosferatu", "nosferatu: a symphony of horror"),
    ("bahubali: the beginning", "bahubali 2: the conclusion"),
    ("üç harfliler: fal", "uc harfliler fal"),
    ("mission: impossible - dead reckoning", "mission: impossible - dead reckoning part one"),
    ("obi-wan kenobi", "obi-wan kenobi: a jedi's return"),
    ("whiplash", "whiplash"),
    ("a", "b"),
    ("", ""),
]


@pytest.mark.parametrize(("left", "right"), PAIRS)
def test_similarity_agrees_with_postgres(test_engine: Engine, left: str, right: str):
    with test_engine.connect() as conn:
        expected = conn.execute(
            text("SELECT similarity(:a, :b)"), {"a": left, "b": right}
        ).scalar_one()
    assert similarity(left, right) == pytest.approx(expected, rel=FLOAT4_TOLERANCE)


@pytest.mark.parametrize(
    "text_value",
    [
        "gangs of wasseypur - part 1",
        "wall-e",
        "star wars: episode ii - attack of the clones",
        "the godfather",
        "a",
    ],
)
def test_trigram_sets_agree_with_postgres(test_engine: Engine, text_value: str):
    """ASCII only: show_trgm hashes multibyte trigrams, so the sets are not comparable."""
    with test_engine.connect() as conn:
        expected = conn.execute(text("SELECT show_trgm(:t)"), {"t": text_value}).scalar_one()
    assert trigrams(text_value) == frozenset(expected)


@pytest.mark.parametrize(
    "raw",
    [
        "Gangs of Wasseypur – Part 1",
        "Star Wars: Episode II – Attack of the Clones",
        "Alien³",
        "Salò, or the 120 Days of Sodom",
    ],
)
def test_normalized_titles_agree_with_postgres(test_engine: Engine, raw: str):
    """Normalization then similarity, end to end, on the real problem titles."""
    normalized = normalize_title(raw)
    loose = loose_title(raw)
    with test_engine.connect() as conn:
        expected = conn.execute(
            text("SELECT similarity(:a, :b)"), {"a": normalized, "b": loose}
        ).scalar_one()
    assert similarity(normalized, loose) == pytest.approx(expected, rel=FLOAT4_TOLERANCE)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("salò, or the 120 days of sodom", "salò o le 120 giornate di sodoma"),
        ("üç harfliler: fal", "uc harfliler fal"),
        ("léon: the professional", "léon"),
        ("bāhubali: the beginning", "bāhubali 2: the conclusion"),
        ("wall·e", "burn·e"),
        ("les misérables", "les miserables"),
    ],
)
def test_multibyte_similarity_values_agree(test_engine: Engine, left: str, right: str):
    """Hashing does not change the score, only the printed representation."""
    with test_engine.connect() as conn:
        expected = conn.execute(
            text("SELECT similarity(:a, :b)"), {"a": left, "b": right}
        ).scalar_one()
    assert similarity(left, right) == pytest.approx(expected, rel=FLOAT4_TOLERANCE)


def test_two_empty_strings_score_zero_like_postgres(test_engine: Engine):
    with test_engine.connect() as conn:
        expected = conn.execute(text("SELECT similarity('', '')")).scalar_one()
    assert expected == 0.0
    assert similarity("", "") == 0.0
