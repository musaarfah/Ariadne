import pytest

from ariadne.core.catalog.normalize import (
    comparison_forms,
    loose_title,
    normalize_title,
    without_leading_article,
)
from ariadne.core.catalog.similarity import similarity, trigrams

# --- normalization ---------------------------------------------------------------------


def test_en_dash_folds_to_hyphen():
    """The single most valuable fold: 12 titles in the reference export need it."""
    assert normalize_title("Gangs of Wasseypur – Part 1") == "gangs of wasseypur - part 1"


@pytest.mark.parametrize("dash", ["‐", "‑", "‒", "–", "—", "―", "−"])
def test_every_dash_variant_folds(dash: str):
    assert normalize_title(f"A{dash}B") == "a-b"


def test_no_break_space_becomes_an_ordinary_space():
    """Invisible on screen, and present in two Star Wars titles in the export."""
    assert normalize_title("Episode II – Attack") == "episode ii - attack"


def test_superscript_folds_to_a_digit():
    assert normalize_title("Alien³") == "alien3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Léon: The Professional", "léon: the professional"),
        ("Salò, or the 120 Days of Sodom", "salò, or the 120 days of sodom"),
        ("Bāhubali: The Beginning", "bāhubali: the beginning"),
        ("Üç Harfliler: Fal", "üç harfliler: fal"),
    ],
)
def test_accents_are_preserved(raw: str, expected: str):
    """TMDB stores the same codepoints, so folding accents away would lose information."""
    assert normalize_title(raw) == expected


def test_whitespace_is_collapsed_and_trimmed():
    assert normalize_title("  The   Thing \n") == "the thing"


def test_loose_form_drops_punctuation():
    assert loose_title("Léon: The Professional") == "léon the professional"
    assert loose_title("WALL·E") == "wall e"


def test_article_removal_is_only_a_comparison_form():
    assert without_leading_article("the godfather") == "godfather"
    assert without_leading_article("a serious man") == "serious man"
    assert without_leading_article("an education") == "education"
    # Not an article, must be left alone.
    assert without_leading_article("theatre of blood") == "theatre of blood"


def test_comparison_forms_are_deduplicated_and_ordered():
    forms = comparison_forms("The Godfather")
    assert forms[0] == "the godfather"
    assert "godfather" in forms
    assert len(forms) == len(set(forms))


def test_comparison_forms_never_empty_for_real_titles():
    for title in ["Alien³", "WALL·E", "The Thing", "Salò, or the 120 Days of Sodom"]:
        assert comparison_forms(title)


# --- similarity ------------------------------------------------------------------------


def test_identical_strings_score_one():
    assert similarity("whiplash", "whiplash") == 1.0


def test_disjoint_strings_score_zero():
    assert similarity("a", "b") == 0.0


def test_trigrams_pad_each_word_independently():
    # Matches pg_trgm: two leading spaces, one trailing, per word.
    assert trigrams("a") == frozenset({"  a", " a "})


def test_trigrams_treat_punctuation_as_a_separator():
    assert trigrams("wall-e") == trigrams("wall e")


def test_accented_letters_are_word_characters_not_separators():
    """Postgres treats them as alphanumeric, so we must too or thresholds diverge."""
    assert " sa" in trigrams("salò")
    # Same length word, so the same number of trigrams. If the accent were treated as a
    # separator, "salò" would split into "sal" and produce fewer.
    assert len(trigrams("salò")) == len(trigrams("salo"))


def test_thresholds_separate_the_measured_cases():
    """The calibration that makes the resolver's ACCEPT_SIMILARITY of 0.75 correct."""
    must_accept = similarity(
        "mission impossible dead reckoning", "mission impossible dead reckoning part one"
    )
    must_reject_documentary = similarity("obi wan kenobi", "obi wan kenobi a jedi's return")
    must_reject_backstage = similarity(
        "salò or the 120 days of sodom", "backstage on the set of salò or the 120 days of sodom"
    )

    assert must_accept > 0.75
    assert must_reject_documentary < 0.75
    assert must_reject_backstage < 0.75


def test_empty_strings_score_zero():
    # Matches Postgres, which returns 0 rather than 1 for two empty strings.
    assert similarity("", "") == 0.0
    assert similarity("", "something") == 0.0
