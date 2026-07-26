import pytest

from ariadne.core.catalog.normalize import (
    comparison_forms,
    loose_title,
    normalize_title,
    sequence_markers,
    titles_are_distinct,
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


# --- distinguishing sequels from their siblings ----------------------------------------


@pytest.mark.parametrize(
    ("query", "candidate"),
    [
        ("Gangs of Wasseypur – Part 2", "Gangs of Wasseypur - Part 1"),
        ("Kill Bill: Vol. 1", "Kill Bill: Vol. 2"),
        ("Nymphomaniac: Vol. II", "Nymphomaniac: Vol. I"),
        (
            "Harry Potter and the Deathly Hallows: Part 1",
            "Harry Potter and the Deathly Hallows: Part 2",
        ),
        ("Back to the Future Part III", "Back to the Future Part II"),
        ("Jatt & Juliet 2", "Jatt & Juliet"),
        ("Justice League Dark", "Justice League"),
    ],
)
def test_siblings_are_distinct(query: str, candidate: str):
    """Every one of these was a real wrong match in the first full-library run.

    Trigram similarity cannot separate them — Back to the Future Part III against Part II
    scores 0.963 — so the numeral has to be compared on its own.
    """
    assert similarity(normalize_title(query), normalize_title(candidate)) > 0.7
    assert titles_are_distinct(query, candidate)


@pytest.mark.parametrize(
    ("query", "candidate"),
    [
        # TMDB holding a longer title for the same film is normal and must not be rejected.
        ("Mission: Impossible – Dead Reckoning", "Mission: Impossible - Dead Reckoning Part One"),
        ("Glass Onion", "Glass Onion: A Knives Out Mystery"),
        ("Whiplash", "Whiplash"),
        ("2001: A Space Odyssey", "2001: A Space Odyssey"),
        ("Alien³", "Alien³"),
        (
            "Star Wars: Episode II – Attack of the Clones",
            "Star Wars: Episode II - Attack of the Clones",
        ),
        # "Vol." and "Volume" are the same word.
        ("Kill Bill: Vol. 1", "Kill Bill: Volume 1"),
        # British/American spelling. Rejecting this cost a correct match in the second run.
        ("Three Colours: Blue", "Three Colors: Blue"),
        ("The Colour of Pomegranates", "The Color of Pomegranates"),
    ],
)
def test_same_film_is_not_distinct(query: str, candidate: str):
    assert not titles_are_distinct(query, candidate)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Back to the Future Part III", {"3"}),
        ("Kill Bill: Vol. 1", {"1"}),
        ("Nymphomaniac: Vol. II", {"2"}),
        ("Mission: Impossible - Dead Reckoning Part One", {"1"}),
        ("2001: A Space Odyssey", {"2001"}),
        ("Whiplash", set()),
        # NFKC folds the superscript, but "alien3" is one token, not a numeral.
        ("Alien³", set()),
    ],
)
def test_sequence_markers(title: str, expected: set[str]):
    assert sequence_markers(title) == expected


def test_word_order_is_distinguishing():
    """ "The Breaking Ice" is not "Breaking the Ice", but trigram similarity scores them 1.000."""
    assert (
        similarity(normalize_title("The Breaking Ice"), normalize_title("Breaking the Ice")) == 1.0
    )
    assert titles_are_distinct("The Breaking Ice", "Breaking the Ice")


@pytest.mark.parametrize(
    ("query", "candidate"),
    [
        # A word substitution among insignificant words, not a reordering.
        ("Kill Bill: Vol. 1", "Kill Bill: Volume 1"),
        # One extra article, not a reordering.
        ("Beauty and the Beast", "The Beauty and the Beast"),
    ],
)
def test_word_order_rule_needs_a_true_permutation(query: str, candidate: str):
    assert not titles_are_distinct(query, candidate)
