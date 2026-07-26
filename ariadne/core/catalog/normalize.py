"""Title folding, applied identically to Letterboxd titles and TMDB candidates.

What the reference export actually contains, measured rather than assumed:

- En dash, 12 times. Letterboxd writes "Gangs of Wasseypur – Part 1"; TMDB writes a hyphen.
  Folding dashes is the single most valuable step.
- No-break space (U+00A0), twice, inside "Star Wars: Episode II –\xa0Attack of the Clones".
  Invisible on screen. NFKC folds it to an ordinary space.
- Superscript three, once, in "Alien³". NFKC folds it to "3". Harmless because the same
  folding is applied to TMDB's title, which also stores "Alien³".
- Accented letters (é, ā, ò, Ü, ç) are left alone by NFKC, and TMDB stores the same
  codepoints, so they need no special handling.
"""

import re
import unicodedata

# Everything that means "dash" in a title but is not the ASCII hyphen TMDB tends to use.
DASH_CHARACTERS = frozenset("‐‑‒–—―−－")

LEADING_ARTICLES = ("the ", "a ", "an ")

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_title(title: str) -> str:
    """Canonical comparable form. This is what lands in films.normalized_title.

    NFKC first, because it is what turns an invisible no-break space into an ordinary one
    and a superscript into a digit. Dash folding second, since NFKC leaves dashes alone.
    """
    folded = unicodedata.normalize("NFKC", title)
    folded = "".join("-" if character in DASH_CHARACTERS else character for character in folded)
    return _WHITESPACE.sub(" ", folded.casefold()).strip()


def loose_title(title: str) -> str:
    """Punctuation-free form, for similarity scoring only.

    Colons, commas and apostrophes are where TMDB and Letterboxd disagree most often and
    matter least — "Léon: The Professional" versus "Leon the Professional".
    """
    stripped = _PUNCTUATION.sub(" ", normalize_title(title))
    return _WHITESPACE.sub(" ", stripped).strip()


def without_leading_article(normalized: str) -> str:
    """Drop a leading article from an already-normalized title.

    Used as an extra comparison form, never as the canonical one: dropping the article from
    the stored title would make "The Thing" and "Thing" indistinguishable.
    """
    for article in LEADING_ARTICLES:
        if normalized.startswith(article):
            return normalized[len(article) :]
    return normalized


def comparison_forms(title: str) -> tuple[str, ...]:
    """Every form a title may legitimately be matched on, most exact first."""
    normalized = normalize_title(title)
    loose = loose_title(title)
    forms = [
        normalized,
        loose,
        without_leading_article(normalized),
        without_leading_article(loose),
    ]

    seen: list[str] = []
    for form in forms:
        if form and form not in seen:
            seen.append(form)
    return tuple(seen)


# Roman numerals and number words folded to digits, so "Part One", "Part I" and "Part 1" are
# recognised as the same marker rather than three different ones.
_NUMERAL_ALIASES = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

# Words TMDB and Letterboxd drop or add freely, so their absence proves nothing.
_INSIGNIFICANT = frozenset(
    {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "part", "vol", "volume"}
)

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

# A leftover word this close to one the candidate carries is a spelling variant, not a
# distinguishing word. Measured: colours/colors is 0.50; dark against justice or league is 0.00.
SPELLING_VARIANT_SIMILARITY = 0.4


def _tokens(title: str) -> list[str]:
    return _TOKEN.findall(normalize_title(title))


def sequence_markers(title: str) -> frozenset[str]:
    """Digits and numerals that distinguish one instalment from another.

    "Back to the Future Part III" yields {"3"}. Trigram similarity scores that against
    "Part II" at 0.963, so the numeral has to be compared separately or sequels collapse onto
    each other.
    """
    return frozenset(
        _NUMERAL_ALIASES.get(token, token) for token in _tokens(title) if _is_numeral(token)
    )


def _is_numeral(token: str) -> bool:
    return token.isdigit() or token in _NUMERAL_ALIASES


def titles_are_distinct(query: str, candidate: str) -> bool:
    """Whether two titles name different films despite scoring as similar.

    Both tests are deliberately asymmetric, because TMDB routinely holds a *longer* title for
    the same film — "Glass Onion: A Knives Out Mystery", "... Dead Reckoning Part One" — while
    the reverse, our side carrying a word TMDB lacks, means we would be discarding the thing
    that identifies the film.
    """
    query_markers = sequence_markers(query)
    candidate_markers = sequence_markers(candidate)

    # A numeral on our side that the candidate lacks, or a different one: different instalment.
    if query_markers - candidate_markers:
        return True
    if query_markers and candidate_markers and query_markers != candidate_markers:
        return True

    # Same words in a different order name different films: "The Breaking Ice" is not
    # "Breaking the Ice". Trigram similarity scores those at 1.000 because it compares
    # per-word trigram sets and cannot see order at all.
    # Only a true permutation counts: the same tokens in a different order. Requiring an equal
    # multiset rather than an equal set matters, because "Kill Bill: Vol. 1" against
    # "Kill Bill: Volume 1" is a word substitution, not a reordering, and "Beauty and the Beast"
    # against "The Beauty and the Beast" simply has one more article.
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if query_tokens != candidate_tokens and sorted(query_tokens) == sorted(candidate_tokens):
        return True

    query_words = {t for t in _tokens(query) if t not in _INSIGNIFICANT and not _is_numeral(t)}
    candidate_words = {
        t for t in _tokens(candidate) if t not in _INSIGNIFICANT and not _is_numeral(t)
    }

    # "Justice League Dark" against "Justice League": dropping "dark" changes the film. But a
    # leftover word that is merely a spelling variant of one the candidate has is not
    # distinguishing — "Three Colours: Blue" is TMDB's "Three Colors: Blue". Compared per word,
    # colours/colors scores 0.50 while dark against justice or league scores 0.00.
    from ariadne.core.catalog.similarity import similarity

    for word in query_words - candidate_words:
        if not any(
            similarity(word, other) >= SPELLING_VARIANT_SIMILARITY for other in candidate_words
        ):
            return True
    return False
