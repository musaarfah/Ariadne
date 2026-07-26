"""Trigram similarity matching PostgreSQL's pg_trgm.

Resolution happens in two places: against TMDB search results, which arrive over HTTP and
are scored in memory, and against our own catalog, which is scored by Postgres using the GIN
trigram index. Those two paths must agree, or a threshold tuned on one would be wrong on the
other. So this reimplements pg_trgm's definition rather than reaching for difflib, and an
integration test asserts the two produce identical numbers.

pg_trgm's definition: split on non-alphanumeric characters, pad each word with two leading
spaces and one trailing space, take every 3-character window, then
similarity = |intersection| / |union| of the resulting sets.
"""

PAD_LEADING = "  "
PAD_TRAILING = " "


def _words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []

    # str.isalnum() is unicode-aware, which matches how Postgres treats accented letters as
    # word characters rather than separators.
    for character in text.lower():
        if character.isalnum():
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []

    if current:
        words.append("".join(current))
    return words


def trigrams(text: str) -> frozenset[str]:
    grams: set[str] = set()
    for word in _words(text):
        padded = f"{PAD_LEADING}{word}{PAD_TRAILING}"
        grams.update(padded[index : index + 3] for index in range(len(padded) - 2))
    return frozenset(grams)


def similarity(left: str, right: str) -> float:
    left_grams = trigrams(left)
    right_grams = trigrams(right)

    # Postgres returns 0 for two empty strings rather than 1. Matching that matters more than
    # arguing about which is philosophically right, since agreement is this module's purpose.
    union = left_grams | right_grams
    if not union:
        return 0.0
    return len(left_grams & right_grams) / len(union)
