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
