"""Resolving a Letterboxd film to a TMDB id.

The highest-risk step in the pipeline. A wrong match injects a stranger's crew into a taste
profile, and every downstream metric silently inherits the error — so this module prefers
refusing to answer over guessing.

Two rules earn their complexity, both from measured evidence:

Year is ranked, not filtered (F12). Letterboxd and TMDB disagree about release years —
Salò is 1975 on one and 1976 on the other — so strict equality rejects correct films. But a
blanket one-year tolerance is also wrong, because Whiplash 2013 (a short) and 2014 (the
feature) are different films with identical titles one year apart. So: an exact year with an
exact title wins outright; a one-year gap is acceptable only when exactly one candidate sits
in that window; two or more years is a rejection.

Title similarity alone can never accept a match (F13). Movie search for the miniseries
Obi-Wan Kenobi returns a documentary about it, which scores well on title. A candidate must
clear the title threshold *and* the year rule.
"""

from dataclasses import dataclass
from typing import Any

from ariadne.core.catalog.normalize import (
    comparison_forms,
    normalize_title,
    titles_are_distinct,
)
from ariadne.core.catalog.similarity import similarity
from ariadne.db.models import MatchMethod

# Candidates below this are not even worth listing in an audit.
CONSIDER_SIMILARITY = 0.30

# A non-exact title must clear this to be accepted. Calibrated against measured cases rather
# than picked: TMDB renamed "Mission: Impossible – Dead Reckoning" to "... Part One" (0.79,
# must accept), while "Obi-Wan Kenobi" only offers "Obi-Wan Kenobi: A Jedi's Return" (0.45,
# must reject) and "Salò" offers "Backstage on the Set of Salò..." (0.65, must reject).
ACCEPT_SIMILARITY = 0.75

# A candidate this close is treated as the same title, letting the year rule decide.
EXACT_TITLE_SIMILARITY = 0.999

# Years beyond this distance are a rejection, never a match.
MAX_YEAR_GAP = 1

# When several candidates share a title and year, TMDB usually holds one real entry plus
# duplicates or obscure namesakes. Accept the leader only if it overwhelms the runner-up.
# Measured: Aladdin 1992 is 12120 votes against 68, Frozen 2010 is 2029 against 2, Beauty and
# the Beast 2017 is 16079 against 0. Popularity here disambiguates *identity*, which is a
# different thing from using popularity to predict taste.
DOMINANCE_RATIO = 10.0
MIN_DOMINANT_VOTES = 50

# Confidence recorded when the title matched exactly but the year was off by one.
NEAR_YEAR_CONFIDENCE = 0.9


@dataclass(frozen=True)
class Candidate:
    tmdb_id: int
    title: str
    original_title: str | None
    year: int | None
    vote_count: int
    title_similarity: float
    year_gap: int | None

    @property
    def has_exact_title(self) -> bool:
        return self.title_similarity >= EXACT_TITLE_SIMILARITY

    @property
    def has_exact_year(self) -> bool:
        return self.year_gap == 0


@dataclass(frozen=True)
class ResolutionOutcome:
    tmdb_id: int | None
    method: MatchMethod
    confidence: float | None
    reason: str
    candidate: Candidate | None = None

    # True only when no plausible film existed at all. False when a film was found but the
    # choice between several was ambiguous. The television check may only run in the first
    # case: "Missing" (2018) failed because two films share that title and year, and calling
    # that television because a series is also named Missing would be a different claim
    # entirely.
    no_film_candidate: bool = False

    @property
    def resolved(self) -> bool:
        return self.tmdb_id is not None


def _release_year(payload: dict[str, Any]) -> int | None:
    release_date = payload.get("release_date") or ""
    head = release_date[:4]
    return int(head) if head.isdigit() else None


def _best_similarity(query_forms: tuple[str, ...], candidate_titles: list[str]) -> float:
    best = 0.0
    for query in query_forms:
        for title in candidate_titles:
            for candidate_form in comparison_forms(title):
                best = max(best, similarity(query, candidate_form))
                if best >= EXACT_TITLE_SIMILARITY:
                    return best
    return best


def build_candidate(payload: dict[str, Any], title: str, year: int | None) -> Candidate | None:
    tmdb_id = payload.get("id")
    candidate_title = payload.get("title")
    if not isinstance(tmdb_id, int) or not isinstance(candidate_title, str):
        return None

    original_title = payload.get("original_title")
    titles = [candidate_title]
    if isinstance(original_title, str) and original_title != candidate_title:
        titles.append(original_title)

    # Drop any title form that names a different film despite scoring as similar — a sequel,
    # a volume, or a namesake missing a distinguishing word. Trigram similarity cannot see
    # the difference: "Back to the Future Part III" against "Part II" scores 0.963.
    comparable = [t for t in titles if not titles_are_distinct(title, t)]

    candidate_year = _release_year(payload)
    return Candidate(
        tmdb_id=tmdb_id,
        title=candidate_title,
        original_title=original_title if isinstance(original_title, str) else None,
        year=candidate_year,
        vote_count=payload.get("vote_count") or 0,
        title_similarity=(
            _best_similarity(comparison_forms(title), comparable) if comparable else 0.0
        ),
        year_gap=None if (year is None or candidate_year is None) else abs(candidate_year - year),
    )


def _dominant(candidates: list[Candidate]) -> Candidate | None:
    """The one candidate that overwhelms the others by vote count, if there is one."""
    ranked = sorted(candidates, key=lambda c: c.vote_count, reverse=True)
    leader, runner_up = ranked[0], ranked[1]

    if leader.vote_count < MIN_DOMINANT_VOTES:
        return None
    if leader.vote_count < DOMINANCE_RATIO * max(runner_up.vote_count, 1):
        return None
    return leader


def _resolve_exact_titles(exact_titles: list[Candidate]) -> ResolutionOutcome:
    """Decide among candidates whose titles all match exactly. Year is the discriminator."""
    exact_year = [c for c in exact_titles if c.has_exact_year]

    if len(exact_year) == 1:
        return ResolutionOutcome(
            exact_year[0].tmdb_id, MatchMethod.EXACT, 1.0, "exact title and year", exact_year[0]
        )

    if len(exact_year) > 1:
        leader = _dominant(exact_year)
        if leader is not None:
            return ResolutionOutcome(
                leader.tmdb_id,
                MatchMethod.DOMINANT,
                leader.title_similarity,
                f"{len(exact_year)} share title and year; {leader.vote_count} votes dominate",
                leader,
            )
        # Flag a strong near-year alternative so the hand audit can act without re-deriving
        # the case. Deliberately not resolved automatically: preferring a popular film one
        # year off would mean picking Whiplash 2014 for a 2013 query, and a silent wrong
        # match is worse than a visible refusal.
        near = [c for c in exact_titles if not c.has_exact_year]
        hint = ""
        if near:
            strongest = max(near, key=lambda c: c.vote_count)
            if strongest.vote_count >= DOMINANCE_RATIO * max(
                (c.vote_count for c in exact_year), default=1
            ):
                hint = (
                    f"; but {strongest.tmdb_id} ({strongest.year}, {strongest.vote_count} votes) "
                    f"is far more popular and one year off"
                )
        return ResolutionOutcome(
            None,
            MatchMethod.UNRESOLVED,
            None,
            f"{len(exact_year)} candidates share title and year, none dominant{hint}",
        )

    # Exact title, but every candidate is a year off. Acceptable only if unambiguous.
    if len(exact_titles) == 1:
        best = exact_titles[0]
        return ResolutionOutcome(
            best.tmdb_id,
            MatchMethod.TRIGRAM,
            NEAR_YEAR_CONFIDENCE,
            f"exact title, year off by {best.year_gap}",
            best,
        )

    leader = _dominant(exact_titles)
    if leader is not None:
        return ResolutionOutcome(
            leader.tmdb_id,
            MatchMethod.DOMINANT,
            NEAR_YEAR_CONFIDENCE,
            f"{len(exact_titles)} exact titles near the year; {leader.vote_count} votes dominate",
            leader,
        )
    return ResolutionOutcome(
        None, MatchMethod.UNRESOLVED, None, f"{len(exact_titles)} exact titles near the year"
    )


def choose(candidates: list[Candidate], year: int | None) -> ResolutionOutcome:
    """Pick a candidate, or refuse.

    Title exactness dominates the year, not the other way round. An earlier version ranked
    exact-year above exact-title and resolved "Salò, or the 120 Days of Sodom" (Letterboxd
    1975) to a making-of documentary that happened to carry a 1975 date, in preference to the
    film itself dated 1976. Refusing is cheap; a wrong match is invisible and permanent.
    """
    plausible = [c for c in candidates if c.title_similarity >= CONSIDER_SIMILARITY]
    if not plausible:
        return ResolutionOutcome(
            None,
            MatchMethod.UNRESOLVED,
            None,
            "no candidate above threshold",
            no_film_candidate=True,
        )

    if year is None:
        exact = [c for c in plausible if c.has_exact_title]
        if len(exact) == 1:
            return ResolutionOutcome(
                exact[0].tmdb_id, MatchMethod.EXACT, 1.0, "exact title, no year available", exact[0]
            )
        return ResolutionOutcome(
            None,
            MatchMethod.UNRESOLVED,
            None,
            "no year, and title is not uniquely exact",
            no_film_candidate=True,
        )

    in_window = [c for c in plausible if c.year_gap is not None and c.year_gap <= MAX_YEAR_GAP]
    if not in_window:
        return ResolutionOutcome(
            None,
            MatchMethod.UNRESOLVED,
            None,
            f"no candidate within {MAX_YEAR_GAP} year(s)",
            no_film_candidate=True,
        )

    exact_titles = [c for c in in_window if c.has_exact_title]
    if exact_titles:
        return _resolve_exact_titles(exact_titles)

    # No exact title anywhere in the window, so a fuzzy match must clear a high bar. This is
    # the branch that must reject the Obi-Wan Kenobi documentary.
    acceptable = [c for c in in_window if c.title_similarity >= ACCEPT_SIMILARITY]
    if not acceptable:
        best = max(in_window, key=lambda c: c.title_similarity)
        return ResolutionOutcome(
            None,
            MatchMethod.UNRESOLVED,
            None,
            f"best title similarity {best.title_similarity:.2f} below {ACCEPT_SIMILARITY}",
            no_film_candidate=True,
        )

    exact_year = [c for c in acceptable if c.has_exact_year]
    pool = exact_year or acceptable
    if len(pool) == 1:
        best = pool[0]
        return ResolutionOutcome(
            best.tmdb_id,
            MatchMethod.TRIGRAM,
            best.title_similarity,
            f"fuzzy title {best.title_similarity:.2f}, year off by {best.year_gap}",
            best,
        )

    leader = _dominant(pool)
    if leader is not None:
        return ResolutionOutcome(
            leader.tmdb_id,
            MatchMethod.DOMINANT,
            leader.title_similarity,
            f"{len(pool)} fuzzy candidates; {leader.vote_count} votes dominate",
            leader,
        )
    return ResolutionOutcome(
        None, MatchMethod.UNRESOLVED, None, f"{len(pool)} fuzzy candidates, none dominant"
    )


def resolve_from_results(
    results: list[dict[str, Any]], title: str, year: int | None
) -> ResolutionOutcome:
    candidates = [c for c in (build_candidate(r, title, year) for r in results) if c is not None]
    return choose(candidates, year)


def normalized_for_storage(payload: dict[str, Any]) -> str:
    title = payload.get("title") or payload.get("original_title") or ""
    return normalize_title(title)
