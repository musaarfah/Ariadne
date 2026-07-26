"""Sampling resolutions for hand audit, and scoring the verdicts that come back.

Resolution rate says how many films got an answer. It says nothing about whether the answers
are right, and a wrong match is invisible downstream — so precision has to be measured by
looking. This module produces the sample and reads back the verdicts.

Sampling is stratified rather than uniform. A uniform sample of 100 would be ~93 exact matches
and a handful of everything else, which measures the safe path precisely and the risky paths
not at all. Instead every non-exact outcome is audited exhaustively, plus a sample of exact
ones to bound the bulk.
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ariadne.constants import RANDOM_SEED
from ariadne.core.catalog.normalize import normalize_title
from ariadne.db.models import Film, MatchMethod, Rating, Resolution

TMDB_FILM_URL = "https://www.themoviedb.org/movie/{tmdb_id}"

# Verdict vocabulary for the review file.
VERDICT_CORRECT = "correct"
VERDICT_WRONG = "wrong"
VERDICT_UNSURE = "unsure"
VERDICTS = (VERDICT_CORRECT, VERDICT_WRONG, VERDICT_UNSURE)

FIELDS = (
    "letterboxd_uri",
    "lb_name",
    "lb_year",
    "method",
    "confidence",
    "tmdb_id",
    "tmdb_title",
    "tmdb_year",
    "votes",
    "tmdb_url",
    "auto",
    "reason",
    "verdict",
    "note",
)


@dataclass(frozen=True)
class AuditCase:
    letterboxd_uri: str
    lb_name: str
    lb_year: int | None
    method: str
    confidence: float | None
    tmdb_id: int | None
    tmdb_title: str | None
    tmdb_year: int | None
    votes: int | None
    reason: str | None

    @property
    def auto_verdict(self) -> str:
        """A pre-verdict for cases decidable without judgement.

        Only claims `correct` when the method is EXACT *and* the folded titles and years are
        identical — the one situation with nothing left to interpret.

        DOMINANT is deliberately never auto-passed even though its titles and years always
        match: several TMDB entries sharing a title and year is precisely what made the case
        ambiguous, so identical title and year is evidence of nothing. Auto-passing it would
        have waved through 53 of the 54 riskiest matches in the sample.
        """
        if self.method != MatchMethod.EXACT.value:
            return "review"
        if self.tmdb_id is None or self.tmdb_title is None:
            return "review"
        if self.lb_year is None or self.tmdb_year is None:
            return "review"
        titles_match = normalize_title(self.lb_name) == normalize_title(self.tmdb_title)
        return "correct" if titles_match and self.lb_year == self.tmdb_year else "review"


def sample_cases(
    session: Session,
    upload_id: object,
    exact_sample: int = 50,
    seed: int = RANDOM_SEED,
) -> list[AuditCase]:
    """Every non-exact outcome, plus a seeded random sample of exact ones."""
    rows = session.execute(
        select(
            Resolution.letterboxd_uri,
            Rating.name,
            Rating.year,
            Resolution.match_method,
            Resolution.confidence,
            Resolution.tmdb_id,
            Film.title,
            Film.year,
            Film.vote_count,
            Resolution.reason,
        )
        .join(Rating, Rating.letterboxd_uri == Resolution.letterboxd_uri)
        .outerjoin(Film, Film.tmdb_id == Resolution.tmdb_id)
        .where(Rating.upload_id == upload_id)
        .order_by(Resolution.letterboxd_uri)
    ).all()

    cases = [
        AuditCase(
            letterboxd_uri=uri,
            lb_name=lb_name,
            lb_year=lb_year,
            method=method.value,
            confidence=confidence,
            tmdb_id=tmdb_id,
            tmdb_title=tmdb_title,
            tmdb_year=tmdb_year,
            votes=votes,
            reason=reason,
        )
        for (
            uri,
            lb_name,
            lb_year,
            method,
            confidence,
            tmdb_id,
            tmdb_title,
            tmdb_year,
            votes,
            reason,
        ) in rows
    ]

    risky = [c for c in cases if c.method != MatchMethod.EXACT.value]
    exact = [c for c in cases if c.method == MatchMethod.EXACT.value]

    rng = random.Random(seed)
    sampled_exact = rng.sample(exact, min(exact_sample, len(exact)))

    return sorted(risky + sampled_exact, key=lambda c: (c.method, c.lb_name))


def write_review_file(cases: list[AuditCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "letterboxd_uri": case.letterboxd_uri,
                    "lb_name": case.lb_name,
                    "lb_year": case.lb_year if case.lb_year is not None else "",
                    "method": case.method,
                    "confidence": "" if case.confidence is None else f"{case.confidence:.2f}",
                    "tmdb_id": case.tmdb_id if case.tmdb_id is not None else "",
                    "tmdb_title": case.tmdb_title or "",
                    "tmdb_year": case.tmdb_year if case.tmdb_year is not None else "",
                    "votes": case.votes if case.votes is not None else "",
                    "tmdb_url": (
                        TMDB_FILM_URL.format(tmdb_id=case.tmdb_id) if case.tmdb_id else ""
                    ),
                    "auto": case.auto_verdict,
                    "reason": case.reason or "",
                    "verdict": "",
                    "note": "",
                }
            )


# Methods that produced a film match. Precision is measured over these alone: a refusal is not
# a wrong answer, and mixing refusals in would report recall as though it were precision.
RESOLVED_METHODS = frozenset(
    {MatchMethod.EXACT.value, MatchMethod.TRIGRAM.value, MatchMethod.DOMINANT.value}
)


@dataclass
class Tally:
    correct: int = 0
    wrong: int = 0
    unsure: int = 0
    unfilled: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.unsure + self.unfilled

    @property
    def judged(self) -> int:
        return self.correct + self.wrong

    @property
    def accuracy(self) -> float:
        """`unsure` is excluded rather than counted either way."""
        return self.correct / self.judged if self.judged else 0.0


@dataclass
class AuditResult:
    by_method: dict[str, Tally] = field(default_factory=dict)
    wrong_cases: list[tuple[str, str, str]] = field(default_factory=list)
    unsure_cases: list[tuple[str, str, str]] = field(default_factory=list)

    def tally(self, method: str) -> Tally:
        return self.by_method.setdefault(method, Tally())

    def _combine(self, methods: frozenset[str] | set[str]) -> Tally:
        combined = Tally()
        for method, tally in self.by_method.items():
            if method in methods:
                combined.correct += tally.correct
                combined.wrong += tally.wrong
                combined.unsure += tally.unsure
                combined.unfilled += tally.unfilled
        return combined

    @property
    def resolved(self) -> Tally:
        """The precision measurement: film matches only."""
        return self._combine(RESOLVED_METHODS)

    @property
    def refusals(self) -> Tally:
        """Television and unresolved — whether declining to match was the right call."""
        return self._combine({MatchMethod.TELEVISION.value, MatchMethod.UNRESOLVED.value})

    @property
    def total(self) -> int:
        return sum(tally.total for tally in self.by_method.values())


def read_verdicts(path: Path) -> AuditResult:
    result = AuditResult()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            tally = result.tally(method)
            verdict = (row.get("verdict") or "").strip().lower()
            label = f"{row['lb_name']} ({row['lb_year']})"
            note = row.get("note") or ""

            if verdict == VERDICT_CORRECT:
                tally.correct += 1
            elif verdict == VERDICT_WRONG:
                tally.wrong += 1
                result.wrong_cases.append((label, method, note))
            elif verdict == VERDICT_UNSURE:
                tally.unsure += 1
                result.unsure_cases.append((label, method, note))
            else:
                tally.unfilled += 1
    return result
