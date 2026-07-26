"""Track 1: shrunken residual effects for below-the-line crew.

Effects are fitted on residuals against consensus expectation, never on raw ratings, so a person
cannot look preferred merely for working on well-regarded films (D6). Each person's mean residual
is then shrunk toward zero in proportion to how little evidence supports it (D7).

The predictor combines roles by averaging rather than summing. A film crediting a liked editor and
a liked composer is not evidence of twice the effect — the two are correlated, since good films
tend to be well-made throughout — and summing would let a film with many credited roles drift far
from any observed rating.

Inseparability is detected rather than resolved. A cinematographer who has only ever shot for one
director in this library cannot be told apart from that director by any amount of arithmetic, and
saying so is more useful than picking one (D9).
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ariadne.core.catalog.roles import BELOW_THE_LINE, DIRECTOR
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.taste.expectation import Expectation, fit_expectation
from ariadne.core.taste.shrinkage import ShrunkEffect, shrink

# Below this many films a person's effect goes in the insufficient-data bucket rather than being
# reported as a result. **Measured, not chosen** (F44): the synthetic sweep finds that a +1.00-star
# effect needs 12 films to be recovered in four of five roles, a +0.75-star effect is recoverable
# only for editors and only at 12, and anything at or below +0.50 stars is undetectable at every
# film count this library contains.
#
# The consequence is uncomfortable and deliberate. Only 3 editors, 1 cinematographer, 4 writers and
# 1 production designer have 12 films here; composer alone has 29. So most roles will report one to
# four names, not ten (D64), and the insufficient-data bucket is the majority of every list.
MIN_FILMS_TO_REPORT = 12

# A person credited on at least this many films, all sharing one director, is flagged as
# inseparable from that director. Two films is enough to make the point; one is not a pattern.
MIN_FILMS_FOR_INSEPARABILITY = 2


@dataclass(frozen=True)
class CrewEffect:
    person_id: int
    role: str
    effect: float
    raw_mean: float
    n_films: int
    stderr: float
    inseparable_from: int | None = None
    director_overlap: float = 0.0

    @property
    def reportable(self) -> bool:
        return self.n_films >= MIN_FILMS_TO_REPORT


@dataclass
class CrewModel:
    """Fits and applies per-role shrunken effects.

    `roles` defaults to below-the-line only, because the thesis is that those explain taste better
    than the director alone. Passing ALL_ROLES gives the crew-plus-director variant, which is
    informative but does not test the claim.
    """

    roles: tuple[str, ...] = BELOW_THE_LINE
    name: str = "crew"
    description: str = "shrunken below-the-line crew effects on residuals"

    _expectation: Expectation | None = field(default=None, repr=False)
    _effects: dict[str, dict[int, CrewEffect]] = field(default_factory=dict, repr=False)
    _directors_of_person: dict[str, dict[int, list[int]]] = field(default_factory=dict, repr=False)

    def fit(self, train: list[RatedFilm]) -> None:
        self._expectation = fit_expectation(train)
        residuals = self._expectation.residuals(train)

        self._effects = {}
        self._directors_of_person = {}

        for role in self.roles:
            grouped: dict[int, list[float]] = defaultdict(list)
            directors: dict[int, list[int]] = defaultdict(list)

            for film, residual in zip(train, residuals, strict=True):
                film_directors = film.people_in(DIRECTOR)
                for person_id in film.people_in(role):
                    grouped[person_id].append(float(residual))
                    directors[person_id].extend(film_directors)

            shrunk = shrink(dict(grouped))
            self._effects[role] = {
                person_id: self._build(person_id, role, effect, directors[person_id])
                for person_id, effect in shrunk.items()
            }
            self._directors_of_person[role] = dict(directors)

    def _build(
        self, person_id: int, role: str, effect: ShrunkEffect, directors: list[int]
    ) -> CrewEffect:
        inseparable, overlap = _inseparability(effect.n, directors)
        return CrewEffect(
            person_id=person_id,
            role=role,
            effect=effect.shrunk,
            raw_mean=effect.raw_mean,
            n_films=effect.n,
            stderr=effect.stderr,
            inseparable_from=inseparable,
            director_overlap=overlap,
        )

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        if self._expectation is None:
            raise RuntimeError("fit before predict")

        base = self._expectation.predict(films)
        adjustments = np.zeros(len(films), dtype=float)

        for index, film in enumerate(films):
            per_role: list[float] = []
            for role in self.roles:
                known = [
                    self._effects[role][p].effect
                    for p in film.people_in(role)
                    if p in self._effects.get(role, {})
                ]
                if known:
                    per_role.append(float(np.mean(known)))
            if per_role:
                adjustments[index] = float(np.mean(per_role))

        return base + adjustments

    def effects(self, role: str) -> list[CrewEffect]:
        """Every person fitted in a role, strongest effect first."""
        return sorted(self._effects.get(role, {}).values(), key=lambda e: -abs(e.effect))

    def reportable_effects(self, role: str) -> list[CrewEffect]:
        return [e for e in self.effects(role) if e.reportable]

    def all_effects(self) -> list[CrewEffect]:
        return [effect for role in self.roles for effect in self.effects(role)]

    def max_absolute_effect(self) -> float:
        effects = self.all_effects()
        return max((abs(e.effect) for e in effects), default=0.0)


def _inseparability(n_films: int, directors: list[int]) -> tuple[int | None, float]:
    """Whether a person's films all share one director, and how concentrated they are.

    Returns the director's id when every one of the person's films shares them, plus the overlap
    share so a near-miss (nine of ten films) is visible rather than hidden by an exact test.
    """
    if not directors or n_films < MIN_FILMS_FOR_INSEPARABILITY:
        return None, 0.0

    counts: dict[int, int] = defaultdict(int)
    for director_id in directors:
        counts[director_id] += 1

    top_director, top_count = max(counts.items(), key=lambda kv: kv[1])
    overlap = top_count / n_films
    return (top_director if top_count == n_films else None), overlap
