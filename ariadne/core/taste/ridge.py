"""Track 2: ridge regression over sparse crew indicators.

Track 1 estimates each person independently, which is why it cannot separate collaborators. If a
cinematographer has only ever shot for one director, their shrunken mean is that director's effect
wearing a different name.

Ridge fits everyone at once. When two people always appear together the penalty splits the shared
effect between them rather than handing it to both, so comparing a person's Track 1 effect against
their ridge coefficient *with the director included as a feature* answers a question Track 1 cannot:
does this person's effect survive once the director is accounted for?

Features are keyed by (role, person), so someone who edits one film and writes another gets two
coefficients — the same treatment Track 1 gives them.

People below a minimum film count are excluded as features. A person appearing once has a
coefficient that can absorb that film's entire residual, which is fitting noise by construction, and
tens of thousands of such columns would swamp the ones that carry information.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeCV

from ariadne.core.catalog.roles import ALL_ROLES, BELOW_THE_LINE
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.taste.expectation import (
    AnyExpectation,
    fit_expectation,
    fit_rich_expectation,
)

# A person needs at least this many films to become a feature. Two is the minimum at which a
# coefficient is estimated from more than one observation.
MIN_FILMS_FOR_FEATURE = 2

# An effect smaller than this is not worth attributing: the retained-share ratio becomes a division
# by almost nothing, and "survives" would be claiming that a non-effect withstood competition.
MIN_EFFECT_TO_ATTRIBUTE = 0.05

# Share of the Track 1 effect a coefficient must keep, with directors in the model, to count as the
# person's own rather than their director's.
SURVIVAL_SHARE = 0.5

# Candidate penalties for RidgeCV. Wide, because the right value depends on how sparse the matrix
# turns out to be, and cross-validation happens inside the training block only.
ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


@dataclass
class RidgeCoefficient:
    role: str
    person_id: int
    coefficient: float
    n_films: int


@dataclass
class RidgeCrewModel:
    """Jointly fitted crew effects.

    `roles` defaults to below-the-line only so the go/no-go compares like with like against Track 1.
    Pass ALL_ROLES to include directors as features, which is what makes the attribution comparison
    possible.
    """

    roles: tuple[str, ...] = BELOW_THE_LINE
    alpha: float | None = None
    min_films: int = MIN_FILMS_FOR_FEATURE
    expectation: str = "rich"
    name: str = "crew_ridge"
    description: str = "ridge over sparse crew indicators, jointly fitted"

    _expectation: AnyExpectation | None = field(default=None, repr=False)
    _columns: dict[tuple[str, int], int] = field(default_factory=dict, repr=False)
    _counts: dict[tuple[str, int], int] = field(default_factory=dict, repr=False)
    _coefficients: np.ndarray | None = field(default=None, repr=False)
    chosen_alpha: float = 0.0

    def _count_people(self, films: list[RatedFilm]) -> dict[tuple[str, int], int]:
        counts: dict[tuple[str, int], int] = {}
        for film in films:
            for role in self.roles:
                for person_id in film.people_in(role):
                    key = (role, person_id)
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def _design(self, films: list[RatedFilm]) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        for index, film in enumerate(films):
            for role in self.roles:
                for person_id in film.people_in(role):
                    column = self._columns.get((role, person_id))
                    if column is not None:
                        rows.append(index)
                        cols.append(column)
        data = np.ones(len(rows), dtype=float)
        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(films), max(len(self._columns), 1))
        )

    def fit(self, train: list[RatedFilm]) -> None:
        self._expectation = (
            fit_rich_expectation(train) if self.expectation == "rich" else fit_expectation(train)
        )
        residuals = self._expectation.residuals(train)

        self._counts = self._count_people(train)
        self._columns = {
            key: index
            for index, key in enumerate(
                sorted(k for k, n in self._counts.items() if n >= self.min_films)
            )
        }

        design = self._design(train)
        if design.shape[1] == 0 or design.nnz == 0:
            self._coefficients = np.zeros(design.shape[1], dtype=float)
            self.chosen_alpha = 0.0
            return

        if self.alpha is None:
            # Cross-validated inside the training block only, so no test information leaks.
            search = RidgeCV(alphas=ALPHA_GRID, fit_intercept=True)
            search.fit(design, residuals)
            self.chosen_alpha = float(search.alpha_)
            self._coefficients = np.asarray(search.coef_, dtype=float)
        else:
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(design, residuals)
            self.chosen_alpha = self.alpha
            self._coefficients = np.asarray(model.coef_, dtype=float)

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        if self._expectation is None or self._coefficients is None:
            raise RuntimeError("fit before predict")

        base = self._expectation.predict(films)
        if not self._columns:
            return base

        adjustment = self._design(films) @ self._coefficients
        return base + np.asarray(adjustment, dtype=float)

    def coefficients(self, role: str | None = None) -> list[RidgeCoefficient]:
        """Fitted coefficients, largest magnitude first."""
        if self._coefficients is None:
            return []
        found = [
            RidgeCoefficient(
                role=key[0],
                person_id=key[1],
                coefficient=float(self._coefficients[column]),
                n_films=self._counts.get(key, 0),
            )
            for key, column in self._columns.items()
            if role is None or key[0] == role
        ]
        return sorted(found, key=lambda c: -abs(c.coefficient))

    def coefficient_for(self, role: str, person_id: int) -> float | None:
        if self._coefficients is None:
            return None
        column = self._columns.get((role, person_id))
        return None if column is None else float(self._coefficients[column])

    @property
    def n_features(self) -> int:
        return len(self._columns)


@dataclass
class Attribution:
    """One person's effect before and after the director is accounted for."""

    role: str
    person_id: int
    n_films: int
    track1_effect: float
    ridge_without_director: float
    ridge_with_director: float

    @property
    def attributable(self) -> bool:
        """Whether there is an effect worth attributing in the first place.

        Below this, the ratio below is a division by almost nothing and produces values like 14.8
        or -50.2 that mean nothing at all.
        """
        return abs(self.track1_effect) >= MIN_EFFECT_TO_ATTRIBUTE

    @property
    def survives(self) -> bool:
        """Whether the effect keeps its sign and most of its size once directors compete."""
        if not self.attributable:
            return False
        same_sign = np.sign(self.ridge_with_director) == np.sign(self.track1_effect)
        retained = abs(self.ridge_with_director) >= SURVIVAL_SHARE * abs(self.track1_effect)
        return bool(same_sign and retained)

    @property
    def retained_share(self) -> float | None:
        """None rather than a meaningless ratio when there was no effect to retain."""
        if not self.attributable:
            return None
        return self.ridge_with_director / self.track1_effect

    @property
    def verdict(self) -> str:
        if not self.attributable:
            return "no effect to attribute"
        return "survives" if self.survives else "absorbed by director"


def attribute(
    films: list[RatedFilm], track1_effects: dict[tuple[str, int], tuple[float, int]]
) -> list[Attribution]:
    """Refit with and without directors as features, and compare against Track 1.

    A person whose coefficient collapses once directors are in the model had an effect that was
    really their director's — which is the quantitative form of the inseparability Track 1 can only
    flag structurally (D8, D9).
    """
    without = RidgeCrewModel(roles=BELOW_THE_LINE)
    without.fit(films)

    with_director = RidgeCrewModel(roles=ALL_ROLES)
    with_director.fit(films)

    results: list[Attribution] = []
    for (role, person_id), (effect, n_films) in track1_effects.items():
        results.append(
            Attribution(
                role=role,
                person_id=person_id,
                n_films=n_films,
                track1_effect=effect,
                ridge_without_director=without.coefficient_for(role, person_id) or 0.0,
                ridge_with_director=with_director.coefficient_for(role, person_id) or 0.0,
            )
        )
    return sorted(results, key=lambda a: -abs(a.track1_effect))
