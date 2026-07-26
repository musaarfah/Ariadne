"""The baseline ladder.

A model result quoted without the ladder is incomplete. Each rung is deliberately given the
strongest version of itself: the popularity baseline fits its own mapping from TMDB's scale, and
the genre and director baselines get the same shrinkage the real model uses. A baseline built as
a strawman proves nothing when beaten.

Rung 4, director-only, is the project's gate. The claim is that below-the-line crew explains
taste better than the director alone. If the crew model cannot beat this on the temporal split,
the thesis is false and that is the finding.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ariadne.core.catalog.roles import ACTOR, DIRECTOR
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.taste.expectation import (
    AnyExpectation,
    fit_expectation,
    fit_rich_expectation,
)
from ariadne.core.taste.shrinkage import shrink


class Predictor(Protocol):
    name: str
    description: str

    def fit(self, train: list[RatedFilm]) -> None: ...

    def predict(self, films: list[RatedFilm]) -> np.ndarray: ...


@dataclass
class GlobalMean:
    """Rung 1: the floor. Predict the user's average and nothing else."""

    name: str = "global_mean"
    description: str = "the user's mean rating"
    _mean: float = 0.0

    def fit(self, train: list[RatedFilm]) -> None:
        self._mean = float(np.mean([f.rating for f in train])) if train else 0.0

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        return np.full(len(films), self._mean, dtype=float)


@dataclass
class Popularity:
    """Rung 2: predict from the film's global average alone, on a fitted scale."""

    name: str = "popularity"
    description: str = "TMDB vote_average, mapped to the user's scale by least squares"
    _expectation: AnyExpectation | None = None

    def fit(self, train: list[RatedFilm]) -> None:
        self._expectation = fit_expectation(train)

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        if self._expectation is None:
            raise RuntimeError("fit before predict")
        return self._expectation.predict(films)


@dataclass
class Context:
    """Rung 2b: the film's context — popularity, country, decade, genre — and nothing personal.

    Added because the simple popularity model left a +0.744-star mean residual on Japanese films.
    A crew model built on the richer expectation has to be compared against a *baseline* built on
    the same expectation, or the apparent improvement would just be the better baseline showing
    through.
    """

    name: str = "context"
    description: str = "vote_average, vote_count, country, decade and genre"
    _expectation: AnyExpectation | None = None

    def fit(self, train: list[RatedFilm]) -> None:
        self._expectation = fit_rich_expectation(train)

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        if self._expectation is None:
            raise RuntimeError("fit before predict")
        return self._expectation.predict(films)


class _PersonEffects:
    """Shared machinery for the genre and crew-style baselines.

    Effects are fitted on residuals against the popularity expectation, never on raw ratings, so
    a person cannot look preferred merely for working on well-regarded films (D6).
    """

    def __init__(
        self,
        name: str,
        description: str,
        role: str | None,
        expectation: str | Callable[[list[RatedFilm]], AnyExpectation] = "rich",
    ):
        self.name = name
        self.description = description
        self._role = role
        # Either of the two standard expectations by name, or a factory. The decomposition needs
        # person effects measured against a context model with some features ablated, and a second
        # copy of the shrinkage-on-residuals machinery is how two versions of it drift apart.
        self._expectation_kind = expectation
        self._expectation: AnyExpectation | None = None
        self._effects: dict[int, float] = {}

    def _keys(self, film: RatedFilm) -> tuple[int, ...]:
        assert self._role is not None  # noqa: S101
        return film.people_in(self._role)

    def fit(self, train: list[RatedFilm]) -> None:
        if callable(self._expectation_kind):
            self._expectation = self._expectation_kind(train)
        elif self._expectation_kind == "rich":
            self._expectation = fit_rich_expectation(train)
        else:
            self._expectation = fit_expectation(train)
        residuals = self._expectation.residuals(train)

        grouped: dict[int, list[float]] = defaultdict(list)
        for film, residual in zip(train, residuals, strict=True):
            for key in self._keys(film):
                grouped[key].append(float(residual))

        self._effects = {key: effect.shrunk for key, effect in shrink(dict(grouped)).items()}

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        if self._expectation is None:
            raise RuntimeError("fit before predict")

        base = self._expectation.predict(films)
        adjustments = np.zeros(len(films), dtype=float)
        for index, film in enumerate(films):
            known = [self._effects[k] for k in self._keys(film) if k in self._effects]
            # Mean rather than sum: two credited editors are not twice the effect, and a film
            # with none gets no adjustment rather than a penalty.
            if known:
                adjustments[index] = float(np.mean(known))
        return base + adjustments


class GenreOnly(_PersonEffects):
    """Rung 3: shrunken genre effects over the popularity expectation.

    Deliberately built on the *simple* expectation. On the rich one this rung was degenerate — genre
    is already a feature there, so it fitted genre effects on residuals from a model that had
    already used genre, and scored identically to `context` (D84). The rung then measured nothing
    and could not take its place in the decomposition, which needs each layer to add one thing
    (D108).
    """

    def __init__(self) -> None:
        super().__init__(
            "genre_only",
            "shrunken genre effects on the popularity expectation",
            role=None,
            expectation="simple",
        )
        self._genre_ids: dict[str, int] = {}

    def _keys(self, film: RatedFilm) -> tuple[int, ...]:
        keys = []
        for genre in film.genres:
            if genre not in self._genre_ids:
                self._genre_ids[genre] = len(self._genre_ids) + 1
            keys.append(self._genre_ids[genre])
        return tuple(keys)


class DirectorOnly(_PersonEffects):
    """Rung 4: the gate. Shrunken director effects, nothing below the line."""

    def __init__(self) -> None:
        super().__init__("director_only", "shrunken director effects on residuals", role=DIRECTOR)


class CastOnly(_PersonEffects):
    """Shrunken actor effects. Not a research rung — cast is product scope only (D97).

    Present because the decomposition has to answer "how much of my taste is the cast", which is the
    question users actually ask, and because it is the one person-layer with enough recurrence to
    clear the 12-film threshold in numbers (F69).
    """

    def __init__(self) -> None:
        super().__init__("cast_only", "shrunken actor effects on residuals", role=ACTOR)


def ladder() -> list[Predictor]:
    """The baselines, floor first."""
    return [GlobalMean(), Popularity(), Context(), GenreOnly(), DirectorOnly()]


def full_ladder() -> list[Predictor]:
    """The baselines plus both crew tracks — the ladder the go/no-go is judged on.

    `crew` is Track 1, shrunken per-person means. `crew_ridge` is Track 2, jointly fitted. Both
    exclude directors, so the comparison against `director_only` tests the actual claim rather than
    a model that has been handed the director for free.
    """
    from ariadne.core.taste.crew import CrewModel
    from ariadne.core.taste.ridge import RidgeCrewModel

    return [*ladder(), CrewModel(), RidgeCrewModel()]
