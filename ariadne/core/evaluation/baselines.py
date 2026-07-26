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
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ariadne.core.catalog.roles import DIRECTOR
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.taste.expectation import Expectation, fit_expectation
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
    _expectation: Expectation | None = None

    def fit(self, train: list[RatedFilm]) -> None:
        self._expectation = fit_expectation(train)

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        if self._expectation is None:
            raise RuntimeError("fit before predict")
        return self._expectation.predict(films)


class _PersonEffects:
    """Shared machinery for the genre and crew-style baselines.

    Effects are fitted on residuals against the popularity expectation, never on raw ratings, so
    a person cannot look preferred merely for working on well-regarded films (D6).
    """

    def __init__(self, name: str, description: str, role: str | None):
        self.name = name
        self.description = description
        self._role = role
        self._expectation: Expectation | None = None
        self._effects: dict[int, float] = {}

    def _keys(self, film: RatedFilm) -> tuple[int, ...]:
        assert self._role is not None  # noqa: S101
        return film.people_in(self._role)

    def fit(self, train: list[RatedFilm]) -> None:
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
    """Rung 3: shrunken genre effects over the popularity expectation."""

    def __init__(self) -> None:
        super().__init__("genre_only", "shrunken genre effects on residuals", role=None)
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


def ladder() -> list[Predictor]:
    """The four baselines, floor first. Rung 5 is the crew model, built in 1.7."""
    return [GlobalMean(), Popularity(), GenreOnly(), DirectorOnly()]
