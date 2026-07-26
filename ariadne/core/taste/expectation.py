"""What rating a user was expected to give a film, before their taste is considered.

The residual against this expectation is the target everything else is fitted on. Consensus
carries no personal information: a 5.0 for The Godfather is what almost everyone gives it and
says nothing about the viewer, while a 5.0 for a film the world is lukewarm on is dense with
signal. Fitting on raw ratings would let a crew member look preferred merely for working on
well-regarded films.

TMDB's vote_average is on a 0-10 scale against the user's 0.5-5.0, and the relationship is not
simply half. It is fitted per user by least squares, so the baseline gets the strongest form of
itself rather than a strawman — the whole point of a baseline ladder is that beating it means
something.
"""

from dataclasses import dataclass

import numpy as np

from ariadne.core.evaluation.dataset import RatedFilm


@dataclass(frozen=True)
class Expectation:
    intercept: float
    slope: float
    fallback: float
    fitted_on: int

    def predict_one(self, film: RatedFilm) -> float:
        if film.vote_average is None:
            return self.fallback
        return self.intercept + self.slope * film.vote_average

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        return np.array([self.predict_one(film) for film in films], dtype=float)

    def residuals(self, films: list[RatedFilm]) -> np.ndarray:
        actual = np.array([film.rating for film in films], dtype=float)
        result: np.ndarray = actual - self.predict(films)
        return result


def fit_expectation(train: list[RatedFilm]) -> Expectation:
    ratings = np.array([film.rating for film in train], dtype=float)
    fallback = float(ratings.mean()) if len(ratings) else 0.0

    usable = [film for film in train if film.vote_average is not None]
    if len(usable) < 2:
        return Expectation(intercept=fallback, slope=0.0, fallback=fallback, fitted_on=0)

    x = np.array([film.vote_average for film in usable], dtype=float)
    y = np.array([film.rating for film in usable], dtype=float)

    # Degenerate if every film shares a vote_average; fall back rather than divide by zero.
    if float(x.std()) == 0.0:
        return Expectation(intercept=fallback, slope=0.0, fallback=fallback, fitted_on=len(usable))

    slope, intercept = np.polyfit(x, y, 1)
    return Expectation(
        intercept=float(intercept),
        slope=float(slope),
        fallback=fallback,
        fitted_on=len(usable),
    )
