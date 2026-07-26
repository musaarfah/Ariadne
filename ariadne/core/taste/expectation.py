"""What rating a user was expected to give a film, before their taste is considered.

The residual against this expectation is the target everything else is fitted on. Consensus
carries no personal information: a 5.0 for The Godfather is what almost everyone gives it and
says nothing about the viewer, while a 5.0 for a film the world is lukewarm on is dense with
signal. Fitting on raw ratings would let a crew member look preferred merely for working on
well-regarded films.

Two models live here.

`Expectation` uses TMDB's vote_average alone, fitted by least squares because the relationship
between a 0-10 consensus and a 0.5-5.0 personal scale is not simply half. It was the original.

`RichExpectation` adds vote_count, country, decade and genre. It exists because the simple model
left a **+0.744-star mean residual on Japanese films** — larger than any crew effect this project
has reported. Every crew member working predominantly in one national cinema was inheriting the
model's failure to predict that cinema, which is a bias in the crew effects rather than noise in
them. After the fix Japan sits near +0.10 and Korea near +0.10.

Categories are learned from training data only, and a category unseen in training contributes
nothing at prediction time rather than being dropped or guessed.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge

from ariadne.core.evaluation.dataset import RatedFilm

# Penalty for the rich model. Modest: roughly sixty mostly-sparse indicator columns against a
# thousand films, so some regularisation is needed but the signal is real and should survive.
RICH_ALPHA = 10.0

# Below this many films the indicators outnumber anything they could learn from.
MIN_FILMS_FOR_RICH = 20


@dataclass(frozen=True)
class Expectation:
    """The simple model: a line through TMDB's vote_average."""

    intercept: float
    slope: float
    fallback: float
    fitted_on: int
    name: str = "simple"

    def predict_one(self, film: RatedFilm) -> float:
        if film.vote_average is None:
            return self.fallback
        return self.intercept + self.slope * film.vote_average

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        return np.array([self.predict_one(film) for film in films], dtype=float)

    def residuals(self, films: list[RatedFilm], target: str = "rating") -> np.ndarray:
        result: np.ndarray = target_values(films, target) - self.predict(films)
        return result


def target_values(films: list[RatedFilm], target: str = "rating") -> np.ndarray:
    """The quantity being predicted: the raw rating, or rating plus a rewatch bonus."""
    if target == "preference":
        return np.array([film.preference for film in films], dtype=float)
    return np.array([film.rating for film in films], dtype=float)


def fit_expectation(train: list[RatedFilm], target: str = "rating") -> Expectation:
    ratings = target_values(train, target)
    fallback = float(ratings.mean()) if len(ratings) else 0.0

    usable = [film for film in train if film.vote_average is not None]
    if len(usable) < 2:
        return Expectation(intercept=fallback, slope=0.0, fallback=fallback, fitted_on=0)

    x = np.array([film.vote_average for film in usable], dtype=float)
    y = target_values(usable, target)

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


@dataclass
class RichExpectation:
    """Consensus plus the film's context: popularity, country, decade, genre.

    Not a taste model. Every feature here describes the film rather than the viewer, so a residual
    against it still means "more or less than this sort of film usually gets from this person".
    """

    countries: tuple[str, ...] = ()
    decades: tuple[int, ...] = ()
    genres: tuple[str, ...] = ()
    fallback: float = 0.0

    # Only ever true when the target carries a rewatch bonus.
    #
    # On the temporal split, in_diary is 16% of train and 69% of test — the diary begins
    # 2023-11-26 and the split cuts at 2024-01-01 — so as a feature it is close to a "this row is
    # in the test set" indicator. Including it when the target is the raw rating buys nothing and
    # leaks split membership. Including it when the target is `preference` is necessary, because
    # only diary-covered films can carry the bonus at all.
    use_diary_flag: bool = False

    name: str = "rich"
    _model: Ridge | None = field(default=None, repr=False)

    def _row(self, film: RatedFilm) -> list[float]:
        values = [
            film.vote_average if film.vote_average is not None else 0.0,
            float(np.log1p(film.vote_count or 0)),
            # An explicit missing-vote_average flag, so the zero above is not read as a rating.
            0.0 if film.vote_average is not None else 1.0,
        ]
        if self.use_diary_flag:
            values.append(1.0 if film.in_diary else 0.0)
        country = film.country or "??"
        decade = (film.year // 10) * 10 if film.year else 0
        values += [1.0 if country == c else 0.0 for c in self.countries]
        values += [1.0 if decade == d else 0.0 for d in self.decades]
        values += [1.0 if g in film.genres else 0.0 for g in self.genres]
        return values

    def _design(self, films: list[RatedFilm]) -> np.ndarray:
        return np.array([self._row(film) for film in films], dtype=float)

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        if self._model is None:
            return np.full(len(films), self.fallback, dtype=float)
        return np.asarray(self._model.predict(self._design(films)), dtype=float)

    def residuals(self, films: list[RatedFilm], target: str = "rating") -> np.ndarray:
        result: np.ndarray = target_values(films, target) - self.predict(films)
        return result


def fit_rich_expectation(train: list[RatedFilm], target: str = "rating") -> RichExpectation:
    """Fit the context model. Categories come from the training block only."""
    ratings = target_values(train, target)
    fallback = float(ratings.mean()) if len(ratings) else 0.0

    if len(train) < MIN_FILMS_FOR_RICH:
        return RichExpectation(fallback=fallback)

    model = RichExpectation(
        countries=tuple(sorted({film.country or "??" for film in train})),
        decades=tuple(sorted({(film.year // 10) * 10 if film.year else 0 for film in train})),
        genres=tuple(sorted({genre for film in train for genre in film.genres})),
        fallback=fallback,
        use_diary_flag=target == "preference",
    )
    ridge = Ridge(alpha=RICH_ALPHA, fit_intercept=True)
    ridge.fit(model._design(train), ratings)
    model._model = ridge
    return model


# Anything fitted on residuals accepts either model.
AnyExpectation = Expectation | RichExpectation
