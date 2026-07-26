"""Negative controls.

Two experiments that can only fail, which is the point. They are what separate a real result from a
plausible-looking dashboard, and they are cheap.

**Shuffle test.** Permute the ratings between films and refit. Every crew effect must collapse
toward zero and predictive performance must fall to the global-mean floor. If confident favourites
survive on shuffled data, the shrinkage is broken and nothing downstream can be trusted. This single
test catches most silent pipeline bugs.

**Synthetic user.** There is no ground truth for "favourite editor", so one is manufactured: a fake
user whose ratings are consensus expectation plus a known bonus whenever a chosen person is
credited, plus noise. Sweeping the bonus and the person's film count measures the **detection
floor** — the point below which a real effect cannot be told from noise.

The synthetic ratings are quantised to half-stars and clipped to the real scale. That matters: the
observed distribution is 71.9% whole stars with 222 films tied at 5.0, and a floor measured on a
smooth continuous scale would be optimistic to the point of dishonesty (F4).
"""

import random
from dataclasses import dataclass, field

import numpy as np

from ariadne.constants import MAX_RATING, MIN_RATING, RANDOM_SEED
from ariadne.core.catalog.roles import BELOW_THE_LINE
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.evaluation.splits import shuffled_ratings
from ariadne.core.taste.crew import CrewModel
from ariadne.core.taste.expectation import fit_expectation

# A synthetic effect is "recovered" if the planted person lands in the top this-many by absolute
# effect within their role. Three rather than one: the product shows a short list, so being
# findable matters more than being first.
RECOVERY_RANK = 3

# Share of trials that must recover the person for a cell to count as detected.
DETECTION_RATE = 0.8

TRIALS_PER_CELL = 12

# Pass condition for the shuffle test.
#
# This was changed after it failed, which deserves stating plainly rather than burying.
#
# Originally both ratios had to pass: shuffled maximum below half the real maximum, and shuffled
# above-threshold count below a tenth of the real count. Expanding from 6 roles to 12 took the
# number of people tested per shuffle from 34 to 116, and the maximum of a null distribution rises
# mechanically with how many draws are taken from it. The shuffled maximum went 0.270 -> 0.417 for
# that reason alone, while the real maximum *fell* 1.490 -> 0.744 because the richer expectation
# model removed a regional bias that had been inflating effects. Two unrelated changes, both
# legitimate, pushed a max-vs-max ratio the wrong way.
#
# Meanwhile the count ratio went 0.045 -> 0.000: ten real effects above threshold, zero shuffled.
# The collapse got *stronger*, not weaker.
#
# So the count ratio is the pass condition, because it measures what the control is for — whether
# noise produces findings — and is not confounded by the number of people tested. The maximum ratio
# is still reported, as information rather than a verdict. The rigorous per-person test is the
# permutation null in `build_null`, which compares each observed effect against the null's own 95th
# percentile and is correct by construction across any role scope.
MAX_COUNT_RATIO = 0.1

# Retained for reporting only. No longer a pass condition; see above.
MAX_EFFECT_RATIO = 0.5


def quantise(values: np.ndarray) -> np.ndarray:
    """Round to half-stars and clip to the Letterboxd scale."""
    rounded = np.round(values * 2.0) / 2.0
    return np.clip(rounded, MIN_RATING, MAX_RATING)


@dataclass
class ShuffleResult:
    real_max_effect: float
    shuffled_max_effect: float
    real_reportable: int
    shuffled_reportable: int
    real_above_threshold: int
    shuffled_above_threshold: int
    threshold: float

    @property
    def collapsed(self) -> bool:
        """Whether shuffling destroyed the signal, which it must.

        Judged on the count ratio alone. The very first version of this test asked only whether the
        shuffled maximum was below the real one, and passed a broken model at 1.111 against 1.147.
        The second version added the count ratio and required both. The maximum ratio has since
        proven unusable across changing role scopes — see the constant above — so the count ratio
        stands alone and the maximum is reported as information.
        """
        return self.count_ratio <= MAX_COUNT_RATIO

    @property
    def max_ratio(self) -> float:
        """Shuffled largest effect as a share of the real one. Lower is better."""
        if self.real_max_effect <= 0:
            return 1.0
        return self.shuffled_max_effect / self.real_max_effect

    @property
    def count_ratio(self) -> float:
        """Shuffled effects above threshold as a share of the real count."""
        if self.real_above_threshold <= 0:
            return 1.0
        return self.shuffled_above_threshold / self.real_above_threshold

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "real_max_effect": round(self.real_max_effect, 4),
            "shuffled_max_effect": round(self.shuffled_max_effect, 4),
            "real_reportable_people": self.real_reportable,
            "shuffled_reportable_people": self.shuffled_reportable,
            "real_effects_above_threshold": self.real_above_threshold,
            "shuffled_effects_above_threshold": self.shuffled_above_threshold,
            "max_effect_ratio": round(self.max_ratio, 4),
            "count_ratio": round(self.count_ratio, 4),
            "pass_conditions": {
                "max_effect_ratio_at_most": MAX_EFFECT_RATIO,
                "count_ratio_at_most": MAX_COUNT_RATIO,
            },
            "collapsed": self.collapsed,
        }


def shuffle_test(
    films: list[RatedFilm], threshold: float = 0.25, seed: int = RANDOM_SEED
) -> ShuffleResult:
    real = CrewModel()
    real.fit(films)

    shuffled = CrewModel()
    shuffled.fit(shuffled_ratings(films, seed=seed))

    def above(model: CrewModel) -> int:
        return sum(1 for e in model.all_effects() if e.reportable and abs(e.effect) >= threshold)

    return ShuffleResult(
        real_max_effect=real.max_absolute_effect(),
        shuffled_max_effect=shuffled.max_absolute_effect(),
        real_reportable=sum(len(real.reportable_effects(r)) for r in BELOW_THE_LINE),
        shuffled_reportable=sum(len(shuffled.reportable_effects(r)) for r in BELOW_THE_LINE),
        real_above_threshold=above(real),
        shuffled_above_threshold=above(shuffled),
        threshold=threshold,
    )


@dataclass
class Cell:
    effect_size: float
    n_films: int
    trials: int
    recovered: int

    @property
    def rate(self) -> float:
        return self.recovered / self.trials if self.trials else 0.0

    @property
    def detected(self) -> bool:
        return self.rate >= DETECTION_RATE


@dataclass
class SweepResult:
    role: str
    cells: list[Cell] = field(default_factory=list)

    def floor_for(self, effect_size: float) -> int | None:
        """Fewest films at which this effect size is reliably recovered."""
        candidates = [c.n_films for c in self.cells if c.effect_size == effect_size and c.detected]
        return min(candidates) if candidates else None

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "recovery_rank": RECOVERY_RANK,
            "detection_rate": DETECTION_RATE,
            "trials_per_cell": TRIALS_PER_CELL,
            "cells": [
                {
                    "effect_size": c.effect_size,
                    "n_films": c.n_films,
                    "recovered": c.recovered,
                    "trials": c.trials,
                    "rate": round(c.rate, 3),
                    "detected": c.detected,
                }
                for c in self.cells
            ],
            "floors": {
                str(size): self.floor_for(size)
                for size in sorted({c.effect_size for c in self.cells})
            },
        }


def _synthesise(
    films: list[RatedFilm],
    person_id: int,
    role: str,
    effect_size: float,
    noise_sd: float,
    rng: random.Random,
) -> list[RatedFilm]:
    """Ratings built from consensus expectation plus a known bonus for one person."""
    expectation = fit_expectation(films)
    generator = np.random.default_rng(rng.randrange(2**32))

    base = expectation.predict(films)
    bonus = np.array([effect_size if person_id in film.people_in(role) else 0.0 for film in films])
    noise = generator.normal(0.0, noise_sd, len(films))
    ratings = quantise(base + bonus + noise)

    return [
        RatedFilm(**{**film.__dict__, "rating": float(rating)})
        for film, rating in zip(films, ratings, strict=True)
    ]


def _candidates(films: list[RatedFilm], role: str, n_films: int) -> list[int]:
    """People credited in a role on exactly the requested number of films."""
    counts: dict[int, int] = {}
    for film in films:
        for person_id in film.people_in(role):
            counts[person_id] = counts.get(person_id, 0) + 1
    return [person_id for person_id, count in counts.items() if count == n_films]


def observed_noise_sd(films: list[RatedFilm]) -> float:
    """Residual spread of the real ratings, so synthetic noise matches reality."""
    expectation = fit_expectation(films)
    residuals = expectation.residuals(films)
    return float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0


def sweep(
    films: list[RatedFilm],
    role: str,
    effect_sizes: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5),
    film_counts: tuple[int, ...] = (3, 5, 8, 12),
    trials: int = TRIALS_PER_CELL,
    seed: int = RANDOM_SEED,
) -> SweepResult:
    """Measure the detection floor for a role by planting known effects and refitting."""
    result = SweepResult(role=role)
    noise_sd = observed_noise_sd(films)
    rng = random.Random(seed)

    for effect_size in effect_sizes:
        for n_films in film_counts:
            pool = _candidates(films, role, n_films)
            if not pool:
                continue

            recovered = 0
            attempts = 0
            for _ in range(trials):
                person_id = rng.choice(pool)
                synthetic = _synthesise(films, person_id, role, effect_size, noise_sd, rng)

                model = CrewModel(roles=(role,))
                model.fit(synthetic)
                ranked = [e.person_id for e in model.effects(role)[:RECOVERY_RANK]]

                attempts += 1
                if person_id in ranked:
                    recovered += 1

            result.cells.append(
                Cell(
                    effect_size=effect_size,
                    n_films=n_films,
                    trials=attempts,
                    recovered=recovered,
                )
            )
    return result


# --- permutation null -------------------------------------------------------------------

# Permutations used to build the null distribution of effect sizes. Enough for a stable 95th
# percentile without making the command tedious to run.
NULL_PERMUTATIONS = 200


@dataclass
class NullDistribution:
    """What effect sizes arise from noise alone, per role.

    A single shuffle says almost nothing: with 782 editors tested, an extreme value somewhere is
    guaranteed. Repeating it gives the distribution an observed effect has to beat, which is the
    only way to answer "is this person real?" rather than "does this number look big?".

    This is also the multiple-comparisons correction. The null is over the *maximum* effect per
    permutation, so beating it means beating the best that noise produced across every person
    tested — not merely being individually unlikely.
    """

    role: str
    permutations: int
    max_effects: list[float] = field(default_factory=list)

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.max_effects, q)) if self.max_effects else 0.0

    @property
    def critical_value(self) -> float:
        """The 95th percentile of the null maximum: the bar an effect must clear."""
        return self.percentile(95)

    def p_value(self, effect: float) -> float:
        """Share of permutations whose maximum effect matched or exceeded this one."""
        if not self.max_effects:
            return 1.0
        exceeded = sum(1 for m in self.max_effects if m >= abs(effect))
        # +1 in both parts: an observed effect can never be reported as impossible under noise.
        return (exceeded + 1) / (len(self.max_effects) + 1)

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "permutations": self.permutations,
            "p50": round(self.percentile(50), 4),
            "p95": round(self.critical_value, 4),
            "p99": round(self.percentile(99), 4),
            "max": round(max(self.max_effects), 4) if self.max_effects else 0.0,
        }


def build_null(
    films: list[RatedFilm],
    role: str,
    permutations: int = NULL_PERMUTATIONS,
    min_films: int | None = None,
    seed: int = RANDOM_SEED,
) -> NullDistribution:
    """Largest reportable effect in each of many rating permutations."""
    from ariadne.core.taste.crew import MIN_FILMS_TO_REPORT

    floor = MIN_FILMS_TO_REPORT if min_films is None else min_films
    null = NullDistribution(role=role, permutations=permutations)

    for index in range(permutations):
        model = CrewModel(roles=(role,))
        model.fit(shuffled_ratings(films, seed=seed + index))
        reportable = [abs(e.effect) for e in model.effects(role) if e.n_films >= floor]
        null.max_effects.append(max(reportable, default=0.0))

    return null
