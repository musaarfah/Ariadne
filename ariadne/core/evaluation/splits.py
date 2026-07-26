"""Train/test splits, and the drift between the halves.

The temporal split is the honest headline, but it cannot be taken at face value on this data.
58% of the reference account's ratings were entered in a 2023 backfill burst, rated from memory,
and the live-logged half is measurably more generous — mean 3.59 against 3.35, with nearly double
the share of 5.0s. Error therefore moves between the halves for reasons that have nothing to do
with model quality, so the drift is computed and reported alongside every metric rather than
left for someone to discover (F3/D12).

The random split is reported too, and it leaks: same era, same wave, same collaborators either
side of the line. It is the optimistic number, and saying so is the point of having both.
"""

import random
from dataclasses import dataclass
from datetime import date

from ariadne.constants import RANDOM_SEED, TEMPORAL_SPLIT_DATE
from ariadne.core.evaluation.dataset import RatedFilm

RANDOM_TEST_FRACTION = 0.3


@dataclass(frozen=True)
class BlockStats:
    n: int
    mean: float
    sd: float
    whole_star_share: float
    top_rating_share: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "mean": round(self.mean, 4),
            "sd": round(self.sd, 4),
            "whole_star_share": round(self.whole_star_share, 4),
            "top_rating_share": round(self.top_rating_share, 4),
        }


@dataclass(frozen=True)
class Drift:
    train: BlockStats
    test: BlockStats

    @property
    def mean_shift(self) -> float:
        return self.test.mean - self.train.mean

    @property
    def top_rating_share_shift(self) -> float:
        return self.test.top_rating_share - self.train.top_rating_share

    def as_dict(self) -> dict[str, object]:
        return {
            "train": self.train.as_dict(),
            "test": self.test.as_dict(),
            "mean_shift": round(self.mean_shift, 4),
            "top_rating_share_shift": round(self.top_rating_share_shift, 4),
        }


@dataclass(frozen=True)
class Split:
    name: str
    train: list[RatedFilm]
    test: list[RatedFilm]
    note: str = ""
    dropped: int = 0

    def drift(self) -> Drift:
        return Drift(train=describe(self.train), test=describe(self.test))


def describe(films: list[RatedFilm]) -> BlockStats:
    if not films:
        return BlockStats(0, 0.0, 0.0, 0.0, 0.0)
    ratings = [f.rating for f in films]
    n = len(ratings)
    mean = sum(ratings) / n
    variance = sum((r - mean) ** 2 for r in ratings) / (n - 1) if n > 1 else 0.0
    return BlockStats(
        n=n,
        mean=mean,
        sd=variance**0.5,
        whole_star_share=sum(1 for r in ratings if r == int(r)) / n,
        top_rating_share=sum(1 for r in ratings if r >= 5.0) / n,
    )


def temporal_split(films: list[RatedFilm], cut: date = TEMPORAL_SPLIT_DATE) -> Split:
    """Split on log date. Films with no log date are dropped and counted, never guessed."""
    dated = [f for f in films if f.logged_date is not None]
    dropped = len(films) - len(dated)

    train = [f for f in dated if f.logged_date is not None and f.logged_date < cut]
    test = [f for f in dated if f.logged_date is not None and f.logged_date >= cut]
    return Split(
        name=f"temporal@{cut.isoformat()}",
        train=train,
        test=test,
        note=(
            "log date, not watch date. The pre-cut block is one undifferentiated backfill and "
            "carries no reliable internal chronology."
        ),
        dropped=dropped,
    )


def random_split(
    films: list[RatedFilm],
    test_fraction: float = RANDOM_TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> Split:
    shuffled = list(films)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_fraction))
    return Split(
        name=f"random@{test_fraction:g}",
        train=shuffled[:cut],
        test=shuffled[cut:],
        note="leaks era, wave and collaborators across the split; the optimistic number",
    )


def shuffled_ratings(films: list[RatedFilm], seed: int = RANDOM_SEED) -> list[RatedFilm]:
    """The same films with their ratings permuted between them.

    The negative control from §7 Level 4: on this data every crew effect must collapse toward
    zero and every model must fall back to the global-mean baseline. If one does not, the
    shrinkage is broken.
    """
    ratings = [f.rating for f in films]
    random.Random(seed).shuffle(ratings)
    return [
        RatedFilm(**{**film.__dict__, "rating": rating})
        for film, rating in zip(films, ratings, strict=True)
    ]
