"""What explains your taste — tier 2, so every number carries an interval.

The question is how much of a rating is accounted for by each kind of information about a film: what
everyone else thought of it, when and where it was made, what kind of film it is, and who made it.

Two rules make this honest rather than decorative.

**Contributions are marginal, not sequential.** Each layer is measured by adding it *alone* to the
same consensus base. A sequential chain would make each number depend on the order the layers were
listed in, and the order would then be a design choice presented as a result. The cost is that the
contributions do not sum to the total, because the layers share information — genre and country
overlap, and a director carries their own era. `Decomposition.combined` reports the total separately
so the gap is visible instead of hidden.

**Two metrics, because they answer different questions.** Variance explained answers "how much of
your rating does this account for", over every held-out film. Gate precision answers "would this
make the recommendations better", over the top 100 only. F73 showed these can disagree — gradient
boosting was the better regressor and the worse ranker — so reporting one alone would let a layer
look useful for a job it does not do.

Layers are allowed to come out at zero or below. After F70–F73 the crew layer is expected to, and
that is the measured result rather than a failure of the display.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ariadne.constants import TEMPORAL_SPLIT_DATE
from ariadne.core.catalog.roles import ACTOR, BELOW_THE_LINE, DIRECTOR
from ariadne.core.evaluation.baselines import Predictor, _PersonEffects
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.evaluation.metrics import (
    GATE_K,
    GATE_THRESHOLD,
    Comparison,
    compare,
    precision_at_k,
    variance_explained_centred,
)
from ariadne.core.evaluation.splits import Split, random_split, temporal_split
from ariadne.core.taste.crew import CrewModel
from ariadne.core.taste.expectation import AnyExpectation, fit_rich_expectation

# Fewer resamples than the gate comparison uses. There are eleven comparisons per split here rather
# than one, and the interval is read to two decimals.
DECOMPOSITION_RESAMPLES = 1000


@dataclass(frozen=True)
class Layer:
    """One kind of information, and what it adds to the base.

    Both fields are differences against the same base predictor, never absolute scores. An absolute
    score would be dominated by the consensus term every layer shares.
    """

    label: str
    detail: str
    explained: Comparison
    ranking: Comparison

    @property
    def helps(self) -> bool:
        """True only when the variance-explained interval clears zero."""
        return self.explained.ci_low > 0.0

    @property
    def hurts(self) -> bool:
        return self.explained.ci_high < 0.0


@dataclass
class Decomposition:
    split: str
    note: str
    test_n: int

    # What the consensus base alone scores. Every layer is a difference against these, and must be
    # measured in the same metric — `base_gate` is the gate configuration, not the product one.
    base_explained: float
    base_gate: float

    layers: list[Layer] = field(default_factory=list)

    # All layers together, so the gap against the sum of the marginals is visible.
    combined: Layer | None = None

    @property
    def sum_of_layers(self) -> float:
        return sum(layer.explained.observed_diff for layer in self.layers)

    @property
    def overlap(self) -> float:
        """How much the marginal contributions double-count.

        Positive means the layers share information: measured one at a time they add up to more
        than they deliver together. This is why the layers are not presented as a pie chart.
        """
        if self.combined is None:
            return 0.0
        return self.sum_of_layers - self.combined.explained.observed_diff


def _context(features: tuple[str, ...]) -> Callable[[list[RatedFilm]], AnyExpectation]:
    return lambda train: fit_rich_expectation(train, features=features)


@dataclass
class _Expectation:
    """A context model used directly as a predictor, with no person effects on top."""

    name: str
    description: str
    features: tuple[str, ...]
    _fitted: AnyExpectation | None = None

    def fit(self, train: list[RatedFilm]) -> None:
        self._fitted = fit_rich_expectation(train, features=self.features)

    def predict(self, films: list[RatedFilm]) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("fit before predict")
        return self._fitted.predict(films)


def _layer_specs() -> list[tuple[str, str, Predictor]]:
    """Each layer as (label, detail, the predictor that is the base plus this layer alone).

    Person layers sit on the *full* context model, not the consensus base. Fitting them on
    consensus alone would let a director absorb the era and country of their own films and be
    credited for it — the prestige confound the whole project is built to avoid (D6).
    """
    full = ("decade", "country", "genre")
    return [
        ("when it was made", "decade", _Expectation("decade", "consensus + decade", ("decade",))),
        (
            "where it comes from",
            "country",
            _Expectation("country", "consensus + country", ("country",)),
        ),
        (
            "what kind of film it is",
            "genre",
            _Expectation("genre", "consensus + genre", ("genre",)),
        ),
        (
            "who directed it",
            "shrunken director effects",
            _PersonEffects("director", "director effects", DIRECTOR, _context(full)),
        ),
        (
            "who acted in it",
            "shrunken effects for the top 10 billed",
            _PersonEffects("cast", "actor effects", ACTOR, _context(full)),
        ),
        (
            "who else made it",
            "editor, DP, composer, writer and seven more roles",
            CrewModel(roles=BELOW_THE_LINE, expectation="rich"),
        ),
    ]


def _compare_both(
    label: str,
    detail: str,
    predicted: np.ndarray,
    base_predicted: np.ndarray,
    actual: np.ndarray,
    resamples: int,
) -> Layer:
    return Layer(
        label=label,
        detail=detail,
        explained=compare(
            label,
            predicted,
            "base",
            base_predicted,
            actual,
            resamples=resamples,
            metric=variance_explained_centred,
        ),
        ranking=compare(label, predicted, "base", base_predicted, actual, resamples=resamples),
    )


def decompose_split(split: Split, resamples: int = DECOMPOSITION_RESAMPLES) -> Decomposition:
    actual = np.array([f.rating for f in split.test], dtype=float)

    base = _Expectation("consensus", "vote_average and vote_count only", ())
    base.fit(split.train)
    base_predicted = base.predict(split.test)

    result = Decomposition(
        split=split.name,
        note=split.note,
        test_n=len(split.test),
        base_explained=variance_explained_centred(base_predicted, actual),
        base_gate=precision_at_k(base_predicted, actual, GATE_K, GATE_THRESHOLD),
    )

    for label, detail, predictor in _layer_specs():
        predictor.fit(split.train)
        result.layers.append(
            _compare_both(
                label, detail, predictor.predict(split.test), base_predicted, actual, resamples
            )
        )

    everything = CrewModel(roles=(DIRECTOR, ACTOR, *BELOW_THE_LINE), expectation="rich")
    everything.fit(split.train)
    result.combined = _compare_both(
        "everything at once",
        "all context features and all person effects",
        everything.predict(split.test),
        base_predicted,
        actual,
        resamples,
    )
    return result


def decompose(
    films: list[RatedFilm],
    resamples: int = DECOMPOSITION_RESAMPLES,
    cut: date = TEMPORAL_SPLIT_DATE,
) -> list[Decomposition]:
    """Both splits, temporal first: it is the honest headline."""
    return [
        decompose_split(temporal_split(films, cut), resamples),
        decompose_split(random_split(films), resamples),
    ]
