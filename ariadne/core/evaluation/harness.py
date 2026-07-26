"""Running predictors through splits and recording what happened.

Built before the model on purpose (PHASES 1.6 before 1.7). With the harness in place first, every
model change is scored the moment it is made, against every baseline, on both splits — and there
is no room to decide by intuition whether something helped.

Every run is persisted to analysis_runs.metrics and never overwritten, so the writeup can show how
numbers moved rather than only where they ended up.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ariadne.constants import RANDOM_SEED
from ariadne.core.evaluation.baselines import Predictor, ladder
from ariadne.core.evaluation.dataset import RatedFilm
from ariadne.core.evaluation.metrics import DEFAULT_K, Metrics, score
from ariadne.core.evaluation.splits import Drift, Split, random_split, temporal_split

MODEL_VERSION = "1.6-baselines"


@dataclass
class PredictorResult:
    name: str
    description: str
    metrics: Metrics

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            **self.metrics.as_dict(),
        }


@dataclass
class SplitResult:
    split: str
    note: str
    dropped: int
    train_n: int
    test_n: int
    drift: Drift
    results: list[PredictorResult] = field(default_factory=list)

    def by_name(self, name: str) -> PredictorResult | None:
        return next((r for r in self.results if r.name == name), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "note": self.note,
            "dropped_no_log_date": self.dropped,
            "train_n": self.train_n,
            "test_n": self.test_n,
            "drift": self.drift.as_dict(),
            "predictors": [r.as_dict() for r in self.results],
        }


def evaluate_split(split: Split, predictors: list[Predictor], k: int = DEFAULT_K) -> SplitResult:
    actual = np.array([f.rating for f in split.test], dtype=float)
    result = SplitResult(
        split=split.name,
        note=split.note,
        dropped=split.dropped,
        train_n=len(split.train),
        test_n=len(split.test),
        drift=split.drift(),
    )

    for predictor in predictors:
        predictor.fit(split.train)
        predicted = predictor.predict(split.test)
        result.results.append(
            PredictorResult(
                name=predictor.name,
                description=predictor.description,
                metrics=score(predicted, actual, k=k),
            )
        )
    return result


def evaluate(
    films: list[RatedFilm],
    predictors: list[Predictor] | None = None,
    k: int = DEFAULT_K,
) -> list[SplitResult]:
    """Score every predictor on both splits. Temporal first: it is the honest headline."""
    chosen = ladder() if predictors is None else predictors
    return [
        evaluate_split(temporal_split(films), chosen, k=k),
        evaluate_split(random_split(films), chosen, k=k),
    ]


def run_payload(
    results: list[SplitResult], films: int, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The JSONB blob stored on analysis_runs."""
    payload: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "seed": RANDOM_SEED,
        "films": films,
        "splits": [r.as_dict() for r in results],
    }
    if extra:
        payload.update(extra)
    return payload
