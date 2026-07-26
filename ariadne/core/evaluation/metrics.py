"""Scoring a set of predictions.

Precision@20 is the primary metric, not MAE. 16.5% of the reference account's ratings are exactly
5.0 and 71.9% are whole stars, so effective resolution is about five levels with an unordered
222-film mass at the top. A model can win on MAE by predicting 3.5 forever and be useless
(F4/D11). The product ranks, so a ranking metric measures the actual job.

Precision@20 is always reported beside the base rate — the share of the test set that clears the
threshold anyway. Without it the number is uninterpretable: 0.70 is excellent if 30% of films are
liked and worthless if 70% are.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

# Two configurations, for two different jobs.
#
# The product recommends about twenty films, so P@20 at >=4.0 is what a user experiences. But as
# a *measurement* it is nearly useless here: the director-only baseline already reaches 0.950 on
# the temporal split, leaving one film of headroom, and each film moves the number by 0.05. A
# metric that cannot resolve an improvement cannot adjudicate the go/no-go.
PRODUCT_K = 20
PRODUCT_THRESHOLD = 4.0

# The gate metric. At k=100 and >=4.5 the director-only baseline sits at 0.690 — 31 films of
# headroom, and each film moves the number by 0.01. Base rate is 0.300, so the number is also
# interpretable. Chosen from the measured grid *before the crew model existed*, which is the only
# point at which choosing it is honest (F39/D68).
GATE_K = 100
GATE_THRESHOLD = 4.5

LIKED_THRESHOLD = PRODUCT_THRESHOLD
DEFAULT_K = PRODUCT_K
CALIBRATION_BINS = 5

# Reported in full so the choice above is inspectable rather than asserted.
GRID_KS = (10, 20, 50, 100, 200)
GRID_THRESHOLDS = (4.0, 4.5, 5.0)


@dataclass(frozen=True)
class Metrics:
    n: int
    precision_at_k: float
    base_rate: float
    lift: float
    spearman: float
    mae: float
    rmse: float
    mae_centred: float
    gate_precision: float
    gate_base_rate: float
    gate_lift: float
    calibration: tuple[tuple[float, float, int], ...]
    k: int = DEFAULT_K

    def as_dict(self) -> dict[str, object]:
        # Every float passes through _finite: a single NaN anywhere in this payload makes the
        # whole run unwritable, and losing a run to a formatting detail is not acceptable.
        return {
            "n": self.n,
            "k": self.k,
            "precision_at_k": round(_finite(self.precision_at_k), 4),
            "base_rate": round(_finite(self.base_rate), 4),
            "lift": round(_finite(self.lift), 4),
            "gate_k": GATE_K,
            "gate_threshold": GATE_THRESHOLD,
            "gate_precision": round(_finite(self.gate_precision), 4),
            "gate_base_rate": round(_finite(self.gate_base_rate), 4),
            "gate_lift": round(_finite(self.gate_lift), 4),
            "spearman": round(_finite(self.spearman), 4),
            "mae": round(_finite(self.mae), 4),
            "rmse": round(_finite(self.rmse), 4),
            "mae_centred": round(_finite(self.mae_centred), 4),
            "calibration": [
                {"predicted": round(_finite(p), 3), "actual": round(_finite(a), 3), "n": n}
                for p, a, n in self.calibration
            ],
        }


def precision_at_k(
    predicted: np.ndarray,
    actual: np.ndarray,
    k: int = DEFAULT_K,
    threshold: float = LIKED_THRESHOLD,
) -> float:
    if len(predicted) == 0:
        return 0.0
    top = np.argsort(-predicted)[: min(k, len(predicted))]
    return float((actual[top] >= threshold).mean())


def _calibration(
    predicted: np.ndarray, actual: np.ndarray, bins: int = CALIBRATION_BINS
) -> tuple[tuple[float, float, int], ...]:
    """Mean actual rating per bin of predicted rating.

    A model can be accurate on average and badly overconfident at the top end, which is exactly
    where recommendations are drawn from.
    """
    if len(predicted) < bins:
        return ()
    order = np.argsort(predicted)
    chunks = np.array_split(order, bins)
    return tuple(
        (float(predicted[chunk].mean()), float(actual[chunk].mean()), int(len(chunk)))
        for chunk in chunks
        if len(chunk)
    )


def _is_constant(values: np.ndarray) -> bool:
    return bool(values.min() == values.max())


def _finite(value: float) -> float:
    """Zero rather than NaN or infinity: metrics are persisted as JSON, which has neither."""
    return value if np.isfinite(value) else 0.0


def score(predicted: np.ndarray, actual: np.ndarray, k: int = DEFAULT_K) -> Metrics:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must be the same length")
    if len(predicted) == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (), k)

    errors = predicted - actual
    base_rate = float((actual >= PRODUCT_THRESHOLD).mean())
    p_at_k = precision_at_k(predicted, actual, k)

    gate_base = float((actual >= GATE_THRESHOLD).mean())
    gate_p = precision_at_k(predicted, actual, GATE_K, GATE_THRESHOLD)

    # Spearman is undefined when either side is constant, which the global-mean baseline is by
    # design. Tested with min == max rather than std() == 0: the standard deviation of 100
    # identical floats comes back as 1.3e-15, so an exact comparison to zero never fires and the
    # NaN reaches the database, where JSON has no way to represent it.
    constant = _is_constant(predicted) or _is_constant(actual)
    rho = 0.0 if constant else float(spearmanr(predicted, actual).statistic)
    if not np.isfinite(rho):
        rho = 0.0

    # Removing each side's mean shows how much error is the block-level shift the temporal split
    # introduces (F3) rather than genuine misranking.
    centred = (predicted - predicted.mean()) - (actual - actual.mean())

    return Metrics(
        n=len(actual),
        precision_at_k=p_at_k,
        base_rate=base_rate,
        lift=p_at_k - base_rate,
        gate_precision=gate_p,
        gate_base_rate=gate_base,
        gate_lift=gate_p - gate_base,
        spearman=rho,
        mae=float(np.abs(errors).mean()),
        rmse=float(np.sqrt((errors**2).mean())),
        mae_centred=float(np.abs(centred).mean()),
        calibration=_calibration(predicted, actual),
        k=k,
    )


def precision_grid(predicted: np.ndarray, actual: np.ndarray) -> dict[float, dict[int, float]]:
    """P@k across thresholds and k, so the chosen configuration can be checked."""
    return {
        threshold: {k: precision_at_k(predicted, actual, k, threshold) for k in GRID_KS}
        for threshold in GRID_THRESHOLDS
    }
