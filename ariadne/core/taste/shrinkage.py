"""Empirical Bayes shrinkage for per-person effects.

With 1,345 films the modal crew member appears once, and a person with two films rated 5.0 would
otherwise outrank one with forty films rated 4.2. Shrinkage pulls each group's mean toward zero
in proportion to how little evidence supports it.

The strength is estimated from the data rather than chosen. k is the ratio of within-group noise
to genuine between-group spread, so a population whose people really do differ gets shrunk
gently, and one whose apparent differences are noise gets shrunk hard.

Hand-written rather than taken from a library (D7): it is a dozen lines, and a hierarchical
modelling framework would be over-engineering until Stage 2 needs cross-user pooling.
"""

from dataclasses import dataclass

import numpy as np

# Floor on the estimated between-group variance. Without it, a population with no real spread
# produces k = infinity and every effect collapses to exactly zero, which is correct but
# numerically awkward downstream.
MIN_BETWEEN_VARIANCE = 1e-9


@dataclass(frozen=True)
class ShrunkEffect:
    key: int
    raw_mean: float
    shrunk: float
    n: int
    stderr: float


def estimate_k(group_values: dict[int, list[float]]) -> float:
    """The shrinkage constant: within-group variance over between-group variance.

    Uses the one-way random-effects (ANOVA) moment estimator. An earlier version corrected the
    observed spread of group means by within/mean_n, which is wrong when group sizes are as skewed
    as they are here: roughly 700 of 782 editors appear on a single film, so mean_n was about 2,
    the noise correction was far too small, real between-person spread was overstated, and k came
    out near 3 — leaving 68% of each raw mean unshrunk. Shuffled ratings then produced effects
    almost as large as real ones.

    MSW is within-group mean square, MSB between-group, and n0 the effective group size that
    accounts for the imbalance.
    """
    counts = np.array([len(v) for v in group_values.values()], dtype=float)
    means = np.array([float(np.mean(v)) for v in group_values.values()], dtype=float)
    flat = np.array([x for values in group_values.values() for x in values], dtype=float)

    n_total = float(counts.sum())
    n_groups = len(counts)
    if n_groups < 2 or n_total <= n_groups:
        return float("inf")

    grand_mean = float(flat.mean())
    ss_within = float(
        sum(((np.array(v, dtype=float) - np.mean(v)) ** 2).sum() for v in group_values.values())
    )
    ms_within = ss_within / (n_total - n_groups)

    ss_between = float((counts * (means - grand_mean) ** 2).sum())
    ms_between = ss_between / (n_groups - 1)

    # Effective group size under imbalance. With every group the same size this reduces to n.
    n0 = (n_total - float((counts**2).sum()) / n_total) / (n_groups - 1)
    if n0 <= 0:
        return float("inf")

    between = max((ms_between - ms_within) / n0, MIN_BETWEEN_VARIANCE)
    if ms_within <= 0:
        return 0.0
    return ms_within / between


def shrink(group_values: dict[int, list[float]], k: float | None = None) -> dict[int, ShrunkEffect]:
    """Shrink each group's mean toward zero by n / (n + k)."""
    if not group_values:
        return {}

    constant = estimate_k(group_values) if k is None else k
    flat = np.array([x for values in group_values.values() for x in values], dtype=float)
    within = float(flat.var(ddof=1)) if len(flat) > 1 else 0.0

    effects: dict[int, ShrunkEffect] = {}
    for key, values in group_values.items():
        n = len(values)
        raw = float(np.mean(values))
        weight = 0.0 if constant == float("inf") else n / (n + constant)
        stderr = float(np.sqrt(within / n)) if n and within else 0.0
        effects[key] = ShrunkEffect(key=key, raw_mean=raw, shrunk=raw * weight, n=n, stderr=stderr)
    return effects
