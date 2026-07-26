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
    """The shrinkage constant: within-group variance over between-group variance."""
    counts = np.array([len(v) for v in group_values.values()], dtype=float)
    means = np.array([float(np.mean(v)) for v in group_values.values()], dtype=float)

    flat = np.array([x for values in group_values.values() for x in values], dtype=float)
    if len(flat) < 2 or len(means) < 2:
        return float("inf")

    within = float(flat.var(ddof=1))

    # Observed spread of group means overstates the truth by roughly within/n, because each mean
    # carries its own sampling noise. Subtracting that leaves the genuine spread.
    observed = float(means.var(ddof=1))
    mean_n = float(counts.mean())
    between = max(observed - within / mean_n, MIN_BETWEEN_VARIANCE)

    return within / between


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
