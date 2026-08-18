"""Statistical methodology matching Section 7.7:

- Two-sided, two-sample proportion z-tests with pooled sample variance,
  comparing Textual Compliance (TC) and Action Compliance (AC) rates.
- Benjamini-Hochberg FDR correction across the family of simultaneous tests.
- Cohen's h effect size for differences between two proportions:
      h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ZTestResult:
    p1: float
    p2: float
    n1: int
    n2: int
    z: float
    p_value: float


def two_proportion_z_test(count1: int, n1: int, count2: int, n2: int) -> ZTestResult:
    """Two-sided, two-sample proportion z-test with pooled variance.

    `count1`/`n1` and `count2`/`n2` are (successes, trials) for each group,
    e.g. (# action-compliant trajectories, # trials) vs (# text-compliant
    trajectories, # trials).
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive")

    p1 = count1 / n1
    p2 = count2 / n2
    pooled = (count1 + count2) / (n1 + n2)

    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = (p1 - p2) / se
        p_value = 2 * (1 - _standard_normal_cdf(abs(z)))

    return ZTestResult(p1=p1, p2=p2, n1=n1, n2=n2, z=z, p_value=p_value)


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for a difference between two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns BH-adjusted q-values in the
    same order as the input `p_values` (not sorted), matching how Tables 4
    and 8 report a q-value alongside each row's raw p-value.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    q_sorted = [0.0] * m

    prev_q = 1.0
    for rank in range(m - 1, -1, -1):
        original_index, p = indexed[rank]
        candidate_q = p * m / (rank + 1)
        prev_q = min(prev_q, candidate_q)
        q_sorted[rank] = min(prev_q, 1.0)

    q_values = [0.0] * m
    for rank, (original_index, _p) in enumerate(indexed):
        q_values[original_index] = q_sorted[rank]

    return q_values
