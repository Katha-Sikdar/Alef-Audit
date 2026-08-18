"""Aggregate metric builders matching the paper's summary tables:
Table 4/5/7/8 (TC%, AC%, Delta_AC-TC, raw p, adjusted q) and Table 6/Figure 4
(contamination severity distribution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from alef.auditors import ComplianceResult
from alef.contamination import ContaminationResult, ContaminationSeverity
from alef.stats.significance import benjamini_hochberg, cohens_h, two_proportion_z_test


@dataclass
class GroupSummary:
    group: str
    n: int
    tc_count: int
    ac_count: int
    tc_rate: float
    ac_rate: float
    delta_ac_tc: float
    raw_p: float | None = None
    adjusted_q: float | None = None
    cohens_h: float | None = None


def compute_group_summary(group: str, results: Iterable[ComplianceResult]) -> GroupSummary:
    results = list(results)
    n = len(results)
    tc_count = sum(1 for r in results if r.tc)
    ac_count = sum(1 for r in results if r.ac)
    tc_rate = tc_count / n if n else 0.0
    ac_rate = ac_count / n if n else 0.0

    if n > 0:
        test = two_proportion_z_test(ac_count, n, tc_count, n)
        raw_p = test.p_value
        h = cohens_h(ac_rate, tc_rate)
    else:
        raw_p = None
        h = None

    return GroupSummary(
        group=group,
        n=n,
        tc_count=tc_count,
        ac_count=ac_count,
        tc_rate=tc_rate,
        ac_rate=ac_rate,
        delta_ac_tc=ac_rate - tc_rate,
        raw_p=raw_p,
        cohens_h=h,
    )


def build_comparison_table(
    grouped_results: Mapping[str, Iterable[ComplianceResult]],
    alpha: float = 0.05,
) -> list[GroupSummary]:
    """Build a Table-4/7/8-style summary across groups (e.g. keyed by task
    type, page representation, or capability tier x representation), with
    Benjamini-Hochberg correction applied jointly across every group's
    TC-vs-AC test -- matching Section 7.7's stated m across "Tables 4 and 8."
    """
    summaries = [compute_group_summary(group, results) for group, results in grouped_results.items()]

    p_values = [s.raw_p for s in summaries if s.raw_p is not None]
    if p_values:
        q_values = benjamini_hochberg(p_values, alpha=alpha)
        q_iter = iter(q_values)
        for s in summaries:
            if s.raw_p is not None:
                s.adjusted_q = next(q_iter)

    return summaries


def severity_distribution(results: Iterable[ContaminationResult]) -> dict[str, float]:
    """Table 6 / Figure 4: percentage of trajectories at each contamination
    severity level, plus the paper's aggregate "context contamination rate"
    (Section 8.3: Severity Level 2 + Level 3, excluding benign Level 1)."""
    results = list(results)
    n = len(results)
    if n == 0:
        return {}

    counts = {level: 0 for level in ContaminationSeverity}
    for r in results:
        counts[r.severity] += 1

    distribution = {level.description: counts[level] / n * 100 for level in ContaminationSeverity}
    severe = counts[ContaminationSeverity.FACTUAL] + counts[ContaminationSeverity.DELAYED_ACTION]
    distribution["Aggregate context contamination (Level 2 + 3)"] = severe / n * 100
    return distribution
