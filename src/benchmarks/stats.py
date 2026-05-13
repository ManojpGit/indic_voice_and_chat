"""Statistical analysis helpers.

One-way ANOVA across provider groups (pure numpy / stdlib — no scipy).
Useful for deciding whether observed WER / latency differences between
providers are statistically meaningful or noise.

Reports F-statistic + a coarse p-value bucket (``"<0.001"`` / ``"<0.05"`` /
``">=0.05"``) by comparing F to a static table for typical k=2..5,
n=10..200. Not a substitute for scipy.stats.f.sf, but adequate for
"is this difference real?" gating and keeps the project dep-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence


@dataclass
class ANOVAResult:
    f_statistic: float
    df_between: int
    df_within: int
    significance: str       # "<0.001" | "<0.05" | ">=0.05"
    group_means: list[float]
    grand_mean: float


# Static critical-F table for one-way ANOVA (α=0.05 row, α=0.001 row).
# Indexed by (df_between, df_within) — common cases. We bin to the nearest
# tabulated cell.
_F_TABLE_05 = {
    (1, 10): 4.96, (1, 30): 4.17, (1, 60): 4.00, (1, 120): 3.92,
    (2, 10): 4.10, (2, 30): 3.32, (2, 60): 3.15, (2, 120): 3.07,
    (3, 10): 3.71, (3, 30): 2.92, (3, 60): 2.76, (3, 120): 2.68,
    (4, 10): 3.48, (4, 30): 2.69, (4, 60): 2.53, (4, 120): 2.45,
}
_F_TABLE_001 = {
    (1, 10): 21.0, (1, 30): 13.3, (1, 60): 12.0, (1, 120): 11.4,
    (2, 10): 14.9, (2, 30): 8.77, (2, 60): 7.77, (2, 120): 7.32,
    (3, 10): 12.6, (3, 30): 7.05, (3, 60): 6.17, (3, 120): 5.78,
    (4, 10): 11.3, (4, 30): 6.12, (4, 60): 5.31, (4, 120): 4.95,
}


def _nearest_critical(table: dict[tuple[int, int], float], df_b: int, df_w: int) -> float:
    if df_b <= 0 or df_w <= 0:
        return float("inf")
    # Clamp df_b to range [1, 4]; df_w to nearest tabulated cell.
    b = max(1, min(4, df_b))
    for cell in (10, 30, 60, 120):
        if df_w <= cell:
            return table[(b, cell)]
    return table[(b, 120)]


def one_way_anova(groups: Sequence[Sequence[float]]) -> ANOVAResult:
    """One-way ANOVA across ``groups`` (one provider per group).

    Returns the F statistic + a significance bucket. Raises if fewer than
    two groups or any group is empty.
    """
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")
    cleaned = [list(g) for g in groups]
    if any(len(g) == 0 for g in cleaned):
        raise ValueError("ANOVA groups must all be non-empty")

    group_means = [mean(g) for g in cleaned]
    sizes = [len(g) for g in cleaned]
    n = sum(sizes)
    k = len(cleaned)
    grand = sum(sum(g) for g in cleaned) / n

    # Between-groups sum of squares
    ss_between = sum(size * (m - grand) ** 2 for m, size in zip(group_means, sizes))
    # Within-groups sum of squares
    ss_within = sum(
        sum((x - m) ** 2 for x in g)
        for g, m in zip(cleaned, group_means)
    )

    df_between = k - 1
    df_within = n - k
    if df_within <= 0:
        # Pathological: only 1 sample per group. F undefined.
        return ANOVAResult(
            f_statistic=float("inf"),
            df_between=df_between,
            df_within=df_within,
            significance=">=0.05",
            group_means=group_means,
            grand_mean=grand,
        )

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f = ms_between / ms_within if ms_within > 0 else float("inf")

    crit_05 = _nearest_critical(_F_TABLE_05, df_between, df_within)
    crit_001 = _nearest_critical(_F_TABLE_001, df_between, df_within)
    if f >= crit_001:
        sig = "<0.001"
    elif f >= crit_05:
        sig = "<0.05"
    else:
        sig = ">=0.05"

    return ANOVAResult(
        f_statistic=f,
        df_between=df_between,
        df_within=df_within,
        significance=sig,
        group_means=group_means,
        grand_mean=grand,
    )
