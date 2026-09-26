"""Seed statistics (plan §3.2): mean ± std over seeds, Welch's t-test vs a reference configuration.

A variant is called "improved" only when p < 0.05 AND the mean gain exceeds the larger of the two stds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    agg = g[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = g.size()
    return agg.reset_index()


def welch(a, b) -> tuple[float, float]:
    """Welch's t-test (unequal variances). Returns (t, p); nan when either side has < 2 values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def compare_to_reference(df: pd.DataFrame, key: str, reference: str, metric: str,
                         within: list[str] = ("dataset",), alpha: float = 0.05) -> pd.DataFrame:
    """For every value of ``key`` (e.g. variant) compare ``metric`` against ``reference`` within each group."""
    rows = []
    for grp, sub in df.groupby(list(within)):
        grp = grp if isinstance(grp, tuple) else (grp,)
        ref = sub.loc[sub[key] == reference, metric].to_numpy(float)
        for val, s in sub.groupby(key):
            x = s[metric].to_numpy(float)
            t, p = welch(x, ref)
            delta = np.nanmean(x) - np.nanmean(ref) if len(ref) else float("nan")
            sd = max(np.nanstd(x, ddof=1) if len(x) > 1 else 0, np.nanstd(ref, ddof=1) if len(ref) > 1 else 0)
            rows.append({**dict(zip(within, grp)), key: val, "metric": metric, "mean": np.nanmean(x),
                         "std": np.nanstd(x, ddof=1) if len(x) > 1 else float("nan"), "n": len(x),
                         "delta_vs_ref": delta, "t": t, "p": p,
                         "improved": bool(val != reference and p < alpha and delta > sd)})
    return pd.DataFrame(rows)
