"""Result figures for the slides. Every figure uses :mod:`sdd.reporting.style` (one palette, entity-stable colours)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import style
from .tables import with_labels

TIDE_MAIN = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    return with_labels(df)


def _too_many(models) -> bool:
    """More series than distinguishable colours -> switch to a single-hue ranked form (never invent hues)."""
    return len(models) > len(style.CATEGORICAL)


def metric_ranked(df: pd.DataFrame, metric: str, title: str | None = None, reference: str = "yolo11n"):
    """One panel per dataset, one horizontal bar per model sorted by the metric (single hue); the reference model
    (e.g. the phase-1 YOLO11n baseline) is drawn in the second colour and as a dashed line."""
    d = _labelled(df)
    datasets = list(dict.fromkeys(d.dataset))
    n = d.Model.nunique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 0.32 * n + 1.3), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = d[d.dataset == ds].groupby("Model")[metric].agg(["mean", "std"]).sort_values("mean")
        colors = [style.CATEGORICAL[1] if m == reference else style.CATEGORICAL[0] for m in g.index]
        ax.barh(g.index, g["mean"] * 100, color=colors, height=0.62,
                xerr=g["std"].fillna(0) * 100 if g["std"].notna().any() else None,
                error_kw={"lw": 1, "capsize": 2, "ecolor": style.INK_2})
        for y, v in enumerate(g["mean"] * 100):
            ax.text(v, y, f" {v:.1f}", va="center", fontsize=8, color=style.INK_2)
        if reference in g.index:
            ax.axvline(g.loc[reference, "mean"] * 100, color=style.MUTED, lw=1, ls="--")
        ax.set_title(ds.upper(), fontsize=10)
        ax.set_xlabel(f"{metric} (%)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(1, (g["mean"] * 100).max() * 1.15))
    fig.suptitle(title or f"{metric} by model" + f"  (orange = {reference})", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def metric_by_model(df: pd.DataFrame, metric: str = "AP_s", title: str | None = None):
    """Grouped bars: datasets on x, one bar per model (mean over seeds, error bar = std when > 1 seed).
    With more models than colours it falls back to :func:`metric_ranked`."""
    d = _labelled(df)
    models = list(dict.fromkeys(d.Model))
    if _too_many(models):
        return metric_ranked(df, metric, (title or f"{metric} by model") + "  (orange = yolo11n baseline)")
    cmap = style.color_map(models)
    g = d.groupby(["dataset", "Model"], sort=False)[metric].agg(["mean", "std"]).reset_index()
    datasets = list(dict.fromkeys(d.dataset))
    x = np.arange(len(datasets))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(datasets) + 2), 3.6))
    for k, m in enumerate(models):
        s = g[g.Model == m].set_index("dataset").reindex(datasets)
        ax.bar(x + (k - (len(models) - 1) / 2) * w, s["mean"] * 100, w * 0.92, color=cmap[m], label=m,
               yerr=s["std"].fillna(0) * 100 if s["std"].notna().any() else None, error_kw={"lw": 1, "capsize": 2,
                                                                                            "ecolor": style.INK_2})
    ax.set_xticks(x, [ds.upper() for ds in datasets])
    ax.set_ylabel(f"{metric} (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(len(models), 4), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(title or f"{metric} by model", pad=28)
    return fig


def tide_by_model(df: pd.DataFrame, dataset: str):
    """TIDE main-error dAP50 per model (small multiples: one panel per error type, shared x)."""
    d = _labelled(df)
    d = d[d.dataset == dataset]
    cols = [f"TIDE_{e}" for e in TIDE_MAIN if f"TIDE_{e}" in d]
    if not cols:
        return None
    g = d.groupby("Model", sort=False)[cols].mean() * 100
    models = list(g.index)
    cmap = {m: style.CATEGORICAL[0] for m in models} if _too_many(models) else style.color_map(models)
    fig, axes = plt.subplots(1, len(cols), figsize=(2.1 * len(cols), 0.4 * len(models) + 1.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.barh(models[::-1], g[c].values[::-1], color=[cmap[m] for m in models[::-1]], height=0.6)
        ax.set_title(c.replace("TIDE_", ""), fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(1, g[cols].values.max() * 1.1))
    fig.suptitle(f"{dataset.upper()}: TIDE dAP50 by error type (higher = more AP lost)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def transfer_curves(df: pd.DataFrame, metric: str = "AP_s"):
    """Metric vs % target data, one line per init (T1/T2/T3), band = ±1 std over seeds, panel per finetune mode."""
    d = df[df.phase == "p3"]
    inits = sorted(d.init.dropna().unique())
    cmap = style.color_map(inits)
    modes = list(dict.fromkeys(d.finetune))
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 3.6), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for init in inits:
            s = d[(d.init == init) & (d.finetune == mode)].groupby("fraction")[metric].agg(["mean", "std"])
            if s.empty:
                continue
            xs = s.index.values * 100
            ax.plot(xs, s["mean"] * 100, marker="o", color=cmap[init], label=init)
            ax.fill_between(xs, (s["mean"] - s["std"].fillna(0)) * 100, (s["mean"] + s["std"].fillna(0)) * 100,
                            color=cmap[init], alpha=0.15, lw=0)
            ax.annotate(init, (xs[-1], s["mean"].iloc[-1] * 100), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=style.INK_2)
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
        ax.minorticks_off()
        ax.set_xlabel("target training data")
        ax.set_title(f"fine-tune: {mode}")
    axes[0][0].set_ylabel(f"{metric} (%)")
    axes[0][-1].legend(loc="lower right")
    fig.suptitle(f"Transfer to target: {metric} vs data amount", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def cost_vs_accuracy(df: pd.DataFrame, metric: str = "AP_s", cost: str = "GFLOPs", dataset: str | None = None):
    """Accuracy/cost trade-off: one dot per model (mean over seeds), direct labels."""
    d = _labelled(df)
    d = d if dataset is None else d[d.dataset == dataset]
    g = d.groupby("Model", sort=False)[[metric, cost]].mean()
    cmap = ({m: style.CATEGORICAL[0] for m in g.index} if _too_many(g.index)  # identity via the direct labels
            else style.color_map(list(g.index)))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m, r in g.iterrows():
        ax.scatter(r[cost], r[metric] * 100, s=60, color=cmap[m], edgecolor=style.SURFACE, linewidth=2, zorder=3)
        ax.annotate(m, (r[cost], r[metric] * 100), xytext=(6, 3), textcoords="offset points", fontsize=9,
                    color=style.INK_2)
    ax.set_xlabel(cost)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"{(dataset or 'all datasets').upper()}: {metric} vs {cost}")
    return fig
