"""Result tables for the three reports (plan §4.5, §5.5, §6.3). Metrics shown in % with one decimal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.stats import compare_to_reference, mean_std

PCT = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AP75_s", "AP_rel_small", "AP75_rel_small", "R_rel_small",
       "R_small", "P", "R", "img_detection_rate", "false_alarm_rate", "img_AP"]


def _na(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def model_label(model: str, variant: str, infer: str | None = None, train_tile=None, data=None,
                matched=None) -> str:
    """e.g. "yolo11n-p2", "yolo11n [orig, iter-matched]", "yolo11n (tiles 640) +SAHI".
    ``data`` is shown only when it differs from the phase's main data version (set to None by the caller)."""
    s = model if _na(variant) or variant == "base" else f"{model}-{variant}"
    if not _na(data):
        s += f" [{data}{', iter-matched' if not _na(matched) and matched else ''}]"
    if not _na(train_tile):
        s += f" (tiles {int(train_tile)})"
    if infer == "sliced":
        s += " +SAHI"
    return s


def main_data_version(df: pd.DataFrame) -> str | None:
    """Most common data version of a set of runs (e.g. 'bal_v1' in phase 1)."""
    return df["data_version"].mode().iloc[0] if "data_version" in df and df["data_version"].notna().any() else None


def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Model`` and group derived datasets (gc10_bal_v1, pcb_t640) under the dataset they are evaluated on.
    Runs on a data version other than the main one of their phase are tagged, e.g. "[orig, iter-matched]"."""
    d = df.copy()
    get = lambda c: d[c] if c in d else pd.Series([None] * len(d), index=d.index)  # noqa: E731
    main = {ph: main_data_version(g) for ph, g in d.groupby("phase")} if "phase" in d else {}
    data = [None if _na(v) or v == main.get(ph) else v for v, ph in zip(get("data_version"), get("phase"))]
    d["Model"] = [model_label(m, v, i, t, dv, mt) for m, v, i, t, dv, mt in
                  zip(d.model, d.variant, get("infer"), get("train_tile"), data, get("match_epochs_to"))]
    if "eval_dataset" in d:
        d["dataset"] = d["eval_dataset"].fillna(d["dataset"])
    return d


def _fmt(v, pct: bool) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v * 100:.1f}" if pct else (f"{v:.1f}" if isinstance(v, float) else str(v))


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(to_markdown(df), encoding="utf-8")
    (out / f"{name}.tex").write_text(df.to_latex(index=False, escape=True), encoding="utf-8")


def table_phase1(df: pd.DataFrame, phase: str = "p1", main_data_only: bool = False) -> pd.DataFrame:
    """Plan §4.5 table A. ``main_data_only`` drops control runs (e.g. original-data runs in a _bal_v1 phase)."""
    d = df[df.phase == phase]
    if main_data_only and main_data_version(d):
        d = d[d.data_version == main_data_version(d)]
    d = with_labels(d)
    metrics = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AP_rel_small", "R_rel_small", "params_M", "GFLOPs",
               "fps_fp16", "e2e_sliced_total_ms"]  # sliced runs: FPS is per 640 forward, see the ms/img column
    metrics = [m for m in metrics if m in d]
    g = d.groupby(["dataset", "Model"], sort=False)[metrics].mean().reset_index()
    for m in metrics:
        g[m] = [_fmt(v, m in PCT) for v in g[m]]
    return g.rename(columns={"dataset": "Dataset"})


def _pm(mean, std, pct=True) -> str:
    if np.isnan(mean):
        return "-"
    s = f"{mean * 100:.1f}" if pct else f"{mean:.2f}"
    if not np.isnan(std):
        s += f" ± {std * 100:.1f}" if pct else f" ± {std:.2f}"
    return s


def table_ablation(df: pd.DataFrame, phase: str = "p2", reference: str = "A0", test_metric: str = "AP_s") -> pd.DataFrame:
    d = df[df.phase == phase]
    metrics = [m for m in ["AP", "AP_s", "AP75_s", "AP_rel_small", "R_rel_small"] if m in d]
    ms = mean_std(d, ["dataset", "variant"], metrics + [c for c in ["params_M", "GFLOPs", "fps_fp16"] if c in d])
    cmp = compare_to_reference(d, "variant", reference, test_metric)
    out = ms[["dataset", "variant", "n_seeds"]].copy()
    for m in metrics:
        out[m] = [_pm(a, b) for a, b in zip(ms[f"{m}_mean"], ms[f"{m}_std"])]
    for m in ["params_M", "GFLOPs", "fps_fp16"]:
        if f"{m}_mean" in ms:
            out[m] = [f"{v:.2f}" if m != "fps_fp16" else f"{v:.0f}" for v in ms[f"{m}_mean"]]
    out = out.merge(cmp[["dataset", "variant", "delta_vs_ref", "p", "improved"]], on=["dataset", "variant"], how="left")
    out[f"Δ{test_metric}"] = [_fmt(v, True) for v in out.pop("delta_vs_ref")]
    out[f"p vs {reference}"] = [f"{p:.3f}" if not np.isnan(p) else "-" for p in out.pop("p")]
    return out


def table_transfer(df: pd.DataFrame, phase: str = "p3") -> pd.DataFrame:
    d = df[df.phase == phase]
    metrics = [m for m in ["AP", "AP_s", "AP_rel_small", "img_detection_rate", "false_alarm_rate", "img_AP"] if m in d]
    ms = mean_std(d, ["dataset", "init", "finetune", "fraction"], metrics)
    out = ms[["dataset", "init", "finetune", "fraction", "n_seeds"]].copy()
    for m in metrics:
        out[m] = [_pm(a, b) for a, b in zip(ms[f"{m}_mean"], ms[f"{m}_std"])]
    return out.sort_values(["dataset", "finetune", "fraction", "init"])


def table_balance(df: pd.DataFrame, phase: str = "p1", model: str = "yolo11n") -> pd.DataFrame:
    """Plan §4.5 table B: effect of the imbalance handling for the base model (original vs _aug vs _bal)."""
    d = df[(df.phase == phase) & (df.model == model) & (df.variant == "base")]
    if "data_version" not in d or d.data_version.nunique() < 2:
        return pd.DataFrame()
    d = with_labels(d)
    d = d[d.dataset.isin(d[d.data_version == "orig"].dataset)]  # datasets that have a control run
    rows = []
    for (ds, dv), g in d.groupby(["dataset", "data_version"], sort=True):
        r = g.iloc[0]
        rare = [c for c in str(r.get("rare_classes", "")).split(";") if c]
        per_rare = ", ".join(f"{c} {_fmt(g[f'AP_cls/{c}'].mean(), True)}" for c in rare if f"AP_cls/{c}" in g)
        rows.append({
            "Dataset": ds,
            "Data": dv + (" (iter-matched)" if not _na(r.get("match_epochs_to")) else ""),
            "train images": int(r.get("n_train_images", 0) or 0),
            **{m: _fmt(g[m].mean(), True) for m in ("AP", "AP_rare", "AP_common", "R_rel_small") if m in g},
            "AP of rare classes": per_rare or "-",
        })
    return pd.DataFrame(rows)


def table_per_class(df: pd.DataFrame, phase: str, dataset: str) -> pd.DataFrame:
    """Per-class AP of every model with the number of test boxes (plan §2.5.3d); rare / unstable classes flagged."""
    d = with_labels(df[df.phase == phase])
    d = d[d.dataset == dataset]
    if d.empty:
        return pd.DataFrame()
    classes = [c.split("/", 1)[1] for c in d.columns if c.startswith("n_test_cls/")]
    r0 = d.iloc[0]
    rare = set(str(r0.get("rare_classes", "")).split(";"))
    unstable = set(str(r0.get("unstable_classes", "")).split(";"))
    out = pd.DataFrame({"class": classes,
                        "test boxes": [int(r0[f"n_test_cls/{c}"]) for c in classes],
                        "flag": ["rare" * (c in rare) + (", " if c in rare and c in unstable else "")
                                 + "unstable" * (c in unstable) for c in classes]})
    for model, g in d.groupby("Model", sort=False):
        out[model] = [_fmt(g[f"AP_cls/{c}"].mean(), True) if f"AP_cls/{c}" in g else "-" for c in classes]
    return out
