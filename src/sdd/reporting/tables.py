"""Result tables for the three reports (plan §4.5, §5.5, §6.3). Metrics shown in % with one decimal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.stats import compare_to_reference, mean_std

PCT = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AP75_s", "AP_rel_small", "AP75_rel_small", "R_rel_small",
       "R_small", "P", "R", "img_detection_rate", "false_alarm_rate", "img_AP"]


def model_label(model: str, variant: str, infer: str | None = None, train_tile=None) -> str:
    s = model if variant in ("base", None) or pd.isna(variant) else f"{model}-{variant}"
    if train_tile is not None and not pd.isna(train_tile):
        s += f" (tiles {int(train_tile)})"
    if infer == "sliced":
        s += " +SAHI"
    return s


def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Model`` and group tiled datasets (pcb_t640) under the dataset they are evaluated on (pcb)."""
    d = df.copy()
    get = lambda c: d[c] if c in d else pd.Series([None] * len(d), index=d.index)  # noqa: E731
    d["Model"] = [model_label(m, v, i, t) for m, v, i, t in zip(d.model, d.variant, get("infer"), get("train_tile"))]
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


def table_phase1(df: pd.DataFrame, phase: str = "p1") -> pd.DataFrame:
    d = with_labels(df[df.phase == phase])
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
