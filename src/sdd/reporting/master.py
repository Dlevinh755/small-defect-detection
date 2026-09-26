"""``master_results.csv``: one row per finished run, rebuilt from every ``runs/*/metrics.json``.

Rebuilding (instead of appending) keeps the table correct when several processes / sessions write runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..env import paths

FRONT = ["run", "phase", "dataset", "model", "variant", "seed", "init", "fraction", "finetune", "smoke",
         "AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AP75_s", "AP_rel_small", "AP75_rel_small",
         "R_rel_small", "R_rel_medium", "R_rel_large", "P", "R", "params_M", "GFLOPs", "fps_fp16",
         "train_hours"]


def collect(results: Path | None = None) -> pd.DataFrame:
    results = results or paths().results
    rows = []
    for f in sorted((results / "runs").glob("*/metrics.json")):
        if (f.parent / "done.json").exists():
            rows.append(json.loads(f.read_text()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = [c for c in FRONT if c in df] + sorted(c for c in df if c not in FRONT)
    return df[cols]


def write_master(results: Path | None = None, include_smoke: bool = False) -> Path:
    results = results or paths().results
    df = collect(results)
    if not df.empty and not include_smoke and "smoke" in df:
        df = df[~df["smoke"].astype(bool)]
    out = results / "master_results.csv"
    df.to_csv(out, index=False)
    return out


def load_master(results: Path | None = None) -> pd.DataFrame:
    p = (results or paths().results) / "master_results.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
