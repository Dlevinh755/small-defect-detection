"""Experiment grids (``configs/experiments/*.yaml``) -> ordered list of runs; resumable execution.

Order: runs (as listed) -> datasets -> [p3: inits -> fractions -> finetune] -> seeds, so the plan's priority
(e.g. YOLO11n before RT-DETR) holds even if the session ends early. ``--shard i/n`` splits the list between
processes (e.g. one per T4 on a Kaggle "T4 x2" machine).
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from ..config import CONFIGS, load_yaml
from ..env import paths
from .run import RunSpec, run

log = logging.getLogger(__name__)


def grid_path(name: str) -> Path:
    p = Path(name)
    return p if p.suffix == ".yaml" and p.exists() else CONFIGS / "experiments" / f"{name}.yaml"


def expand(grid: dict, smoke: bool = False) -> list[RunSpec]:
    """Run entries may override ``datasets``; ``data_suffix`` is appended to every dataset id (e.g. "_bal_v1");
    ``match_epochs_to`` may contain "{ds}". Phase-3 grids (``inits``) add fractions x bg_ratios x finetune."""
    specs = []
    for r in grid["runs"]:
        r = dict(r)
        model = r.pop("model")
        for base in r.pop("datasets", grid["datasets"]):
            ds = base + r.get("data_suffix", grid.get("data_suffix", ""))
            common = dict(phase=grid["phase"], dataset=ds, model=model, smoke=smoke, epochs=r.get("epochs"),
                          batch=r.get("batch"), extra=r.get("extra", {}))
            if "inits" in grid:
                for init_name, init in grid["inits"].items():
                    for frac in grid.get("fractions", [1.0]):
                        for bg in grid.get("bg_ratios", [None]):
                            for ft in grid.get("finetune", ["full"]):
                                for seed in grid["seeds"]:
                                    specs.append(RunSpec(
                                        **common, variant=init.get("variant", r.get("variant", "base")), seed=seed,
                                        init=init_name, init_from=init.get("from", "coco"), fraction=frac,
                                        bg_ratio=bg, finetune=ft, freeze_layers=grid.get("freeze_layers", 11),
                                        freeze_epochs=grid.get("freeze_epochs", 10)))
            else:
                match = r.get("match_epochs_to")
                for seed in grid["seeds"]:
                    specs.append(RunSpec(**common, variant=r.get("variant", "base"), seed=seed,
                                         infer=r.get("infer", "full"), reuse_phase=r.get("reuse_phase"),
                                         match_epochs_to=match.format(ds=base) if match else None))
    return specs


def status(specs: list[RunSpec]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        rd = paths().run_dir(s.name)
        st = "done" if (rd / "done.json").exists() else "partial" if rd.exists() else "todo"
        out.append((s.name, st))
    return out


def run_grid(name: str, device: str = "0", shard: tuple[int, int] = (0, 1), smoke: bool = False,
             datasets: list[str] | None = None, models: list[str] | None = None, max_hours: float | None = None,
             stop_on_error: bool = False) -> dict:
    grid = load_yaml(grid_path(name))
    specs = expand(grid, smoke)
    if datasets:
        specs = [s for s in specs if s.dataset in datasets]
    if models:
        specs = [s for s in specs if s.model in models or s.variant in models]
    i, n = shard
    specs = [s for k, s in enumerate(specs) if k % n == i]
    t0, summary = time.time(), {"done": [], "failed": [], "skipped_time": []}
    for s in specs:
        if max_hours and (time.time() - t0) / 3600 > max_hours:
            summary["skipped_time"].append(s.name)
            continue
        log.info("=== %s", s.name)
        try:
            run(s, device=device)
            summary["done"].append(s.name)
        except Exception as e:
            log.error("run %s failed: %s\n%s", s.name, e, traceback.format_exc())
            summary["failed"].append(s.name)
            (paths().run_dir(s.name)).mkdir(parents=True, exist_ok=True)
            (paths().run_dir(s.name) / "error.txt").write_text(traceback.format_exc())
            if stop_on_error:
                raise
    return summary
