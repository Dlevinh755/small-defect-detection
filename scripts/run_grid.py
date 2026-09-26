"""Run (or inspect) an experiment grid from configs/experiments/. Finished runs are skipped, interrupted runs resume.

    python scripts/run_grid.py p1 --status
    python scripts/run_grid.py p1 --smoke --datasets neu            # pipeline check
    python scripts/run_grid.py p1 --device 0 --shard 0/2 &          # Kaggle T4 x2: one process per GPU
    python scripts/run_grid.py p1 --device 1 --shard 1/2
    python scripts/run_grid.py p1 --max-hours 11                    # stop starting runs near the session limit
"""

import argparse
import json

import _bootstrap  # noqa: F401

from sdd.config import load_yaml
from sdd.engine.grid import expand, grid_path, run_grid, status
from sdd.reporting.master import write_master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", help="grid name (p1, p2_ablation, ...) or path to a yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th run starting at i")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*", help="filter by model or variant name")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--status", action="store_true", help="list runs and their state, do not train")
    ap.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args()
    if a.status:
        specs = expand(load_yaml(grid_path(a.grid)), a.smoke)
        for name, st in status(specs):
            print(f"{st:8s} {name}")
        return
    i, n = map(int, a.shard.split("/"))
    summary = run_grid(a.grid, a.device, (i, n), a.smoke, a.datasets, a.models, a.max_hours, a.stop_on_error)
    print(json.dumps(summary, indent=2))
    print("master table:", write_master(include_smoke=a.smoke))


if __name__ == "__main__":
    main()
