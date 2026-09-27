"""Whole study in one command, for a machine without session limits (lab server, workstation):

    python scripts/run_all.py                         # everything: data -> P1 -> P2 -> P3 -> report
    python scripts/run_all.py --dry-run               # print the plan only
    python scripts/run_all.py --smoke                 # every training stage 1 epoch on 10% of NEU
    python scripts/run_all.py --only p1 report        # a subset of stages (see STAGES)
    python scripts/run_all.py --from p2_ablation      # resume from a stage (finished runs are skipped anyway)
    nohup python scripts/run_all.py > run_all.log 2>&1 &     # keep it running after logout (or use tmux)

Uses every visible GPU (one grid shard per GPU, scripts/run_session.py). Every stage is resumable: re-running the
command skips finished runs and continues interrupted ones from last.pt, so a crash or reboot costs little.

Phase-2/3 choices are taken as they are in the configs - no pause to look at results: A2/A3/A7 come from
configs/variants.yaml (default SimAM / NWD), the phase-2 data suffix from the p2 grids (default _bal_v1), and the
phase-3 improved source model T3 is A7 (configs/experiments/p3_*.yaml). Edit those files first if you already
decided otherwise.

Local runs need the raw data (scripts/download_data.py; Kaggle sources need ~/.kaggle/kaggle.json on a server).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable
SOURCES = ["neu", "gc10", "pcb", "mt"]
TARGET = "ksdd2"


def stages(hours: float, smoke: bool) -> list[tuple[str, list[list[str]]]]:
    def grid(name):
        cmd = [PY, str(SCRIPTS / "run_session.py"), name, "--hours", str(hours)]
        return cmd + (["--smoke"] if smoke else [])

    src = " ".join(SOURCES).split()
    return [
        ("download", [[PY, str(SCRIPTS / "download_data.py"), "--only", *src, TARGET]]),
        ("data", [[PY, str(SCRIPTS / "prepare_data.py"), "--datasets", *src, TARGET, "--n-vis", "12"]]),
        ("eda", [[PY, str(SCRIPTS / "eda.py"), "--datasets", *src]]),
        ("balance", [[PY, str(SCRIPTS / "prepare_data.py"), "--balance", *src, "--n-vis", "0"],
                     [PY, str(SCRIPTS / "prepare_data.py"), "--balance", "gc10", "--balance-method", "aug",
                      "--n-vis", "0"]]),
        ("p1", [grid("p1")]),
        ("p2_screen", [grid("p2_candidates")]),
        ("p2_tiling", [grid("p2_tiling")]),
        ("p2_ablation", [grid("p2_ablation"), grid("p2_ablation_rest")]),
        ("p3", [[PY, str(SCRIPTS / "prepare_data.py"), "--merge", *src, "--name", "merged", "--n-vis", "0"],
                grid("p3_source"), grid("p3_transfer")]),
        ("report", [[PY, str(SCRIPTS / "make_report.py")] + (["--include-smoke"] if smoke else [])]),
    ]


def main():
    names = [n for n, _ in stages(1, False)]
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=names, help="run just these stages")
    ap.add_argument("--from", dest="start", choices=names, help="start at this stage")
    ap.add_argument("--skip", nargs="*", default=[], choices=names)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hours", type=float, default=10_000.0, help="time budget per grid (default: unlimited)")
    ap.add_argument("--keep-going", action="store_true", help="continue with later stages when one fails")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = stages(a.hours, a.smoke)
    if a.start:
        plan = plan[names.index(a.start):]
    plan = [(n, c) for n, c in plan if (not a.only or n in a.only) and n not in a.skip]
    t0 = time.time()
    for name, cmds in plan:
        print(f"\n===== {name} ({(time.time() - t0) / 3600:.2f} h elapsed)", flush=True)
        for cmd in cmds:
            print("$", " ".join(cmd), flush=True)
            if a.dry_run:
                continue
            rc = subprocess.run(cmd).returncode
            if rc:
                print(f"!! stage {name} failed (exit {rc})", flush=True)
                if not a.keep_going:
                    raise SystemExit(f"stopped at stage '{name}'. Fix it and resume with: "
                                     f"python scripts/run_all.py --from {name}")
    print(f"\nall done in {(time.time() - t0) / 3600:.2f} h - tables in results/tables, figures in results/figures")


if __name__ == "__main__":
    main()
