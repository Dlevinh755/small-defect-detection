"""Run an experiment grid on every GPU of the machine, within a hard session deadline.

    python scripts/run_session.py p1 --deadline 1760000000      # unix time at which everything must be stopped
    python scripts/run_session.py p1 --hours 10.5 --smoke
    python scripts/run_session.py p1 --hours 11 --session-shard 1/2   # this machine does the 2nd half of the grid

One ``run_grid.py`` process per GPU. ``--session-shard i/m`` splits a grid between m machines / Kaggle accounts
(machine i takes every m-th slice): with G GPUs each, GPU g runs shard ``i*G + g`` of ``m*G``. Every machine must
use the same code version (same grid order) and the same GPU count. New runs are not started past the deadline, and runs still
going at the deadline are STOPPED (process group SIGTERM, then SIGKILL): Ultralytics / Faster R-CNN keep
``last.pt`` from the last finished epoch, so the next session resumes them. Stopping ourselves - instead of being
killed by Kaggle at 12 h - lets the notebook finish normally, so the report runs and the output is saved.
Logs: <logs>/<grid>_gpu<g>.log.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def stop(procs, grace: float = 60.0) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, AttributeError):
                p.terminate()
    t0 = time.time()
    while any(p.poll() is None for p in procs) and time.time() - t0 < grace:
        time.sleep(2)
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, AttributeError):
                p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--deadline", type=float, help="unix time: stop everything at this moment")
    g.add_argument("--hours", type=float, help="deadline = now + hours")
    ap.add_argument("--session-shard", default="0/1", help="i/m: this machine's part when m machines share the grid")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--logs", default=os.environ.get("SDD_LOGS"), help="default: <results>/../logs on Kaggle "
                    "(/kaggle/working/logs), else <results>/logs")
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args()

    deadline = a.deadline if a.deadline else time.time() + a.hours * 3600
    left_h = (deadline - time.time()) / 3600
    if left_h <= 0.1:
        print(f"skip {a.grid}: no time left in this session")
        return
    n_gpu = gpu_count()
    n = max(n_gpu, 1)
    from sdd.env import is_kaggle, paths

    logs = Path(a.logs) if a.logs else (Path("/kaggle/working/logs") if is_kaggle() else paths().results / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("run_grid.py")
    si, sm = map(int, a.session_shard.split("/"))
    procs = []
    for i in range(n):
        cmd = [sys.executable, str(script), a.grid, "--device", str(i) if n_gpu else "cpu",
               "--shard", f"{si * n + i}/{sm * n}",
               "--max-hours", f"{left_h:.3f}"]
        if a.smoke:
            cmd += ["--smoke", "--datasets", *(a.datasets or ["neu"])]
        elif a.datasets:
            cmd += ["--datasets", *a.datasets]
        log = open(logs / f"{a.grid}_gpu{i}.log", "a")
        print("$", " ".join(cmd), ">", log.name, flush=True)
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    stopped = False
    while any(p.poll() is None for p in procs):
        if time.time() >= deadline:
            print(f"deadline reached: stopping {a.grid} (interrupted runs resume from last.pt next session)", flush=True)
            stop(procs)
            stopped = True
            break
        time.sleep(a.poll)
    for i in range(n):
        tail = (logs / f"{a.grid}_gpu{i}.log").read_text(errors="replace").splitlines()[-12:]
        print(f"--- {a.grid} gpu{i} (last lines)\n" + "\n".join(tail))
    subprocess.run([sys.executable, str(script), a.grid, "--status"] + (["--smoke"] if a.smoke else []))
    if stopped:
        print("session deadline hit - re-run the notebook with PREV_RESULTS = this version's output")


if __name__ == "__main__":
    main()
