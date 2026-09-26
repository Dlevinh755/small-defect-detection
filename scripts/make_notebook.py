"""Generates notebooks/sdd_end_to_end.ipynb (run from the repo root: python scripts/make_notebook.py). Edit the
notebook here, not in the .ipynb, so the two never diverge."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append(("markdown", s.strip("\n")))


def code(s):
    cells.append(("code", s.strip("\n")))


md(r"""
# Small surface defect detection - end-to-end pipeline

One notebook for the whole study (plan: `docs/KE_HOACH_NGHIEN_CUU_LOI_BE_MAT_NHO.md`):
data -> EDA -> class-imbalance handling -> phase 1 baselines -> phase 2 screening / large images / ablation -> phase 3 transfer -> tables & figures.

**How to use on Kaggle**
1. Settings: **Accelerator = GPU T4 x2** (T4 x1 also works), **Internet = on**.
2. Raw data, **once**: set `RAW_BUNDLE = "<kaggle-username>/sdd-raw"`, add Kaggle Secrets `KAGGLE_USERNAME` and
   `KAGGLE_KEY` (Add-ons -> Secrets), keep only `RUN["download"]` on and run: all 5 datasets (~4 GB) are downloaded,
   checked and uploaded as your PRIVATE dataset `sdd-raw`. **Every later session**: Add Data -> `sdd-raw`, set
   `RAW_BUNDLE = ""` - the data is found under `/kaggle/input` and nothing is downloaded. From the 2nd training
   session on, also add the output of the previous version of this notebook (`PREV_RESULTS`).
3. Pick the stages in the **Config** cell, then **Save Version -> Save & Run All** (a background run of up to
   12 h; do not rely on an interactive session, it can stop when the browser is closed).

The full study does not fit in one 12 h session. Every run is resumable: finished runs are skipped, interrupted runs
continue from `last.pt`. Set `PREV_RESULTS` to the previous version's output and run again. The whole notebook
shares one time budget (`SESSION_HOURS`): at `SESSION_HOURS - REPORT_MARGIN_H` training is stopped (even a run in
the middle of an epoch - it resumes next session), the report runs, and the notebook ends before Kaggle's 12 h limit.
""")

md("## 1. Config")
code(r"""
import os, time
T0 = time.time()

# --- code & data ---------------------------------------------------------------------------------------
SDD_REPO       = "https://github.com/Dlevinh755/small-defect-detection.git"  # "" -> use SDD_CODE_INPUT
SDD_CODE_INPUT = "/kaggle/input/sdd-code"            # used when SDD_REPO is empty
PREV_RESULTS   = ""                                  # e.g. "/kaggle/input/sdd-e2e-v3/results" (resume)
RAW_BUNDLE     = ""                                  # ONE-TIME: "<kaggle-username>/sdd-raw" -> download + upload raw data

SOURCE_DATASETS = ["neu", "gc10", "pcb", "mt"]
TARGET_DATASET  = "ksdd2"

# --- stages (all resumable; re-running a finished stage only re-reads its results) ---------------------
SMOKE = False   # True: every model 1 epoch on 10% of NEU - pipeline check only (~15 min)
RUN = {
    "download":    True,   # fetch missing raw datasets (skipped when already attached / downloaded)
    "data":        True,   # convert datasets + draw labels (check them before training!)
    "eda":         True,
    "balance":     True,   # <ds>_bal_v1: class-aware augmentation + copy-paste, TRAIN only (plan §2.5.5), CPU
    "p1":          True,   # baselines + P2 (plan §4)
    "p2_screen":   False,  # single modules, 1 seed -> pick A2/A3 in configs/variants.yaml (plan §5.2)
    "p2_tiling":   False,  # PCB / GC10: SAHI on the P1 checkpoints + training on 640 tiles
    "p2_ablation": False,  # A0-A7 x 3 seeds (edit variants.yaml first!)
    "p3":          False,  # source models + transfer grid on the target set (plan §6)
    "report":      True,   # tables / figures / feature maps / error galleries
}
# Several Kaggle accounts in parallel: every account runs this notebook with the same stages and its own
# SESSION_SHARD ("0/2" on account A, "1/2" on account B, both T4 x2); the next session attaches ALL outputs:
# PREV_RESULTS = "/kaggle/input/<out-A>/results /kaggle/input/<out-B>/results" (space-separated).
SESSION_SHARD = "0/1"
SESSION_HOURS = 11.5       # Kaggle limit is 12 h. Training stops at SESSION_HOURS - REPORT_MARGIN_H (runs resume
REPORT_MARGIN_H = 0.5      # next session from last.pt), leaving time for the report and for Kaggle to save output
""")

md("## 2. Setup")
code(r"""
import shutil, subprocess
os.environ.update(SDD_PREV=PREV_RESULTS)

# 1) get the code: reuse an existing clone (also a manual `git clone` into /kaggle/working), else clone / copy it
REPO_DIR = next((d for d in ("/kaggle/working/sdd", "/kaggle/working/small-defect-detection")
                 if os.path.exists(f"{d}/scripts/kaggle_setup.sh")), "/kaggle/working/sdd")
if not os.path.exists(f"{REPO_DIR}/scripts/kaggle_setup.sh"):
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    if SDD_REPO:
        subprocess.run(["git", "clone", "-q", SDD_REPO, REPO_DIR], check=True)
    elif os.path.exists(f"{SDD_CODE_INPUT}/scripts/kaggle_setup.sh"):
        shutil.copytree(SDD_CODE_INPUT, REPO_DIR)
    else:
        raise FileNotFoundError("no code: set SDD_REPO or attach the repo as a dataset at SDD_CODE_INPUT")
elif os.path.exists(f"{REPO_DIR}/.git"):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "-q", "--ff-only"])  # latest code on every session
print("code:", REPO_DIR, subprocess.run(["git", "-C", REPO_DIR, "log", "-1", "--oneline"],
                                        capture_output=True, text=True).stdout.strip())

# 2) install + sync previous results
!bash {REPO_DIR}/scripts/kaggle_setup.sh
%cd {REPO_DIR}
""")
code(r'''
import subprocess, sys
from pathlib import Path
import pandas as pd
import torch
from IPython.display import Image, Markdown, display

R = Path("/kaggle/working/results")
LOGS = Path("/kaggle/working/logs"); LOGS.mkdir(exist_ok=True)
N_GPU = max(torch.cuda.device_count(), 1)
print(f"GPUs: {N_GPU}")


def hours_left():
    return SESSION_HOURS - (time.time() - T0) / 3600


def sh(cmd):
    print("$", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        print(f"!! exit code {r.returncode}")


def grid(name):
    """Run a grid on all GPUs until the session deadline (scripts/run_session.py). Runs still going at the deadline
    are stopped and resume from last.pt next session; the notebook itself always finishes, so the output is saved."""
    deadline = T0 + (SESSION_HOURS - REPORT_MARGIN_H) * 3600
    sh(f"python scripts/run_session.py {name} --deadline {deadline:.0f} --session-shard {SESSION_SHARD}"
       + (" --smoke" if SMOKE else ""))


def show(pattern, width=720):
    for p in sorted(R.glob(pattern)):
        display(Markdown(f"**{p.relative_to(R)}**"))
        display(Image(filename=str(p), width=width))


def table(name):
    p = R / "tables" / f"{name}.md"
    display(Markdown(p.read_text(encoding="utf-8") if p.exists() else f"_{name}: no results yet_"))
''')

md(r"""
## 3. Data
Downloads whatever is missing (Kaggle datasets via kagglehub, Magnetic Tile from GitHub, KolektorSDD2 from ViCoS),
checks each dataset (XML present, original resolution, boards, masks, counts) and writes `raw/MANIFEST.md`
(source, date, license, counts - copy it into `data/README.md`). Then converts every dataset to YOLO + COCO json with the fixed splits in `splits/`. **Look at the drawn labels before
trusting any training** (plan §8). If the splits were created in this session (first run ever), download
`/kaggle/working/sdd/splits/` and commit it, so every later session uses the same split.
""")
code(r"""
if RUN["download"]:
    extra = ""
    if RAW_BUNDLE:  # one-time: real files + private Kaggle dataset (needs the API key in Kaggle Secrets)
        try:
            from kaggle_secrets import UserSecretsClient
            sec = UserSecretsClient()
            os.environ["KAGGLE_USERNAME"] = sec.get_secret("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = sec.get_secret("KAGGLE_KEY")
            extra = f" --bundle {RAW_BUNDLE} --upload"
        except Exception as e:
            print(f"no Kaggle API secrets ({e}): bundling without upload - after Save Version use "
                  "Output -> New Dataset, or upload /kaggle/working/raw by hand")
            extra = f" --bundle {RAW_BUNDLE}"
    sh(f"python scripts/download_data.py --only {' '.join(SOURCE_DATASETS + [TARGET_DATASET])}{extra}")
    display(Markdown(Path("/kaggle/working/raw/MANIFEST.md").read_text(encoding="utf-8")))
if RUN["data"]:
    sh(f"python scripts/prepare_data.py --datasets {' '.join(SOURCE_DATASETS + [TARGET_DATASET])} --n-vis 12")
    show("figures/labels/*/train_*.jpg", width=320)
""")

md("## 4. EDA (plan §2.4)")
code(r"""
if RUN["eda"]:
    sh(f"python scripts/eda.py --datasets {' '.join(SOURCE_DATASETS)}")
    display(pd.read_csv(R / "eda/datasets_summary.csv").set_index("dataset").T)
    display(Markdown("**Class imbalance (plan table 2.5.2)**"))
    table("imbalance_2_5_2")
    for ds in SOURCE_DATASETS:
        show(f"eda/{ds}/*.png")
        display(Markdown(f"**{ds}**: test boxes per candidate relative-size bins - choose bins with >= ~50 boxes "
                         "per group, then set `eval.rel_bins` in `configs/protocol.yaml`"))
        display(pd.read_csv(R / f"eda/{ds}/rel_bin_candidates.csv"))
""")

md(r"""
## 5. Class-imbalance handling (plan §2.5.5)
Builds `<ds>_bal_v1` for every source dataset: class-aware augmentation (repeat factor, flips/rotations + photometric)
and copy-paste of rare defects, on the **train split only** (val/test untouched, <= +50% images, seed 42).
Datasets with `mode: "auto"` in `configs/datasets.yaml` stay unchanged when max/min < 3. The optional ablation
data `gc10_aug_v1` (augmentation only) is built too. CPU only - the same cell also works in a CPU session.

**QA before training** (plan checklist): boxes must sit on the defects in `qa_aug`, pasted patches must show no
seam in `qa_cp`, and val/test sizes must be unchanged.
""")
code(r"""
if RUN["balance"]:
    sh(f"python scripts/prepare_data.py --balance {' '.join(SOURCE_DATASETS)} --n-vis 0")
    sh("python scripts/prepare_data.py --balance gc10 --balance-method aug --n-vis 0")
    for ds in SOURCE_DATASETS:
        f = R / f"figures/balance/{ds}_bal_v1/balance_counts.csv"
        if f.exists():
            display(Markdown(f"**{ds}_bal_v1**")); display(pd.read_csv(f))
    show("figures/balance/*/balance_counts.png")
    show("figures/balance/*/qa_*.jpg", width=1000)
""")

md(r"""
## 6. Phase 1 - baselines + class balancing + small improvement (plan §4)
All models on `<ds>_bal_v1` (table A); YOLO11n on the original data with iteration-matched epochs as the control for
the balancing (table B); YOLO11n-P2 as the small model improvement (`configs/experiments/p1.yaml`).
Balanced datasets are also built automatically here if section 5 was skipped.
""")
code(r"""
if RUN["p1"]:
    grid("p1")
""")

md(r"""
## 7. Phase 2 (plan §5)
**7a. Screening** - each candidate module alone, 1 seed. With the phase-1 TIDE diagnosis (decision table §5.1),
use it to set A2/A3 in `configs/variants.yaml` before the ablation.
""")
code(r"""
if RUN["p2_screen"]:
    grid("p2_candidates")
""")
md(r"""
**7b. Large images (PCB, GC10)** - SAHI-style sliced inference on the phase-1 checkpoints (no training) and
training on 640 tiles; scored on the same full test images as phase 1.
""")
code(r"""
if RUN["p2_tiling"]:
    grid("p2_tiling")
""")
md("**7c. Ablation A0-A7** - A0, A1-A3, A7 with 3 seeds first; A4-A6 with 1 seed if time allows.")
code(r"""
if RUN["p2_ablation"]:
    grid("p2_ablation")
    grid("p2_ablation_rest")
""")

md(r"""
## 8. Phase 3 - transfer learning to the target set (plan §6)
Source models (T2 = base, T3 = improved) on the merged source data, then T1/T2/T3 x {10, 25, 50, 100}% x
{full, freeze} x 3 seeds on the target set.
""")
code(r"""
if RUN["p3"]:
    sh(f"python scripts/prepare_data.py --merge {' '.join(SOURCE_DATASETS)} --name merged --n-vis 0")
    grid("p3_source")
    grid("p3_transfer")
""")

md("## 9. Report - tables, figures, feature maps, error galleries")
code(r"""
if RUN["report"]:
    SM = " --include-smoke" if SMOKE else ""
    sh("python scripts/make_report.py" + SM)
    master = R / "master_results.csv"
    runs = pd.read_csv(master) if master.exists() and master.stat().st_size > 1 else pd.DataFrame()
    done = set(runs.run) if len(runs) else set()
    # error galleries for the phase-1 YOLO11n runs
    gal = sorted(r for r in done if r.removeprefix("smoke_").startswith("p1_")
                 and r.endswith("_bal_v1_yolo11n_base_s0"))
    if gal:
        sh("python scripts/make_report.py" + SM + " --gallery " + " ".join(gal))
    # feature maps: screened backbones if available, else phase-1 base vs P2
    for ds in SOURCE_DATASETS:
        pre = "smoke_" if SMOKE else ""
        fm = [r for r in (f"{pre}p2s_{ds}_yolo11n_{v}_s0" for v in ("base", "simam", "spd")) if r in done]
        fm = fm or [r for r in (f"{pre}p1_{ds}_bal_v1_yolo11n_base_s0", f"{pre}p1_{ds}_bal_v1_yolo11n_p2_s0")
                    if r in done]
        if fm:
            sh(f"python scripts/feature_maps.py --dataset {ds}_bal_v1 --n-images 2 --runs {' '.join(fm)}")
""")
code(r"""
if RUN["report"]:
    for t in ["imbalance_2_5_2", "p1_results", "p1_balance_effect", "p2s_results", "p2t_results", "p2_ablation",
              "p3_transfer"]:
        display(Markdown(f"### {t}"))
        table(t)
    for f in sorted((R / "tables/per_class").glob("p1_*.md")):  # per-class AP + test boxes (plan §2.5.3d)
        display(Markdown(f"### per-class AP: {f.stem}"))
        display(Markdown(f.read_text(encoding="utf-8")))
    show("figures/p1/*.png"); show("figures/p2*/*.png"); show("figures/p3/*.png")
    show("figures/feature_maps/*/compare_*.png", width=900)
    show("figures/errors/*/*.jpg", width=900)
""")
code(r"""
# slide material in one small zip (full runs incl. weights stay in /kaggle/working/results for the next session)
!cd /kaggle/working && zip -qr report_bundle.zip results/master_results.csv results/tables results/figures results/eda 2>/dev/null; ls -lh /kaggle/working/*.zip
print(f"session time used: {(time.time() - T0) / 3600:.2f} h")
""")

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": s} if t == "markdown" else
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/sdd_end_to_end.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(cells), "cells")
