# Small Surface Defect Detection

Experiment code for the research plan in [docs/KE_HOACH_NGHIEN_CUU_LOI_BE_MAT_NHO.md](docs/KE_HOACH_NGHIEN_CUU_LOI_BE_MAT_NHO.md):
baselines on industrial surface-defect datasets, small-object improvements (P2 head, attention / SPD-Conv
backbone, NWD / Wise-IoU / Inner-IoU losses) and transfer learning to a small real-world target set.
Built to run on **Kaggle T4** (single or T4 x2), resumable across 12-hour sessions.

| Phase | What | Grid(s) | `RUN` flag |
|---|---|---|---|
| data | class-imbalance handling: `<ds>_bal_v1` = class-aware augmentation + copy-paste (train only) | - | `balance` |
| 1 (Thường kỳ) | YOLO11n, Faster R-CNN R50-FPN v2, RT-DETR-l + YOLO11n-P2 on `<ds>_bal_v1`; YOLO11n on original data (iteration-matched control) | `p1` | `p1` |
| 2 (Giữa kỳ) | module screening, then ablation A0-A7 x 3 seeds, Welch t-test | `p2_candidates`, `p2_ablation`, `p2_ablation_rest` | `p2_screen`, `p2_ablation` |
| 2, large images | SAHI-style sliced inference on P1 checkpoints; training on 640 tiles (PCB, GC10) | `p2_tiling` | `p2_tiling` |
| 3 (Cuối kỳ) | source models on merged data, then T1/T2/T3 x 10-100% x full/freeze on KolektorSDD2 | `p3_source`, `p3_transfer` | `p3` |

## Layout

```
configs/
  paths.yaml            raw dataset roots (Kaggle / local profiles), work & results dirs
  datasets.yaml         dataset registry: reader, classes, class-name mapping, split ratios, mask settings
  protocol.yaml         shared protocol: imgsz, epochs/batch per family, seeds, eval thresholds, size bins
  variants.yaml         model variants (P2, SimAM, CBAM, CA, SPD, NWD, WIoU, Inner-IoU, ablation A0-A7)
  models/               custom Ultralytics yamls (yolo11n-p2, yolo11n-p2p4)
  experiments/          run grids per phase
splits/<ds>/            fixed train/val/test id lists (seed 42) - created once, commit them
src/sdd/
  data/                 raw readers -> splits -> YOLO + COCO layout; balance (imbalance), tiling, merge (phase 3),
                        subsample (fractions, defect-free ratio), resolve (derived dataset ids), label drawing
  eda/                  EDA tables and figures
  models/               attention / SPD modules, backbone surgery, losses, custom Ultralytics trainer, Faster R-CNN
  engine/               one run end to end (run.py), prediction export, grid runner
  evaluation/           COCO AP (abs + relative size), operating-point P/R + recall by size, TIDE,
                        params/GFLOPs/FPS, image-level metrics, seed statistics
  reporting/            master_results.csv, tables (csv/md/tex), figures, error gallery
scripts/                CLIs: prepare_data, eda, train, run_grid, make_report, feature_maps, sync_results,
                        kaggle_setup.sh
notebooks/              sdd_end_to_end.ipynb: the single Kaggle notebook (data -> EDA -> P1 -> P2 -> P3 -> report)
tests/                  CPU tests on synthetic data
```

## Data download

`python scripts/download_data.py` fetches and checks all five datasets into one folder (`data/raw` locally,
`/kaggle/working/raw` on Kaggle); sources, licenses and expected counts are in `configs/sources.yaml`:

| id | dataset | source | notes |
|---|---|---|---|
| neu | NEU-DET | Kaggle `kaustubhdikshit/neu-surface-defect-database` | needs ANNOTATIONS/*.xml |
| gc10 | GC10-DET | Kaggle `alex000kim/gc10det` | only ~2300 of ~3570 images have XML; the rest is skipped |
| pcb | PKU-Market-PCB | Kaggle `akhatova/pcb-defects` | original high-res boards (not the Roboflow 3x export); board ids recorded |
| mt | Magnetic Tile | GitHub `abin24/Magnetic-tile-defect-datasets.` (zip) | masks -> boxes; defect images with an empty mask are skipped |
| ksdd2 | KolektorSDD2 | data.vicos.si `KolektorSDD2.zip` (~850 MB) | CC BY-NC-SA 4.0, target set |

Datasets already present (attached input or earlier download) and passing the checks are skipped; `--check` only
checks; the result table goes to `<raw>/MANIFEST.md`.

**Download once, reuse as a Kaggle dataset** (recommended):
1. One-time session: in the notebook set `RAW_BUNDLE = "<kaggle-username>/sdd-raw"`, add Kaggle Secrets
   `KAGGLE_USERNAME` / `KAGGLE_KEY` (Add-ons -> Secrets; key from kaggle.com -> Settings -> API), keep only
   `RUN["download"]` on, run. This is `download_data.py --bundle <id> --upload`: everything is downloaded as real
   files (no symlinks), checked, and uploaded as a PRIVATE dataset (a new version if it already exists) together
   with `MANIFEST.md`. A failed dataset stops the upload. Without API secrets the bundle is still prepared;
   after Save Version use Output -> "New Dataset".
2. Every later session: Add Data -> `sdd-raw`, `RAW_BUNDLE = ""`. `configs/paths.yaml` finds each dataset with
   globs such as `/kaggle/input/**/NEU-DET`, whatever the mount layout (`/kaggle/input/<slug>/` or
   `/kaggle/input/datasets/<owner>/<slug>/`), so the download step only re-checks and skips.

## Quickstart (local)

```bash
python -m venv .venv && .venv/Scripts/pip install -e .[dev]      # Linux/Mac: .venv/bin/pip
pytest -q                                                         # synthetic-data tests, CPU only
python scripts/download_data.py --only neu                        # or all; see "Data download"
python scripts/prepare_data.py --datasets neu
python scripts/eda.py --datasets neu
python scripts/train.py --dataset neu --model yolo11n --variant p2 --smoke --device cpu
```

## On Kaggle

1. Code: https://github.com/Dlevinh755/small-defect-detection (already set as `SDD_REPO` in the notebook; a
   private repo needs a token in the URL, or upload the repo folder as a Kaggle dataset `sdd-code` instead).
2. Import `notebooks/sdd_end_to_end.ipynb`; settings: GPU **T4 x2** (x1 works), Internet **on**; add the raw
   datasets and edit their slugs in `configs/paths.yaml` (or set `SDD_RAW_<DS>`).
3. In the **Config** cell choose the stages (`RUN = {...}`), optionally `SMOKE = True` for a ~15 min pipeline
   check first, then **Save Version -> Save & Run All**. Setup clones the repo (or reuses an existing clone and
   pulls), then runs `scripts/kaggle_setup.sh` (
   install, sync previous results); each grid runs one process per GPU (`--shard i/n`), and all stages share one
   session budget (`SESSION_HOURS`), so no new run starts when time is nearly up.
4. The study needs several sessions: add the previous version's output as an input, set
   `PREV_RESULTS=/kaggle/input/<that-output>/results` and run again - finished runs (`done.json`) are skipped,
   unfinished ones resume from `last.pt`. Enable the phase-2/3 flags once the previous phase is done
   (and after editing A2/A3 in `configs/variants.yaml`).

## On a server (no session limit)

```bash
git clone https://github.com/Dlevinh755/small-defect-detection.git && cd small-defect-detection
pip install -r requirements.txt && pip install -e . --no-deps
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/            # for the three Kaggle-hosted datasets
nohup python scripts/run_all.py > run_all.log 2>&1 &       # data -> P1 -> P2 -> P3 -> report, all GPUs
tail -f run_all.log                                        # progress; per-GPU logs in results/logs/
```

`run_all.py` runs the stages in order (`--dry-run` prints them, `--only` / `--from` / `--skip` select, `--smoke`
checks the pipeline first). If it stops, run the same command again: finished runs are skipped, interrupted ones
resume from `last.pt`. It does not pause for the phase-2/3 decisions: A2/A3 come from `configs/variants.yaml`
(default SimAM / NWD), T3 = A7 - edit those first if you decided otherwise.

## Runs and results

Run name: `<phase>_<dataset>_<model>_<variant>[_<init>_f<pct>_<finetune>][_bg<r>][_itm][_sahi]_s<seed>`, e.g.
`p1_gc10_bal_v1_yolo11n_p2_s0`, `p1_gc10_yolo11n_base_itm_s0`,
`p3_ksdd2_yolo11n_A7_T3_f025_freeze_s1`. Each run directory (`results/runs/<name>/`) holds `config.yaml`
(spec, resolved variant, protocol, library versions), the framework training output, `test_predictions.json`,
`metrics.json` and `done.json`. `scripts/make_report.py` rebuilds `results/master_results.csv` (one row per run)
and writes `results/tables/*` and `results/figures/*`.

Main metrics (test split, once, checkpoint chosen on val):

- COCO `AP, AP50, AP75, AP_s/m/l, AR*`, plus `AP50_s`, `AP75_s` and per-class AP
- relative-size AP / AR (`AP_rel_small`, ...; box area / image area, bins in `protocol.yaml -> eval.rel_bins`)
- at the operating point (`op_conf`, IoU 0.5): `P, R, F1`, recall per relative and absolute size group
- TIDE dAP50 per error type (`TIDE_Cls, Loc, Both, Dupe, Bkg, Miss`)
- `params_M`, `GFLOPs` (torch flop counter, all frameworks), batch-1 latency / FPS in FP32 and FP16
- image level (phase 3): detection rate, false-alarm rate on defect-free images, image AUROC / AP

## Implementation notes

- **Class imbalance (plan §2.5).** Splits are stratified by the rarest class in each image (optional `group_regex`
  for board-level splits). `<ds>_bal_v1` is built on demand from `<ds>`: images with rare classes get
  `round(max r_c) - 1` extra copies (`r_c = min(cap, max(1, sqrt(t / f_c)))`), each with a different flip / rotation
  + photometric change, then rare defects are copy-pasted (native size, feathered background ring, brightness
  matched, no overlap) until they reach 1/3 of the largest class; at most +50% train images, seed 42, val/test are
  the source dataset's files. `_aug_v1` (augmentation only) and `_rfs_v1` (identical repeats) exist for ablation.
  Knobs: `protocol.yaml -> balance`, per dataset `datasets.yaml -> balance` (mode `"auto"` balances only when
  max/min >= 3). `prepare_data.py --balance` writes before/after counts and QA grids to `figures/balance/`.
  Every run reports `AP_rare` / `AP_common` (rare = < 1/3 of the largest class in the original train split),
  per-class AP with test-box counts (`unstable_classes` < 10 test boxes); the phase-1 control run uses the original
  data with epochs scaled by `|train_bal| / |train|` (`match_epochs_to`, run name `..._itm_...`). Loss-level
  options for phase 2: variants `clsw` (Ultralytics `cls_pw`, 1/sqrt(freq)) and `focal`. Phase 3: `bg_ratios`
  (defect-free : defective train images) in `p3_transfer.yaml`.

- **Ultralytics is not forked.** `sdd.models.yolo_trainer.make_trainer(variant)` returns a `DetectionTrainer`
  subclass whose `get_model` builds the yaml, loads COCO weights, then edits the built network
  (`sdd.models.surgery`: attention after backbone C3k2 layers, SPD-Conv replacing strided convs). Layer indices
  never change, so `freeze=`, head routing and weight transfer keep working. The box loss / assigner are swapped
  in `SDDDetectionModel.init_criterion` (`sdd.models.losses`). The code is pinned to `ultralytics==8.4.163`;
  after an upgrade run `pytest tests/test_models.py`.
- NWD is computed in input pixels (Ultralytics' loss works in grid units; the stride is applied back), so
  `nwd_C` in `variants.yaml` is a pixel constant.
- Predictions are exported by our own code (not Ultralytics `save_json`) with the image / category ids of
  `coco/test.json`, for all three frameworks.
- YOLO / RT-DETR use Ultralytics' default recipe; Faster R-CNN uses the torchvision reference recipe - "framework
  defaults, no per-model tuning" (plan §3.1). Record any epoch cut (e.g. RT-DETR) with `epochs:` in the grid.
- **Large images.** A dataset id `<ds>_t<size>` (e.g. `pcb_t640`) is built on demand from `<ds>`: train/val are
  cut into overlapping tiles (boxes kept if >= `min_visibility` inside, a share of empty tiles kept), while the
  test split stays the original full images. A run with `infer: sliced` predicts on tiles + the whole image and
  merges with class-wise NMS; `reuse_phase: p1` re-scores an existing checkpoint that way without training.
  Tables group `pcb_t640` rows under `pcb`, labelled `(tiles 640)` / `+SAHI`; the FPS column is per 640 forward,
  `e2e_sliced_total_ms` is the real per-image cost of sliced inference.
- **Feature maps.** `scripts/feature_maps.py --runs <run> <run> ... --dataset <ds>` puts the same test images
  (by default those with the smallest defects) through several runs and plots the channel-mean activation of
  backbone layers 2/4/6/8 (stride 4-32) with the GT boxes; `--channels-layer` adds the per-channel grid,
  `--pre-attention` reads wrapped layers before their attention module.
- The freeze fine-tune mode is two Ultralytics runs (frozen backbone for `freeze_epochs`, then all layers from
  that checkpoint); the LR schedule restarts in the second stage.
