"""EDA, tables, figures and error gallery on synthetic data / synthetic metrics (no model is run)."""

import json

import numpy as np
import pandas as pd

from sdd.data.build import build_dataset, load_coco_json
from sdd.eda.analysis import run_eda
from sdd.env import paths
from sdd.reporting import plots
from sdd.reporting.error_gallery import gallery
from sdd.reporting.style import save
from sdd.reporting.tables import save_table, table_ablation, table_phase1, table_transfer


def test_eda_outputs(raw_data):
    d = build_dataset("neu")
    out = paths().results / "eda" / "neu"
    s = run_eda(d, out, "neu")
    assert s["images"] == 30 and s["boxes"] > 30
    for f in ("boxes.csv", "summary.json", "rel_bin_candidates.csv", "area_hist.png", "class_distribution.png",
              "rel_size_by_class.png"):
        assert (out / f).exists(), f
    assert list((out / "smallest").glob("*.png"))


def _fake_master():
    rng = np.random.default_rng(0)
    rows = []
    for ph, variants, seeds in (("p1", ["base", "p2"], [0]), ("p2", ["A0", "A1", "A7"], [0, 1, 2])):
        for ds in ("neu", "gc10"):
            for model in (["yolo11n", "frcnn"] if ph == "p1" else ["yolo11n"]):
                for v in (variants if model == "yolo11n" else ["base"]):
                    for s in seeds:
                        ap_s = 0.2 + 0.05 * variants.index(v) + rng.normal(0, 0.005)
                        rows.append({"phase": ph, "dataset": ds, "model": model, "variant": v, "seed": s,
                                     "AP": ap_s + 0.2, "AP50": 0.7, "AP75": 0.4, "AP_s": ap_s, "AP_m": 0.4,
                                     "AP_l": 0.5, "AP75_s": ap_s / 2, "AP_rel_small": ap_s, "R_rel_small": 0.5,
                                     "params_M": 2.6, "GFLOPs": 6.4, "fps_fp16": 300.0, "TIDE_Miss": 0.1,
                                     "TIDE_Loc": 0.05, "TIDE_Cls": 0.02})
    for init, gain in (("T1", 0.0), ("T2", 0.03), ("T3", 0.05)):
        for ft in ("full", "freeze"):
            for frac in (0.1, 0.25, 0.5, 1.0):
                for s in (0, 1, 2):
                    rows.append({"phase": "p3", "dataset": "ksdd2", "model": "yolo11n", "variant": "base",
                                 "init": init, "finetune": ft, "fraction": frac, "seed": s,
                                 "AP": 0.3 + gain + 0.1 * frac, "AP_s": 0.2 + gain + 0.1 * frac,
                                 "img_detection_rate": 0.8, "false_alarm_rate": 0.1, "img_AP": 0.9})
    return pd.DataFrame(rows)


def test_tables_and_figures(tmp_path):
    df = _fake_master()
    t1 = table_phase1(df)
    assert set(t1.Model) == {"yolo11n", "yolo11n-p2", "frcnn"}
    ab = table_ablation(df)
    assert ab.set_index(["dataset", "variant"]).loc[("neu", "A7"), "improved"]
    tr = table_transfer(df)
    assert len(tr) == 3 * 2 * 4
    save_table(ab, tmp_path, "ab")
    assert (tmp_path / "ab.md").read_text(encoding="utf-8").startswith("| dataset")
    save(plots.metric_by_model(df[df.phase == "p1"], "AP_s"), tmp_path / "a.png")
    save(plots.tide_by_model(df[df.phase == "p1"], "neu"), tmp_path / "b.png")
    save(plots.transfer_curves(df, "AP_s"), tmp_path / "c.png")
    save(plots.cost_vs_accuracy(df[df.phase == "p1"], dataset="neu"), tmp_path / "d.png")
    assert all((tmp_path / f"{x}.png").stat().st_size > 1000 for x in "abcd")


def test_error_gallery_categories(raw_data, tmp_path):
    d = build_dataset("neu")
    gt = load_coco_json(d / "coco" / "test.json")
    dets = []
    for k, a in enumerate(gt["annotations"]):
        x, y, w, h = a["bbox"]
        if k % 3 == 0:
            continue                                            # missed
        if k % 3 == 1:
            dets.append({"image_id": a["image_id"], "category_id": (a["category_id"] + 1) % 6,
                         "bbox": [x, y, w, h], "score": 0.9})   # misclassified
        else:
            dets.append({"image_id": a["image_id"], "category_id": a["category_id"],
                         "bbox": [x + w * 0.6, y, w, h], "score": 0.9})  # mislocalised (IoU = 0.25)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "test_predictions.json").write_text(json.dumps(dets))
    counts = gallery(run_dir, d, tmp_path / "gal")
    assert sum(counts.values()) > 0
    assert set(counts) == {"missed", "mislocalised", "misclassified"}


def test_many_models_use_ranked_single_hue_chart(tmp_path):
    rng = np.random.default_rng(1)
    variants = ["p2p4", "simam", "cbam", "ca", "spd", "nwd", "nwd_assign", "wiou", "inner", "clsw", "focal"]
    rows = [{"phase": "p2s", "dataset": f"{ds}_bal_v1", "eval_dataset": ds, "model": "yolo11n", "variant": v,
             "AP_s": 0.2 + rng.uniform(0, 0.1), "GFLOPs": 6 + rng.uniform(0, 3), "TIDE_Miss": 0.1, "TIDE_Loc": 0.05}
            for ds in ("neu", "gc10") for v in ["base"] + variants]
    df = pd.DataFrame(rows)
    save(plots.metric_by_model(df, "AP_s", "P2S: AP_s by model"), tmp_path / "ranked.png")
    save(plots.tide_by_model(df, "gc10"), tmp_path / "tide.png")
    save(plots.cost_vs_accuracy(df, dataset="gc10"), tmp_path / "cost.png")
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("ranked.png", "tide.png", "cost.png"))
