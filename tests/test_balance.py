"""Class-imbalance handling (plan §2.5): transforms, copy-paste, balanced datasets, splits, metrics, grids.
No model is trained."""

import json

import numpy as np
import pandas as pd
import pytest

from sdd.config import load_yaml
from sdd.data.balance import balance_dataset, geo_transform, imbalance_summary, paste, parse_balanced
from sdd.data.build import build_dataset, samples_from_coco
from sdd.data.readers import Sample
from sdd.data.resolve import base_dataset, data_version, ensure_dataset
from sdd.data.splits import group_split, rarest_class_keys
from sdd.data.subsample import subsample_train
from sdd.engine.grid import expand, grid_path
from sdd.engine.run import RunSpec, _train_params
from sdd.evaluation.imbalance import imbalance_metrics
from sdd.reporting.balance_report import balance_report
from sdd.reporting.tables import table_balance, with_labels


@pytest.mark.parametrize("op", ["hflip", "vflip", "rot90", "rot180"])
def test_geo_transform_boxes_follow_pixels(op):
    img = np.zeros((60, 100, 3), np.uint8)  # non-square on purpose
    box = np.array([[10, 5, 30, 15]], float)
    img[5:15, 10:30] = 255
    out, nb = geo_transform(img, box, op)
    x1, y1, x2, y2 = nb[0].astype(int)
    assert out[y1:y2, x1:x2].min() == 255                       # the box covers the white block...
    assert out.sum() == img.sum() and (out > 0).sum() == (x2 - x1) * (y2 - y1) * 3  # ...and nothing else


def test_paste_keeps_defect_and_feathers_background():
    target = np.full((80, 80), 100, np.uint8)
    patch = np.full((20, 20), 50, np.uint8)
    patch[6:14, 6:14] = 250                                      # defect inside a 6 px darker ring
    out = paste(target, patch, (6, 6, 14, 14), 30, 30, margin=6)
    assert out[36:44, 36:44].min() > 240                         # defect kept (ring brightness shifted +50)
    assert abs(int(out[30, 30]) - 100) <= 2                      # outer ring corner blends into the target
    assert (out[:30] == 100).all()                               # nothing outside the patch changed


def test_rarest_class_key_and_group_split():
    s = [Sample(f"i{k}", None, 10, 10, np.zeros((len(c), 4)), np.array(c, int)) for k, c in
         enumerate([[0], [0], [0], [0, 1], [1], []])]
    keys = rarest_class_keys(s)
    assert keys["i3"] == "1" and keys["i0"] == "0" and keys["i5"] == "clean"
    groups = {x.uid: g for x, g in zip(s, "aabbcc")}
    sp = group_split(s, groups, {"train": 0.66, "val": 0.17, "test": 0.17}, 0)
    where = {u: k for k, v in sp.items() for u in v}
    assert all(where[a] == where[b] for a, b in [("i0", "i1"), ("i2", "i3"), ("i4", "i5")])


def test_balanced_dataset(raw_data):
    src = build_dataset("gc10")
    d = ensure_dataset("gc10_bal_v1")
    meta = json.loads((d / "meta.json").read_text())
    n0 = meta["train_images_original"]
    assert meta["applied"] and 0 < meta["split_sizes"]["train"] - n0 <= 0.5 * n0 + 1
    assert set(meta["generated"]) <= {"aug", "cp"} and meta["generated"].get("aug") and meta["generated"].get("cp")
    for split in ("val", "test"):  # val / test untouched
        assert (d / "coco" / f"{split}.json").read_text() == (src / "coco" / f"{split}.json").read_text()
    b, a = meta["before"], meta["after"]
    for c in b["rare_classes"]:
        assert a["images_per_class"][c] > b["images_per_class"][c]
    for s in samples_from_coco(d, "train"):
        assert (s.boxes[:, 0] >= 0).all() and (s.boxes[:, 2] <= s.width + 1e-6).all()
        assert (s.boxes[:, 1] >= 0).all() and (s.boxes[:, 3] <= s.height + 1e-6).all()
    gen = (d / "generated.csv").read_text()
    balance_dataset("gc10", "bal", 1, force=True)                  # deterministic rebuild
    assert (d / "generated.csv").read_text() == gen
    rep = balance_report(d, d.parent / "_qa")
    assert (d.parent / "_qa" / "balance_counts.png").exists() and set(rep["qa"]) == {"aug", "cp"}


def test_aug_only_rfs_and_auto_noop(raw_data):
    aug = json.loads((ensure_dataset("gc10_aug_v1") / "meta.json").read_text())
    assert set(aug["generated"]) == {"aug"}
    rfs_dir = ensure_dataset("gc10_rfs_v1")
    rfs = [s for s in samples_from_coco(rfs_dir, "train") if "__rep" in s.uid]
    assert rfs and all(s.img_path.resolve().read_bytes() for s in rfs[:2])
    neu = json.loads((ensure_dataset("neu_bal_v1") / "meta.json").read_text())
    if neu["before"]["max_min_ratio"] < 3:                       # mode auto: balanced data -> train unchanged
        assert not neu["applied"] and neu["split_sizes"]["train"] == neu["train_images_original"]
    assert parse_balanced("pcb_bal_v1") == ("pcb", "bal", 1)
    assert base_dataset("gc10_bal_v1_t640") == "gc10" and data_version("gc10_bal_v1_t640") == "bal_v1"


def test_iteration_matched_control_and_grid(raw_data):
    specs = {s.name: s for s in expand(load_yaml(grid_path("p1")))}
    assert "p1_gc10_bal_v1_yolo11n_base_s0" in specs and "p1_gc10_yolo11n_base_itm_s0" in specs
    assert "p1_neu_yolo11n_base_itm_s0" not in specs
    ctl = specs["p1_gc10_yolo11n_base_itm_s0"]
    assert ctl.match_epochs_to == "gc10_bal_v1"
    fam = _train_params(ctl)
    ratio = fam["epochs_matched_to"]["ratio"]
    assert ratio > 1 and fam["epochs"] == round(100 * ratio)
    assert RunSpec("p3", "ksdd2", "yolo11n", init="T1", bg_ratio=1).name.endswith("_bg1_s0")


def test_bg_ratio_subsample(raw_data):
    d = build_dataset("ksdd2")
    tr = samples_from_coco(d, "train")
    kept = set(map(str, subsample_train(d, 1.0, 0, bg_ratio=1)))
    kept_s = [s for s in tr if str(s.img_path) in kept]
    n_def = sum(len(s.boxes) > 0 for s in tr)
    assert sum(len(s.boxes) > 0 for s in kept_s) == n_def
    assert sum(len(s.boxes) == 0 for s in kept_s) == min(n_def, sum(len(s.boxes) == 0 for s in tr))


def test_imbalance_metrics_and_table(raw_data):
    build_dataset("gc10")
    gt = json.loads((build_dataset("gc10") / "coco" / "test.json").read_text())
    names = [c["name"] for c in gt["categories"]]
    summ = imbalance_summary(samples_from_coco(build_dataset("gc10"), "train"), names, 0.333)
    rare = summ["rare_classes"]
    m = {f"AP_cls/{n}": (0.2 if n in rare else 0.6) for n in names[:4]}
    out = imbalance_metrics(m, gt, "gc10")
    assert abs(out["AP_rare"] - 0.2) < 1e-9 and abs(out["AP_common"] - 0.6) < 1e-9
    assert out["unstable_classes"]  # tiny synthetic test split -> every class < 10 boxes
    rows = []
    for ds, dv, ap_r, match in (("gc10", "orig", 0.2, "gc10_bal_v1"), ("gc10_bal_v1", "bal_v1", 0.35, None)):
        rows.append({"phase": "p1", "dataset": ds, "eval_dataset": "gc10", "data_version": dv, "model": "yolo11n",
                     "variant": "base", "match_epochs_to": match, "AP": 0.5, "AP_rare": ap_r, "AP_common": 0.6,
                     "R_rel_small": 0.4, "n_train_images": 100, "rare_classes": ";".join(rare),
                     **{f"AP_cls/{c}": ap_r for c in rare}})
    df = pd.DataFrame(rows)
    tb = table_balance(df)
    assert list(tb.Data) == ["bal_v1", "orig (iter-matched)"] and list(tb.AP_rare) == ["35.0", "20.0"]
    labels = set(with_labels(df).Model)
    assert labels == {"yolo11n", "yolo11n [orig, iter-matched]"} or len(labels) == 2


def test_per_class_table_ignores_other_datasets_classes():
    from sdd.reporting.tables import table_per_class

    df = pd.DataFrame([
        {"phase": "p1", "dataset": "neu_bal_v1", "eval_dataset": "neu", "model": "yolo11n", "variant": "base",
         "n_test_cls/crazing": 30, "AP_cls/crazing": 0.5, "n_test_cls/crease": np.nan, "AP_cls/crease": np.nan},
        {"phase": "p1", "dataset": "gc10_bal_v1", "eval_dataset": "gc10", "model": "yolo11n", "variant": "base",
         "n_test_cls/crazing": np.nan, "AP_cls/crazing": np.nan, "n_test_cls/crease": 8, "AP_cls/crease": 0.2},
    ])
    t = table_per_class(df, "p1", "neu")
    assert list(t["class"]) == ["crazing"] and list(t["test boxes"]) == [30]
    assert list(table_per_class(df, "p1", "gc10")["class"]) == ["crease"]
