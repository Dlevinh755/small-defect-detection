import json

import numpy as np

from sdd.config import load_yaml
from sdd.data.build import build_dataset, samples_from_coco
from sdd.data.merge import merge_datasets
from sdd.data.readers import mask_to_boxes
from sdd.data.splits import load_splits, stratified_split
from sdd.data.subsample import fraction_data_yaml, subsample_train


def test_split_is_deterministic_and_stratified():
    keys = {f"{c}_{i}": c for c in "abc" for i in range(10)}
    a = stratified_split(keys, {"train": 0.8, "val": 0.1, "test": 0.1}, 42)
    b = stratified_split(keys, {"train": 0.8, "val": 0.1, "test": 0.1}, 42)
    assert a == b
    assert [len(a[s]) for s in ("train", "val", "test")] == [24, 3, 3]
    for c in "abc":
        assert sum(u.startswith(c) for u in a["val"]) == 1


def test_mask_to_boxes_drops_small_components():
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 5:9] = 255
    m[40, 40] = 255
    boxes, dropped = mask_to_boxes(m, thr=0, min_area=4)
    assert dropped == 1
    np.testing.assert_array_equal(boxes, [[5, 10, 9, 20]])


def test_build_voc_roundtrip(raw_data):
    d = build_dataset("neu")
    meta = json.loads((d / "meta.json").read_text())
    assert meta["split_sizes"] == {"train": 24, "val": 3, "test": 3}
    assert load_splits("neu") is not None
    # YOLO label -> pixels must equal the COCO box
    for s in samples_from_coco(d, "train")[:5]:
        lines = (d / "labels" / "train" / f"{s.uid}.txt").read_text().splitlines()
        assert len(lines) == len(s.boxes)
        for line, box in zip(lines, s.boxes):
            c, cx, cy, w, h = map(float, line.split())
            px = [(cx - w / 2) * s.width, (cy - h / 2) * s.height, (cx + w / 2) * s.width, (cy + h / 2) * s.height]
            np.testing.assert_allclose(px, box, atol=1e-3)
    coco = json.loads((d / "coco" / "test.json").read_text())
    assert all(isinstance(im["id"], int) and im["uid"] for im in coco["images"])
    assert load_yaml(d / "data.yaml")["names"][0] == "crazing"


def test_rebuild_keeps_splits(raw_data):
    before = load_splits("neu")
    build_dataset("neu", force=True)
    assert load_splits("neu") == before


def test_ksdd2_official_split_and_clean_images(raw_data):
    d = build_dataset("ksdd2")
    meta = json.loads((d / "meta.json").read_text())
    assert meta["split_sizes"]["test"] == 6
    assert meta["read_report"]["dropped_small"] == 6  # one 2-px speck per defective image
    tr = samples_from_coco(d, "train") + samples_from_coco(d, "val")
    assert sum(len(s.boxes) == 0 for s in tr) == 8
    assert all(s.uid.startswith("train_") for s in tr)


def test_subsample_and_fraction_yaml(raw_data):
    d = build_dataset("ksdd2")
    n = len(samples_from_coco(d, "train"))
    sub = subsample_train(d, 0.5, seed=0)
    assert 0 < len(sub) < n and sub == subsample_train(d, 0.5, seed=0)
    y = fraction_data_yaml(d, 0.5, 0)
    assert y.name == "data_f050_s0.yaml" and (d / "data.yaml").exists()
    assert load_yaml(y)["train"] == "train_f050_s0.txt"


def test_merge_prefixes_classes(raw_data):
    d = merge_datasets(["neu", "ksdd2"], "merged_test", force=True)
    names = load_yaml(d / "data.yaml")["names"]
    assert names[0] == "neu/crazing" and names[6] == "ksdd2/defect"
    ks = [s for s in samples_from_coco(d, "train") if s.uid.startswith("ksdd2__") and len(s.classes)]
    assert ks and all((s.classes == 6).all() for s in ks)
