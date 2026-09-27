"""Tiling, sliced-inference merging, grid expansion and feature-map figures (no training)."""

import json

import numpy as np

from sdd.config import load_yaml
from sdd.data.build import samples_from_coco
from sdd.data.tiling import clip_boxes_to_window, parse_tiled, tile_dataset, tile_positions, tile_windows
from sdd.engine.grid import expand, grid_path
from sdd.engine.predict import merge_detections
from sdd.engine.run import RunSpec


def test_tile_positions_cover_and_align():
    assert tile_positions(500, 640, 0.2) == [0]
    p = tile_positions(2048, 640, 0.2)
    assert p[0] == 0 and p[-1] == 2048 - 640 and all(b - a <= 512 for a, b in zip(p, p[1:]))
    assert len(tile_windows(2048, 1000, 640, 0.2)) == len(p) * len(tile_positions(1000, 640, 0.2))


def test_clip_boxes_visibility():
    boxes = np.array([[10, 10, 30, 30], [90, 10, 130, 30], [150, 150, 160, 160]], float)
    b, c = clip_boxes_to_window(boxes, np.array([0, 1, 2]), (0, 0, 100, 100), 0.5)
    # box 2 is 25% inside -> dropped; box 3 outside -> dropped
    np.testing.assert_array_equal(c, [0])
    b, c = clip_boxes_to_window(boxes, np.array([0, 1, 2]), (80, 0, 180, 100), 0.5)
    np.testing.assert_allclose(b, [[10, 10, 50, 30]])


def test_tile_dataset_keeps_full_test_split(raw_data):
    d = tile_dataset("ksdd2", 256)
    meta = json.loads((d / "meta.json").read_text())
    src_test = samples_from_coco(d.parent / "ksdd2", "test")
    assert meta["tiling_stats"]["test"]["images"] == len(src_test)
    assert (d / "coco" / "test.json").read_text() == (d.parent / "ksdd2" / "coco" / "test.json").read_text()
    tiles = samples_from_coco(d, "train")
    assert tiles and all(t.width <= 256 and t.height <= 256 for t in tiles)
    for t in tiles:
        if len(t.boxes):
            assert (t.boxes[:, 2] <= t.width).all() and (t.boxes[:, 3] <= t.height).all()
    # every defect of the source train split survives in at least one tile (blobs are 8x20 px < tile)
    assert meta["tiling_stats"]["train"]["boxes_tiles"] >= meta["tiling_stats"]["train"]["boxes_src"]
    assert parse_tiled("pcb_t640") == ("pcb", 640) and parse_tiled("pcb") is None


def test_merge_detections_is_classwise_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    classes = np.array([0, 0, 1, 0])
    keep = merge_detections(boxes, scores, classes, 0.5, 300)
    assert sorted(keep.tolist()) == [0, 2, 3]  # duplicate of the same class removed, other class kept
    assert merge_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), 0.5, 300).size == 0


def test_tiling_grid_and_reuse_names():
    specs = expand(load_yaml(grid_path("p2_tiling")))
    names = [s.name for s in specs]
    assert "p2t_pcb_bal_v1_yolo11n_base_sahi_s0" in names and "p2t_gc10_t640_yolo11n_A0_sahi_s0" in names
    a0 = next(s for s in specs if s.reuse_phase == "p2")
    assert a0.reuse_source() == f"p2_{a0.dataset}_yolo11n_A0_s0"  # SAHI on the phase-2 A0 checkpoint
    s = next(s for s in specs if s.reuse_phase)
    assert s.reuse_source() == f"p1_{s.dataset}_yolo11n_{s.variant}_s0" and s.dataset.endswith("_bal_v1")
    assert RunSpec("p1", "neu", "yolo11n").name == "p1_neu_yolo11n_base_s0"


def test_feature_map_figures(raw_data, tmp_path):
    import cv2

    from sdd.config import variant_cfg
    from sdd.models.yolo_trainer import build_yolo
    from sdd.reporting.feature_maps import capture, channel_grid, compare_runs, letterbox
    from sdd.reporting.style import save

    base = build_yolo("yolo11n.yaml", nc=1, variant=variant_cfg("base"), verbose=False)
    simam = build_yolo("yolo11n.yaml", nc=1, variant=variant_cfg("simam"), verbose=False)
    s = samples_from_coco(tile_dataset("ksdd2", 256).parent / "ksdd2", "test")[0]
    img = cv2.imread(str(s.img_path))
    feats = capture(simam, letterbox(img)[0], [2, 8], pre_attention=True)
    assert feats[2].shape[-1] == 160 and feats[8].shape[-1] == 20
    save(compare_runs({"base": base, "simam": simam}, img, [2, 4, 6, 8], s.boxes), tmp_path / "cmp.png")
    save(channel_grid(feats[8]), tmp_path / "grid.png")
    assert (tmp_path / "cmp.png").stat().st_size > 5000 and (tmp_path / "grid.png").exists()
