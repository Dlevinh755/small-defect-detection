import numpy as np
import pandas as pd

from sdd.evaluation.coco_eval import coco_metrics
from sdd.evaluation.image_level import image_level_metrics
from sdd.evaluation.matching import dets_by_image, gt_from_coco, operating_point_metrics, recall_by_size
from sdd.evaluation.stats import compare_to_reference, welch

GT = {
    "images": [{"id": 1, "uid": "a", "width": 200, "height": 200}, {"id": 2, "uid": "b", "width": 200, "height": 200},
               {"id": 3, "uid": "c", "width": 200, "height": 200}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 8, 8], "area": 64, "iscrowd": 0},      # small
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 60, 60], "area": 3600, "iscrowd": 0},  # medium
        {"id": 3, "image_id": 2, "category_id": 0, "bbox": [0, 0, 150, 150], "area": 22500, "iscrowd": 0},  # large
    ],
    "categories": [{"id": 0, "name": "x"}, {"id": 1, "name": "y"}],
}


def _perfect():
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
            for a in GT["annotations"]]


def test_perfect_predictions_give_ap_one():
    m = coco_metrics(GT, _perfect())
    for k in ("AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AP75_s", "AP_rel_small"):
        assert abs(m[k] - 1) < 1e-6, k


def test_missing_small_box_only_hurts_small_group():
    dets = [d for d in _perfect() if d["bbox"][2] > 10]
    m = coco_metrics(GT, dets)
    assert m["AP_s"] == 0 and abs(m["AP_m"] - 1) < 1e-6 and abs(m["AP_l"] - 1) < 1e-6
    assert m["AP_rel_small"] == 0
    G = gt_from_coco(GT)
    op = operating_point_metrics(G, dets_by_image(dets, G), 0.5, 0.25, [0, 0.005, 0.2, 1], ["t", "m", "l"])
    assert op["R_t"] == 0 and op["R_m"] == 1 and op["R_l"] == 1 and op["P"] == 1


def test_recall_by_size_appendix_interface():
    gts = {"i": [(0, 0, 10, 10, 0), (0, 0, 100, 100, 0)]}
    preds = {"i": [(0, 0, 10, 10, 0, 0.9), (0, 0, 100, 100, 1, 0.9)]}  # second has the wrong class
    r = recall_by_size(gts, preds, {"i": 200 * 200}, [0, 0.01, 1])
    assert r["[0,0.01)"] == (1.0, 1) and r["[0.01,1)"] == (0.0, 1)


def test_image_level():
    G = gt_from_coco(GT)
    dets = _perfect() + [{"image_id": 3, "category_id": 0, "bbox": [1, 1, 5, 5], "score": 0.5}]
    m = image_level_metrics(G, dets_by_image(dets, G))
    assert m["img_detection_rate"] == 1 and m["false_alarm_rate"] == 1 and m["n_clean_imgs"] == 1


def test_welch_and_reference_comparison():
    t, p = welch([0.30, 0.31, 0.32], [0.20, 0.21, 0.19])
    assert p < 0.01
    df = pd.DataFrame({"dataset": ["d"] * 6, "variant": ["A0"] * 3 + ["A1"] * 3,
                       "AP_s": [0.20, 0.21, 0.19, 0.30, 0.31, 0.32]})
    r = compare_to_reference(df, "variant", "A0", "AP_s").set_index("variant")
    assert r.loc["A1", "improved"] and not r.loc["A0", "improved"]
    assert np.isclose(r.loc["A1", "delta_vs_ref"], 0.11)
