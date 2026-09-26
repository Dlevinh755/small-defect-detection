"""Qualitative error gallery for small defects (plan §4.4 day 3): missed / mislocalised / misclassified.

For each GT box in the smallest relative-size group, at the operating point (conf >= op_conf):
    missed        no detection overlaps it with IoU >= 0.1
    mislocalised  best same-class IoU in [0.1, match_iou)
    misclassified a different-class detection has IoU >= match_iou and no same-class match
GT is drawn green, the relevant detection red; crops are upscaled for the slides.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import protocol
from ..data.build import load_coco_json
from ..evaluation.matching import dets_by_image, gt_from_coco, iou_matrix
from ..evaluation.sizes import rel_group


def categorize(gt: dict, dets: list[dict], op_conf: float, match_iou: float, group: int = 0):
    G, D = gt_from_coco(gt), dets_by_image(dets, [im["id"] for im in gt["images"]])
    cases = {"missed": [], "mislocalised": [], "misclassified": []}
    for i, g in G.items():
        if not len(g.boxes):
            continue
        d = D[i]
        keep = d.scores >= op_conf
        db, dc = d.boxes[keep], d.classes[keep]
        ious = iou_matrix(g.boxes, db)
        rel = np.prod(g.boxes[:, 2:] - g.boxes[:, :2], 1) / (g.width * g.height)
        for k in np.where(rel_group(rel) == group)[0]:
            iou_k = ious[k] if ious.size else np.zeros(0)
            same = dc == g.classes[k]
            if not iou_k.size or iou_k.max() < 0.1:
                cases["missed"].append((i, k, None))
            elif same.any() and iou_k[same].max() >= match_iou:
                continue
            elif (iou_k[~same] >= match_iou).any():
                j = np.where(~same)[0][np.argmax(iou_k[~same])]
                cases["misclassified"].append((i, k, db[j]))
            elif same.any() and iou_k[same].max() >= 0.1:
                j = np.where(same)[0][np.argmax(iou_k[same])]
                cases["mislocalised"].append((i, k, db[j]))
    return G, cases


def _crop(img, gt_box, det_box, pad=24, size=160):
    boxes = [gt_box] + ([det_box] if det_box is not None else [])
    x1 = int(max(min(b[0] for b in boxes) - pad, 0))
    y1 = int(max(min(b[1] for b in boxes) - pad, 0))
    x2 = int(min(max(b[2] for b in boxes) + pad, img.shape[1]))
    y2 = int(min(max(b[3] for b in boxes) + pad, img.shape[0]))
    c = img[y1:y2, x1:x2].copy()
    s = size / max(c.shape[:2])
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    def draw(b, col):
        cv2.rectangle(c, (int((b[0] - x1) * s), int((b[1] - y1) * s)), (int((b[2] - x1) * s), int((b[3] - y1) * s)), col, 1)
    draw(gt_box, (0, 200, 0))
    if det_box is not None:
        draw(det_box, (40, 40, 230))
    canvas = np.full((size, size, 3), 250, np.uint8)
    canvas[: c.shape[0], : c.shape[1]] = c[:size, :size]
    return canvas


def gallery(run_dir: Path, data_dir: Path, out_dir: Path, n: int = 12, split: str = "test") -> dict:
    ev = protocol()["eval"]
    gt = load_coco_json(data_dir / "coco" / f"{split}.json")
    dets = json.loads((run_dir / "test_predictions.json").read_text())
    G, cases = categorize(gt, dets, ev["op_conf"], ev["match_iou"])
    files = {im["id"]: data_dir / "images" / split / im["file_name"] for im in gt["images"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, items in cases.items():
        counts[name] = len(items)
        tiles = []
        for i, k, db in items[:n]:
            img = cv2.imread(str(files[i]))
            if img is not None:
                tiles.append(_crop(img, G[i].boxes[k], db))
        if tiles:
            cols = min(6, len(tiles))
            rows = [np.hstack(tiles[r * cols:(r + 1) * cols] + [np.full_like(tiles[0], 250)] *
                              (cols - len(tiles[r * cols:(r + 1) * cols]))) for r in range(int(np.ceil(len(tiles) / cols)))]
            cv2.imwrite(str(out_dir / f"{name}.jpg"), np.vstack(rows))
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2))
    return counts
