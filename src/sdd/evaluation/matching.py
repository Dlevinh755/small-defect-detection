"""Greedy class-aware matching at an operating point (conf >= op_conf, IoU >= match_iou).

Used for precision / recall, recall per size group (plan Appendix A.3), image-level metrics and the
error gallery. Everything works on per-image arrays in original-pixel xyxy coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sizes import ABS_NAMES, abs_group, rel_bins, rel_group


@dataclass
class ImageGT:
    boxes: np.ndarray    # (N, 4) xyxy
    classes: np.ndarray  # (N,)
    width: int
    height: int
    uid: str = ""


@dataclass
class ImageDet:
    boxes: np.ndarray    # (M, 4) xyxy
    classes: np.ndarray
    scores: np.ndarray


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def gt_from_coco(coco: dict) -> dict[int, ImageGT]:
    out = {im["id"]: ImageGT(np.zeros((0, 4)), np.zeros(0, int), im["width"], im["height"], im.get("uid", ""))
           for im in coco["images"]}
    tmp: dict[int, list] = {}
    for a in coco["annotations"]:
        tmp.setdefault(a["image_id"], []).append(a)
    for i, anns in tmp.items():
        b = np.array([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)], float)
        out[i] = ImageGT(b, np.array([a["category_id"] for a in anns], int), out[i].width, out[i].height, out[i].uid)
    return out


def dets_by_image(dets: list[dict], image_ids) -> dict[int, ImageDet]:
    tmp: dict[int, list] = {i: [] for i in image_ids}
    for d in dets:
        tmp.setdefault(d["image_id"], []).append(d)
    out = {}
    for i, ds in tmp.items():
        b = np.array([[x, y, x + w, y + h] for x, y, w, h in (d["bbox"] for d in ds)], float).reshape(-1, 4)
        out[i] = ImageDet(b, np.array([d["category_id"] for d in ds], int), np.array([d["score"] for d in ds], float))
    return out


def match_image(g: ImageGT, d: ImageDet, iou_thr: float, conf_thr: float, class_aware: bool = True):
    """Returns (gt_hit[N] bool, det_tp[M'] bool, kept det indices[M'], best_iou_per_gt[N])."""
    keep = np.where(d.scores >= conf_thr)[0]
    keep = keep[np.argsort(-d.scores[keep], kind="stable")]
    ious = iou_matrix(d.boxes[keep], g.boxes)  # (M', N)
    same = d.classes[keep][:, None] == g.classes[None, :] if class_aware else np.ones_like(ious, bool)
    gt_hit = np.zeros(len(g.boxes), bool)
    det_tp = np.zeros(len(keep), bool)
    for j in range(len(keep)):
        cand = np.where(same[j] & ~gt_hit & (ious[j] >= iou_thr))[0]
        if cand.size:
            k = cand[np.argmax(ious[j, cand])]
            gt_hit[k] = det_tp[j] = True
    best = ious.max(0) if ious.size else np.zeros(len(g.boxes))
    return gt_hit, det_tp, keep, best


def operating_point_metrics(gts: dict[int, ImageGT], dets: dict[int, ImageDet], iou_thr: float = 0.5,
                            conf_thr: float = 0.25, bins=None, names=None) -> dict:
    """Precision, recall, F1 and recall per absolute / relative size group at one operating point."""
    if bins is None:
        bins, names = rel_bins()
    tp = fp = 0
    hits, rel_g, abs_g = [], [], []
    for i, g in gts.items():
        d = dets.get(i, ImageDet(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)))
        gt_hit, det_tp, _, _ = match_image(g, d, iou_thr, conf_thr)
        tp += int(det_tp.sum())
        fp += int((~det_tp).sum())
        if len(g.boxes):
            area = np.prod(g.boxes[:, 2:] - g.boxes[:, :2], axis=1)
            hits.append(gt_hit)
            rel_g.append(rel_group(area / (g.width * g.height), bins))
            abs_g.append(abs_group(area))
    hits = np.concatenate(hits) if hits else np.zeros(0, bool)
    rel_g = np.concatenate(rel_g) if rel_g else np.zeros(0, int)
    abs_g = np.concatenate(abs_g) if abs_g else np.zeros(0, int)
    n_gt = len(hits)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = hits.sum() / n_gt if n_gt else float("nan")
    out = {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r and not np.isnan(p + r) else float("nan"),
           "TP": tp, "FP": fp, "n_gt": n_gt}
    for k, n in enumerate(names):
        m = rel_g == k
        out[f"R_{n}"] = float(hits[m].mean()) if m.any() else float("nan")
        out[f"n_{n}"] = int(m.sum())
    for k, n in enumerate(ABS_NAMES):
        m = abs_g == k
        out[f"R_{n}"] = float(hits[m].mean()) if m.any() else float("nan")
        out[f"n_{n}"] = int(m.sum())
    return out


def recall_by_size(gts, preds, img_areas, bins, iou_thr=0.5, conf_thr=0.25) -> dict:
    """Plan Appendix A.3 interface: gts[img] = [(x1,y1,x2,y2,cls)], preds[img] = [(x1,y1,x2,y2,cls,score)]."""
    G, D = {}, {}
    for img, gl in gts.items():
        a = np.asarray(gl, float).reshape(-1, 5)
        # img_areas only matters through W*H; use W=area, H=1
        G[img] = ImageGT(a[:, :4], a[:, 4].astype(int), img_areas[img], 1)
        p = np.asarray(preds.get(img, []), float).reshape(-1, 6)
        D[img] = ImageDet(p[:, :4], p[:, 4].astype(int), p[:, 5])
    names = [f"[{bins[i]},{bins[i + 1]})" for i in range(len(bins) - 1)]
    m = operating_point_metrics(G, D, iou_thr, conf_thr, bins, names)
    return {n: (m[f"R_{n}"], m[f"n_{n}"]) for n in names}
