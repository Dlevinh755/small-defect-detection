"""Image-level metrics for the industrial target (plan §6.2): is a defective part flagged, is a good part not?"""

from __future__ import annotations

import numpy as np

from .matching import ImageDet, ImageGT, match_image


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1.0
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def image_level_metrics(gts: dict[int, ImageGT], dets: dict[int, ImageDet], iou_thr: float = 0.5,
                        conf_thr: float = 0.25) -> dict:
    """
    img_detection_rate: defective images with >= 1 correctly localised box (class-aware, IoU >= iou_thr)
    img_flag_rate:      defective images with any detection >= conf_thr (what a reject gate would do)
    false_alarm_rate:   defect-free images with any detection >= conf_thr
    img_AUROC / img_AP: image score = max detection score
    """
    det_ok, flagged, fa, labels, scores = [], [], [], [], []
    empty = ImageDet(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
    for i, g in gts.items():
        d = dets.get(i, empty)
        s = float(d.scores.max()) if len(d.scores) else 0.0
        defective = len(g.boxes) > 0
        labels.append(defective)
        scores.append(s)
        if defective:
            hit, _, _, _ = match_image(g, d, iou_thr, conf_thr)
            det_ok.append(hit.any())
            flagged.append(s >= conf_thr)
        else:
            fa.append(s >= conf_thr)
    labels, scores = np.array(labels, bool), np.array(scores)
    nan = float("nan")
    return {
        "img_detection_rate": float(np.mean(det_ok)) if det_ok else nan,
        "img_flag_rate": float(np.mean(flagged)) if flagged else nan,
        "false_alarm_rate": float(np.mean(fa)) if fa else nan,
        "img_AUROC": _auroc(scores[labels], scores[~labels]),
        "img_AP": _ap(labels.astype(float), scores),
        "n_defective_imgs": int(labels.sum()),
        "n_clean_imgs": int((~labels).sum()),
    }
