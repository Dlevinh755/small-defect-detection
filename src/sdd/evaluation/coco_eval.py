"""COCO-style AP / AR, overall, per absolute size (COCO areas), per relative size and per class (plan §3.2).

Detections are COCO result dicts ``{"image_id", "category_id", "bbox": [x, y, w, h], "score"}`` in
original-image pixels, with the numeric ``image_id`` of the processed ``coco/<split>.json``.
"""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..data.build import load_coco_json
from .sizes import rel_bins

REL_SCALE = 1e6  # relative areas are multiplied by this so pycocotools' areaRng machinery can be reused


def coco_from_dict(d: dict) -> COCO:
    c = COCO()
    c.dataset = copy.deepcopy(d)
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def _load_gt(gt) -> dict:
    return load_coco_json(gt) if isinstance(gt, (str, Path)) else gt


def _run(gt: COCO, dets: list[dict], area_rng=None, area_lbl=None) -> COCOeval:
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(copy.deepcopy(dets))
        if area_rng is not None:
            # relative mode: dt areas were already rescaled by the caller through the 'area' field
            for a in dt.dataset["annotations"]:
                a["area"] = a["rel_area"]
            dt.createIndex()
        E = COCOeval(gt, dt, "bbox")
        if area_rng is not None:
            E.params.areaRng, E.params.areaRngLbl = area_rng, area_lbl
        E.evaluate()
        E.accumulate()
        E.summarize()
    return E


def _ap(E: COCOeval, iou: float | None = None, area: int = 0, cat: int | None = None) -> float:
    """Mean precision from E.eval['precision'] [T, R, K, A, M] at maxDets=100; -1 -> nan."""
    p = E.eval["precision"]
    t = slice(None) if iou is None else np.where(np.isclose(E.params.iouThrs, iou))[0]
    k = slice(None) if cat is None else [cat]
    s = p[t, :, k, area, -1]
    s = s[s > -1]
    return float(s.mean()) if s.size else float("nan")


def _ar(E: COCOeval, area: int = 0) -> float:
    r = E.eval["recall"][:, :, area, -1]
    r = r[r > -1]
    return float(r.mean()) if r.size else float("nan")


def coco_metrics(gt, dets: list[dict], rel: bool = True, per_class: bool = True) -> dict:
    """Return a flat dict of metrics (fractions in [0, 1], nan where a group has no GT)."""
    gt_d = _load_gt(gt)
    out: dict[str, float] = {}
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    G = coco_from_dict(gt_d)
    E = _run(G, dets)
    names = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR1", "AR10", "AR100", "AR_s", "AR_m", "AR_l"]
    out.update({n: float(v) if v > -1 else float("nan") for n, v in zip(names, E.stats)})
    for a, lbl in ((1, "s"), (2, "m"), (3, "l")):
        out[f"AP50_{lbl}"] = _ap(E, 0.5, a)
        out[f"AP75_{lbl}"] = _ap(E, 0.75, a)
    if per_class:
        for k, cid in enumerate(E.params.catIds):
            name = G.cats[cid]["name"]
            out[f"AP_cls/{name}"] = _ap(E, None, 0, k)
            out[f"AP50_cls/{name}"] = _ap(E, 0.5, 0, k)
    if rel:
        out.update(relative_size_metrics(gt_d, dets))
    return out


def relative_size_metrics(gt_d: dict, dets: list[dict], bins=None, names=None) -> dict:
    """AP / AP50 / AP75 / AR per relative-size group (box area / image area)."""
    if bins is None:
        bins, names = rel_bins()
    wh = {im["id"]: im["width"] * im["height"] for im in gt_d["images"]}
    g = copy.deepcopy(gt_d)
    for a in g["annotations"]:
        a["area"] = a["bbox"][2] * a["bbox"][3] / wh[a["image_id"]] * REL_SCALE
    d = copy.deepcopy(dets)
    for x in d:
        x["rel_area"] = x["bbox"][2] * x["bbox"][3] / wh[x["image_id"]] * REL_SCALE
    area_rng = [[0, REL_SCALE * 1.01]] + [[lo * REL_SCALE, hi * REL_SCALE] for lo, hi in zip(bins[:-1], bins[1:])]
    area_rng[-1][1] *= 1.01  # include boxes covering the full image
    E = _run(coco_from_dict(g), d, area_rng, ["all", *names])
    out = {}
    for i, n in enumerate(names, start=1):
        out[f"AP_{n}"] = _ap(E, None, i)
        out[f"AP50_{n}"] = _ap(E, 0.5, i)
        out[f"AP75_{n}"] = _ap(E, 0.75, i)
        out[f"AR_{n}"] = _ar(E, i)
    return out
