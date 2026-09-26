"""Imbalance-aware evaluation (plan §2.5.3d, §2.5.5 "Đánh giá tác dụng").

* rare classes are fixed from the ORIGINAL train split of the evaluated dataset (< rare_ratio x the largest class),
  so balanced and original runs are judged on the same class groups
* ``AP_rare`` / ``AP_common`` (and AP50 variants): macro means of per-class AP over each group
* ``n_test_cls/<class>``: test boxes per class; classes below ``unstable_test_boxes`` are listed in
  ``unstable_classes`` - do not draw per-class conclusions from them
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import protocol
from ..data.balance import imbalance_summary
from ..data.build import class_names, samples_from_coco
from ..env import paths


@lru_cache
def rare_classes(dataset: str) -> tuple[str, ...]:
    d = paths().data_dir(dataset)
    return tuple(imbalance_summary(samples_from_coco(d, "train"), class_names(d),
                                   protocol()["balance"]["rare_ratio"])["rare_classes"])


def imbalance_metrics(m: dict, gt: dict, base: str) -> dict:
    names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    n_test = np.bincount([a["category_id"] for a in gt["annotations"]], minlength=len(names))
    rare = set(rare_classes(base))
    thr = protocol()["balance"]["unstable_test_boxes"]
    out = {f"n_test_cls/{n}": int(k) for n, k in zip(names, n_test)}
    out["rare_classes"] = ";".join(sorted(rare))
    out["unstable_classes"] = ";".join(n for n, k in zip(names, n_test) if k < thr)
    for key in ("AP", "AP50"):
        per = {n: m.get(f"{key}_cls/{n}", np.nan) for n in names}
        r = [v for n, v in per.items() if n in rare and not np.isnan(v)]
        c = [v for n, v in per.items() if n not in rare and not np.isnan(v)]
        out[f"{key}_rare"] = float(np.mean(r)) if r else float("nan")
        out[f"{key}_common"] = float(np.mean(c)) if c else float("nan")
    return out
