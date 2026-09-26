"""TIDE error decomposition (Bolya et al. 2020): dAP50 recoverable by fixing each error type (plan §3.2, §5.1).

Main errors: Cls, Loc, Both, Dupe, Bkg, Miss. Special: FalsePos, FalseNeg. Values are dAP50 in [0, 1].
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def tide_errors(gt_json: str | Path, dets: list[dict], name: str = "run") -> dict:
    try:
        from tidecv import TIDE, datasets
    except ImportError:
        log.warning("tidecv not installed - skipping TIDE")
        return {}
    if not dets:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        dp = Path(tmp) / f"{name}.json"
        dp.write_text(json.dumps(dets))
        gt = json.loads(Path(gt_json).read_text())
        for a in gt["annotations"]:  # tidecv's COCO loader converts a mask even in box mode: give it the box polygon
            x, y, w, h = a["bbox"]
            a.setdefault("segmentation", [[x, y, x + w, y, x + w, y + h, x, y + h]])
        gt_json = Path(tmp) / "gt.json"
        gt_json.write_text(json.dumps(gt))
        tide = TIDE()
        with contextlib.redirect_stdout(io.StringIO()):
            tide.evaluate(datasets.COCO(str(gt_json)), datasets.COCOResult(str(dp)), mode=TIDE.BOX, name=name)
            main = tide.get_main_errors()[name]
            special = tide.get_special_errors()[name]
            ap50 = tide.runs[name].ap
    out = {"TIDE_AP50": ap50 / 100}
    out.update({f"TIDE_{k}": v / 100 for k, v in main.items()})
    out.update({f"TIDE_{k}": v / 100 for k, v in special.items()})
    return out
