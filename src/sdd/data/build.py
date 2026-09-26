"""Processed dataset layout (plan §2.3)::

    <work>/data/<ds>/
        images/{train,val,test}/<uid>.<ext>    symlinks to the raw images (copies where symlinks are unavailable)
        labels/{train,val,test}/<uid>.txt      YOLO: cls cx cy w h (normalised); empty file = defect-free image
        coco/{train,val,test}.json             original-pixel boxes, numeric image ids, category ids 0..K-1
        data.yaml                              Ultralytics dataset config
        meta.json                              read report, split sizes, class names

COCO ``image_id`` is numeric (pycocotools requirement); every image entry also stores its ``uid`` (= file stem),
which is what prediction exporters use to look the id up - GT and predictions share one mapping by construction.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from ..config import dataset_cfg, datasets_cfg, save_yaml
from ..env import paths
from .readers import Sample, read_dataset
from .splits import SPLIT_NAMES, get_or_create_splits

log = logging.getLogger(__name__)


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def yolo_lines(s: Sample) -> list[str]:
    lines = []
    for (x1, y1, x2, y2), c in zip(s.boxes, s.classes):
        cx, cy = (x1 + x2) / 2 / s.width, (y1 + y2) / 2 / s.height
        w, h = (x2 - x1) / s.width, (y2 - y1) / s.height
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def coco_dict(samples: list[Sample], class_names: list[str]) -> dict:
    coco = {
        "info": {"description": "sdd", "date_created": str(date.today())},
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n, "supercategory": "defect"} for i, n in enumerate(class_names)],
    }
    ann_id = 1
    for img_id, s in enumerate(sorted(samples, key=lambda s: s.uid), start=1):
        coco["images"].append(
            {"id": img_id, "file_name": s.img_path.name, "uid": s.uid, "width": s.width, "height": s.height}
        )
        for (x1, y1, x2, y2), c in zip(s.boxes, s.classes):
            w, h = float(x2 - x1), float(y2 - y1)
            coco["annotations"].append(
                {"id": ann_id, "image_id": img_id, "category_id": int(c), "bbox": [float(x1), float(y1), w, h],
                 "area": w * h, "iscrowd": 0}
            )
            ann_id += 1
    return coco


def write_split(out: Path, split: str, samples: list[Sample], class_names: list[str]) -> None:
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    renamed = []
    for s in samples:
        dst = img_dir / f"{s.uid}{s.img_path.suffix.lower()}"
        _link(s.img_path, dst)
        (lbl_dir / f"{s.uid}.txt").write_text("\n".join(yolo_lines(s)))
        renamed.append(Sample(s.uid, dst, s.width, s.height, s.boxes, s.classes, s.meta))
    (out / "coco").mkdir(exist_ok=True)
    (out / "coco" / f"{split}.json").write_text(json.dumps(coco_dict(renamed, class_names)))


def write_data_yaml(out: Path, class_names: list[str], train: str = "images/train", filename: str = "data.yaml") -> Path:
    y = out / filename
    save_yaml(
        {"path": str(out), "train": train, "val": "images/val", "test": "images/test",
         "names": {i: n for i, n in enumerate(class_names)}},
        y,
    )
    return y


def build_dataset(name: str, force: bool = False) -> Path:
    """Read raw data, apply the fixed split, write the processed layout. Idempotent unless ``force``."""
    out = paths().data_dir(name)
    if (out / "meta.json").exists() and not force:
        log.info("%s already built at %s", name, out)
        return out
    if force and out.exists():
        shutil.rmtree(out)
    cfg = dataset_cfg(name)
    samples, rep = read_dataset(paths().raw_dir(name), cfg)
    splits = get_or_create_splits(name, samples, cfg, datasets_cfg()["split_seed"])
    by_uid = {s.uid: s for s in samples}
    for split in SPLIT_NAMES:
        write_split(out, split, [by_uid[u] for u in splits[split]], cfg["classes"])
    write_data_yaml(out, cfg["classes"])
    meta = {
        "dataset": name,
        "classes": cfg["classes"],
        "raw_dir": str(paths().raw_dir(name)),
        "built": str(date.today()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "boxes_per_split": {k: int(sum(len(by_uid[u].boxes) for u in v)) for k, v in splits.items()},
        "read_report": {k: v for k, v in asdict(rep).items() if k != "notes"},
        "notes": rep.notes[:50],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Built %s: %s", name, meta["split_sizes"])
    return out


# ------------------------------------------------------------------------------------------- reading back
def load_coco_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def samples_from_coco(data_dir: Path, split: str) -> list[Sample]:
    """Reconstruct samples (with processed image paths) from a built split."""
    coco = load_coco_json(data_dir / "coco" / f"{split}.json")
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    out = []
    for im in coco["images"]:
        a = anns.get(im["id"], [])
        boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (x["bbox"] for x in a)], float).reshape(-1, 4)
        cls = np.array([x["category_id"] for x in a], int)
        out.append(Sample(im["uid"], data_dir / "images" / split / im["file_name"], im["width"], im["height"], boxes, cls))
    return out


def class_names(data_dir: Path) -> list[str]:
    return json.loads((data_dir / "meta.json").read_text())["classes"]
