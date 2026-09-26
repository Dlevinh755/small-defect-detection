"""Tiled training data for large images (plan §5.1: "tăng imgsz / SAHI").

``tile_dataset("pcb", 640)`` builds ``<work>/data/pcb_t640`` from the built ``pcb`` dataset:

* train / val: every image is cut into overlapping ``size x size`` tiles (written as files, tiles cannot be
  symlinks). Boxes are clipped to the tile and kept if at least ``min_visibility`` of their area is inside.
  Only ``keep_empty`` of the defect-free tiles are kept (seeded), the rest would swamp training with background.
* test: the ORIGINAL full images and ``coco/test.json`` of the source dataset, so a tiled model evaluated with
  sliced inference is compared with full-image models on exactly the same test set.

A tiled model sees defects at native resolution instead of shrunk by the resize to 640, which is the point for
PKU-PCB (~3000x1600) and GC10 (2048x1000).
"""

from __future__ import annotations

import json
import random
import re
import shutil
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from ..config import protocol
from ..env import paths
from .build import build_dataset, class_names, samples_from_coco, write_data_yaml, write_split
from .readers import Sample

TILED_RE = re.compile(r"^(?P<base>.+)_t(?P<size>\d+)$")


def tile_positions(length: int, size: int, overlap: float) -> list[int]:
    """Start offsets covering [0, length) with tiles of ``size``; the last tile is aligned to the border."""
    if length <= size:
        return [0]
    stride = max(1, int(size * (1 - overlap)))
    pos = list(range(0, length - size, stride))
    return pos + [length - size]


def tile_windows(w: int, h: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """xyxy windows (clipped to the image when it is smaller than a tile)."""
    return [(x, y, min(x + size, w), min(y + size, h))
            for y in tile_positions(h, size, overlap) for x in tile_positions(w, size, overlap)]


def clip_boxes_to_window(boxes: np.ndarray, classes: np.ndarray, win, min_visibility: float):
    """Boxes in window coordinates, keeping those with >= min_visibility of their area inside the window."""
    if not len(boxes):
        return boxes.reshape(0, 4), classes
    x1, y1, x2, y2 = win
    c = boxes.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(x1, x2)
    c[:, [1, 3]] = c[:, [1, 3]].clip(y1, y2)
    area = np.prod(boxes[:, 2:] - boxes[:, :2], 1)
    vis = np.prod(np.clip(c[:, 2:] - c[:, :2], 0, None), 1) / np.maximum(area, 1e-9)
    keep = vis >= min_visibility
    return c[keep] - [x1, y1, x1, y1], classes[keep]


def _tile_split(src: list[Sample], out_img_dir: Path, size: int, overlap: float, min_vis: float, keep_empty: float,
                rng: random.Random) -> list[Sample]:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for s in src:
        img = None
        for win in tile_windows(s.width, s.height, size, overlap):
            boxes, cls = clip_boxes_to_window(s.boxes, s.classes, win, min_vis)
            if not len(boxes) and rng.random() >= keep_empty:
                continue
            if img is None:
                img = cv2.imread(str(s.img_path), cv2.IMREAD_UNCHANGED)
            x1, y1, x2, y2 = win
            uid = f"{s.uid}__{x1}_{y1}"
            path = out_img_dir / f"{uid}{s.img_path.suffix.lower()}"
            cv2.imwrite(str(path), img[y1:y2, x1:x2])
            tiles.append(Sample(uid, path, x2 - x1, y2 - y1, boxes, cls, {"parent": s.uid, "window": list(win)}))
    return tiles


def tile_dataset(base: str, size: int, force: bool = False) -> Path:
    name = f"{base}_t{size}"
    out = paths().data_dir(name)
    if (out / "meta.json").exists() and not force:
        return out
    if out.exists():
        shutil.rmtree(out)
    cfg = protocol()["tiling"]
    built = paths().data_dir(base)  # may itself be derived (e.g. pcb_bal_v1), built by sdd.data.resolve
    src_dir = built if (built / "meta.json").exists() else build_dataset(base)
    names = class_names(src_dir)
    rng = random.Random(0)
    stats = {}
    raw_tiles = out / "_tiles"
    for split in ("train", "val"):
        src = samples_from_coco(src_dir, split)
        tiles = _tile_split(src, raw_tiles / split, size, cfg["overlap"], cfg["min_visibility"], cfg["keep_empty"], rng)
        write_split(out, split, tiles, names)
        stats[split] = {"images": len(src), "tiles": len(tiles),
                        "boxes_src": int(sum(len(s.boxes) for s in src)),
                        "boxes_tiles": int(sum(len(t.boxes) for t in tiles))}
    test = samples_from_coco(src_dir, "test")
    write_split(out, "test", test, names)  # full images, unchanged
    stats["test"] = {"images": len(test), "tiles": 0}
    write_data_yaml(out, names)
    (out / "meta.json").write_text(json.dumps(
        {"dataset": name, "source": base, "tile": size, "classes": names, "built": str(date.today()),
         "tiling": cfg, "split_sizes": {k: v.get("tiles") or v["images"] for k, v in stats.items()},
         "tiling_stats": stats}, indent=2))
    return out


def parse_tiled(name: str) -> tuple[str, int] | None:
    m = TILED_RE.match(name)
    return (m["base"], int(m["size"])) if m else None
