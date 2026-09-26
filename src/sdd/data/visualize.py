"""Draw GT (and optionally predicted) boxes - conversion sanity check before any training (plan §4.4, §8)."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from .build import class_names, samples_from_coco

_PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
            (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0)]


def color(c: int) -> tuple[int, int, int]:
    return _PALETTE[int(c) % len(_PALETTE)]


def draw_boxes(img: np.ndarray, boxes, classes, names, scores=None, thickness: int | None = None) -> np.ndarray:
    img = img.copy()
    t = thickness or max(1, round(max(img.shape[:2]) / 400))
    for i, ((x1, y1, x2, y2), c) in enumerate(zip(boxes, classes)):
        p1, p2 = (int(x1), int(y1)), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, color(c), t)
        label = names[int(c)] if names else str(c)
        if scores is not None:
            label += f" {scores[i]:.2f}"
        cv2.putText(img, label, (p1[0], max(p1[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, color(c), max(1, t // 2))
    return img


def visualize_split(data_dir: Path, out_dir: Path, split: str = "train", n: int = 20, seed: int = 0,
                    only_defects: bool = True) -> list[Path]:
    names = class_names(data_dir)
    samples = samples_from_coco(data_dir, split)
    if only_defects:
        samples = [s for s in samples if len(s.boxes)]
    random.Random(seed).shuffle(samples)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for s in samples[:n]:
        img = cv2.imread(str(s.img_path))
        if img is None:
            continue
        p = out_dir / f"{split}_{s.uid}.jpg"
        cv2.imwrite(str(p), draw_boxes(img, s.boxes, s.classes, names))
        written.append(p)
    return written
