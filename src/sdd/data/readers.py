"""Readers turning each raw dataset layout into a list of :class:`Sample` (original-image pixel coordinates).

All readers search the raw root recursively, so the path in ``configs/paths.yaml`` can point to the top folder
of whatever Kaggle mirror is attached. Known layouts:

* NEU-DET        ``**/ANNOTATIONS/*.xml`` + ``**/IMAGES/*.jpg`` (or ``train|validation/annotations|images/...``)
* GC10-DET       ``**/lable|label/*.xml`` + ``**/<class_no>/*.jpg``
* PKU-Market-PCB ``**/Annotations/<Class>/*.xml`` + ``**/images/<Class>/*.jpg``
* Magnetic Tile  ``**/MT_<Class>/Imgs/<stem>.jpg`` + mask ``<stem>.png``; ``MT_Free`` = defect-free
* KolektorSDD2   ``**/train|test/<id>.png`` + mask ``<id>_GT.png``
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    uid: str                 # unique id inside the dataset; becomes the file stem in the processed layout
    img_path: Path
    width: int
    height: int
    boxes: np.ndarray        # (N, 4) float xyxy, original pixels
    classes: np.ndarray      # (N,) int class ids
    meta: dict = field(default_factory=dict)

    @property
    def strat_key(self) -> str:
        """Stratification key: dominant class, or 'clean' for defect-free images."""
        if len(self.classes) == 0:
            return "clean"
        return str(Counter(self.classes.tolist()).most_common(1)[0][0])


@dataclass
class ReadReport:
    n_images: int = 0
    n_boxes: int = 0
    dropped_small: int = 0       # mask components < min_area
    dropped_degenerate: int = 0  # zero-size / out-of-image boxes
    missing_images: int = 0      # annotation without image
    duplicates: int = 0          # duplicated annotation files (same stem)
    duplicate_boxes: int = 0     # identical (class, box) repeated inside one image -> kept once
    notes: list = field(default_factory=list)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)


def _clip_boxes(boxes: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes.reshape(0, 4), np.zeros(0, bool)
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    keep = (b[:, 2] - b[:, 0] > 0) & (b[:, 3] - b[:, 1] > 0)
    return b, keep


def _norm(name: str) -> str:
    return name.strip().lower()


def _class_lookup(cfg: dict) -> dict[str, int]:
    classes = [_norm(c) for c in cfg["classes"]]
    idx = {c: i for i, c in enumerate(classes)}
    for raw, target in (cfg.get("class_map") or {}).items():
        idx[_norm(str(raw))] = classes.index(_norm(target))
    return idx


# ----------------------------------------------------------------------------------------------------------- VOC
def _index_images(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            index[p.stem].append(p)
    return index


def _pick_image(xml: Path, candidates: list[Path]) -> Path:
    """When a stem matches several images, prefer the one sharing the longest path suffix with the xml."""
    if len(candidates) == 1:
        return candidates[0]

    def score(p: Path) -> int:
        return len(set(p.parent.parts) & set(xml.parent.parts))

    return max(candidates, key=score)


def read_voc(root: Path, cfg: dict) -> tuple[list[Sample], ReadReport]:
    lookup = _class_lookup(cfg)
    ignore = {_norm(str(c)) for c in cfg.get("ignore_classes") or []}
    images = _index_images(root)
    rep = ReadReport()
    samples, seen, unknown, ignored = [], set(), Counter(), Counter()
    for xml in sorted(root.rglob("*.xml")):
        if xml.stem in seen:
            rep.duplicates += 1
            continue
        if xml.stem not in images:
            rep.missing_images += 1
            continue
        seen.add(xml.stem)
        img = _pick_image(xml, images[xml.stem])
        w, h = _image_size(img)
        r = ET.parse(xml).getroot()
        boxes, classes = [], []
        for o in r.findall("object"):
            name = _norm(o.findtext("name", ""))
            if name in ignore:  # known annotation errors listed in datasets.yaml -> object dropped, reported
                ignored[name] += 1
                rep.notes.append(f"ignored object '{name}' in {xml.name}")
                continue
            if name not in lookup:
                unknown[name] += 1
                continue
            b = o.find("bndbox")
            boxes.append([float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            classes.append(lookup[name])
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if len(boxes):  # identical duplicates (NEU has a few): Ultralytics drops them in training, so must the GT
            _, first = np.unique(np.column_stack([classes, boxes]), axis=0, return_index=True)
            first = np.sort(first)
            rep.duplicate_boxes += len(boxes) - len(first)
            boxes, classes = boxes[first], [classes[i] for i in first]
        # VOC is 1-based inclusive; convert to 0-based [x1, x2) pixel extents
        boxes[:, :2] -= 1
        boxes, keep = _clip_boxes(boxes, w, h)
        rep.dropped_degenerate += int((~keep).sum())
        samples.append(Sample(xml.stem, img, w, h, boxes[keep], np.asarray(classes, int)[keep]))
    if unknown:
        raise ValueError(f"Unknown class names {dict(unknown)} in {root}; add them to class_map in datasets.yaml "
                         "(or to ignore_classes if they are annotation errors)")
    rep.dropped_degenerate += sum(ignored.values())
    rep.n_images, rep.n_boxes = len(samples), sum(len(s.boxes) for s in samples)
    return samples, rep


# ---------------------------------------------------------------------------------------------------------- masks
def mask_to_boxes(mask: np.ndarray, thr: int = 0, min_area: int = 4) -> tuple[np.ndarray, int]:
    """Connected components of a binary mask -> xyxy boxes. Returns (boxes, n_dropped_below_min_area)."""
    m = (mask > thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes, dropped = [], 0
    for x, y, w, h, area in stats[1:]:  # label 0 = background
        if area >= min_area:
            boxes.append([x, y, x + w, y + h])
        else:
            dropped += 1
    return np.asarray(boxes, float).reshape(-1, 4), dropped


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"Cannot read mask {path}")
    return m


def read_magnetic_tile(root: Path, cfg: dict) -> tuple[list[Sample], ReadReport]:
    lookup = _class_lookup(cfg)
    rep, samples = ReadReport(), []
    dirs = sorted(d for d in root.rglob("MT_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No MT_* folders under {root}")
    for d in dirs:
        cls_name = _norm(d.name[3:])
        clean = cls_name == "free"
        if clean and not cfg.get("include_clean", True):
            continue
        if not clean and cls_name not in lookup:
            raise ValueError(f"Unknown Magnetic Tile class folder {d.name}")
        for img in sorted(d.rglob("*.jpg")):
            mask_path = img.with_suffix(".png")
            w, h = _image_size(img)
            if clean or not mask_path.exists():
                if not clean:
                    rep.notes.append(f"missing mask {mask_path}")
                boxes, dropped = np.zeros((0, 4)), 0
            else:
                boxes, dropped = mask_to_boxes(_read_mask(mask_path), cfg.get("mask_thr", 127), cfg.get("min_area", 4))
            rep.dropped_small += dropped
            if not clean and not len(boxes):
                # a defect-folder image whose mask has no component >= min_area must not become a "clean" background
                rep.notes.append(f"skipped {d.name}/{img.name}: no mask component >= min_area")
                rep.dropped_degenerate += 1
                continue
            cls = np.full(len(boxes), -1 if clean else lookup[cls_name], int)
            samples.append(Sample(f"{cls_name}_{img.stem}", img, w, h, boxes, cls, {"folder": d.name}))
    rep.n_images, rep.n_boxes = len(samples), sum(len(s.boxes) for s in samples)
    return samples, rep


def read_ksdd2(root: Path, cfg: dict) -> tuple[list[Sample], ReadReport]:
    rep, samples = ReadReport(), []
    for split in ("train", "test"):
        dirs = [d for d in root.rglob(split) if d.is_dir()]
        if not dirs:
            raise FileNotFoundError(f"No '{split}' folder under {root}")
        d = dirs[0]
        for img in sorted(p for p in d.glob("*.png") if not p.stem.endswith("_GT")):
            mask_path = img.with_name(f"{img.stem}_GT.png")
            w, h = _image_size(img)
            if not mask_path.exists():
                # every official KSDD2 image has a mask (all-zero for defect-free parts); an image without one is a
                # stray file (the official zip contains e.g. "10301 (copy).png") -> skipped, never a "clean" image
                rep.notes.append(f"skipped {split}/{img.name}: no mask {mask_path.name}")
                continue
            boxes, dropped = mask_to_boxes(_read_mask(mask_path), cfg.get("mask_thr", 0), cfg.get("min_area", 4))
            rep.dropped_small += dropped
            samples.append(
                Sample(f"{split}_{img.stem}", img, w, h, boxes, np.zeros(len(boxes), int), {"official_split": split})
            )
    rep.n_images, rep.n_boxes = len(samples), sum(len(s.boxes) for s in samples)
    return samples, rep


READERS = {"voc": read_voc, "magnetic_tile": read_magnetic_tile, "ksdd2": read_ksdd2}


def read_dataset(root: Path, cfg: dict) -> tuple[list[Sample], ReadReport]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {root} (edit configs/paths.yaml or set SDD_RAW_*)")
    samples, rep = READERS[cfg["reader"]](root, cfg)
    uids = Counter(s.uid for s in samples)
    dup = [u for u, c in uids.items() if c > 1]
    if dup:
        raise ValueError(f"Non-unique sample ids, e.g. {dup[:5]}")
    log.info("%s: %d images, %d boxes, report=%s", cfg.get("name", ""), rep.n_images, rep.n_boxes, rep)
    return samples, rep
