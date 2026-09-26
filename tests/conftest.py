"""Synthetic raw datasets + isolated work/results/splits dirs, so tests never touch real data or splits/."""

import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_TMP = Path(os.environ.setdefault("SDD_TEST_TMP", str(Path(__file__).resolve().parents[1] / "work" / "_pytest")))
shutil.rmtree(_TMP, ignore_errors=True)  # fresh synthetic data + splits every session
os.environ["SDD_PROFILE"] = "local"
os.environ["SDD_WORK"] = str(_TMP / "work")
os.environ["SDD_RESULTS"] = str(_TMP / "results")
os.environ["SDD_SPLITS"] = str(_TMP / "splits")
os.environ["SDD_RAW_NEU"] = str(_TMP / "raw" / "neu")
os.environ["SDD_RAW_KSDD2"] = str(_TMP / "raw" / "ksdd2")
os.environ["SDD_RAW_GC10"] = str(_TMP / "raw" / "gc10")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def _voc(path: Path, w: int, h: int, objs):
    ann = ET.Element("annotation")
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text, ET.SubElement(size, "height").text = str(w), str(h)
    for name, (x1, y1, x2, y2) in objs:
        o = ET.SubElement(ann, "object")
        ET.SubElement(o, "name").text = name
        b = ET.SubElement(o, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2)):
            ET.SubElement(b, k).text = str(v)
    ET.ElementTree(ann).write(path)


def make_neu(root: Path, per_class: int = 5, size: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    (root / "IMAGES").mkdir(parents=True, exist_ok=True)
    (root / "ANNOTATIONS").mkdir(parents=True, exist_ok=True)
    for ci, c in enumerate(NEU_CLASSES):
        for k in range(per_class):
            img = np.full((size, size, 3), 120, np.uint8)
            objs = []
            for _ in range(int(rng.integers(1, 3))):
                w, h = int(rng.integers(6, 60)), int(rng.integers(6, 60))
                x, y = int(rng.integers(1, size - w)), int(rng.integers(1, size - h))
                cv2.rectangle(img, (x, y), (x + w, y + h), (30 + 30 * ci, 200, 60), -1)
                objs.append((c, (x + 1, y + 1, x + w, y + h)))  # VOC 1-based inclusive
            stem = f"{c}_{k + 1}"
            cv2.imwrite(str(root / "IMAGES" / f"{stem}.jpg"), img)
            _voc(root / "ANNOTATIONS" / f"{stem}.xml", size, size, objs)


def make_ksdd2(root: Path, n_train: int = 12, n_test: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("test", n_test)):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((630, 230, 3), 90, np.uint8)
            mask = np.zeros((630, 230), np.uint8)
            if i % 3 == 0:  # defective: one real blob + one 2-px speck (dropped by min_area)
                x, y = int(rng.integers(10, 180)), int(rng.integers(10, 580))
                mask[y : y + 20, x : x + 8] = 255
                mask[5, 5:7] = 255
                img[mask > 0] = 200
            cv2.imwrite(str(d / f"{split}{i:04d}.png"), img)
            cv2.imwrite(str(d / f"{split}{i:04d}_GT.png"), mask)


GC10_COUNTS = {"1_chongkong": 24, "2_hanfeng": 12, "3_yueyawan": 5, "4_shuiban": 3}  # imbalanced on purpose


def make_gc10(root: Path, seed: int = 0):
    """Imbalanced VOC dataset with GC10 raw class names, non-square 400x300 textured images, small defects."""
    rng = np.random.default_rng(seed)
    (root / "lable").mkdir(parents=True, exist_ok=True)
    k = 0
    for ci, (name, n) in enumerate(GC10_COUNTS.items()):
        for _ in range(n):
            img = rng.integers(90, 140, (300, 400, 3), dtype=np.uint8)
            objs = []
            for _ in range(int(rng.integers(1, 3))):
                w, h = int(rng.integers(8, 30)), int(rng.integers(8, 30))
                x, y = int(rng.integers(1, 400 - w)), int(rng.integers(1, 300 - h))
                cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (20 + 50 * ci, 230, 40), -1)
                objs.append((name, (x + 1, y + 1, x + w, y + h)))
            k += 1
            d = root / str(ci + 1)
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"img_{k:03d}.jpg"), img)
            _voc(root / "lable" / f"img_{k:03d}.xml", 400, 300, objs)


@pytest.fixture(scope="session")
def raw_data():
    make_neu(Path(os.environ["SDD_RAW_NEU"]))
    make_ksdd2(Path(os.environ["SDD_RAW_KSDD2"]))
    make_gc10(Path(os.environ["SDD_RAW_GC10"]))
    return _TMP
