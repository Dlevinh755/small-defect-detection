"""Offline class-imbalance handling on the TRAIN split (plan §2.5, Appendix A.5-A.7).

``balance_dataset("gc10", "bal", 1)`` builds ``<work>/data/gc10_bal_v1`` from the built ``gc10`` dataset:

* ``bal`` - class-aware augmentation + copy-paste of rare defects (main phase-1 data)
* ``aug`` - class-aware augmentation only (ablation: which of the two helps)
* ``rfs`` - plain repeat-factor sampling: identical copies (fallback)

Rules (plan §2.5.5): generated once with a fixed seed; only train is changed and copy-paste sources come from train
only; val/test are the untouched splits of the source dataset; no downscaling / random crop (small defects must
survive); at most ``max_increase`` (+50%) new train images.

Class-aware augmentation: every train image with a defect gets ``round(max_c r_c) - 1`` extra copies (stochastic
rounding, seeded), ``r_c = min(cap, max(1, sqrt(t / f_c)))`` with ``f_c`` the share of train images containing class c
(repeat factor, Gupta et al. 2019). Each copy uses a different geometric op + a random photometric change, so no
image is duplicated verbatim.

Copy-paste: rare-class defects are cut with a ``margin`` px background ring and pasted, at native size, onto other
train images where they do not overlap existing boxes; the patch background is shifted to the brightness of the
target region and the ring is feathered, so no hard seam is created. Classes are filled up to ``target_ratio`` x the
image count of the largest class, within the remaining budget.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import re
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from ..config import dataset_cfg, protocol
from ..env import paths
from .build import build_dataset, class_names, samples_from_coco, write_data_yaml, write_split
from .readers import Sample

log = logging.getLogger(__name__)
BALANCED_RE = re.compile(r"^(?P<base>.+)_(?P<method>bal|aug|rfs)_v(?P<version>\d+)$")
GEO_OPS = ("hflip", "vflip", "rot90", "rot180")


def parse_balanced(name: str) -> tuple[str, str, int] | None:
    m = BALANCED_RE.match(name)
    return (m["base"], m["method"], int(m["version"])) if m else None


# ------------------------------------------------------------------------------------------ statistics
def class_counts(samples: list[Sample], nc: int) -> tuple[np.ndarray, np.ndarray]:
    """(boxes per class, images containing each class)."""
    boxes, images = np.zeros(nc, int), np.zeros(nc, int)
    for s in samples:
        np.add.at(boxes, s.classes, 1)
        np.add.at(images, np.unique(s.classes), 1)
    return boxes, images


def imbalance_summary(samples: list[Sample], names: list[str], rare_ratio: float) -> dict:
    """Imbalance of an (original) train split: box counts, max/min ratio, rare classes (plan §2.5.2)."""
    boxes, images = class_counts(samples, len(names))
    present = boxes > 0
    ratio = float(boxes.max() / boxes[present].min()) if present.any() else float("nan")
    rare = [names[i] for i in range(len(names)) if present[i] and boxes[i] < rare_ratio * boxes.max()]
    return {
        "boxes_per_class": dict(zip(names, boxes.tolist())),
        "images_per_class": dict(zip(names, images.tolist())),
        "max_class": names[int(boxes.argmax())],
        "min_class": names[int(np.where(present, boxes, boxes.max() + 1).argmin())] if present.any() else None,
        "max_min_ratio": ratio,
        "rare_classes": rare,
        "absent_classes": [names[i] for i in range(len(names)) if not present[i]],
        "defect_images": int(sum(len(s.boxes) > 0 for s in samples)),
        "clean_images": int(sum(len(s.boxes) == 0 for s in samples)),
    }


def repeat_factors(samples: list[Sample], nc: int, t: float, cap: float) -> np.ndarray:
    _, images = class_counts(samples, nc)
    f = images / max(len(samples), 1)
    with np.errstate(divide="ignore"):
        r = np.sqrt(t / np.where(f > 0, f, np.inf))
    return np.clip(r, 1.0, cap)


# ------------------------------------------------------------------------------------------ transforms
def geo_transform(img: np.ndarray, boxes: np.ndarray, op: str) -> tuple[np.ndarray, np.ndarray]:
    """Box-preserving geometric op (no resize). ``boxes`` (N, 4) xyxy pixels."""
    H, W = img.shape[:2]
    b = np.asarray(boxes, float).reshape(-1, 4)
    x1, y1, x2, y2 = b.T
    if op == "hflip":
        img, nb = img[:, ::-1], np.stack([W - x2, y1, W - x1, y2], 1)
    elif op == "vflip":
        img, nb = img[::-1, :], np.stack([x1, H - y2, x2, H - y1], 1)
    elif op == "rot90":  # 90 deg counter-clockwise: (x, y) -> (y, W - x)
        img, nb = np.rot90(img, 1), np.stack([y1, W - x2, y2, W - x1], 1)
    elif op == "rot180":
        img, nb = img[::-1, ::-1], np.stack([W - x2, H - y2, W - x1, H - y1], 1)
    elif op == "identity":
        nb = b
    else:
        raise ValueError(f"unknown geometric op {op}")
    return np.ascontiguousarray(img), nb.reshape(-1, 4)


def photo_transform(img: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Contrast / brightness / gamma / light noise / light blur on uint8 images (plan §2.5.5)."""
    if img.dtype != np.uint8:
        return img
    f = img.astype(np.float32) * rng.uniform(*cfg["contrast"]) + rng.uniform(-cfg["brightness"], cfg["brightness"])
    if rng.random() < cfg["p_gamma"]:
        f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, rng.uniform(*cfg["gamma"]))
    if rng.random() < cfg["p_noise"]:
        f = f + rng.normal(0, rng.uniform(*cfg["noise_sigma"]), f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < cfg["p_blur"]:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def _read(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cannot read {path}")
    return img


def _match_channels(patch: np.ndarray, img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and patch.ndim == 2:
        return cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    if img.ndim == 2 and patch.ndim == 3:
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return patch


def feather_mask(h: int, w: int, box, margin: int) -> np.ndarray:
    """1 over the defect box, decaying linearly to 0 across the background ring."""
    if margin <= 0:
        return np.ones((h, w), np.float32)
    inside = np.zeros((h, w), np.uint8)
    bx1, by1 = int(box[0]), int(box[1])
    bx2, by2 = int(math.ceil(box[2])), int(math.ceil(box[3]))
    inside[by1:by2, bx1:bx2] = 1
    dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3)
    return np.clip(1.0 - dist / margin, 0.0, 1.0).astype(np.float32)


def paste(img: np.ndarray, patch: np.ndarray, inner_box, ox: int, oy: int, margin: int) -> np.ndarray:
    """Blend ``patch`` into ``img`` at (ox, oy); the patch background is shifted to the target brightness."""
    patch = _match_channels(patch, img)
    ph, pw = patch.shape[:2]
    region = img[oy : oy + ph, ox : ox + pw].astype(np.float32)
    pf = patch.astype(np.float32)
    ring = np.ones((ph, pw), bool)
    ring[int(inner_box[1]) : int(math.ceil(inner_box[3])), int(inner_box[0]) : int(math.ceil(inner_box[2]))] = False
    if ring.any():
        pf = pf + (region.mean() - pf[ring].mean())
    m = feather_mask(ph, pw, inner_box, margin)
    if img.ndim == 3:
        m = m[..., None]
    out = img.copy()
    out[oy : oy + ph, ox : ox + pw] = np.clip(m * pf + (1 - m) * region, 0, 255).astype(img.dtype)
    return out


def _free(cand, boxes: np.ndarray, gap: int) -> bool:
    if not len(boxes):
        return True
    b = np.asarray(boxes)
    sep = ((cand[2] + gap <= b[:, 0]) | (b[:, 2] + gap <= cand[0]) | (cand[3] + gap <= b[:, 1]) | (b[:, 3] + gap <= cand[1]))
    return bool(sep.all())


# ------------------------------------------------------------------------------------------ generators
def plan_augmentation(samples: list[Sample], r_c: np.ndarray, rng: random.Random) -> list[tuple[int, int, float]]:
    """Jobs (sample index, copy number, priority) from repeat factors with seeded stochastic rounding."""
    jobs = []
    for i, s in enumerate(samples):
        if not len(s.classes):
            continue
        r = float(r_c[s.classes].max())
        extra = r - 1.0
        n = int(extra) + int(rng.random() < extra - int(extra))
        jobs += [(i, k, r) for k in range(1, n + 1)]
    return jobs


def class_aware_augment(samples: list[Sample], jobs, out_dir: Path, geo_ops, photo_cfg: dict, seed: int,
                        identical: bool = False) -> list[Sample]:
    """Generate the planned copies. ``identical=True`` -> plain repeat-factor sampling (same file, new name)."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ops_per_sample: dict[int, list[str]] = {}
    new = []
    for i, k, _ in jobs:
        s = samples[i]
        uid = f"{s.uid}__{'rep' if identical else 'aug'}{k}"
        if identical:
            new.append(Sample(uid, s.img_path, s.width, s.height, s.boxes, s.classes,
                              {"origin": "rfs", "source": s.uid, "op": "identity"}))
            continue
        ops = ops_per_sample.setdefault(i, list(rng.permutation(list(geo_ops))))
        op = ops[(k - 1) % len(ops)]
        img, boxes = geo_transform(_read(s.img_path), s.boxes, op)
        img = photo_transform(img, rng, photo_cfg)
        path = out_dir / f"{uid}{s.img_path.suffix.lower()}"
        cv2.imwrite(str(path), img)
        new.append(Sample(uid, path, img.shape[1], img.shape[0], boxes, s.classes.copy(),
                          {"origin": "aug", "source": s.uid, "op": op}))
    return new


def copy_paste_rare(samples: list[Sample], rare: list[int], deficit: dict[int, int], budget: int, out_dir: Path,
                    cfg: dict, seed: int) -> list[Sample]:
    """Paste rare-class defects onto other train images until each rare class's deficit (images) is filled."""
    rng = random.Random(seed)
    margin, gap = cfg["margin"], cfg["gap"]
    pool: dict[int, list] = {c: [] for c in rare}
    for s in samples:
        if not any(c in pool for c in s.classes.tolist()):
            continue
        img = None
        for (x1, y1, x2, y2), c in zip(s.boxes, s.classes):
            if c not in pool or (x2 - x1) * (y2 - y1) > cfg["max_rel_area"] * s.width * s.height:
                continue
            img = _read(s.img_path) if img is None else img
            X1, Y1 = int(max(0, x1 - margin)), int(max(0, y1 - margin))
            X2, Y2 = int(min(s.width, math.ceil(x2) + margin)), int(min(s.height, math.ceil(y2) + margin))
            pool[int(c)].append((img[Y1:Y2, X1:X2].copy(), (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1), s.uid))
    pool = {c: v for c, v in pool.items() if v}
    deficit = {c: d for c, d in deficit.items() if c in pool and d > 0}
    if not deficit:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    new, attempts = [], 0
    while deficit and len(new) < budget and attempts < budget * 5:
        attempts += 1
        target = rng.choice(samples)
        img = _read(target.img_path)
        boxes, classes = [list(b) for b in target.boxes], target.classes.tolist()
        wanted = sorted(deficit, key=lambda c: -deficit[c])
        chosen = [wanted[0]] + rng.sample(wanted[1:], min(len(wanted) - 1, rng.randint(0, cfg["max_paste"] - 1)))
        pasted = []
        for c in chosen:
            patch, inner, src = rng.choice(pool[c])
            ph, pw = patch.shape[:2]
            if src == target.uid or ph >= target.height or pw >= target.width:
                continue
            for _ in range(20):
                ox, oy = rng.randint(0, target.width - pw), rng.randint(0, target.height - ph)
                if _free((ox, oy, ox + pw, oy + ph), np.array(boxes).reshape(-1, 4), gap):
                    break
            else:
                continue
            img = paste(img, patch, inner, ox, oy, margin)
            boxes.append([ox + inner[0], oy + inner[1], ox + inner[2], oy + inner[3]])
            classes.append(c)
            pasted.append(c)
        if not pasted:
            continue
        uid = f"{target.uid}__cp{len(new)}"
        path = out_dir / f"{uid}{target.img_path.suffix.lower()}"
        cv2.imwrite(str(path), img)
        new.append(Sample(uid, path, target.width, target.height, np.array(boxes, float), np.array(classes, int),
                          {"origin": "cp", "source": target.uid, "op": "copy_paste:" + ",".join(map(str, pasted))}))
        for c in set(pasted):
            deficit[c] -= 1
            if deficit[c] <= 0:
                del deficit[c]
    return new


# ------------------------------------------------------------------------------------------ dataset
def balance_dataset(base: str, method: str = "bal", version: int = 1, force: bool = False) -> Path:
    name = f"{base}_{method}_v{version}"
    out = paths().data_dir(name)
    if (out / "meta.json").exists() and not force:
        return out
    if out.exists():
        shutil.rmtree(out)
    B, dcfg = protocol()["balance"], dataset_cfg(base).get("balance", {"mode": "auto"})
    src_dir = build_dataset(base)
    names = class_names(src_dir)
    nc = len(names)
    train = samples_from_coco(src_dir, "train")
    before = imbalance_summary(train, names, B["rare_ratio"])
    mode = {True: "on", False: "off"}.get(dcfg.get("mode", "auto"), dcfg.get("mode", "auto"))  # YAML: on -> True
    apply = mode == "on" or (mode == "auto" and before["max_min_ratio"] >= B["imbalance_threshold"])
    budget = int(B["max_increase"] * len(train))
    rng = random.Random(B["seed"])
    generated: list[Sample] = []
    notes = []
    if not apply:
        notes.append(f"mode={mode}, max/min={before['max_min_ratio']:.2f} < {B['imbalance_threshold']}: train unchanged")
    else:
        use_cp = method == "bal" and dcfg.get("copy_paste", False)
        r_c = repeat_factors(train, nc, B["t"], B["cap"])
        jobs = plan_augmentation(train, r_c, rng)
        aug_budget = budget // 2 if use_cp else budget
        if len(jobs) > aug_budget:  # keep the rarest-first jobs (ties in seeded order)
            rng.shuffle(jobs)
            jobs = sorted(jobs, key=lambda j: -j[2])[:aug_budget]
            notes.append(f"augmentation capped at {aug_budget} images (budget)")
        generated += class_aware_augment(train, jobs, out / "_gen" / "train", dcfg.get("geo_ops", GEO_OPS),
                                         B["photometric"], B["seed"], identical=method == "rfs")
        if use_cp:
            _, imgs_now = class_counts(train + generated, nc)
            target = int(math.ceil(B["target_ratio"] * imgs_now.max()))
            rare = [names.index(c) for c in before["rare_classes"]]
            deficit = {c: target - int(imgs_now[c]) for c in rare}
            generated += copy_paste_rare(train, rare, deficit, budget - len(generated), out / "_gen" / "train",
                                         B["copy_paste"], B["seed"])
        report = {"repeat_factor": dict(zip(names, np.round(r_c, 2).tolist()))}
    write_split(out, "train", train + generated, names)
    for split in ("val", "test"):
        write_split(out, split, samples_from_coco(src_dir, split), names)
    write_data_yaml(out, names)
    after = imbalance_summary(train + generated, names, B["rare_ratio"])
    origin = Counter(s.meta.get("origin") for s in generated)
    meta = {
        "dataset": name, "source": base, "method": method, "version": version, "classes": names,
        "built": str(date.today()), "applied": apply, "notes": notes,
        "config": {"protocol": B, "dataset": dcfg},
        "split_sizes": {"train": len(train) + len(generated), "val": len(samples_from_coco(src_dir, "val")),
                        "test": len(samples_from_coco(src_dir, "test"))},
        "train_images_original": len(train), "generated": dict(origin),
        "increase": round(len(generated) / max(len(train), 1), 3),
        "before": before, "after": after, **(report if apply else {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    with open(out / "generated.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "origin", "source", "op", "boxes"])
        for s in generated:
            w.writerow([s.uid, s.meta["origin"], s.meta["source"], s.meta["op"], len(s.boxes)])
    log.info("%s: +%d train images (%s), rare=%s", name, len(generated), dict(origin), before["rare_classes"])
    return out


def balance_table(meta: dict) -> list[dict]:
    """Before / after images and boxes per class (for the slides, plan §2.5.5 QA)."""
    b, a = meta["before"], meta["after"]
    return [{"class": c, "images_before": b["images_per_class"][c], "images_after": a["images_per_class"][c],
             "boxes_before": b["boxes_per_class"][c], "boxes_after": a["boxes_per_class"][c],
             "rare": c in b["rare_classes"]} for c in meta["classes"]]
