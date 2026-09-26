"""Exploratory data analysis for the phase-1 report (plan §2.4).

Everything is computed from the processed COCO json (original-pixel boxes), so EDA sees exactly what the
models are evaluated on. Outputs go to ``<results>/eda/<ds>/``: CSV tables + PNG figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..data.build import class_names, load_coco_json
from ..evaluation.sizes import ABS_NAMES, abs_group, rel_bins, rel_group
from ..reporting import style
from ..reporting.style import save

SPLITS = ("train", "val", "test")


def box_table(data_dir: Path, imgsz: int = 640) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per box and one row per image, over all splits."""
    names = class_names(data_dir)
    bins, rnames = rel_bins()
    boxes, images = [], []
    for split in SPLITS:
        coco = load_coco_json(data_dir / "coco" / f"{split}.json")
        ims = {im["id"]: im for im in coco["images"]}
        counts = {i: 0 for i in ims}
        for a in coco["annotations"]:
            im = ims[a["image_id"]]
            counts[a["image_id"]] += 1
            x, y, w, h = a["bbox"]
            boxes.append({"split": split, "uid": im["uid"], "cls": a["category_id"], "x": x, "y": y, "w": w, "h": h,
                          "img_w": im["width"], "img_h": im["height"]})
        images += [{"split": split, "uid": ims[i]["uid"], "img_w": ims[i]["width"], "img_h": ims[i]["height"],
                    "n_boxes": n} for i, n in counts.items()]
    b = pd.DataFrame(boxes)
    img = pd.DataFrame(images)
    if b.empty:
        return b, img
    b["class"] = b["cls"].map(dict(enumerate(names)))
    b["area"] = b.w * b.h
    b["rel_area"] = b.area / (b.img_w * b.img_h)
    b["aspect"] = np.maximum(b.w, b.h) / np.maximum(np.minimum(b.w, b.h), 1e-6)
    scale = imgsz / np.maximum(b.img_w, b.img_h)  # letterbox resize used by all models
    b["w_640"], b["h_640"] = b.w * scale, b.h * scale
    b["area_640"] = b.w_640 * b.h_640
    b["abs_size"] = pd.Categorical(np.array(ABS_NAMES)[abs_group(b.area)], ABS_NAMES)
    b["abs_size_640"] = pd.Categorical(np.array(ABS_NAMES)[abs_group(b.area_640)], ABS_NAMES)
    b["rel_size"] = pd.Categorical(np.array(rnames)[rel_group(b.rel_area, bins)], rnames)
    return b, img


def summary(b: pd.DataFrame, img: pd.DataFrame, dataset: str) -> dict:
    out = {
        "dataset": dataset,
        "images": len(img),
        "clean_images": int((img.n_boxes == 0).sum()),
        "boxes": len(b),
        "classes": int(b["cls"].nunique()) if len(b) else 0,
        "boxes_per_img_mean": round(float(img.n_boxes.mean()), 2),
        "boxes_per_img_max": int(img.n_boxes.max()),
        "img_size": ", ".join(sorted({f"{w}x{h}" for w, h in zip(img.img_w, img.img_h)})[:3]),
    }
    for col in ("abs_size", "abs_size_640", "rel_size"):
        for k, v in b[col].value_counts(normalize=True).items():
            out[f"{col}:{k}"] = round(float(v), 3)
    for split in SPLITS:
        out[f"{split}_images"] = int((img.split == split).sum())
    return out


def suggest_rel_bins(b: pd.DataFrame, candidates=None, split: str = "test") -> pd.DataFrame:
    """Test-set box counts per group for candidate relative bins (choose one with >= ~50 boxes/group)."""
    candidates = candidates or [
        [0, 0.001, 0.01, 1], [0, 0.0025, 0.01, 1], [0, 0.005, 0.02, 1], [0, 0.01, 0.05, 1],
    ]
    t = b[b.split == split]
    rows = []
    for c in candidates:
        counts = np.bincount(rel_group(t.rel_area, c), minlength=len(c) - 1)
        rows.append({"bins": str(c), **{f"g{i}": int(n) for i, n in enumerate(counts)}})
    q = t.rel_area.quantile([1 / 3, 2 / 3]).round(5).tolist()
    rows.append({"bins": f"tertiles {[0, *q, 1]}", **{f"g{i}": len(t) // 3 for i in range(3)}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- figures
def fig_class_distribution(b, img, names, title):
    counts = b["class"].value_counts().reindex(names, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1.2))
    ax.barh(counts.index[::-1], counts.values[::-1], color=style.CATEGORICAL[0], height=0.6)
    for y, v in enumerate(counts.values[::-1]):
        ax.text(v, y, f" {v}", va="center", fontsize=8, color=style.INK_2)
    ax.set_xlabel("boxes")
    ax.grid(axis="y", visible=False)
    n_clean = int((img.n_boxes == 0).sum())
    ax.set_title(f"{title}: boxes per class" + (f"  (+{n_clean} defect-free images)" if n_clean else ""))
    return fig


def fig_area_hist(b, title, imgsz=640):
    """Absolute area before/after resize to 640 (log x) + relative area, with the COCO/relative thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    lo = max(b[["area", "area_640"]].min().min(), 1)
    hi = b[["area", "area_640"]].max().max()
    edges = np.logspace(np.log10(lo), np.log10(hi), 40)
    ax = axes[0]
    # outlines, not translucent fills: overlapping fills would blend into a third, meaningless colour
    ax.hist(b.area, bins=edges, histtype="step", lw=2, color=style.CATEGORICAL[0], label="original resolution")
    if not np.allclose(b.area, b.area_640):
        ax.hist(b.area_640, bins=edges, histtype="step", lw=2, color=style.CATEGORICAL[1],
                label=f"after resize to {imgsz}")
    for t, lbl in ((32**2, "32²"), (96**2, "96²")):
        ax.axvline(t, color=style.MUTED, lw=1, ls="--")
        ax.text(t, ax.get_ylim()[1] * 0.95, f" {lbl}", color=style.INK_2, fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("box area (px², log)")
    ax.set_ylabel("boxes")
    ax.set_title(f"{title}: absolute box area", pad=24)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)

    ax = axes[1]
    rel = b.rel_area.clip(lower=1e-6)
    ax.hist(rel, bins=np.logspace(np.log10(rel.min()), 0, 40), color=style.CATEGORICAL[0], alpha=0.85)
    for t in rel_bins()[0][1:-1]:
        ax.axvline(t, color=style.MUTED, lw=1, ls="--")
        ax.text(t, ax.get_ylim()[1] * 0.95, f" {t:g}", color=style.INK_2, fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("box area / image area (log)")
    ax.set_title("relative box area", pad=24)
    fig.tight_layout()
    return fig


def fig_size_by_class(b, names, col, title):
    """Share of small/medium/large boxes per class (ordinal -> one-hue ramp)."""
    tab = pd.crosstab(b["class"], b[col], normalize="index").reindex(names).fillna(0)
    cats = list(b[col].cat.categories)
    ramp = [style.SEQUENTIAL[i] for i in np.linspace(0, len(style.SEQUENTIAL) - 1, len(cats)).round().astype(int)][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.4))
    left = np.zeros(len(tab))
    for c, colr in zip(cats, ramp):
        v = tab[c].values if c in tab else np.zeros(len(tab))
        ax.barh(tab.index[::-1], v[::-1], left=left[::-1], color=colr, height=0.6, label=c,
                edgecolor=style.SURFACE, linewidth=1)
        left += v
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of boxes")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=len(cats), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title(f"{title}: {col.replace('_', ' ')} per class", pad=22)
    return fig


def fig_aspect(b, title):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    a = b.aspect.clip(upper=50)
    ax.hist(a, bins=np.logspace(0, np.log10(max(a.max(), 1.01)), 30), color=style.CATEGORICAL[0])
    ax.set_xscale("log")
    ax.set_xlabel("box aspect ratio max(w,h)/min(w,h) (log)")
    ax.set_ylabel("boxes")
    ax.set_title(f"{title}: box elongation")
    return fig


def smallest_per_class(data_dir: Path, b: pd.DataFrame, out: Path, pad: int = 24, zoom: int = 128) -> list[Path]:
    """Crop the smallest box of every class (context padding, upscaled) for the slides."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for cls_name, g in b.groupby("class"):
        r = g.loc[g.area.idxmin()]
        files = list((data_dir / "images" / r.split).glob(f"{r.uid}.*"))
        if not files:
            continue
        im = cv2.imread(str(files[0]))
        x1, y1 = int(max(r.x - pad, 0)), int(max(r.y - pad, 0))
        x2, y2 = int(min(r.x + r.w + pad, r.img_w)), int(min(r.y + r.h + pad, r.img_h))
        crop = im[y1:y2, x1:x2].copy()
        cv2.rectangle(crop, (int(r.x - x1), int(r.y - y1)), (int(r.x - x1 + r.w), int(r.y - y1 + r.h)), (56, 56, 255), 1)
        s = zoom / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        p = out / f"smallest_{str(cls_name).replace('/', '_')}_{r.w:.0f}x{r.h:.0f}.png"
        cv2.imwrite(str(p), crop)
        written.append(p)
    return written


def run_eda(data_dir: Path, out_dir: Path, dataset: str) -> dict:
    names = class_names(data_dir)
    b, img = box_table(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b.to_csv(out_dir / "boxes.csv", index=False)
    s = summary(b, img, dataset)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    suggest_rel_bins(b).to_csv(out_dir / "rel_bin_candidates.csv", index=False)
    per_class = b.groupby("class").agg(boxes=("area", "size"), area_median=("area", "median"),
                                       rel_area_median=("rel_area", "median"), aspect_median=("aspect", "median"))
    per_class.to_csv(out_dir / "per_class.csv")
    title = dataset.upper()
    save(fig_class_distribution(b, img, names, title), out_dir / "class_distribution.png")
    save(fig_area_hist(b, title), out_dir / "area_hist.png")
    save(fig_size_by_class(b, names, "abs_size", title), out_dir / "abs_size_by_class.png")
    save(fig_size_by_class(b, names, "abs_size_640", title), out_dir / "abs_size_640_by_class.png")
    save(fig_size_by_class(b, names, "rel_size", title), out_dir / "rel_size_by_class.png")
    save(fig_aspect(b, title), out_dir / "aspect_ratio.png")
    smallest_per_class(data_dir, b, out_dir / "smallest")
    return s
