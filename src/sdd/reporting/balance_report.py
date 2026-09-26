"""QA and slide material for a balanced dataset (plan §2.5.5 "Kiểm tra chất lượng"):

* ``balance_counts.png`` / ``.csv``: images and boxes per class before / after (rare classes marked *)
* ``qa_aug.jpg`` / ``qa_cp.jpg``: generated images with their boxes drawn (15 each by default) - check that boxes
  follow the defects after flips / rotations and that pasted patches show no seam
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..data.balance import balance_table
from ..data.build import samples_from_coco
from ..data.visualize import draw_boxes
from . import style
from .style import save


def counts_figure(meta: dict):
    t = pd.DataFrame(balance_table(meta))
    labels = [f"{c} *" if r else c for c, r in zip(t["class"], t.rare)]
    y = np.arange(len(t))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(t) + 1.6), sharey=True)
    for ax, what in zip(axes, ("images", "boxes")):
        h = 0.38
        ax.barh(y + h / 2, t[f"{what}_before"], h, color=style.CATEGORICAL[0], label="original train")
        ax.barh(y - h / 2, t[f"{what}_after"], h, color=style.CATEGORICAL[1], label=meta["dataset"])
        for yy, v in zip(y - h / 2, t[f"{what}_after"]):
            ax.text(v, yy, f" {v}", va="center", fontsize=7, color=style.INK_2)
        ax.set_title(f"{what} per class", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.suptitle(f"{meta['source'].upper()}: train split before / after balancing (+{meta['increase']:.0%} images, "
                 "* = rare class)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def _grid(tiles: list[np.ndarray], cols: int = 5, size: int = 220) -> np.ndarray:
    fitted = []
    for t in tiles:
        s = size / max(t.shape[:2])
        t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        c = np.full((size, size, 3), 250, np.uint8)
        c[: t.shape[0], : t.shape[1]] = t if t.ndim == 3 else cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        fitted.append(c)
    fitted += [np.full((size, size, 3), 250, np.uint8)] * (-len(fitted) % cols)
    return np.vstack([np.hstack(fitted[i : i + cols]) for i in range(0, len(fitted), cols)])


def qa_images(data_dir: Path, out: Path, n: int = 15, seed: int = 0) -> dict:
    meta = json.loads((data_dir / "meta.json").read_text())
    names = meta["classes"]
    gen_path = data_dir / "generated.csv"
    origin = {r["uid"]: r["origin"] for r in csv.DictReader(open(gen_path))} if gen_path.exists() else {}
    by_uid = {s.uid: s for s in samples_from_coco(data_dir, "train") if s.uid in origin}
    written = {}
    for kind in ("aug", "cp", "rfs"):
        uids = sorted(u for u, o in origin.items() if o == kind and u in by_uid)
        if not uids:
            continue
        pick = random.Random(seed).sample(uids, min(n, len(uids)))
        tiles = []
        for u in pick:
            s = by_uid[u]
            img = cv2.imread(str(s.img_path), cv2.IMREAD_COLOR)
            if img is not None:
                tiles.append(draw_boxes(img, s.boxes, s.classes, names))
        if tiles:
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"qa_{kind}.jpg"
            cv2.imwrite(str(p), _grid(tiles))
            written[kind] = p
    return written


def balance_report(data_dir: Path, out: Path, n_qa: int = 15) -> dict:
    meta = json.loads((data_dir / "meta.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(balance_table(meta)).to_csv(out / "balance_counts.csv", index=False)
    save(counts_figure(meta), out / "balance_counts.png")
    qa = qa_images(data_dir, out, n_qa)
    return {"increase": meta["increase"], "generated": meta["generated"], "applied": meta["applied"],
            "rare": meta["before"]["rare_classes"], "notes": meta["notes"], "qa": {k: str(v) for k, v in qa.items()}}
