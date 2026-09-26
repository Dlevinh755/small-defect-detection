"""Faster R-CNN R50-FPN v2 baseline (torchvision), trained on the same processed splits (plan §4.2, M2).

Protocol parity with the Ultralytics runs: COCO-pretrained init, 640 input (``min_size = max_size = 640``),
checkpoint chosen by val COCO-AP, test evaluated once. Recipe = torchvision detection reference
(SGD, multi-step LR, linear warmup, horizontal flip), i.e. "framework default" per plan §3.1.
Labels are shifted by +1 because torchvision reserves 0 for background.
"""

from __future__ import annotations

import csv
import logging
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..data.build import class_names, samples_from_coco
from ..data.readers import Sample

log = logging.getLogger(__name__)


def build_frcnn(nc: int, pretrained: bool = True, imgsz: int = 640, max_det: int = 300):
    kw = dict(min_size=imgsz, max_size=imgsz, box_detections_per_img=max_det, box_score_thresh=0.001)
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None,
                                       weights_backbone=None, **kw)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc + 1)
    return model


def load_frcnn(weights: str | Path, device: str = "cpu"):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = build_frcnn(ckpt["nc"], pretrained=False, imgsz=ckpt.get("imgsz", 640))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def _read_rgb(path: Path) -> torch.Tensor:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise OSError(f"cannot read {path}")
    return torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255)


class DetDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool):
        self.samples, self.train = samples, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = _read_rgb(s.img_path)
        boxes = torch.as_tensor(s.boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(s.classes, dtype=torch.int64) + 1
        if self.train and random.random() < 0.5:
            img = img.flip(-1)
            w = img.shape[-1]
            boxes = torch.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
        return img, {"boxes": boxes, "labels": labels}, s.uid


def _collate(batch):
    return tuple(zip(*batch))


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_frcnn(model, samples: list[Sample], uid_to_id: dict[str, int], device: str, batch: int = 8,
                  workers: int = 2, conf: float = 0.001, amp: bool = True) -> tuple[list[dict], dict]:
    model.eval()
    dl = DataLoader(DetDataset(samples, train=False), batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=_collate)
    dets, t_inf, n = [], 0.0, 0
    for imgs, _, uids in dl:
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", enabled=amp and device.startswith("cuda")):
            out = model(imgs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_inf += time.perf_counter() - t0
        n += len(imgs)
        for uid, o in zip(uids, out):
            keep = o["scores"] >= conf
            for (x1, y1, x2, y2), c, s in zip(o["boxes"][keep].tolist(), o["labels"][keep].tolist(),
                                              o["scores"][keep].tolist()):
                dets.append({"image_id": uid_to_id[uid], "category_id": int(c) - 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return dets, {"inference_ms_per_img_batched": t_inf / max(n, 1) * 1000}


def train_frcnn(data_dir: Path, out_dir: Path, *, epochs: int, batch: int, lr: float, momentum: float,
                weight_decay: float, lr_steps: list[int], warmup_iters: int, eval_every: int, seed: int,
                workers: int, amp: bool, device: str, imgsz: int = 640, fraction: float = 1.0) -> Path:
    """Train and return the path of the best-on-val checkpoint. Resumes from ``last.pt`` if present."""
    from ..data.build import load_coco_json
    from ..evaluation.coco_eval import coco_metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    wdir = out_dir / "weights"
    wdir.mkdir(exist_ok=True)
    _seed_all(seed)
    nc = len(class_names(data_dir))
    train_s = samples_from_coco(data_dir, "train")
    if fraction < 1:
        train_s = random.Random(seed).sample(train_s, max(1, int(len(train_s) * fraction)))
    val_s = samples_from_coco(data_dir, "val")
    val_gt = load_coco_json(data_dir / "coco" / "val.json")
    val_ids = {im["uid"]: im["id"] for im in val_gt["images"]}

    model = build_frcnn(nc, pretrained=True, imgsz=imgsz).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    g = torch.Generator()
    g.manual_seed(seed)
    dl = DataLoader(DetDataset(train_s, train=True), batch_size=batch, shuffle=True, num_workers=workers,
                    collate_fn=_collate, generator=g, drop_last=len(train_s) > batch)

    start, best_ap, it = 0, -1.0, 0
    last = wdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best_ap, it = ck["epoch"] + 1, ck["best_ap"], ck["iter"]
        log.info("Resuming Faster R-CNN from epoch %d", start)

    csv_path = out_dir / "results.csv"
    new_csv = not csv_path.exists()
    with open(csv_path, "a", newline="") as fcsv:
        wr = csv.writer(fcsv)
        if new_csv:
            wr.writerow(["epoch", "loss", "lr", "val_AP", "val_AP50", "time_s"])
        for ep in range(start, epochs):
            model.train()
            t0, tot, nb = time.time(), 0.0, 0
            for imgs, targets, _ in dl:
                imgs = [im.to(device, non_blocking=True) for im in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if it < warmup_iters:  # linear warmup from lr/1000
                    f = 1e-3 + (1 - 1e-3) * it / warmup_iters
                    for pg in opt.param_groups:
                        pg["lr"] = lr * f * (0.1 ** sum(ep >= s for s in lr_steps))
                with torch.autocast("cuda", enabled=scaler.is_enabled()):
                    loss = sum(model(imgs, targets).values())
                if not math.isfinite(loss.item()):
                    raise RuntimeError(f"non-finite loss {loss.item()} at epoch {ep}")
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
                nb += 1
                it += 1
            sched.step()
            ap = ap50 = float("nan")
            if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
                dets, _ = predict_frcnn(model, val_s, val_ids, device, batch=batch, workers=workers, amp=amp)
                m = coco_metrics(val_gt, dets, rel=False, per_class=False)
                ap, ap50 = m["AP"], m["AP50"]
                if ap > best_ap:
                    best_ap = ap
                    torch.save({"model": model.state_dict(), "nc": nc, "imgsz": imgsz, "epoch": ep, "val_AP": ap},
                               wdir / "best.pt")
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "epoch": ep, "best_ap": best_ap, "iter": it, "nc": nc, "imgsz": imgsz}, last)
            wr.writerow([ep, tot / max(nb, 1), opt.param_groups[0]["lr"], ap, ap50, round(time.time() - t0, 1)])
            fcsv.flush()
            log.info("frcnn epoch %d/%d loss %.4f val AP %.4f", ep + 1, epochs, tot / max(nb, 1), ap)
    return wdir / "best.pt"
