"""Unified detection export -> COCO result list, for every framework.

Our own exporter is used instead of Ultralytics ``save_json`` so that image ids and category ids come from
the same ``coco/<split>.json`` as the ground truth (plan §8 risk "image_id mismatch"). Boxes are in
original-image pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.build import load_coco_json, samples_from_coco


def split_files(data_dir: Path, split: str) -> tuple[list[Path], dict[str, int]]:
    coco = load_coco_json(data_dir / "coco" / f"{split}.json")
    files = [data_dir / "images" / split / im["file_name"] for im in coco["images"]]
    return files, {im["uid"]: im["id"] for im in coco["images"]}


def predict_ultralytics(weights: str | Path, family: str, data_dir: Path, split: str, *, imgsz: int, conf: float,
                        iou: float, max_det: int, device: str, half: bool = True, batch: int = 16) -> tuple[list, dict]:
    import sdd.models.yolo_trainer  # noqa: F401  (classes pickled in custom checkpoints)
    from ultralytics import RTDETR, YOLO

    model = (RTDETR if family == "rtdetr" else YOLO)(str(weights))
    files, uid_to_id = split_files(data_dir, split)
    dets, speed = [], {"preprocess": [], "inference": [], "postprocess": []}
    for i in range(0, len(files), batch):
        results = model.predict([str(f) for f in files[i : i + batch]], imgsz=imgsz, conf=conf, iou=iou,
                                max_det=max_det, device=device, quantize=16 if half else None, verbose=False,
                                batch=batch)
        for r in results:
            img_id = uid_to_id[Path(r.path).stem]
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xyxy, b.cls.cpu().numpy(), b.conf.cpu().numpy()):
                dets.append({"image_id": img_id, "category_id": int(c),
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
            for k in speed:
                speed[k].append(r.speed.get(k, np.nan))
    return dets, {f"{k}_ms": float(np.nanmean(v)) for k, v in speed.items() if v}


def merge_detections(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float,
                     max_det: int) -> np.ndarray:
    """Class-wise NMS over detections pooled from overlapping tiles (+ full image). Returns kept indices."""
    import torch
    from torchvision.ops import batched_nms

    if not len(boxes):
        return np.zeros(0, int)
    keep = batched_nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores, dtype=torch.float32),
                       torch.as_tensor(classes), iou)
    return keep[:max_det].numpy()


def predict_ultralytics_sliced(weights: str | Path, family: str, data_dir: Path, split: str, *, imgsz: int,
                               conf: float, iou: float, max_det: int, device: str, half: bool = True, tile: int = 640,
                               overlap: float = 0.2, full_image: bool = True, merge_iou: float = 0.5,
                               batch: int = 16) -> tuple[list, dict]:
    """SAHI-style sliced inference (Akyon et al. 2022): predict on overlapping tiles of the original image
    (and optionally the whole image), shift tile boxes back, merge with class-wise NMS."""
    import time

    import cv2
    import torch

    import sdd.models.yolo_trainer  # noqa: F401
    from ultralytics import RTDETR, YOLO

    from ..data.tiling import tile_windows

    model = (RTDETR if family == "rtdetr" else YOLO)(str(weights))
    files, uid_to_id = split_files(data_dir, split)
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, quantize=16 if half else None,
              verbose=False)
    dets, times, n_tiles = [], [], []
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        wins = tile_windows(w, h, tile, overlap)
        crops = [np.ascontiguousarray(img[y1:y2, x1:x2]) for x1, y1, x2, y2 in wins]
        offsets = [(x1, y1) for x1, y1, _, _ in wins]
        if full_image and len(wins) > 1:
            crops.append(img)
            offsets.append((0, 0))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        B, S, C = [], [], []
        for i in range(0, len(crops), batch):
            for r, (ox, oy) in zip(model.predict(crops[i : i + batch], batch=batch, **kw), offsets[i : i + batch]):
                B.append(r.boxes.xyxy.cpu().numpy() + [ox, oy, ox, oy])
                S.append(r.boxes.conf.cpu().numpy())
                C.append(r.boxes.cls.cpu().numpy().astype(int))
        B, S, C = np.concatenate(B), np.concatenate(S), np.concatenate(C)
        keep = merge_detections(B, S, C, merge_iou, max_det)
        times.append((time.perf_counter() - t0) * 1000)
        n_tiles.append(len(crops))
        img_id = uid_to_id[f.stem]
        for (x1, y1, x2, y2), s, c in zip(B[keep], S[keep], C[keep]):
            dets.append({"image_id": img_id, "category_id": int(c),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return dets, {"sliced_total_ms": float(np.mean(times)), "sliced_crops_per_img": float(np.mean(n_tiles))}


def predict_frcnn_split(weights: str | Path, data_dir: Path, split: str, *, conf: float, device: str,
                        batch: int = 8, workers: int = 2) -> tuple[list, dict]:
    from ..models.frcnn import load_frcnn, predict_frcnn

    model = load_frcnn(weights, device)
    _, uid_to_id = split_files(data_dir, split)
    return predict_frcnn(model, samples_from_coco(data_dir, split), uid_to_id, device, batch=batch,
                         workers=workers, conf=conf)
