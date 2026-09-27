"""Params, GFLOPs @640 and batch-1 latency / FPS (plan §3.2).

GFLOPs use ``torch.utils.flop_counter`` for every framework, so YOLO, RT-DETR and Faster R-CNN are counted the
same way (FLOPs = 2 x MACs, the convention Ultralytics prints). Latency is the model forward pass on a
640x640 input after ``warmup`` iterations; end-to-end latency (pre/post-processing, NMS) is recorded
separately by the prediction exporter.
"""

from __future__ import annotations

import copy
import time
from typing import Callable

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_params(model: torch.nn.Module) -> float:
    """Millions of parameters."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def gflops(forward: Callable[[], object]) -> float:
    with FlopCounterMode(display=False) as fc:
        forward()
    return fc.get_total_flops() / 1e9


@torch.no_grad()
def latency_ms(forward: Callable[[], object], warmup: int = 50, iters: int = 200) -> float:
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        forward()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        forward()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def benchmark_module(model: torch.nn.Module, kind: str, imgsz: int = 640, warmup: int = 50, iters: int = 200,
                     device: str | None = None) -> dict:
    """``kind``: 'ultralytics' (tensor batch input) or 'torchvision' (list-of-images input)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = copy.deepcopy(model).float().eval()
    out = {"params_M": count_params(m)}
    if kind == "ultralytics" and hasattr(m, "fuse"):
        m = m.fuse(verbose=False)  # Conv+BN fusion, as in Ultralytics' own inference benchmarks

    def make(mod, dtype):
        # input on the MODEL's device: FLOPs are counted on a CPU copy while latency runs on the GPU
        x = torch.zeros(1, 3, imgsz, imgsz, device=next(mod.parameters()).device, dtype=dtype)
        if kind == "torchvision":
            return lambda: mod([x[0]])
        return lambda: mod(x)

    out["GFLOPs"] = gflops(make(m.cpu(), torch.float32))
    m = m.to(device)
    if device == "cpu":  # keep CPU benchmarks short (tests / local)
        warmup, iters = min(warmup, 2), min(iters, 5)
    out["latency_fp32_ms"] = latency_ms(make(m, torch.float32), warmup, iters)
    if device != "cpu":
        if kind == "torchvision":
            fwd = make(m, torch.float32)
            with torch.autocast("cuda", dtype=torch.float16):
                out["latency_fp16_ms"] = latency_ms(fwd, warmup, iters)
        else:
            out["latency_fp16_ms"] = latency_ms(make(m.half(), torch.float16), warmup, iters)
    out["fps_fp32"] = 1000 / out["latency_fp32_ms"]
    if "latency_fp16_ms" in out:
        out["fps_fp16"] = 1000 / out["latency_fp16_ms"]
    return out
