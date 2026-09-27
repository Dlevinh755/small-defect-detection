"""Variants must build, run forward + loss + backward, keep strides, and survive checkpoint round-trips."""

import io

import pytest
import torch
from ultralytics.cfg import get_cfg

from sdd.config import variant_cfg
from sdd.models.losses import wasserstein_similarity
from sdd.models.modules import AttnWrap, SPDConv
from sdd.models.yolo_trainer import build_yolo

VARIANTS = ["base", "p2", "p2p4", "simam", "cbam", "ca", "spd", "nwd", "nwd_assign", "wiou", "inner", "focal",
            "clsw", "A7"]


def _batch():
    return {"img": torch.rand(2, 3, 128, 128), "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[1.0], [2.0], [0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.05, 0.05], [0.2, 0.3, 0.1, 0.2], [0.7, 0.7, 0.3, 0.3]])}


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_trains_one_step(name):
    cfg = variant_cfg(name)
    m = build_yolo(cfg["model_cfg"], nc=3, variant=cfg, verbose=False)
    m.args = get_cfg()
    m.train()
    loss, _ = m.loss(_batch())
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert m.model[-1].stride.tolist()[0] == (4.0 if "p2" in cfg["model_cfg"] else 8.0)


def test_variant_inheritance():
    a7 = variant_cfg("A7")
    assert a7["model_cfg"].endswith("yolo11n-p2.yaml") and a7["backbone"] == "simam" and a7["loss"] == "nwd"
    assert variant_cfg("A2")["model_cfg"] == "yolo11n.yaml"


def test_surgery_keeps_pretrained_weights_and_pickles():
    base = build_yolo("yolo11n.yaml", nc=3, variant=variant_cfg("base"), verbose=False)
    m = build_yolo("yolo11n.yaml", nc=3, variant=variant_cfg("simam"), weights=base, verbose=False)
    assert isinstance(m.model[2], AttnWrap)
    torch.testing.assert_close(m.model[2].block.cv1.conv.weight, base.model[2].cv1.conv.weight)
    torch.testing.assert_close(m.model[16].cv1.conv.weight, base.model[16].cv1.conv.weight)
    # a surgered source initialises a surgered target (phase-3 T3 path), even with a different nc
    m2 = build_yolo("yolo11n.yaml", nc=5, variant=variant_cfg("simam"), weights=m, verbose=False)
    torch.testing.assert_close(m2.model[2].block.cv1.conv.weight, m.model[2].block.cv1.conv.weight)
    buf = io.BytesIO()
    torch.save({"model": m}, buf)
    buf.seek(0)
    assert torch.load(buf, weights_only=False)["model"].sdd_cfg["backbone"] == "simam"


def test_spd_shape_and_nwd():
    x = torch.rand(1, 8, 32, 32)
    assert SPDConv(8, 16)(x).shape == (1, 16, 16, 16)
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert torch.isclose(wasserstein_similarity(b, b, 12.8), torch.tensor(1.0), atol=1e-3).all()
    assert wasserstein_similarity(b, b + 4, 12.8) < wasserstein_similarity(b, b + 1, 12.8)


def test_benchmark_input_follows_model_device():
    """Regression: FLOPs were counted with the model on CPU but the input on the benchmark device (crashed on GPU)."""
    from sdd.evaluation.efficiency import benchmark_module

    m = build_yolo("yolo11n.yaml", nc=2, variant=variant_cfg("base"), verbose=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = benchmark_module(m, "ultralytics", imgsz=128, warmup=1, iters=1, device=dev)
    assert out["GFLOPs"] > 0 and out["params_M"] > 0 and out["fps_fp32"] > 0
