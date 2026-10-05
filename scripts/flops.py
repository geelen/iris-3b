"""Analytic parameter / FLOPs counter for a model preset.

Runs one forward (batch 1) on the meta device, so nothing is allocated or
computed, with attention kernels stubbed out and counting hooks supplying the
arithmetic:
- every Linear: 2 * tokens * in_features * out_features;
- every self-attention: 4 * B * N^2 * dim at its own width;
- every joint or single-stream attention: 4 * B * (Nx + Ny)^2 * dim.
Norms, activations, interpolation, and rotary phase math are not counted.

Usage:
    python scripts/flops.py --height 256 --width 256
    python scripts/flops.py --preset iris-3b --height 1024 --width 1024
"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch import nn

from iris3b.models.blocks.single_stream import SingleStreamBlock
from iris3b.models.dit import IrisDiT
from iris3b.models.presets import PRESETS, get_preset
from iris3b.nn.attention import JointAttention, SelfAttention


@contextmanager
def identity_sdpa():
    """Replace scaled_dot_product_attention with a pass-through of ``v``."""
    real = F.scaled_dot_product_attention

    def stub(q, k, v, *args, **kwargs):
        if kwargs.get("enable_gqa", False) and q.shape[1] != v.shape[1]:
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        return v

    F.scaled_dot_product_attention = stub
    try:
        yield
    finally:
        F.scaled_dot_product_attention = real


def count_flops(model: IrisDiT, height: int, width: int) -> int:
    """Forward FLOPs of one ``height x width`` sample; ``model`` may live on the meta device."""
    flops = 0

    def linear_hook(module: nn.Linear, inputs, _output):
        nonlocal flops
        tokens = inputs[0].numel() // module.in_features
        flops += 2 * tokens * module.in_features * module.out_features

    def self_attn_hook(_module, inputs):
        nonlocal flops
        batch, n, dim = inputs[0].shape
        flops += 4 * batch * n * n * dim

    def joint_attn_hook(_module, inputs):
        # fused [text, image] attention; the qkv/proj Linears are counted by
        # linear_hook, this adds only the attention matmuls
        nonlocal flops
        x, y = inputs[0], inputs[1]
        s = x.shape[1] + y.shape[1]
        flops += 4 * x.shape[0] * s * s * x.shape[-1]

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, SelfAttention):
            handles.append(module.register_forward_pre_hook(self_attn_hook))
        elif isinstance(module, (JointAttention, SingleStreamBlock)):
            handles.append(module.register_forward_pre_hook(joint_attn_hook))

    cfg = model.cfg
    device = next(model.parameters()).device
    x = torch.randn(1, cfg.in_channels, height, width, device=device)
    t = torch.full((1,), 500.0, device=device)
    if cfg.text_adapter == "lap_blocks2":
        y = torch.randn(1, cfg.text_len, cfg.text_lap_num_layers, cfg.text_dim, device=device)
    else:
        y = torch.randn(1, cfg.text_len, cfg.text_dim, device=device)
    # all-real mask: the analytic cost is the padded worst case either way, and
    # a masked text adapter refuses to run without one
    y_mask = torch.ones(1, cfg.text_len, dtype=torch.int64, device=device)
    try:
        with torch.no_grad(), identity_sdpa():
            model(x, t, y, y_mask=y_mask)
    finally:
        for handle in handles:
            handle.remove()
    return flops


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", default="iris-3b", choices=sorted(PRESETS))
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args(argv)

    cfg = get_preset(args.preset)
    with torch.device("meta"):
        model = IrisDiT(cfg).eval()
    flops = count_flops(model, args.height, args.width)
    print(f"preset={args.preset} size={args.height}x{args.width} text_len={cfg.text_len}")
    print(f"params: {model.num_parameters / 1e6:.1f} M")
    print(f"flops:  {flops / 1e9:.2f} GFLOPs")


if __name__ == "__main__":
    main()
