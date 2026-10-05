from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import iris3b.models.blocks.single_stream as single_stream_module
import iris3b.nn.attention as attention_module
from iris3b.config import load_config
from iris3b.models import IrisDiT, get_preset
from iris3b.models.blocks.mmdit import MMDiTBlock
from iris3b.models.blocks.single_stream import SingleStreamBlock
from iris3b.nn.attention import JointAttention, scaled_dot_product
from tiny_config import tiny_model_config


class _FixedAdaLN(nn.Module):
    def __init__(self, attn_gate: float, mlp_gate: float):
        super().__init__()
        self.attn_gate = attn_gate
        self.mlp_gate = mlp_gate

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(cond)
        attn_gate = torch.full_like(cond, self.attn_gate)
        mlp_gate = torch.full_like(cond, self.mlp_gate)
        return torch.cat([zero, zero, attn_gate, zero, zero, mlp_gate], dim=-1)


def _identity_rope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(single_stream_module, "apply_rope", lambda q, k, _rope: (q, k))


def _identity_joint_rope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "apply_rope", lambda q, k, _rope: (q, k))


@pytest.mark.parametrize("kv_heads", [0, 3, 5])
def test_grouped_query_heads_must_divide_query_heads(kv_heads):
    with pytest.raises(ValueError, match="query heads 4.*KV heads"):
        SingleStreamBlock(64, 4, num_kv_heads=kv_heads)


def test_scaled_dot_product_gqa_matches_explicit_kv_repeat():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 8)
    expected = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
    )
    actual = scaled_dot_product(q, k, v, enable_gqa=True)
    torch.testing.assert_close(actual, expected)


def test_gqa_block_uses_unequal_heads_without_materializing_kv(monkeypatch):
    _identity_rope(monkeypatch)
    observed = {}

    def fake_attention(q, k, v, **kwargs):
        observed.update(q=q.shape, k=k.shape, v=v.shape, enable_gqa=kwargs["enable_gqa"])
        return v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

    monkeypatch.setattr(single_stream_module, "scaled_dot_product", fake_attention)
    block = SingleStreamBlock(64, 4, num_kv_heads=2).eval()
    x = torch.randn(2, 5, 64)
    y = torch.randn(2, 3, 64)
    out_x, out_y = block(x, y, torch.randn(2, 1, 64), None, None)
    assert out_x.shape == x.shape and out_y.shape == y.shape
    assert observed == {
        "q": torch.Size([2, 4, 8, 16]),
        "k": torch.Size([2, 2, 8, 16]),
        "v": torch.Size([2, 2, 8, 16]),
        "enable_gqa": True,
    }


def test_zero_initialized_sigmoid_gate_halves_attention_branch(monkeypatch):
    _identity_rope(monkeypatch)
    monkeypatch.setattr(
        single_stream_module,
        "scaled_dot_product",
        lambda q, _k, _v, **_kwargs: torch.ones_like(q),
    )
    block = SingleStreamBlock(8, 2, qk_norm=False, gated_attention=True).eval()
    block.adaln = _FixedAdaLN(attn_gate=1.0, mlp_gate=0.0)
    nn.init.zeros_(block.attn_gate.weight)
    nn.init.eye_(block.attn_proj.weight)
    nn.init.zeros_(block.attn_proj.bias)
    for parameter in block.mlp.parameters():
        nn.init.zeros_(parameter)

    x = torch.randn(1, 2, 8)
    y = torch.randn(1, 3, 8)
    out_x, out_y = block(x, y, torch.zeros(1, 1, 8), None, None)
    torch.testing.assert_close(out_x, x + 0.5 * torch.ones_like(x))
    torch.testing.assert_close(out_y, y + 0.5 * torch.ones_like(y))


def test_sandwich_norm_normalizes_both_branches_and_receives_gradients(monkeypatch):
    _identity_rope(monkeypatch)
    monkeypatch.setattr(
        single_stream_module,
        "scaled_dot_product",
        lambda q, _k, _v, **_kwargs: torch.ones_like(q),
    )
    block = SingleStreamBlock(16, 4, sandwich_norm=True).train()
    block.adaln = _FixedAdaLN(attn_gate=1.0, mlp_gate=1.0)
    branch_records = []

    def capture_norm(module, inputs, output):
        branch_records.append((module, inputs[0].detach(), output))

    handles = [
        block.attn_post_norm.register_forward_hook(capture_norm),
        block.mlp_post_norm.register_forward_hook(capture_norm),
    ]
    try:
        out_x, out_y = block(
            torch.randn(2, 4, 16),
            torch.randn(2, 3, 16),
            torch.zeros(2, 1, 16),
            None,
            None,
        )
        (out_x.square().mean() + out_y.square().mean()).backward()
    finally:
        for handle in handles:
            handle.remove()

    assert len(branch_records) == 2
    for module, branch_input, output in branch_records:
        mean_square = branch_input.float().square().mean(dim=-1)
        expected_rms = torch.sqrt(mean_square / (mean_square + module.eps))
        actual_rms = output.float().square().mean(dim=-1).sqrt()
        torch.testing.assert_close(actual_rms, expected_rms)
    assert block.attn_post_norm.weight.grad is not None
    assert block.attn_post_norm.weight.grad.abs().sum() > 0
    assert block.mlp_post_norm.weight.grad is not None
    assert block.mlp_post_norm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kv_heads", [0, 3, 5])
def test_dual_stream_grouped_query_heads_must_divide_query_heads(kv_heads):
    with pytest.raises(ValueError, match="query heads 4.*KV heads"):
        MMDiTBlock(64, 4, num_kv_heads=kv_heads)


def test_dual_stream_gqa_shrinks_kv_on_both_streams(monkeypatch):
    _identity_joint_rope(monkeypatch)
    observed = {}

    def fake_attention(q, k, v, **kwargs):
        observed.update(q=q.shape, k=k.shape, v=v.shape, enable_gqa=kwargs["enable_gqa"])
        return v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

    monkeypatch.setattr(attention_module, "scaled_dot_product", fake_attention)
    block = MMDiTBlock(64, 4, num_kv_heads=2).eval()
    x = torch.randn(2, 5, 64)
    y = torch.randn(2, 3, 64)
    out_x, out_y = block(x, y, torch.randn(2, 1, 64), None, None)
    assert out_x.shape == x.shape and out_y.shape == y.shape
    assert observed == {
        "q": torch.Size([2, 4, 8, 16]),
        "k": torch.Size([2, 2, 8, 16]),
        "v": torch.Size([2, 2, 8, 16]),
        "enable_gqa": True,
    }
    kv_out = block.attn.k_proj_x.out_features
    assert kv_out == block.attn.v_proj_y.out_features == 32
    assert block.attn.q_proj_x.out_features == 64


def test_dual_stream_zero_initialized_gate_halves_both_attention_branches(monkeypatch):
    _identity_joint_rope(monkeypatch)
    monkeypatch.setattr(
        attention_module,
        "scaled_dot_product",
        lambda q, _k, _v, **_kwargs: torch.ones_like(q),
    )
    block = MMDiTBlock(8, 2, qk_norm=False, gated_attention=True).eval()
    block.adaln_img = _FixedAdaLN(attn_gate=1.0, mlp_gate=0.0)
    block.adaln_txt = _FixedAdaLN(attn_gate=1.0, mlp_gate=0.0)
    for gate in (block.attn_gate_x, block.attn_gate_y):
        nn.init.zeros_(gate.weight)
    for proj in (block.attn.proj_x, block.attn.proj_y):
        nn.init.eye_(proj.weight)
        nn.init.zeros_(proj.bias)
    for parameter in (*block.mlp_x.parameters(), *block.mlp_y.parameters()):
        nn.init.zeros_(parameter)

    x = torch.randn(1, 2, 8)
    y = torch.randn(1, 3, 8)
    out_x, out_y = block(x, y, torch.zeros(1, 1, 8), None, None)
    torch.testing.assert_close(out_x, x + 0.5 * torch.ones_like(x))
    torch.testing.assert_close(out_y, y + 0.5 * torch.ones_like(y))


def test_dual_stream_sandwich_norm_normalizes_all_four_branches(monkeypatch):
    _identity_joint_rope(monkeypatch)
    monkeypatch.setattr(
        attention_module,
        "scaled_dot_product",
        lambda q, _k, _v, **_kwargs: torch.ones_like(q),
    )
    block = MMDiTBlock(16, 4, sandwich_norm=True).train()
    block.adaln_img = _FixedAdaLN(attn_gate=1.0, mlp_gate=1.0)
    block.adaln_txt = _FixedAdaLN(attn_gate=1.0, mlp_gate=1.0)
    post_norms = (
        block.attn_post_norm_x,
        block.attn_post_norm_y,
        block.mlp_post_norm_x,
        block.mlp_post_norm_y,
    )
    branch_records = []

    def capture_norm(module, inputs, output):
        branch_records.append((module, inputs[0].detach(), output))

    handles = [norm.register_forward_hook(capture_norm) for norm in post_norms]
    try:
        out_x, out_y = block(
            torch.randn(2, 4, 16),
            torch.randn(2, 3, 16),
            torch.zeros(2, 1, 16),
            None,
            None,
        )
        (out_x.square().mean() + out_y.square().mean()).backward()
    finally:
        for handle in handles:
            handle.remove()

    assert len(branch_records) == 4
    for module, branch_input, output in branch_records:
        mean_square = branch_input.float().square().mean(dim=-1)
        expected_rms = torch.sqrt(mean_square / (mean_square + module.eps))
        actual_rms = output.float().square().mean(dim=-1).sqrt()
        torch.testing.assert_close(actual_rms, expected_rms)
    for norm in post_norms:
        assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0


def test_stage1_config_matches_the_preset():
    # the shipped YAML is the launch surface, the preset is what the
    # parameter-count test reads: any drift between them is a silent
    # architecture change
    cfg = load_config("configs/iris3b/stage1_256.yaml")
    assert cfg.model == replace(get_preset("iris-3b"), rope_aspect="square")


def test_dual_stream_gqa_keeps_every_attention_matrix_on_muon():
    """GQA unfuses joint QKV; the router must still split only fused matrices."""
    from iris3b.train.optim import _muon_split_map

    cfg = tiny_model_config(
        block="mmdit",
        num_kv_heads=2,
        gated_attention=True,
        sandwich_norm=True,
    )
    model = IrisDiT(cfg)
    splits = _muon_split_map(model)
    attention = [m for m in model.modules() if isinstance(m, JointAttention)]
    assert attention and all(m.qkv_x is None for m in attention)
    for module in attention:
        for linear in (module.q_proj_x, module.k_proj_x, module.v_proj_x):
            assert id(linear.weight) not in splits
