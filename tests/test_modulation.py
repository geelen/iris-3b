"""Shared-core modulation with per-block low-rank conditional residuals.

``shared_lowrank`` / ``shared_bias`` must be drop-ins for the plain per-block
``Linear(D, 6D)`` on the ``[B, 1, D]`` conditioning path.
"""

import pytest
import torch
from torch import nn

from iris3b.models import IrisDiT
from iris3b.nn.modulation import (
    ModulationBuilder,
    SharedCoreBias,
    SharedCoreModulation,
)
from tiny_config import tiny_model_config

DIM = 64
CPU = torch.device("cpu")


def rope_tables(n_txt: int = 5, grid: tuple[int, int] = (4, 4)):
    """Real RoPE caches for block-level forwards, taken from a tiny model."""
    model = IrisDiT(tiny_model_config())
    return model._fetch_rope_img(grid, CPU), model._fetch_rope_txt(n_txt, CPU)


def block_inputs(n_txt: int = 5, grid: tuple[int, int] = (4, 4)):
    torch.manual_seed(7)
    n_img = grid[0] * grid[1]
    return torch.randn(2, n_img, DIM), torch.randn(2, n_txt, DIM), torch.randn(2, 1, DIM)


def patch_modulation_params(model: IrisDiT) -> int:
    """Every parameter implementing patch-block adaLN, in either mode.

    ``per_block``: ``blocks.i.adaln_{img,txt}.{weight,bias}``. ``shared_lowrank``:
    the shared cores under ``modulation_cores`` plus each block's ``down`` (V)
    and ``adaln_up`` (U). ``shared_bias``: the cores plus one bias per block.
    """
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith("modulation_cores.") or (name.startswith("blocks.") and "adaln" in name)
    )


# -- (a) shared_lowrank drops in for the per-block Linear ---------------------


def test_shared_lowrank_with_zero_residual_reproduces_a_plain_linear():
    cores = nn.ModuleDict()
    build = ModulationBuilder(DIM, mode="shared_lowrank", rank=8, cores=cores)
    mod = build("img")
    assert isinstance(mod, SharedCoreModulation)
    nn.init.zeros_(mod.adaln_up.weight)
    cond = torch.randn(2, 1, DIM)
    out = mod(cond)
    assert out.shape == (2, 1, 6 * DIM)
    assert torch.equal(out, cores["adaln_img"](cond))

    # a live residual perturbs all six chunks, i.e. U really spans 6*D
    nn.init.normal_(mod.adaln_up.weight, std=0.02)
    delta = (mod(cond) - cores["adaln_img"](cond)).chunk(6, dim=-1)
    assert all(chunk.abs().max() > 0 for chunk in delta)


def test_shared_lowrank_cores_are_shared_once_across_blocks():
    cfg = tiny_model_config(modulation="shared_lowrank", modulation_rank=8)
    model = IrisDiT(cfg)
    assert sorted(model.modulation_cores) == ["adaln_img", "adaln_txt"]
    for block in model.blocks:
        assert block.adaln_img.core is model.modulation_cores["adaln_img"]
        assert block.adaln_txt.core is model.modulation_cores["adaln_txt"]
    core_keys = [k for k in model.state_dict() if k.startswith("modulation_cores")]
    assert sorted(core_keys) == [
        "modulation_cores.adaln_img.bias",
        "modulation_cores.adaln_img.weight",
        "modulation_cores.adaln_txt.bias",
        "modulation_cores.adaln_txt.weight",
    ]
    # the shared core must be counted once, not once per block
    n_core = sum(p.numel() for p in model.modulation_cores.parameters())
    assert n_core == 2 * (DIM * 6 * DIM + 6 * DIM)
    assert patch_modulation_params(model) == n_core + model.cfg.depth * 2 * (DIM * 8 + 8 * 6 * DIM)


def test_shared_core_survives_activation_checkpointing():
    """The one core is called inside every checkpointed block region at once.

    It is not one of the checkpointed function's arguments (it is captured), so
    recomputation must still route gradient into it, once per block.
    """
    cfg = tiny_model_config(modulation="shared_lowrank", modulation_rank=8)
    model = IrisDiT(cfg).train()
    model.activation_checkpointing = "full"
    args = (torch.randn(2, 3, 32, 32), torch.tensor([5.0, 900.0]), torch.randn(2, 16, 32))
    model(*args, capture=(cfg.depth,)).features[cfg.depth].square().sum().backward()
    core = model.modulation_cores["adaln_img"]
    assert core.weight.grad is not None and core.weight.grad.abs().max() > 0
    for block in model.blocks:
        assert block.adaln_img.adaln_up.weight.grad.abs().max() > 0

    # same gradient as the uncheckpointed path
    ckpt_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.activation_checkpointing = "none"
    model(*args, capture=(cfg.depth,)).features[cfg.depth].square().sum().backward()
    for name, grad in ckpt_grads.items():
        assert torch.allclose(grad, dict(model.named_parameters())[name].grad, atol=1e-6), name


def test_zero_init_residual_can_train_because_v_is_not_zeroed():
    """Why zero-init zeroes U and the core but leaves V at its default init.

    With U = 0 the block is identity and only U carries gradient; V's gradient
    is gated by U and starts flowing once U moves. Zeroing V as well would make
    BOTH factors' gradients identically zero forever -- a dead residual.
    """
    cfg = tiny_model_config(modulation="shared_lowrank", modulation_rank=8, adaln_zero_init=True)
    model = IrisDiT(cfg)
    mod = model.blocks[0].adaln_img
    args = (torch.randn(1, 3, 32, 32), torch.tensor([500.0]), torch.randn(1, 16, 32))

    def grad_magnitudes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        model(*args, capture=(1,)).features[1].square().sum().backward()
        return (
            mod.down.weight.grad.abs().max(),
            mod.adaln_up.weight.grad.abs().max(),
            model.modulation_cores["adaln_img"].weight.grad.abs().max(),
        )

    g_v, g_u, g_core = grad_magnitudes()
    assert g_u > 0 and g_core > 0  # U and the shared core learn from step 0
    assert g_v == 0  # V's gradient is U-gated, hence zero while U is zero

    with torch.no_grad():  # one optimizer step on U
        mod.adaln_up.weight.normal_(std=0.02)
    g_v, g_u, _ = grad_magnitudes()
    assert g_v > 0 and g_u > 0

    with torch.no_grad():  # the rejected both-factors-zero init
        mod.adaln_up.weight.zero_()
        mod.down.weight.zero_()
    g_v, g_u, _ = grad_magnitudes()
    assert g_v == 0 and g_u == 0


# -- (b) exact parameter counts ----------------------------------------------


def test_exact_parameter_counts_baseline_vs_shared_lowrank():
    rank = 8
    base_model = IrisDiT(tiny_model_config())
    lowrank_model = IrisDiT(tiny_model_config(modulation="shared_lowrank", modulation_rank=rank))
    depth, linear = base_model.cfg.depth, DIM * 6 * DIM + 6 * DIM

    # depth blocks x 2 streams x Linear(D, 6D)
    assert patch_modulation_params(base_model) == depth * 2 * linear
    # 2 shared cores + depth blocks x 2 streams x (V: D x r, U: r x 6D)
    assert patch_modulation_params(lowrank_model) == 2 * linear + depth * 2 * (DIM * rank + rank * 6 * DIM)
    saving = patch_modulation_params(base_model) - patch_modulation_params(lowrank_model)
    assert saving > 0
    # nothing outside the modulation changes
    assert base_model.num_parameters - lowrank_model.num_parameters == saving


# -- (c) identity at init under adaln_zero_init -------------------------------


@pytest.mark.parametrize("mode", ["per_block", "shared_lowrank"])
def test_adaln_zero_init_keeps_blocks_identity_in_both_modes(mode):
    cfg = tiny_model_config(modulation=mode, modulation_rank=8, adaln_zero_init=True)
    model = IrisDiT(cfg)
    rope_img, rope_txt = rope_tables()
    x, y, cond = block_inputs()
    for block in model.blocks:
        out_x, out_y = block(x, y, cond, rope_img, rope_txt)
        assert torch.equal(out_x, x)
        assert torch.equal(out_y, y)


# -- (d) misconfiguration ----------------------------------------------------


def test_unknown_modulation_mode_raises():
    with pytest.raises(ValueError, match="model.modulation must be one of"):
        IrisDiT(tiny_model_config(modulation="fully_shared"))
    with pytest.raises(ValueError, match="model.modulation must be one of"):
        ModulationBuilder(DIM, mode="fully_shared")


def test_non_positive_rank_raises():
    with pytest.raises(ValueError, match="modulation_rank must be positive"):
        IrisDiT(tiny_model_config(modulation="shared_lowrank", modulation_rank=0))


def test_shared_lowrank_requires_a_model_owned_core_container():
    with pytest.raises(ValueError, match="core container"):
        ModulationBuilder(DIM, mode="shared_lowrank")


# -- (e) shared_bias: the PixArt adaLN-single sibling of shared_lowrank -------


def test_shared_bias_matches_its_core_alone_at_init():
    """A fresh SharedCoreBias is exactly its shared core, so the mode extends it."""
    torch.manual_seed(3)
    core = nn.Linear(DIM, 6 * DIM, bias=True)
    mod = SharedCoreBias(DIM, core)
    cond = torch.randn(2, 1, DIM)

    assert torch.all(mod.bias == 0)
    assert torch.equal(mod(cond), core(cond))

    with torch.no_grad():
        mod.bias.normal_()
    assert torch.allclose(mod(cond), core(cond) + mod.bias)


def test_shared_bias_cores_are_shared_once_across_blocks():
    cfg = tiny_model_config(modulation="shared_bias")
    model = IrisDiT(cfg)

    assert set(model.modulation_cores) == {"adaln_img", "adaln_txt"}
    for block in model.blocks:
        assert block.adaln_img.core is model.modulation_cores["adaln_img"]
        assert block.adaln_txt.core is model.modulation_cores["adaln_txt"]

    names = [n for n, _ in model.named_parameters() if "adaln_img" in n and "weight" in n]
    assert names == ["modulation_cores.adaln_img.weight"]  # the core appears exactly once


def test_shared_bias_is_identity_at_init_under_adaln_zero_init():
    cfg = tiny_model_config(modulation="shared_bias", adaln_zero_init=True)
    model = IrisDiT(cfg)
    rope_img, rope_txt = rope_tables()
    x, y, cond = block_inputs()

    for core in model.modulation_cores.values():
        assert torch.all(core.weight == 0) and torch.all(core.bias == 0)
    for block in model.blocks:
        assert torch.all(block.adaln_img.bias == 0) and torch.all(block.adaln_txt.bias == 0)
        out_x, out_y = block(x, y, cond, rope_img, rope_txt)
        assert torch.equal(out_x, x)
        assert torch.equal(out_y, y)


def test_shared_bias_has_no_dead_branch_when_the_core_is_zeroed():
    """Unlike the low-rank residual, a bias always receives gradient."""
    cfg = tiny_model_config(modulation="shared_bias", adaln_zero_init=True)
    model = IrisDiT(cfg)
    rope_img, rope_txt = rope_tables()
    x, y, cond = block_inputs()
    block = model.blocks[0]

    out_x, out_y = block(x, y, cond, rope_img, rope_txt)
    (out_x.pow(2).sum() + out_y.pow(2).sum()).backward()

    assert block.adaln_img.bias.grad.abs().max() > 0
    assert block.adaln_txt.bias.grad.abs().max() > 0


def test_exact_parameter_counts_shared_bias():
    rank = 8
    base_model = IrisDiT(tiny_model_config())
    bias_model = IrisDiT(tiny_model_config(modulation="shared_bias"))
    depth, linear = base_model.cfg.depth, DIM * 6 * DIM + 6 * DIM

    # 2 shared cores + depth blocks x 2 streams x one 6D-wide bias
    assert patch_modulation_params(bias_model) == 2 * linear + depth * 2 * 6 * DIM
    saving = patch_modulation_params(base_model) - patch_modulation_params(bias_model)
    assert saving > 0
    assert base_model.num_parameters - bias_model.num_parameters == saving

    # strictly cheaper than the rank-8 conditional residual
    lowrank_model = IrisDiT(tiny_model_config(modulation="shared_lowrank", modulation_rank=rank))
    assert patch_modulation_params(bias_model) < patch_modulation_params(lowrank_model)


def test_shared_bias_requires_a_model_owned_core_container():
    with pytest.raises(ValueError, match="core container"):
        ModulationBuilder(DIM, mode="shared_bias")
