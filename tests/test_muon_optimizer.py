from dataclasses import replace

import pytest
import torch
from torch import nn

from iris3b.config import OptimizerConfig
from iris3b.models import IrisDiT
from iris3b.train.ckpt import (
    _core_optimizer_state,
    _validate_optimizer_layout,
    load_checkpoint,
    save_checkpoint,
)
from iris3b.train.lr import build_lr_scheduler
from iris3b.train.optim import (
    _muon_distributed_mesh,
    build_muon_param_groups,
    build_optimizer,
)
from tiny_config import tiny_model_config


def _route_by_parameter(groups: list[dict]) -> dict[int, dict]:
    routed: dict[int, dict] = {}
    for group in groups:
        for parameter in group["params"]:
            assert id(parameter) not in routed
            routed[id(parameter)] = group
    return routed


def _debug_model(**overrides) -> IrisDiT:
    return IrisDiT(tiny_model_config(**overrides))


def test_muon_routing_covers_every_parameter_and_keeps_boundaries_on_adamw():
    model = _debug_model()
    auxiliary = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 64))
    params = [*model.parameters(), *auxiliary.parameters()]

    groups = build_muon_param_groups(model, params)
    routed = _route_by_parameter(groups)

    assert set(routed) == {id(parameter) for parameter in params}
    for parameter in (
        model.s_embedder.proj.weight,
        model.y_embedder.proj.weight,
        model.pixel_embedder.proj.weight,
        model.final_layer.linear.weight,
    ):
        assert routed[id(parameter)]["algorithm"] == "adamw"
        assert routed[id(parameter)]["iris_scope"] == "core"

    assert routed[id(model.t_embedder.mlp[0].weight)]["algorithm"] == "adamw"
    assert routed[id(model.t_embedder.mlp[2].weight)]["algorithm"] == "adamw"
    assert routed[id(auxiliary[0].weight)]["algorithm"] == "adamw"
    assert routed[id(auxiliary[0].weight)]["iris_scope"] == "aux"

    block = model.blocks[0]
    qkv_group = routed[id(block.attn.qkv_x.weight)]
    assert qkv_group["split_sizes"] == (64, 64, 64)
    adaln_group = routed[id(block.adaln_img.weight)]
    assert adaln_group["split_sizes"] == (64,) * 6

    pit = model.pixel_blocks[0]
    assert routed[id(pit.adaln.weight)]["algorithm"] == "adamw"
    assert "split_sizes" not in routed[id(pit.adaln.weight)]


def test_muon_routing_splits_single_stream_shared_and_postmod_matrices():
    base = tiny_model_config()
    pixel = replace(base.pixel, modulation="post")
    model = _debug_model(
        block="single_stream",
        modulation="shared_lowrank",
        modulation_rank=8,
        pixel=pixel,
    )

    groups = build_muon_param_groups(model, model.parameters())
    routed = _route_by_parameter(groups)
    block = model.blocks[0]

    assert routed[id(block.qkv.weight)]["split_sizes"] == (64, 64, 64)
    assert routed[id(block.adaln.core.weight)]["split_sizes"] == (64,) * 6
    assert routed[id(block.adaln.adaln_up.weight)]["split_sizes"] == (64,) * 6

    pit = model.pixel_blocks[0]
    assert routed[id(pit.adaln.weight)]["algorithm"] == "adamw"
    assert "split_sizes" not in routed[id(pit.adaln.weight)]


def test_muon_routing_handles_gqa_gates_and_blocks2_text_adapter():
    base = tiny_model_config()
    model = _debug_model(
        block="single_stream",
        num_kv_heads=2,
        gated_attention=True,
        sandwich_norm=True,
        text_adapter="blocks2",
        modulation="shared_lowrank",
        modulation_rank=8,
        pixel=replace(base.pixel, modulation="post"),
    )

    groups = build_muon_param_groups(model, model.parameters())
    routed = _route_by_parameter(groups)
    assert set(routed) == {id(parameter) for parameter in model.parameters()}

    block = model.blocks[0]
    assert block.qkv is None
    for linear in (
        block.q_proj,
        block.k_proj,
        block.v_proj,
        block.attn_gate,
        block.attn_proj,
    ):
        assert routed[id(linear.weight)]["algorithm"] == "muon"
        assert "split_sizes" not in routed[id(linear.weight)]

    assert routed[id(model.y_embedder.proj.weight)]["algorithm"] == "adamw"
    adapter_qkv = model.y_embedder.blocks[0].attn.qkv.weight
    assert routed[id(adapter_qkv)]["split_sizes"] == (64, 64, 64)


def test_muon_routing_handles_lap_shared_bias_and_patch_only_head():
    base = tiny_model_config()
    model = _debug_model(
        text_adapter="lap_blocks2",
        text_lap_num_layers=4,
        text_lap_num_heads=4,
        modulation="shared_bias",
        pixel=replace(base.pixel, enabled=False),
    )

    groups = build_muon_param_groups(model, model.parameters())
    routed = _route_by_parameter(groups)
    assert set(routed) == {id(parameter) for parameter in model.parameters()}
    assert routed[id(model.y_embedder.layer_pool.weight)]["algorithm"] == "adamw"
    assert routed[id(model.y_embedder.refiner.proj.weight)]["algorithm"] == "adamw"
    lap_qkv = model.y_embedder.layer_blocks[0].attn.qkv.weight
    assert routed[id(lap_qkv)]["split_sizes"] == (32, 32, 32)
    for core in model.modulation_cores.values():
        assert routed[id(core.weight)]["split_sizes"] == (64,) * 6
    assert routed[id(model.final_layer.linear.weight)]["algorithm"] == "adamw"


class _FakeMuon(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, {"lr": kwargs["lr"]})

    def step(self, closure=None):
        return None


class _StatefulFakeMuon(torch.optim.AdamW):
    def __init__(self, params, **kwargs):
        super().__init__(
            params,
            lr=kwargs["lr"],
            betas=kwargs["betas"],
            eps=kwargs["epsilon"],
            weight_decay=kwargs["weight_decay"],
        )


def _named_training_params(model: nn.Module, auxiliary: nn.Module) -> tuple[list[str], list]:
    named = [(f"model.{name}", parameter) for name, parameter in model.named_parameters()]
    named += [(f"auxiliary.{name}", parameter) for name, parameter in auxiliary.named_parameters()]
    return [name for name, _ in named], [parameter for _, parameter in named]


def _initialize_optimizer_state(optimizer, params) -> None:
    optimizer.zero_grad(set_to_none=True)
    for parameter in params:
        parameter.grad = torch.full_like(parameter, 1.0e-3)
    optimizer.step()


def test_muon_keeps_aux_group_and_scheduler_topology_stable(monkeypatch):
    model = _debug_model()
    auxiliary = nn.Linear(64, 64)
    cfg = OptimizerConfig(name="muon", auto_lr="none")
    monkeypatch.setattr("iris3b.train.optim._load_dion_muon", lambda: _FakeMuon)

    baseline, _ = build_optimizer(cfg, model.parameters(), effective_batch_size=256, model=model)
    with_aux, _ = build_optimizer(
        cfg,
        [*model.parameters(), *auxiliary.parameters()],
        effective_batch_size=256,
        model=model,
    )

    assert len(baseline.param_groups) == len(with_aux.param_groups)
    assert baseline.param_groups[-1]["iris_scope"] == "aux"
    assert baseline.param_groups[-1]["params"] == []
    assert with_aux.param_groups[-1]["params"] == list(auxiliary.parameters())

    baseline_scheduler = build_lr_scheduler(cfg, baseline, world_size=1, total_steps=10)
    with_aux_scheduler = build_lr_scheduler(cfg, with_aux, world_size=1, total_steps=10)
    baseline.step()
    baseline_scheduler.step()
    state = baseline_scheduler.state_dict()
    assert len(state["base_lrs"]) == len(with_aux_scheduler.state_dict()["base_lrs"])

    with_aux_scheduler.load_state_dict(state)
    baseline.step()
    with_aux.step()
    baseline_scheduler.step()
    with_aux_scheduler.step()
    assert baseline_scheduler.get_last_lr() == with_aux_scheduler.get_last_lr()
    assert len(set(with_aux_scheduler.get_last_lr())) == 1


def test_routed_optimizer_full_and_core_checkpoint_round_trips(tmp_path, monkeypatch):
    cfg = OptimizerConfig(name="muon", lr=1.0e-3, auto_lr="none", warmup_steps=2)
    monkeypatch.setattr("iris3b.train.optim._load_dion_muon", lambda: _StatefulFakeMuon)

    source_model = _debug_model()
    source_aux = nn.Linear(64, 64)
    source_names, source_params = _named_training_params(source_model, source_aux)
    source_optimizer, _ = build_optimizer(
        cfg,
        source_params,
        effective_batch_size=256,
        model=source_model,
        parameter_names=source_names,
    )
    source_scheduler = build_lr_scheduler(cfg, source_optimizer, world_size=1, total_steps=10)
    _initialize_optimizer_state(source_optimizer, source_params)
    source_scheduler.step()
    path = save_checkpoint(
        tmp_path / "source" / "step_1.pth",
        source_model,
        source_optimizer,
        source_scheduler,
        epoch=1,
        step=1,
    )

    full_model = _debug_model()
    full_aux = nn.Linear(64, 64)
    full_names, full_params = _named_training_params(full_model, full_aux)
    full_optimizer, _ = build_optimizer(
        cfg,
        full_params,
        effective_batch_size=256,
        model=full_model,
        parameter_names=full_names,
    )
    full_scheduler = build_lr_scheduler(cfg, full_optimizer, world_size=1, total_steps=10)
    load_checkpoint(path, full_model, full_optimizer, full_scheduler, weights_only_load=False)
    for source, restored in zip(source_model.parameters(), full_model.parameters(), strict=True):
        torch.testing.assert_close(
            source_optimizer.state[source]["exp_avg"],
            full_optimizer.state[restored]["exp_avg"],
        )

    core_model = _debug_model()
    core_aux = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 64))
    core_names, core_params = _named_training_params(core_model, core_aux)
    core_optimizer, _ = build_optimizer(
        cfg,
        core_params,
        effective_batch_size=256,
        model=core_model,
        parameter_names=core_names,
    )
    core_scheduler = build_lr_scheduler(cfg, core_optimizer, world_size=1, total_steps=10)
    load_checkpoint(
        path,
        core_model,
        core_optimizer,
        core_scheduler,
        optimizer_prefix=len(list(core_model.parameters())),
    )
    assert all(parameter in core_optimizer.state for parameter in core_model.parameters())
    assert all(parameter not in core_optimizer.state for parameter in core_aux.parameters())
    assert len(core_scheduler.get_last_lr()) == len(core_optimizer.param_groups)


def test_muon_requires_the_core_model():
    cfg = OptimizerConfig(name="muon")
    parameter = nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError, match="core model"):
        build_optimizer(cfg, [parameter], 256)


def test_muon_takes_the_1d_shard_submesh_under_hybrid():
    """Dion's Muon rejects a 2-D mesh, so hybrid must hand over its shard dim.

    Passing the full (dp_replicate, dp_shard) mesh raises inside Dion with
    "Only 1D DeviceMesh supported", which is a startup crash rather than a
    silent wrong update, but it blocks every hybrid-sharded run.
    """

    class FakeMesh:
        def __init__(self, ndim, names):
            self.ndim = ndim
            self.mesh_dim_names = names

        def __getitem__(self, key):
            return f"submesh:{key}"

    hybrid = FakeMesh(2, ("dp_replicate", "dp_shard"))
    assert _muon_distributed_mesh(hybrid) == "submesh:dp_shard"

    flat = FakeMesh(1, ("dp_shard",))
    assert _muon_distributed_mesh(flat) is flat

    with pytest.raises(ValueError, match="1-D sharded sub-mesh"):
        _muon_distributed_mesh(FakeMesh(2, ("outer", "inner")))


def test_core_resume_uses_routed_group_metadata_and_rejects_route_changes():
    saved = {
        "state": {0: {"momentum": "m0"}, 1: {"exp_avg": "m1"}, 2: {"exp_avg": "aux"}},
        "param_groups": [
            {
                "params": [0],
                "algorithm": "muon",
                "iris_scope": "core",
                "iris_route": "hidden_matrix",
                "split_sizes": (4, 4, 4),
                "iris_param_names": ("model.hidden",),
            },
            {
                "params": [1],
                "algorithm": "adamw",
                "iris_scope": "core",
                "iris_route": "boundary_or_vector",
                "iris_param_names": ("model.boundary",),
            },
            {
                "params": [2],
                "algorithm": "adamw",
                "iris_scope": "aux",
                "iris_route": "auxiliary",
                "iris_param_names": ("repa.projector.weight",),
            },
        ],
    }
    live_groups = [
        {**saved["param_groups"][0], "params": [0]},
        {**saved["param_groups"][1], "params": [1]},
        {**saved["param_groups"][2], "params": [2, 3]},
    ]

    payload, kept = _core_optimizer_state(saved, live_groups, legacy_prefix=2)
    assert kept == 2
    assert payload["state"] == {0: {"momentum": "m0"}, 1: {"exp_avg": "m1"}}
    assert payload["param_groups"] == live_groups

    changed = [{**live_groups[0], "split_sizes": (6, 6)}, *live_groups[1:]]
    with pytest.raises(ValueError, match="same core routing"):
        _core_optimizer_state(saved, changed, legacy_prefix=2)

    renamed = [
        {**live_groups[0], "iris_param_names": ("model.other_hidden",)},
        *live_groups[1:],
    ]
    with pytest.raises(ValueError, match="same core routing"):
        _core_optimizer_state(saved, renamed, legacy_prefix=2)


def test_full_resume_rejects_positional_routing_changes():
    saved = {
        "param_groups": [
            {
                "params": [0, 1],
                "algorithm": "muon",
                "iris_scope": "core",
                "iris_route": "hidden_matrix",
                "iris_param_names": ("model.a", "model.b"),
            },
            {
                "params": [],
                "algorithm": "adamw",
                "iris_scope": "aux",
                "iris_route": "auxiliary",
                "iris_param_names": (),
            },
        ]
    }
    live = [dict(group) for group in saved["param_groups"]]
    _validate_optimizer_layout(saved, live)

    live[0]["iris_param_names"] = ("model.b", "model.a")
    with pytest.raises(ValueError, match="same ordered parameter routing"):
        _validate_optimizer_layout(saved, live)
