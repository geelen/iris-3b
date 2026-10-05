"""Distributed strategy: config gates and shard-EMA math.

CPU-provable surface only. accelerate never promotes a CPU/gloo launch to a
sharded DistributedType, so actual sharding, collectives, and checkpoint
consolidation need a multi-GPU launch and are not covered here.
"""

from dataclasses import replace

import pytest
import torch
from torch import nn

from iris3b.config import (
    DistConfig,
    EMAConfig,
    MeshConfig,
    TrainConfig,
)
from iris3b.train.ema import EMA, ShardEMA


def _dist_train_cfg(strategy: str = "fsdp2", **overrides) -> TrainConfig:
    base = TrainConfig(
        dist=DistConfig(strategy=strategy),
        ema=EMAConfig(enabled=True),
    )
    return replace(base, **overrides)


def test_sharded_gates():
    ok = _dist_train_cfg()
    ok.dist.validate(ok)  # baseline recipe (muon) passes

    norms = _dist_train_cfg(log_block_grad_norms=True)
    with pytest.raises(ValueError, match="log_block_grad_norms"):
        norms.dist.validate(norms)

    core_resume = _dist_train_cfg(resume_optimizer="core")
    with pytest.raises(ValueError, match="resume_optimizer"):
        core_resume.dist.validate(core_resume)


def test_ddp_rejects_sharding_and_mesh_knobs():
    """DDP shards nothing, so a sharding/mesh setting is a misconfiguration
    rather than a no-op."""
    ok = TrainConfig()  # ddp: no sharded gate fires even with norms
    ok.log_block_grad_norms = True
    ok.dist.validate(ok)

    for sharding in ("grad_op", "hybrid", "hybrid_grad_op"):
        train = TrainConfig(dist=DistConfig(sharding=sharding))
        with pytest.raises(ValueError, match="sharding applies to fsdp2 only"):
            train.dist.validate(train)

    for mesh in (MeshConfig(dp_shard=8), MeshConfig(dp_replicate=2)):
        train = TrainConfig(dist=DistConfig(mesh=mesh))
        with pytest.raises(ValueError, match="mesh applies to fsdp2 only"):
            train.dist.validate(train)

    dcp = TrainConfig(dist=DistConfig(checkpoint_io="dcp"))
    with pytest.raises(ValueError, match="requires a sharded strategy"):
        dcp.dist.validate(dcp)


@pytest.mark.parametrize("sharding", ["hybrid", "hybrid_grad_op"])
def test_hybrid_sharding_requires_an_explicit_shard_group(sharding):
    """dp_shard is the replication boundary; guessing it silently changes which
    collectives cross the fabric."""
    train = _dist_train_cfg()
    train.dist.sharding = sharding
    with pytest.raises(ValueError, match="mesh.dp_shard"):
        train.dist.validate(train)

    train.dist.mesh = MeshConfig(dp_shard=8)
    train.dist.validate(train)


def test_enum_fields_are_rejected_by_name():
    bad_strategy = _dist_train_cfg("fsdp3")
    with pytest.raises(ValueError, match="ddp or fsdp2"):
        bad_strategy.dist.validate(bad_strategy)

    bad_sharding = _dist_train_cfg()
    bad_sharding.dist.sharding = "everything"
    with pytest.raises(ValueError, match="sharding must be"):
        bad_sharding.dist.validate(bad_sharding)

    bad_reduce = _dist_train_cfg()
    bad_reduce.dist.reduce_dtype = "fp16"
    with pytest.raises(ValueError, match="reduce_dtype"):
        bad_reduce.dist.validate(bad_reduce)

    bad_unused = _dist_train_cfg()
    bad_unused.dist.ddp.find_unused_parameters = "maybe"
    with pytest.raises(ValueError, match="find_unused_parameters"):
        bad_unused.dist.validate(bad_unused)


def test_wrapper_state_dict_mapping():
    from iris3b.train.trainer import _wrapper_state_dict

    payload = {
        "state_dict": {"blocks.0.qkv.weight": 1},
        "repa_projector": {"0.weight": 2},
    }
    sd = _wrapper_state_dict(payload["state_dict"], payload)
    assert sd == {"model.blocks.0.qkv.weight": 1, "repa.projector.0.weight": 2}


def test_save_checkpoint_accepts_precomputed_dicts(tmp_path):
    from iris3b.train.ckpt import save_checkpoint

    model_sd = {"w": torch.ones(2)}
    optim_sd = {"state": {"model.w": {"exp_avg": torch.zeros(2)}}, "param_groups": []}
    ema_sd = {"w": torch.full((2,), 0.5)}
    path = save_checkpoint(
        tmp_path / "epoch_1_step_10.pth",
        model_sd,
        optimizer=optim_sd,
        ema=ema_sd,
        epoch=1,
        step=10,
    )
    payload = torch.load(path, weights_only=False)
    assert torch.equal(payload["state_dict"]["w"], model_sd["w"])
    assert torch.equal(payload["state_dict_ema"]["w"], ema_sd["w"])
    assert list(payload["optimizer"]["state"]) == ["model.w"]
    assert (tmp_path / "latest.pth").resolve() == path.resolve()


class _FakeShardedModel(nn.Module):
    """Mimics FSDP(use_orig_params=True) outside forward: parameters whose
    ``.data`` are 1-D views into one flat shard, including a size-0 view for
    a parameter owned by another rank."""

    def __init__(self, flat: torch.Tensor, spans: list[tuple[int, int]]):
        super().__init__()
        self.flat = flat
        self.spans = spans
        for i, (off, n) in enumerate(spans):
            p = nn.Parameter(torch.empty(0))
            p.data = flat[off : off + n] if n else torch.empty(0)
            setattr(self, f"p{i}", p)


def _make_pair(seed: int = 0):
    """One full 'model' (reference) and its two fake rank shards."""
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(10, generator=g)
    # rank 0 owns full[0:6], rank 1 owns full[6:10]; three logical params:
    # a=full[0:4], b=full[4:8] (split across ranks), c=full[8:10] (rank 1 only)
    flat0, flat1 = full[:6].clone(), full[6:].clone()
    r0 = _FakeShardedModel(flat0, [(0, 4), (4, 2), (0, 0)])
    r1 = _FakeShardedModel(flat1, [(0, 0), (0, 2), (2, 2)])
    return full, r0, r1


def test_shard_ema_matches_full_tensor_ema():
    full, r0, r1 = _make_pair()
    decay = 0.9
    ema0, ema1 = ShardEMA(r0, decay), ShardEMA(r1, decay)
    ref = full.clone()
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        step = torch.randn(10, generator=g)
        r0.flat.copy_(step[:6])
        r1.flat.copy_(step[6:])
        ema0.update()
        ema1.update()
        ref.mul_(decay).add_(step, alpha=1 - decay)
    gathered = torch.cat(
        [ema0.shadow_params[0], ema0.shadow_params[1], ema1.shadow_params[1], ema1.shadow_params[2]]
    )
    assert torch.allclose(gathered, ref, atol=0, rtol=0)


def test_shard_ema_matches_dense_ema_class():
    """Same trajectory through ShardEMA (sharded) and EMA (dense module)."""
    torch.manual_seed(3)
    dense = nn.Linear(4, 4, bias=False)
    flat = dense.weight.detach().reshape(-1).clone()
    sharded = _FakeShardedModel(flat, [(0, 16)])
    dense_ema = EMA(dense, 0.99)
    shard_ema = ShardEMA(sharded, 0.99)
    for step in range(4):
        with torch.no_grad():
            delta = torch.full((4, 4), 0.5 * (step + 1))
            dense.weight.copy_(delta)
            sharded.flat.copy_(delta.reshape(-1))
        dense_ema.update(dense)
        shard_ema.update()
    assert torch.equal(shard_ema.shadow_params[0], dense_ema.module.weight.detach().reshape(-1))


def test_shard_ema_swapped_writes_flat_storage_and_restores():
    _, r0, _ = _make_pair(seed=7)
    ema = ShardEMA(r0, 0.5)
    live = r0.flat.clone()
    with torch.no_grad():
        r0.flat.add_(1.0)  # diverge live weights from the shadow
    ema.update()
    shadow_now = [s.clone() for s in ema.shadow_params]
    live_now = r0.flat.clone()
    with ema.swapped():
        # inside: the flat storage (what a collective gather would read)
        # holds the averaged values
        assert torch.equal(r0.flat[0:4], shadow_now[0])
        assert torch.equal(r0.flat[4:6], shadow_now[1])
    assert torch.equal(r0.flat, live_now)  # restored exactly
    assert not torch.equal(live, live_now)  # sanity: they had diverged
