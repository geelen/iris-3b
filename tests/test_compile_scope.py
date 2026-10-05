"""Per-block compilation scope, shape budgeting, and dynamo cache limits.

No graph is ever traced here: ``nn.Module.compile`` only installs a compiled
call implementation, so these tests check what gets compiled and with which
settings.
"""

import pytest
import torch
from torch import nn

from iris3b.config import DataConfig, PerfConfig
from iris3b.data.buckets import TRAIN_BUCKETS
from iris3b.models import IrisDiT
from iris3b.train.compile import (
    compile_decision,
    compile_model,
    compile_targets,
    expected_shape_count,
)
from iris3b.train.ema import EMA
from tiny_config import tiny_model_config

_LIMIT_ATTRS = (
    "recompile_limit",
    "cache_size_limit",
    "accumulated_recompile_limit",
    "accumulated_cache_size_limit",
)


@pytest.fixture
def dynamo_limits():
    """Restore dynamo's global cache budget so tests never leak into each other."""
    saved = {n: getattr(torch._dynamo.config, n) for n in _LIMIT_ATTRS}
    yield torch._dynamo.config
    for name, value in saved.items():
        setattr(torch._dynamo.config, name, value)


def _perf(**overrides) -> PerfConfig:
    # "default" keeps torch.compile from touching the inductor autotuner
    return PerfConfig(compile=True, compile_mode="default", **overrides)


def _tiny() -> IrisDiT:
    return IrisDiT(tiny_model_config())


# -- shape budget ------------------------------------------------------------
def test_expected_shape_count_follows_the_shape_policy():
    assert expected_shape_count(DataConfig(image_size=256), 16) == 1
    bucketed = DataConfig(shape_policy="bucket", aspect_ratio_bucket="shared21-512")
    assert expected_shape_count(bucketed, 16) == len(TRAIN_BUCKETS["shared21-512"]) == 21
    assert expected_shape_count(DataConfig(aspect_ratio_bucket="shared21-1024"), 16) == 1
    # "area" derives shapes from the corpus: the set is open
    assert expected_shape_count(DataConfig(shape_policy="area"), 16) == 0


def test_dynamic_is_static_only_for_a_single_shape(dynamo_limits):
    compile_model(_tiny(), _perf(), shape_count=1)
    assert "dynamic=False" in compile_decision()
    compile_model(_tiny(), _perf(), shape_count=40)
    assert "dynamic=True" in compile_decision()
    # an open shape set can never be specialized
    compile_model(_tiny(), _perf(), shape_count=0)
    assert "dynamic=True" in compile_decision()
    assert "shapes=open" in compile_decision()


def test_compile_dynamic_overrides_the_shape_heuristic(dynamo_limits):
    compile_model(_tiny(), _perf(compile_dynamic="true"), shape_count=1)
    assert "dynamic=True" in compile_decision()
    compile_model(_tiny(), _perf(compile_dynamic="false"), shape_count=40)
    assert "dynamic=False" in compile_decision()


def test_recompile_limit_raised_without_the_muon_path(dynamo_limits):
    dynamo_limits.recompile_limit = 8
    perf = _perf()
    compile_model(_tiny(), perf, shape_count=40)
    # dynamo's default 8 would strand a 40-bucket run in eager after 8 shapes
    assert dynamo_limits.recompile_limit >= 40 * 2 + 8
    assert dynamo_limits.recompile_limit >= perf.recompile_limit
    assert f"recompile_limit={dynamo_limits.recompile_limit}" in compile_decision()
    # identical blocks share one code object, so the accumulated cap has to
    # cover instances * shapes or it becomes the next silent cliff
    assert dynamo_limits.accumulated_recompile_limit >= dynamo_limits.recompile_limit * 3


def test_accumulated_cap_scales_with_compiled_instances(dynamo_limits):
    dynamo_limits.recompile_limit = 8
    dynamo_limits.accumulated_recompile_limit = 256
    deep = IrisDiT(tiny_model_config(depth=12))
    compile_model(deep, _perf(), shape_count=40)
    per_frame = dynamo_limits.recompile_limit
    assert dynamo_limits.accumulated_recompile_limit >= per_frame * 13  # 12 blocks + 1 PiT


def test_recompile_limit_is_never_lowered(dynamo_limits):
    dynamo_limits.recompile_limit = 512
    compile_model(_tiny(), _perf(), shape_count=1)
    assert dynamo_limits.recompile_limit == 512


# -- scope -------------------------------------------------------------------
def test_block_scope_targets_every_transformer_stack():
    model = _tiny()
    names = [name for name, _ in compile_targets(model)]
    assert names == ["blocks.0", "blocks.1", "pixel_blocks.0"]

    lap = tiny_model_config(text_adapter="lap_blocks2", text_lap_num_layers=3, text_lap_num_heads=4)
    lap_names = [name for name, _ in compile_targets(IrisDiT(lap))]
    assert lap_names == [
        "blocks.0",
        "blocks.1",
        "pixel_blocks.0",
        "y_embedder.layer_blocks.0",
        "y_embedder.layer_blocks.1",
        "y_embedder.refiner.blocks.0",
        "y_embedder.refiner.blocks.1",
    ]

    plain = tiny_model_config(text_adapter="blocks2")
    plain_names = [name for name, _ in compile_targets(IrisDiT(plain))]
    assert plain_names[-2:] == ["y_embedder.blocks.0", "y_embedder.blocks.1"]


def test_block_scope_compiles_blocks_not_the_model(dynamo_limits):
    model = _tiny()
    compile_model(model, _perf(), shape_count=1)
    assert model._compiled_call_impl is None
    assert all(block._compiled_call_impl is not None for block in model.blocks)
    assert model.pixel_blocks[0]._compiled_call_impl is not None
    assert "scope=block targets=3" in compile_decision()


def test_model_scope_compiles_the_whole_core(dynamo_limits):
    model = _tiny()
    compile_model(model, _perf(compile_scope="model"), shape_count=1)
    assert model._compiled_call_impl is not None
    assert all(block._compiled_call_impl is None for block in model.blocks)
    assert "scope=model targets=1" in compile_decision()


def test_unrecognized_core_falls_back_to_model_scope(dynamo_limits):
    model = nn.Sequential(nn.Linear(4, 4))
    compile_model(model, _perf(), shape_count=1)
    assert model._compiled_call_impl is not None
    assert "scope=model" in compile_decision()


# -- properties the call site depends on -------------------------------------
def test_block_compile_leaves_state_dict_keys_untouched(dynamo_limits):
    model = _tiny()
    before = list(model.state_dict())
    compile_model(model, _perf(), shape_count=40)
    assert list(model.state_dict()) == before
    assert not any("_orig_mod" in key for key in model.state_dict())


def test_ema_shadow_stays_eager_and_keyed_identically(dynamo_limits):
    model = _tiny()
    ema = EMA(model, decay=0.999)
    compile_model(model, _perf(), shape_count=40)
    assert all(block._compiled_call_impl is None for block in ema.module.blocks)
    assert list(ema.module.state_dict()) == list(model.state_dict())
    ema.update(model)  # name-keyed update must still resolve every parameter


# -- config validation -------------------------------------------------------
def test_perf_validate_rejects_unknown_compile_enums():
    with pytest.raises(ValueError, match="compile_scope"):
        PerfConfig(compile_scope="whole").validate()
    with pytest.raises(ValueError, match="compile_dynamic"):
        PerfConfig(compile_dynamic="maybe").validate()
