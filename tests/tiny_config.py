"""A tiny CPU-sized model config for unit tests."""

from dataclasses import replace

from iris3b.config import ModelConfig, PixelStageConfig

TINY_MODEL = ModelConfig(
    block="mmdit",
    dual_depth=0,
    hidden_size=64,
    depth=2,
    num_heads=4,
    num_kv_heads=None,
    gated_attention=False,
    sandwich_norm=False,
    patch_size=4,
    modulation="per_block",
    adaln_zero_init=False,
    rope_aspect="square",
    text_dim=32,
    text_len=16,
    text_adapter="linear",
    text_lap_num_layers=0,
    repa_layer=1,
    pixel=PixelStageConfig(depth=1, hidden_size=8, attn_hidden_size=64, num_heads=4, modulation="pre"),
)


def tiny_model_config(**overrides) -> ModelConfig:
    return replace(TINY_MODEL, **overrides)
