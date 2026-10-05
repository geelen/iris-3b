"""Text adapters behind ``model.text_adapter``.

"linear" is ``Linear + RMSNorm``; "blocks2" adds two unconditioned transformer
blocks at backbone width; "lap_blocks2" first aggregates the selected encoder
layers per token, then refines across tokens with the same two blocks.
"""

from dataclasses import replace

import pytest
import torch

from iris3b.config import (
    Config,
    DataConfig,
    EMAConfig,
    OptimizerConfig,
    TextEncoderConfig,
    TrainConfig,
)
from iris3b.models import IrisDiT
from iris3b.nn.embeddings import LayerwiseTextEmbedder, TransformerTextEmbedder
from iris3b.registry import TEXT_ENCODERS
from iris3b.seeding import mix_seed
from iris3b.text.base import TextEncoder, TextEncoding
from tiny_config import tiny_model_config as tiny


def test_linear_adapter_ignores_the_mask():
    torch.manual_seed(0)
    model = IrisDiT(tiny()).eval()
    args = (torch.randn(2, 3, 32, 32), torch.tensor([7.0, 3.0]), torch.randn(2, 16, 32))
    half = torch.zeros(2, 16, dtype=torch.int64)
    half[:, :4] = 1
    assert torch.equal(model(*args).x, model(*args, y_mask=half).x)
    assert torch.equal(model(*args).x, model(*args, y_mask=torch.ones(2, 16, dtype=torch.int64)).x)


def test_lap_blocks2_rejects_text_states_without_the_configured_layers():
    cfg = tiny(
        text_adapter="lap_blocks2",
        text_lap_num_layers=3,
        text_lap_num_heads=4,
        text_lap_mlp_ratio=1.3,
    )
    model = IrisDiT(cfg).eval()
    mask = torch.ones(2, 16, dtype=torch.int64)
    with pytest.raises(ValueError, match="layerwise text states"):
        model.y_embedder(torch.randn(2, 16, 32), mask)
    with pytest.raises(ValueError, match="layerwise text states"):
        model.y_embedder(torch.randn(2, 16, 2, 32), mask)


def test_blocks2_without_mask_raises_and_names_the_call_site():
    torch.manual_seed(0)
    model = IrisDiT(tiny(text_adapter="blocks2")).eval()
    with pytest.raises(ValueError, match="requires y_mask") as excinfo:
        model(torch.randn(1, 3, 32, 32), torch.tensor([1.0]), torch.randn(1, 16, 32))
    assert "test_text_adapter.py" in str(excinfo.value)


def test_pad_positions_cannot_influence_real_tokens():
    """The load-bearing masking test: perturbing the INPUT at pad positions must
    leave every real-token output bitwise unchanged. Fails maskless, because
    unmasked self-attention mixes the 300-token padded sequence."""
    torch.manual_seed(0)
    adapter = TransformerTextEmbedder(32, 64, num_blocks=2, num_heads=4).eval()
    y = torch.randn(2, 16, 32)
    mask = torch.zeros(2, 16, dtype=torch.int64)
    mask[0, :5] = 1
    mask[1, :11] = 1

    perturbed = y.clone()
    keep = mask.bool()
    perturbed[~keep] = torch.randn_like(perturbed[~keep]) * 7.0
    base, moved = adapter(y, mask), adapter(perturbed, mask)
    assert torch.equal(base[keep], moved[keep])
    # the pad rows themselves DID move, so the test is not vacuous
    assert not torch.equal(base[~keep], moved[~keep])


def test_lap_pad_positions_cannot_influence_real_tokens():
    torch.manual_seed(0)
    adapter = LayerwiseTextEmbedder(32, 64, 3, 4, refiner_num_heads=4).eval()
    y = torch.randn(2, 16, 3, 32)
    mask = torch.zeros(2, 16, dtype=torch.int64)
    mask[0, :5] = 1
    mask[1, :11] = 1
    perturbed = y.clone()
    keep = mask.bool()
    perturbed[~keep] = torch.randn_like(perturbed[~keep]) * 7.0
    base, moved = adapter(y, mask), adapter(perturbed, mask)
    assert torch.equal(base[keep], moved[keep])
    assert not torch.equal(base[~keep], moved[~keep])


def test_all_real_row_is_unaffected_by_masking():
    torch.manual_seed(0)
    adapter = TransformerTextEmbedder(32, 64, num_blocks=2, num_heads=4).eval()
    y = torch.randn(2, 16, 32)
    full = torch.ones(2, 16, dtype=torch.int64)
    torch.testing.assert_close(adapter(y, full), _maskless(adapter, y))
    # a fully padded row degenerates to identity attention instead of NaN
    assert torch.isfinite(adapter(y, torch.zeros(2, 16, dtype=torch.int64))).all()


def _maskless(adapter: TransformerTextEmbedder, y: torch.Tensor) -> torch.Tensor:
    """Reference forward with attention over every position, no mask at all."""
    h = adapter.proj(y)
    for block in adapter.blocks:
        h = block(h)
    return adapter.norm(h)


@TEXT_ENCODERS.register("stub_padded")
class PaddedStubTextEncoder(TextEncoder):
    """Deterministic embeddings with a REAL padded mask: the first
    ``1 + len(prompt) % max_length`` positions are real, the rest is pad. No
    downloads. Used to drive the mask through the trainer and the sampler."""

    def __init__(self, cfg, device="cpu"):
        self.dim = cfg.dim
        self.max_length = cfg.max_length
        self.hidden_layers = tuple(cfg.hidden_layers)
        self.device = torch.device(device)

    def _embed(self, prompts: list[str]) -> TextEncoding:
        rows, masks = [], []
        for p in prompts:
            gen = torch.Generator().manual_seed(mix_seed(len(p), *(ord(c) for c in p[:32])))
            row = torch.randn(self.max_length, self.dim, generator=gen)
            if self.hidden_layers:
                row = torch.stack([row + layer for layer in self.hidden_layers], dim=1)
            rows.append(row)
            real = 1 + len(p) % self.max_length
            m = torch.zeros(self.max_length, dtype=torch.long)
            m[:real] = 1
            masks.append(m)
        return TextEncoding(
            embeddings=torch.stack(rows).to(self.device),
            mask=torch.stack(masks).to(self.device),
        )

    def encode(self, prompts: list[str]) -> TextEncoding:
        return self._embed(list(prompts))

    def null(self, negative_prompt: str = "") -> TextEncoding:
        return self._embed([negative_prompt])

    def to(self, device) -> "PaddedStubTextEncoder":
        self.device = torch.device(device)
        return self


def _smoke_cfg(work_dir, skip_dropped_text: bool) -> Config:
    return Config(
        name="lap_blocks2-smoke",
        work_dir=str(work_dir),
        report_to="none",
        model=tiny(
            text_adapter="lap_blocks2",
            text_lap_num_layers=3,
            text_lap_num_heads=4,
            text_lap_mlp_ratio=1.3,
        ),
        text_encoder=TextEncoderConfig(name="stub_padded", dim=32, max_length=16, hidden_layers=[1, 2, 3]),
        repa=replace(Config().repa, weight=0.0),
        data=DataConfig(
            type="synthetic", image_size=32, num_workers=0, val_data_dirs=["ignored-by-synthetic"]
        ),
        train=TrainConfig(
            batch_size=2,
            num_epochs=1,
            max_steps=2,
            mixed_precision="no",
            text_dropout=0.5,  # exercises the CFG-dropout mask substitution
            seed=7,
            optimizer=OptimizerConfig(name="adamw", lr=1e-3, auto_lr="none", warmup_steps=1),
            ema=EMAConfig(enabled=True, decay=0.999),
            save_every_steps=0,
            save_every_epochs=0,
            sample_every_steps=2,  # exercises generate() incl. the CFG mask batch
            val_every_steps=2,  # exercises the frozen-grid val loss path
            val_samples=4,
            log_every=1000,
            validation_prompts=["a red cube", "a long prompt about a small dog"],
            perf=replace(Config().train.perf, skip_dropped_text=skip_dropped_text),
        ),
    )


@pytest.mark.parametrize("skip_dropped_text", [False, True])
def test_trainer_and_sampler_feed_the_mask(tmp_path, skip_dropped_text):
    """A full trainer step proves every model call transports masks and layered
    embeddings through dropout, validation loss, and CFG sampling."""
    from accelerate.state import AcceleratorState

    from iris3b.train.trainer import Trainer

    AcceleratorState._reset_state()
    cfg = _smoke_cfg(tmp_path, skip_dropped_text)
    Trainer(cfg).run()
    assert sorted(p.name for p in (tmp_path / "log_vis").glob("*.webp"))


def test_adapter_carries_no_positions():
    """No RoPE inside the blocks: with an all-real mask the adapter is
    permutation equivariant over the token axis, so all positional information
    keeps coming from ``y_pos_embedding`` (added after it) and the main model's
    text RoPE."""
    torch.manual_seed(0)
    adapter = TransformerTextEmbedder(32, 64, num_blocks=2, num_heads=4).eval()
    y = torch.randn(1, 16, 32)
    full = torch.ones(1, 16, dtype=torch.int64)
    perm = torch.randperm(16)
    torch.testing.assert_close(adapter(y, full)[:, perm], adapter(y[:, perm], full))


def test_gradients_reach_both_adapter_blocks():
    torch.manual_seed(0)
    cfg = tiny(text_adapter="blocks2")
    model = IrisDiT(cfg).train()
    out = model(
        torch.randn(2, 3, 32, 32),
        torch.tensor([5.0, 1.0]),
        torch.randn(2, 16, 32),
        capture=(cfg.depth,),
        y_mask=torch.ones(2, 16, dtype=torch.int64),
    )
    # the output head is zero-init, so a loss on model output would carry no
    # gradient back to the adapter; read the last block's patch tokens instead
    out.features[cfg.depth].pow(2).sum().backward()
    named = dict(model.y_embedder.named_parameters())
    touched = [n for n, p in named.items() if p.grad is not None and p.grad.abs().sum() > 0]
    for i in (0, 1):
        assert any(n.startswith(f"blocks.{i}.attn.") for n in touched), (i, touched)
        assert any(n.startswith(f"blocks.{i}.mlp.") for n in touched), (i, touched)
        assert any(n.startswith(f"blocks.{i}.norm") for n in touched), (i, touched)
    assert "proj.weight" in touched and "norm.weight" in touched


def test_gradients_reach_lap_and_refiner_blocks():
    torch.manual_seed(0)
    cfg = tiny(
        text_adapter="lap_blocks2",
        text_lap_num_layers=3,
        text_lap_num_heads=4,
        text_lap_mlp_ratio=1.3,
    )
    model = IrisDiT(cfg).train()
    out = model(
        torch.randn(2, 3, 32, 32),
        torch.tensor([5.0, 1.0]),
        torch.randn(2, 16, 3, 32),
        capture=(cfg.depth,),
        y_mask=torch.ones(2, 16, dtype=torch.int64),
    )
    out.features[cfg.depth].pow(2).sum().backward()
    touched = {
        name
        for name, parameter in model.y_embedder.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    }
    for index in (0, 1):
        assert any(name.startswith(f"layer_blocks.{index}.attn.") for name in touched)
        assert any(name.startswith(f"layer_blocks.{index}.mlp.") for name in touched)
        assert any(name.startswith(f"refiner.blocks.{index}.") for name in touched)
    assert "layer_pool.weight" in touched
    assert "refiner.proj.weight" in touched


def test_unknown_text_adapter_raises():
    with pytest.raises(ValueError, match="text_adapter"):
        IrisDiT(tiny(text_adapter="blocks4"))
