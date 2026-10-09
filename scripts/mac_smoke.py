"""Small, weight-free Apple GPU compatibility check; not a quality/speed benchmark."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from iris3b.config import ModelConfig, PixelStageConfig
from iris3b.models import IrisDiT
from iris3b.sampling import generate
from iris3b.text.base import TextEncoder, TextEncoding


class SyntheticText(TextEncoder):
    dim = 32
    max_length = 8

    def encode(self, prompts):
        return TextEncoding(torch.ones(len(prompts), 8, 2, 32), torch.ones(len(prompts), 8, dtype=torch.long))

    def null(self, negative_prompt=""):
        return self.encode([negative_prompt])

    def to(self, device):
        return self


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU unavailable to this Python process")
    cfg = ModelConfig(
        hidden_size=64, depth=2, dual_depth=1, num_heads=4, num_kv_heads=1,
        text_dim=32, text_len=8, text_lap_num_layers=2, text_lap_num_heads=4,
        patch_size=16, repa_layer=0, adaln_zero_init=False,
        pixel=PixelStageConfig(depth=1, hidden_size=8, attn_hidden_size=64, num_heads=4),
    )
    torch.manual_seed(7)
    model = IrisDiT(cfg).eval()
    # Avoid the zero-initialized output head making numeric checks vacuous.
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.02)
    noise = torch.randn(1, 3, 32, 32)
    results = {"torch": torch.__version__, "scope": "random tiny hybrid model + synthetic text, 32px, 2 steps", "runs": []}
    with torch.inference_mode():
        reference = generate(model, SyntheticText(), ["probe"], 32, 32, steps=2, noise=noise, device="cpu")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            start = time.perf_counter()
            try:
                model.to(device="mps", dtype=dtype)
                output = generate(model, SyntheticText(), ["probe"], 32, 32, steps=2, noise=noise, device="mps")
                torch.mps.synchronize()
                error = (output.cpu() - reference).abs().max().item()
                if not output.isfinite().all().item() or error > 0.05:
                    raise RuntimeError(f"Numeric check failed: max absolute CPU difference {error}")
                results["runs"].append({"dtype": str(dtype), "passed": True, "max_abs_cpu_difference": error, "seconds": time.perf_counter() - start})
            except Exception as exc:
                results["runs"].append({"dtype": str(dtype), "passed": False, "error": str(exc)})
    print(json.dumps(results, indent=2))
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    text_cfg = Qwen3VLTextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
        head_dim=16, rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
    )
    text_cfg._attn_implementation = "sdpa"
    with torch.inference_mode():
        decoder = Qwen3VLTextModel(text_cfg).eval()
        ids = torch.tensor([[1, 2, 3, 4, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        reference = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        decoder.to(device="mps", dtype=torch.bfloat16)
        output = decoder(input_ids=ids.to("mps"), attention_mask=mask.to("mps"), use_cache=False, output_hidden_states=True)
        difference = (output.last_hidden_state.float().cpu() - reference).abs().max().item()
        assert output.last_hidden_state.isfinite().all().item() and difference < 0.05
        assert len(output.hidden_states) == 3
        generator = torch.Generator(device="mps").manual_seed(7)
        assert torch.randn(4, generator=generator, device="mps").isfinite().all().item()
    print(json.dumps({"qwen3_vl_tiny_bfloat16": "passed", "max_abs_cpu_difference": difference, "mps_generator": "passed"}, indent=2))
    sys.exit(0 if all(run["passed"] for run in results["runs"]) else 1)


if __name__ == "__main__":
    main()
