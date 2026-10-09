"""Benchmark released IRIS weights on MPS, one image at a time.

Run: HF_HOME="$PWD/output/hf-cache" .venv/bin/python scripts/mac_benchmark.py
Text is encoded once, then Qwen is unloaded before IRIS is loaded. IRIS uses
BF16 parameters; solver state remains FP32. No CPU operator fallback is enabled.
"""

import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psutil
import torch
from torchvision.utils import save_image

from iris3b.config import inference_config
from iris3b.models import IrisDiT
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import generate, load_for_inference
from iris3b.text.base import TextEncoder, TextEncoding

PROMPTS = [
    "a red fox sleeping in fresh snow, golden hour, detailed wildlife photography",
    "a neon-lit Tokyo street in the rain at night, cinematic photography",
    "studio portrait of an elderly fisherman, weathered face, soft window light, detailed photography",
]
RUNS = [(0, 512, 20), (0, 1024, 20), (1, 1024, 20), (2, 1024, 100)]


class CachedText(TextEncoder):
    def __init__(self, cache):
        self.cache = cache

    def encode(self, prompts):
        entries = [self.cache[p] for p in prompts]
        return TextEncoding(torch.cat([e.embeddings for e in entries]), torch.cat([e.mask for e in entries]))

    def null(self, negative_prompt=""):
        return self.cache[negative_prompt]

    def to(self, device):
        return self


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run with Apple GPU access")
    out = Path("output/benchmark")
    out.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "precision": "bfloat16 weights, float32 solver", "cfg_scale": 3, "seed": 42, "runs": []}
    peaks = {"rss_bytes": 0, "mps_driver_bytes": 0, "mps_allocated_bytes": 0}
    stop = threading.Event()

    def memory():
        process = psutil.Process()
        while not stop.wait(0.5):
            for key, value in (("rss_bytes", process.memory_info().rss), ("mps_driver_bytes", torch.mps.driver_allocated_memory()), ("mps_allocated_bytes", torch.mps.current_allocated_memory())):
                peaks[key] = max(peaks[key], value)

    threading.Thread(target=memory, daemon=True).start()

    def persist():
        report["sampled_process_peaks"] = dict(peaks)
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(OmegaConf.load("output/iris-3b/config.yaml"))
    cfg = inference_config(raw, [])
    import iris3b.text  # register Qwen encoder

    print("Loading Qwen text encoder", flush=True)
    start = time.perf_counter()
    encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device="mps")
    torch.mps.synchronize()
    report["text_encoder_load_seconds"] = time.perf_counter() - start
    cache = {}
    start = time.perf_counter()
    for prompt in ["", *PROMPTS]:
        encoding = encoder.encode([prompt])
        cache[prompt] = TextEncoding(encoding.embeddings.cpu(), encoding.mask.cpu())
        print(f"Encoded: {prompt or '(empty prompt)'}", flush=True)
    report["encode_four_prompts_seconds"] = time.perf_counter() - start
    del encoder, encoding
    gc.collect()
    torch.mps.empty_cache()
    text = CachedText(cache)

    print("Loading IRIS in bfloat16", flush=True)
    start = time.perf_counter()
    _, weights = load_for_inference("output/iris-3b")
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().to(device="mps", dtype=torch.bfloat16)
    del weights
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    report["iris_load_seconds"] = time.perf_counter() - start
    persist()

    active = {"step": 0, "label": "warmup", "started": time.perf_counter()}

    def progress(module, args, output):
        active["step"] += 1
        torch.mps.synchronize()
        print(f"{active['label']}: step {active['step']}, {time.perf_counter() - active['started']:.1f}s elapsed", flush=True)

    hook = model.register_forward_hook(progress)

    def sample(index, size, steps):
        return generate(
            model, text, [PROMPTS[index]], height=size, width=size, steps=steps,
            cfg_scale=3, cfg_interval=tuple(cfg.sample.cfg_interval),
            shift=cfg.flow.shift, order=cfg.sample.order,
            generator=torch.Generator(device="mps").manual_seed(42), device="mps",
            num_train_timesteps=cfg.flow.num_train_timesteps, prediction=cfg.flow.prediction,
        )

    try:
        start = time.perf_counter()
        warmup = sample(0, 256, 2)
        del warmup
        report["warmup_seconds"] = time.perf_counter() - start
        for number, (index, size, steps) in enumerate(RUNS, 1):
            active.update(step=0, label=f"image {number}/4 {size}px {steps} steps", started=time.perf_counter())
            image = sample(index, size, steps)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - active["started"]
            if not image.isfinite().all().item():
                raise RuntimeError("Non-finite image pixels")
            file = out / f"{number:02d}_{size}px_{steps}steps.png"
            save_image(image.cpu(), str(file), normalize=True, value_range=(-1, 1))
            row = {"prompt": PROMPTS[index], "width": size, "height": size, "steps": steps, "seconds": elapsed, "seconds_per_step": elapsed / steps, "file": str(file.resolve()), "mps_driver_bytes": torch.mps.driver_allocated_memory()}
            report["runs"].append(row)
            persist()
            print(json.dumps(row), flush=True)
            del image
    finally:
        hook.remove()
        stop.set()
        persist()


if __name__ == "__main__":
    main()
