# Apple GPU compatibility check

Tested on 2026-10-09 on an M4 Pro MacBook Pro with 48 GB unified memory,
using upstream commit `a8d1523` and Python 3.12.12.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-mac-smoke.txt
.venv/bin/python scripts/mac_smoke.py
```

Run from the repository directory in a terminal with access to the Apple GPU.
Codex's sandbox reports MPS unavailable; running the test outside it succeeds.
The test downloads no checkpoints and produces no meaningful image.

## Results

A random reduced IRIS model (one dual-stream and one single-stream block,
grouped-query attention, shared modulation, layerwise text adapter, and pixel
head) completed two guided sampling steps at 32×32 on MPS. Its maximum absolute
output difference from the float32 CPU reference was:

| Model precision | Maximum absolute difference |
| --- | --- |
| float32 | 0.0000001192 |
| float16 | 0.0002046 |
| bfloat16 | 0.002271 |

A random two-layer Qwen3-VL text decoder passed in bfloat16 with masked SDPA
attention and hidden-state output. Maximum absolute difference from CPU was
0.01875. The MPS random generator also passed. No upstream model changes were
required. The reduced models retain component types but shrink dimensions and
layer counts, so these checks do not establish full-model memory use or speed.

## Interpretation

IRIS uses a familiar diffusion-transformer trunk, but its direct pixel output
head and layerwise Qwen conditioning are model-specific. Existing MPS kernels
are reusable immediately. An MLX implementation of FLUX provides useful
reference code, but does not directly load IRIS checkpoints.

The released image-generation weights are roughly 12 GB in float32; the Qwen
checkpoint is an additional download. The stock inference
script accepts `--device mps` but keeps IRIS weights in float32 and enables
bfloat16 autocast only on CUDA.

## Full-checkpoint benchmark

The released weights run on this Mac through PyTorch MPS. The benchmark uses
BF16 model parameters, FP32 solver state, CFG 3, seed 42, and one image per
batch. Qwen encodes all prompts first and is then unloaded before IRIS loads.
CPU operator fallback is explicitly disabled. No upstream model changes were
needed. This is a custom inference harness rather than the stock FP32 script.

| Image | Resolution | Steps | Generation time |
| --- | --- | --- | --- |
| [Fox](benchmarks/m4-pro-2026-10-09/01_512px_20steps.png) | 512×512 | 20 | 47.37 seconds |
| [Fox](benchmarks/m4-pro-2026-10-09/02_1024px_20steps.png) | 1024×1024 | 20 | 166.71 seconds |
| [Tokyo street](benchmarks/m4-pro-2026-10-09/03_1024px_20steps.png) | 1024×1024 | 20 | 165.14 seconds |
| Fisherman portrait | 1024×1024 | 100 | Running |

These are individual synchronized runs. Generation includes every model step
but excludes downloads, model loading, initial prompt encoding, warmup, and
PNG saving. Qwen loading took 16.19 seconds, four prompt encodings took 2.96
seconds, IRIS loading took 8.56 seconds, and a two-step 256px warmup took 3.77
seconds. Per-step synchronization supports progress reporting and may add a
small overhead.

The 512px fox has prominent grid artifacts. The native 1024px fox and street
images are visually much cleaner, although the street's signs contain invented
text. These different resolutions do not establish a precision-related cause
for the artifacts. There is no full-checkpoint FP32 or CUDA quality baseline.

Reproduce after downloading the checkpoints:

```sh
HF_HOME="$PWD/output/hf-cache" PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python -u scripts/mac_benchmark.py
```

Run in a terminal with Apple GPU access. The harness expects IRIS files in
`output/iris-3b` and Qwen in the specified Hugging Face cache. It writes images
and raw measurements to `output/benchmark`. Archived images and portable
measurements are in `benchmarks/m4-pro-2026-10-09`. All three downloaded weight
files passed SHA256 validation against their Hugging Face metadata.

Sources: [IRIS model card](https://huggingface.co/speridlabs/iris-3b),
[Apple's FLUX MLX example](https://github.com/ml-explore/mlx-examples/tree/main/flux).
