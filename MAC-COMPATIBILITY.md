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
checkpoint is an additional download. A full checkpoint run is still needed to
measure memory, 1024px performance, and image quality. The stock inference
script accepts `--device mps` but keeps IRIS weights in float32 and enables
bfloat16 autocast only on CUDA.

Sources: [IRIS model card](https://huggingface.co/speridlabs/iris-3b),
[Apple's FLUX MLX example](https://github.com/ml-explore/mlx-examples/tree/main/flux).
