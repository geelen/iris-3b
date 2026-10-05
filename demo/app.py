"""Iris-3B text-to-image demo."""

import json
import os
import random
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image

import iris3b.text  # noqa: F401  (registers encoders)
from iris3b.config import inference_config
from iris3b.models.dit import IrisDiT
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import generate, load_for_inference

try:
    import spaces

    gpu = spaces.GPU(duration=120)
except ImportError:
    def gpu(fn):
        return fn

MODEL_REPO = os.environ.get("IRIS_MODEL_REPO", "speridlabs/iris-3b")
# (height, width) buckets with real training mass (>=2% of the 1024 stage, >=4% of SFT);
# the rarer training buckets show artifacts
SIZES = {f"{w}×{h}": (h, w) for h, w in [(768, 1344), (832, 1280), (896, 1152), (1024, 1024),
                                          (1152, 896), (1280, 832), (1344, 768)]}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source = MODEL_REPO if os.path.isdir(MODEL_REPO) else snapshot_download(MODEL_REPO, token=os.environ.get("HF_TOKEN"))
raw, weights = load_for_inference(source)
cfg = inference_config(raw, [])
with torch.device("meta"):
    model = IrisDiT(cfg.model)
model.load_state_dict(weights, strict=True, assign=True)
model = model.eval().to(device=DEVICE, dtype=torch.float32)  # FP32 weights, BF16 autocast at inference
text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=DEVICE)


@gpu
def infer(prompt, negative, size, steps, cfg_scale, seed, randomize):
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    height, width = SIZES[size]
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        image = generate(
            model, text_encoder, [prompt], height=height, width=width, steps=int(steps), order=cfg.sample.order,
            cfg_scale=cfg_scale, cfg_interval=tuple(cfg.sample.cfg_interval), shift=cfg.flow.shift,
            negative_prompt=negative, generator=torch.Generator(device=DEVICE).manual_seed(int(seed)), device=DEVICE,
            num_train_timesteps=cfg.flow.num_train_timesteps, prediction=cfg.flow.prediction,
        )[0]
    pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels), seed


ROOT = Path(__file__).parent
# curated gallery images and the exact settings they were sampled with (100 steps, CFG 3.0)
PRESETS = json.loads((ROOT / "examples/presets.json").read_text())


def meta(kind, size, seed, steps, cfg_scale):
    return (f'<div class="sl-meta"><span>{kind}</span><span>{size} · seed {int(seed)} · '
            f"{int(steps)} steps · CFG {cfg_scale:.1f}</span></div>")


def run(prompt, negative, size, steps, cfg_scale, seed, randomize):
    image, seed = infer(prompt, negative, size, steps, cfg_scale, seed, randomize)
    return image, seed, meta("Generated", size, seed, steps, cfg_scale)


def pick(evt: gr.SelectData):
    p = PRESETS[evt.index]
    return (p["prompt"], p["size"], p["steps"], p["cfg"], p["seed"], False, str(ROOT / p["image"]),
            meta("Preset", p["size"], p["seed"], p["steps"], p["cfg"]))


FONT = [gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
MONO = [gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"]
theme = gr.themes.Base(font=FONT, font_mono=MONO, radius_size=gr.themes.sizes.radius_none).set(
    body_background_fill="#ffffff", body_background_fill_dark="#000000",
    body_text_color="#000000", body_text_color_dark="#ffffff",
    block_background_fill="#ffffff", block_background_fill_dark="#000000",
    background_fill_primary="#ffffff", background_fill_primary_dark="#000000",
    background_fill_secondary="#f7f7f7", background_fill_secondary_dark="#111111",
    border_color_primary="#e3e3e3", border_color_primary_dark="rgba(255,255,255,0.15)",
    block_border_color="#e3e3e3", block_border_color_dark="rgba(255,255,255,0.15)", block_shadow="none",
    input_background_fill="#ffffff", input_background_fill_dark="#000000",
    input_border_color="#d4d4d4", input_border_color_dark="rgba(255,255,255,0.25)",
    input_border_color_focus="#000000", input_border_color_focus_dark="#ffffff",
    button_primary_background_fill="#000000", button_primary_background_fill_dark="#ffffff",
    button_primary_background_fill_hover="#d94a1f", button_primary_background_fill_hover_dark="#d94a1f",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#000000",
    color_accent="#d94a1f", color_accent_soft="#fbe7e0", slider_color="#d94a1f", slider_color_dark="#d94a1f",
    checkbox_background_color_selected="#000000", checkbox_background_color_selected_dark="#ffffff",
)
CSS = """
:root { --sl-fg: #000; --sl-fg-2: #444; --sl-fg-3: #6b6b6b; --sl-rule: #e3e3e3; }
.dark { --sl-fg: #fff; --sl-fg-2: rgba(255,255,255,.75); --sl-fg-3: rgba(255,255,255,.55);
  --sl-rule: rgba(255,255,255,.15); }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 32px !important; }
footer { display: none !important; }
.sl-head { padding: 40px 0 28px; border-bottom: 1px solid var(--sl-rule); margin-bottom: 8px; }
.sl-label, .sl-meta, .sl-links a, .sl-links button, .sl-section, .block-label, .label-wrap span, label > span {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important; text-transform: uppercase;
  letter-spacing: .12em; font-size: 10.5px !important; color: var(--sl-fg-3) !important; }
.sl-head h1 { margin: 14px 0 0; font-size: clamp(56px, 6.4vw, 104px); font-weight: 700; line-height: .86;
  letter-spacing: -0.055em; color: var(--sl-fg); }
.sl-row { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 16px 40px; }
.sl-head p { margin: 18px 0 0; max-width: 60ch; font-size: 17px; line-height: 1.5; color: var(--sl-fg-2); }
.sl-head p strong { color: var(--sl-fg); font-weight: 600; }
.sl-head p em { color: #d94a1f; font-style: normal; font-weight: 600; }
.sl-links { display: flex; gap: 10px; }
.sl-links a, .sl-links button { display: inline-flex; align-items: center; height: 40px; padding: 0 20px;
  border: 1px solid var(--sl-fg); background: transparent; color: var(--sl-fg) !important; text-decoration: none;
  cursor: pointer; transition: background .15s, color .15s; }
.sl-links a:hover, .sl-links button:hover { background: var(--sl-fg); color: var(--body-background-fill) !important; }
#sl-theme { width: 40px; padding: 0; justify-content: center; font-size: 15px !important; letter-spacing: 0; }
#sl-theme::before { content: "☾"; }
.dark #sl-theme::before { content: "☀"; }
.sl-meta { display: flex; justify-content: space-between; gap: 16px; padding-top: 6px; }
.sl-meta span:first-child { color: #d94a1f; }
.sl-section { border-top: 1px solid var(--sl-rule); padding-top: 28px !important; margin-top: 28px; }
#generate { height: 48px; font-size: 14px; font-weight: 500; letter-spacing: .01em; }
#result { background: #0c0c0c; }
#result img { object-fit: contain; }
#presets .thumbnail-item { border: 0 !important; box-shadow: none !important; overflow: hidden; }
#presets .thumbnail-item img { transition: transform .6s ease-out; }
#presets .thumbnail-item:hover img { transform: scale(1.04); }
#presets .caption-label { display: none; }
#presets .grid-wrap { max-height: none !important; height: auto !important; overflow: visible !important; }
#presets .thumbnail-item.selected { outline: 2px solid #d94a1f; outline-offset: -2px; }
@media (max-width: 760px) {
  .gradio-container { padding: 0 16px !important; }
  .sl-head { padding-top: 24px; }
  #presets .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
}
"""
# dark by default; the header toggle flips gradio's `dark` class and remembers the choice
THEME_JS = """() => {
  const set = (dark) => { document.body.classList.toggle('dark', dark); localStorage.setItem('iris-theme', dark ? 'dark' : 'light'); };
  set(localStorage.getItem('iris-theme') !== 'light');
  document.addEventListener('click', (e) => { if (e.target.closest('#sl-theme')) set(!document.body.classList.contains('dark')); });
}"""
HEAD = """
<div class="sl-head">
  <span class="sl-label">Speridlabs · Research</span>
  <div class="sl-row">
    <h1>Iris-3B</h1>
    <div class="sl-links">
      <a href="https://speridlabs.com/research/iris" target="_blank">Research post</a>
      <a href="https://github.com/speridlabs/iris-3b" target="_blank">Code</a>
      <button id="sl-theme" type="button" aria-label="Toggle light/dark"></button>
    </div>
  </div>
  <p><strong>Pixel-space text-to-image diffusion.</strong> No VAE: a 3B transformer generates every pixel
  directly. Pick a preset below to load its prompt and settings, or <em>write your own.</em></p>
</div>
"""

first = PRESETS[0]
with gr.Blocks(title="Iris-3B · Speridlabs", theme=theme, css=CSS, js=THEME_JS) as demo:
    gr.HTML(HEAD)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            prompt = gr.Textbox(label="Prompt", lines=4, value=first["prompt"])
            run_btn = gr.Button("Generate", variant="primary", elem_id="generate")
            with gr.Accordion("Settings", open=False):
                negative = gr.Textbox(label="Negative prompt", value=cfg.sample.negative_prompt)
                size = gr.Dropdown(list(SIZES), value=first["size"], label="Size (width × height)")
                steps = gr.Slider(10, 150, value=50, step=1, label="Steps")
                cfg_scale = gr.Slider(1.0, 10.0, value=first["cfg"], step=0.1, label="Guidance scale")
                seed = gr.Number(value=first["seed"], precision=0, label="Seed")
                randomize = gr.Checkbox(value=False, label="Random seed")
        with gr.Column(scale=7):
            image = gr.Image(value=str(ROOT / first["image"]), type="pil", format="png", show_label=False,
                             elem_id="result", height="min(640px, 92vw)")
            info = gr.HTML(meta("Preset", first["size"], first["seed"], first["steps"], first["cfg"]))
    gr.HTML('<div class="sl-section">Presets — click to load prompt and settings</div>')
    presets = gr.Gallery([str(ROOT / p["image"]) for p in PRESETS], columns=6, rows=4, height="auto",
                         object_fit="cover", allow_preview=False, show_label=False, show_share_button=False,
                         show_download_button=False, elem_id="presets")
    inputs = [prompt, negative, size, steps, cfg_scale, seed, randomize]
    run_btn.click(run, inputs, [image, seed, info])
    prompt.submit(run, inputs, [image, seed, info])
    presets.select(pick, None, [prompt, size, steps, cfg_scale, seed, randomize, image, info])

if __name__ == "__main__":
    demo.queue().launch()
