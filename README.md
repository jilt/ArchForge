# Arch Forge

**A one-node SDXL sketch-to-environment enhancer for ComfyUI.** Arch Forge turns rough architectural sketches and base renders into photoreal cyberpunk, sci-fi, and outdoor scenes on practical 8–12 GB VRAM GPUs.

It packages the controls that normally sprawl across a multi-node graph: three LoRA slots, Canny + Depth ControlNet, img2img sampling, upscale and sharpen, plus optional automatic perspective correction.

> **Positioning:** built for fast, local architectural environment exploration — especially sci-fi, cyberpunk, game-art, and outdoor concepts — rather than a full BIM-to-final-video pipeline.

## Why Arch Forge

Many high-end archviz workflows use large FLUX and video models, multiple workflow files, or substantial VRAM. Arch Forge deliberately uses the SDXL ecosystem so it remains practical on common 8–12 GB GPUs.

| | Arch Forge | Multi-stage FLUX/LTX archviz workflows |
|---|---|---|
| Core use | Sketch/render to environment concept | Generate, edit, and animate a full archviz pipeline |
| Form factor | One ComfyUI custom node | Multiple workflow JSON files or subgraphs |
| Model focus | SDXL checkpoints and LoRAs | FLUX-class image models and LTX-class video models |
| Hardware target | Typical 8–12 GB SDXL setup | Often higher VRAM or cloud-oriented |
| Video/editing | Not included | Often includes masking, editing, and image-to-video |
| Best for | Fast local iteration and game-style architectural realism | Long-form technical archviz production |

Arch Forge is not intended to replace advanced editing or camera-controlled video workflows. It is intended to get a structurally guided architectural image from sketch to believable environment quickly, locally, and without building a large graph every time.

## Features

- **One-node workflow** — exposes an architectural enhancement pipeline through one ComfyUI node
- **Low-VRAM SDXL focus** — designed around common 8–12 GB GPU setups rather than FLUX/LTX-scale requirements
- **Three LoRA slots** — independent model and CLIP strengths for each LoRA
- **Dual ControlNet guidance** — Canny preserves edges; Depth preserves volume and scene structure
- **Prompt controls** — full positive and negative prompts, with optional image interrogation support
- **Img2img sampling controls** — seed, steps, CFG, sampler, scheduler, and start/end-step controls
- **Upscale and sharpen** — selectable upscaler plus adjustable radius, sigma, and alpha
- **Optional Nuke Corner Pin** — manual four-point warp for advanced compositing use cases
- **Perspective correction** — optional automatic straightening of converging verticals with a strength control and border crop
- **Custom compact UI** — collapsed control groups keep a complex pipeline manageable

## Installation

### ComfyUI Manager

Once published, search **Arch Forge** in ComfyUI Manager and choose **Install**.

### Manual installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-ArchForge.git arch_forge
cd arch_forge
../../python_embeded/python.exe -m pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Requirements

- ComfyUI 0.19.3 or later recommended
- Python 3.10 or later
- An SDXL-compatible checkpoint
- NVIDIA GPU with 8–12 GB VRAM recommended; actual use depends on resolution, checkpoint, LoRAs, ControlNets, and upscaler
- `opencv-python>=4.8.0` and `numpy>=1.26.0` for optional perspective correction

The perspective helper uses stable OpenCV primitives (`Canny`, probabilistic Hough lines, homography warp) and standard NumPy operations. It works with newer installed versions; for example OpenCV 4.13 and NumPy 2.4 satisfy these minimums.

## Optional node dependencies

Arch Forge itself is a custom node. Some capabilities depend on other ComfyUI node packs; unavailable optional features are skipped rather than blocking the basic SDXL pipeline.

| Dependency | Used for | Required? |
|---|---|---|
| [ComfyUI-Zoe-Depth-Anything](https://github.com/Kosinkadink/ComfyUI-Zoe-Depth-Anything) | Creates a depth image for Depth ControlNet | Recommended; falls back to the input image if unavailable |
| [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) | Optional image interrogation / prompt enhancement, if enabled in your build | Optional |
| [nuke-nodes-comfyui](https://github.com/sumitchatterjee13/nuke-nodes-comfyui) | Manual NukeCornerPin four-point warp | Optional |

### Important: two different correction tools

- **Perspective Correction** is the preferred architecture tool. It detects dominant line structure, estimates a vertical vanishing point, then applies a homography intended to make converging verticals parallel.
- **NukeCornerPin** is an advanced manual four-point warp. It is useful for arbitrary compositing or deliberate target-corner mapping; it does not automatically know which building lines should be straight.

For normal architectural straightening, use **Perspective Correction**. Keep Nuke Corner Pin disabled unless you specifically need manual corner control.

## Quick start

1. Add **Arch Forge** from `arch_forge` in the ComfyUI node menu.
2. Connect a rough architecture sketch, clay render, concept render, or composited base image to `input_image`.
3. Select an SDXL checkpoint, Canny ControlNet, Depth ControlNet, and an upscaler available in your ComfyUI model folders.
4. Start with the default Canny and Depth strengths, then adjust prompt, LoRAs, and denoise timing to match your source image.
5. Queue the workflow.
6. For converging building lines, open **🏛 Perspective**, enable **auto-straighten**, then begin with Strength `1.0` and Auto-crop enabled.

## Workflow examples

The repository includes two starter workflow examples.

### Cyberpunk / sci-fi architecture

`examples/cyberpunk_city_workflow.json`

Use a rough building or city sketch as the source, then guide SDXL with stronger Canny structure and selected sci-fi/environment LoRAs.

Suggested starting point:

- Canny strength: `0.8`
- Depth strength: `0.5`
- Canny timing: `0.00–0.90`
- Depth timing: `0.20–0.70`
- Perspective correction: enable when generated verticals visibly converge

### Realistic outdoor architecture

`examples/outdoor_architecture_workflow.json`

Use an exterior concept, SketchUp-style view, clay render, or simple architectural image as the source; use a realistic SDXL checkpoint and modest Canny/Depth guidance.

Suggested starting point:

- Canny strength: `0.6`
- Depth strength: `0.4`
- Depth environment: `outdoor`
- Perspective correction: use only if the facade should read as a rectified architectural view

## Controls

### Checkpoint and LoRAs

| Parameter | Description |
|---|---|
| `checkpoint_name` | SDXL checkpoint used for generation |
| `lora_name_1` through `lora_name_3` | Up to three optional LoRA selections |
| `lora_strength_*_model` | Strength applied to the model side of each LoRA |
| `lora_strength_*_clip` | Strength applied to the CLIP side of each LoRA |

### Prompts

| Parameter | Description |
|---|---|
| `positive_prompt` | Desired materials, lighting, style, environment, and camera qualities |
| `negative_prompt` | Visual issues and unwanted styles to avoid |

### Canny ControlNet

| Parameter | Description |
|---|---|
| `canny_controlnet` | Edge-guidance ControlNet model |
| `canny_strength` | Overall Canny guidance strength |
| `canny_start` / `canny_end` | Portion of denoising during which Canny is active |
| `canny_enable_threshold` | Enables the Canny threshold filter |
| `canny_threshold_low` / `canny_threshold_high` | Lower and upper Canny thresholds |

### Depth ControlNet

| Parameter | Description |
|---|---|
| `depth_controlnet` | Depth-guidance ControlNet model |
| `depth_strength` | Overall Depth guidance strength |
| `depth_start` / `depth_end` | Portion of denoising during which Depth is active |
| `depth_environment` | `indoor` or `outdoor` depth-preprocessor mode |
| `depth_resolution` | Depth-map processing resolution |

### Post-processing

| Parameter | Description |
|---|---|
| `upscale_model` | Selected model from ComfyUI's `upscale_models` directory |
| `sharpen_radius` | Sharpening radius |
| `sharpen_sigma` | Sharpening sigma |
| `sharpen_alpha` | Sharpening intensity |

### Sampling

| Parameter | Description |
|---|---|
| `seed` | Random seed |
| `steps` | Total sampler steps |
| `cfg` | Classifier-free guidance scale |
| `sampler_name` | ComfyUI sampler selection |
| `scheduler` | ComfyUI scheduler selection |
| `start_at_step` | Img2img denoising start step |
| `end_at_step` | Img2img denoising end step |

### Perspective Correction

| Parameter | Default | Description |
|---|---:|---|
| `enable_perspective_correction` | `false` | Runs the integrated auto-straighten helper on the final image |
| `perspective_strength` | `1.0` | Correction blend strength from 0 to 1; lower it if the full correction feels too aggressive |
| `perspective_auto_crop` | `true` | Removes black borders introduced by the perspective warp |

The helper runs after decode, upscale, sharpen, and any enabled manual Nuke Corner Pin operation. It detects image edges, extracts likely vertical lines, estimates their vanishing point, computes a homography, and applies the transformation using OpenCV.

### Nuke Corner Pin

| Parameter | Description |
|---|---|
| `enable_nuke` | Enables the optional manual NukeCornerPin stage |
| `nuke_to1_*` through `nuke_to4_*` | Destination-corner coordinates for the manual four-point warp |
| `nuke_filter` | Resampling filter used by the Nuke node |

## Perspective correction notes

Auto-straighten is intended for single facades, exteriors, city scenes, and generated images with strong architectural edges. It is not a guaranteed camera-calibration tool.

For best results:

- Use images with clearly visible building edges or vertical facade lines.
- Keep the correction off for intentionally tilted, fisheye, heavily organic, or predominantly landscape images.
- If a correction is too aggressive, lower **Strength**.
- Enable **Auto-crop** to remove empty warped borders.
- If the result is not desirable, disable auto-straighten; the node returns the normal enhanced image when it cannot find sufficient line structure.

## Troubleshooting

### Node does not appear

- Confirm the repository is inside `ComfyUI/custom_nodes/arch_forge`.
- Restart ComfyUI after copying or updating files.
- Read the ComfyUI console for an import error in `arch_forge`.

### Perspective correction does not visibly change the image

- Confirm **Enable auto-straighten** is enabled in **🏛 Perspective**.
- Use an input/output with clear, dominant architectural verticals.
- Try Strength `1.0` for testing.
- The helper intentionally returns the unmodified image when it cannot find enough reliable lines.

### Perspective correction produces an unwanted warp

- Lower **Strength** to `0.25–0.75`.
- Disable the correction for wide-angle, organic, heavily diagonal, or non-architectural images.
- Use the optional manual Nuke Corner Pin only when you need explicit four-corner compositing control.

### Missing optional dependency

- Depth preprocessing: install `ComfyUI-Zoe-Depth-Anything`.
- Manual corner pin: install `nuke-nodes-comfyui`.
- Perspective correction: verify OpenCV and NumPy in ComfyUI's embedded Python:

```bat
.\python_embeded\python.exe -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"
```

## Project structure

```text
arch_forge/
├── __init__.py
├── arch_forge_node.py
├── perspective/
│   ├── __init__.py
│   ├── core.py
│   ├── lines.py
│   └── homography.py
├── web/
│   └── arch_forge.js
├── examples/
│   ├── cyberpunk_city_workflow.json
│   └── outdoor_architecture_workflow.json
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## ComfyUI Registry

Arch Forge is prepared for ComfyUI Registry publication through `pyproject.toml`. Before publishing:

1. Replace GitHub and ComfyUI publisher placeholders.
2. Keep `LICENSE`, `README.md`, `requirements.txt`, and `pyproject.toml` in the repository root.
3. Validate installation in a clean ComfyUI environment.
4. Increment the semantic version in `pyproject.toml` for each published release.

Suggested project description:

```toml
description = "One-node SDXL sketch-to-environment enhancer with auto perspective correction. Built for low-VRAM GPUs (8–12 GB). Dual ControlNet, 3 LoRAs, upscale, straighten verticals."
```

## Changelog

### v1.1.0

- Added integrated auto-straighten perspective correction
- Detects dominant architectural line structure and corrects converging verticals
- Added **🏛 Perspective** UI group: enable, strength, and auto-crop controls
- Uses OpenCV and NumPy only; no additional ComfyUI custom node is required for auto-straighten
- Retained optional NukeCornerPin for advanced manual four-point warps

### v1.0.0

- Initial Arch Forge release
- SDXL architecture/environment enhancement workflow
- Three LoRA slots
- Canny and Depth ControlNet guidance
- Img2img sampling controls
- Upscale and sharpen post-processing

## Credits

- Built by [jeeltcraft](https://www.jeeltcraft.com/)
- Built on ComfyUI, SDXL, ControlNet, and the ComfyUI custom-node ecosystem
- Perspective-correction approach informed by conventional line detection, vanishing-point estimation, and projective homography techniques

## License

MIT — see [LICENSE](LICENSE).
