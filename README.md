
<div align="center">

# ComfyUI-CacheDiT ⚡

**One-Click DiT Model Acceleration for ComfyUI**

[![cache-dit](https://img.shields.io/badge/cache--dit-v1.2.0+-blue)](https://github.com/vipshop/cache-dit)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

**Quality Comparison (Z-Image-Base, 50 steps)**

| w/o Cache-DiT Acceleration | w/ Cache-DiT Acceleration |
|:---:|:---:|
|<img src="./assets/without_cachedit.png" width=200px>| <img src="./assets/with_cachedit.png" width=200px>

**Guidance Video (Click below)**

<a href="https://www.youtube.com/watch?v=nbhxqRu21js">
  <img src="https://img.youtube.com/vi/nbhxqRu21js/maxresdefault.jpg" alt="ComfyUI-CacheDiT Tutorial" width="420">
</a>

*Thanks to Benji for the excellent tutorial!*

</div>

---

<a name="english"></a>

## Overview

ComfyUI-CacheDiT brings **1.4-1.6x speedup** to DiT (Diffusion Transformer) models through intelligent caching, with **zero configuration required**.

Inspired by [**llm-scaler**](https://github.com/intel/llm-scaler), a high-performance GenAI solution for text, image, and video generation on Intel XPU.

### Tested & Verified Models

<div align="center">

| Model | Steps | Speedup | Warmup | Skip_interval |
|-------|-------|---------|---------|--------|
| **Z-Image** | 50 | 1.3x | 10 | 5 |
| **Z-Image-Turbo** | 9 | 1.5x | 3 | 2 |
| **Qwen-Image-2512** | 50 | 1.4-1.6x | 5 | 3 |
| **Flux.2 Klein 4B** | 20 | 1.67x | 4 | 2 |
| **Flux.2 Klein 9B** | 20 | 1.67x | 4 | 2 |
| **MiniMax H3** | 20 | 1.41-1.50x | 3 | N/A (adaptive) |
| **LTX-2 T2V** | 20 | 2.0x | 6 | 4 |
| **LTX-2 I2V** | 20 | 2.0x | 6 | 4 |
| **WAN2.2 14B T2V** | 20 | 1.67x | 4 | 2 |
| **WAN2.2 14B I2V** | 20 | 1.67x | 4 | 2 |

</div>

MiniMax H3 uses residual-difference-based adaptive DBCache decisions rather
than a fixed `skip_interval`, so that column is not applicable to H3.

## Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

### Install Node

**Clone Repository**
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Jasonzzt/ComfyUI-CacheDiT.git
```


## Quick Start

### Ultra-Simple Usage (3 Steps)

**For Image Models (Z-Image, Qwen-Image Flux.2 Klein):**

1. Load your model
2. Connect to **⚡ CacheDiT Accelerator** node
3. Connect to KSampler - **Done!**

```
[Load Checkpoint] → [⚡ CacheDiT Accelerator] → [KSampler]
```

**For Video Models (MiniMax H3, LTX-2, WAN2.2 14B):**

**MiniMax H3:**
```
[Load Diffusion Model] → [⚡ CacheDiT Accelerator] → [Model Sampling / Guider]
```
The `Auto` preset detects `MiniMaxH3Model` and enables Pattern 3 DBCache with
in-place-safe residual boundaries for its joint audio-video transformer.

For explicit H3 tuning, use the optional **⚡ CacheDiT MiniMax H3 Advanced**
node in the same position. It exposes only the four H3 DBCache controls that
are useful for workflow-level quality/speed tuning; `Auto` remains the
recommended starting point.

The initial H3 preset is `F8 / B0 / threshold 0.12 / warmup 3`:

| Control | Initial value | Meaning and initial-setting rationale |
|---------|---------------|---------------------------------------|
| `fn_blocks` | 8 | Always execute the first 8 blocks. This gives the residual comparison an early, current-step signal while leaving most blocks eligible for reuse. |
| `bn_blocks` | 0 | Do not recompute tail blocks on a cache hit. This is the initial performance-oriented setting; raise it if a workflow needs more conservative tail refinement. |
| `residual_diff_threshold` | 0.12 | Reuse the middle-block residual when the normalized residual difference stays below 0.12. Lower values are more conservative; higher values usually increase cache hits and quality risk. |
| `warmup_steps` | 3 | Fully compute steps 0, 1, and 2; adaptive cache reuse becomes eligible at step 3. |

These values are a validated integration baseline, not a model-wide optimum.
T2V, I2V, and R2V workflows can respond differently to the same cache
settings. Tune one control at a time and compare the same workflow, seed,
steps, resolution, frame count, audio settings, and output codec.


**LTX-2 Models:**
```
[Load Checkpoint] → [⚡ LTX2 Cache Optimizer] → [Stage 1 KSampler]
```

**WAN2.2 14B Models (High-Noise + Low-Noise MoE):**
```
[High-Noise Model] → [⚡ Wan Cache Optimizer] → [KSampler]
                                               
[Low-Noise Model]  → [⚡ Wan Cache Optimizer] → [KSampler]
```
*Each expert model gets its own optimizer node with independent cache.*

### Node Parameters

<div align="center">

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | - | Input model (required) |
| `enable` | Boolean | True | Enable/disable acceleration |
| `model_type` | Combo | Auto | Auto-detect or select preset |
| `warmup_steps` | Integer | 0 | Generic node: 0 uses the selected preset |
| `skip_interval` | Integer | 0 | Generic node: 0 uses the selected preset |
| `print_summary` | Boolean | True | Show performance dashboard |

</div>

**That's it!** The generic node configures technical parameters automatically.
Use the H3 Advanced node only when you need explicit H3 workflow tuning.

## How It Works

**Caching Logic**:
```python
# After warmup phase (first 3 steps)
if (current_step - warmup) % skip_interval == 0:
    # Reuse cached result
    result = cache
else:
    # Compute new result
    result = transformer.forward(...)
    cache = result.detach()  # Save to cache
```

## Credits

Based on [**cache-dit**](https://github.com/vipshop/cache-dit) by Vipshop's Machine Learning Platform Team.

Built for [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) - the powerful and modular Stable Diffusion GUI.

## FAQ

**Note for LTX-2**: This audio-visual transformer uses dual latent paths (video + audio). Use the dedicated `⚡ LTX2 Cache Optimizer` node (not the standard CacheDiT node) for optimal temporal consistency and quality.

**Note for WAN2.2 14B**: This model uses a MoE (Mixture of Experts) architecture with High-Noise and Low-Noise models. Use the dedicated `⚡ Wan Cache Optimizer` node (not the standard CacheDiT node) for best results.

Other DiT models should work with auto-detection, but may need manual preset selection.

### Q: Does it support distilled low step models?

**A:** Currently, only **Z-Image-Turbo (9 steps)** has been tested and verified. Other low-step distilled models require further validation. 

For extremely low step counts (< 6 steps), the warmup overhead significantly reduces the benefit - sacrificing quality for minimal speed gains is generally not worthwhile in such cases.

### Q: How can I disable the node without restarting ComfyUI?

**A:** Simply set `enable=False` in the node and run it once. This will cleanly remove the CacheDiT optimization from your model without requiring a restart.

### Q: Performance Dashboard shows 0% cache hit?

**A:** This usually means:
1. Model not properly detected - try manual preset selection
2. Inference steps too short (< 10 steps) - warmup takes most steps
3. Check logs for "Lightweight cache enabled" message

### Q: Does this affect image quality?

**A:** Properly configured (default settings), quality impact is minimal:

---

<div align="center">

Star ⭐ this repo if you find it useful!

</div>
