
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
| **LTX-2 T2V** | 20 | 2.0x | 6 | 4 |
| **LTX-2 I2V** | 20 | 2.0x | 6 | 4 |
| **WAN2.2 14B T2V** | 20 | 1.67x | 4 | 2 |
| **WAN2.2 14B I2V** | 20 | 1.67x | 4 | 2 |

</div>

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

**For Video Models (LTX-2, WAN2.2 14B):**

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
| `print_summary` | Boolean | True | Show performance dashboard |

</div>

**That's it!** All technical parameters (threshold, fn_blocks, warmup, etc.) are automatically configured based on your model type.

## How It Works

### Intelligent Fallback System

ComfyUI-CacheDiT uses a **two-tier acceleration approach**:

1. **Primary**: cache-dit library with DBCache algorithm
2. **Fallback**: Lightweight cache (direct forward hook replacement)

For ComfyUI models (Qwen-Image, Z-Image, etc.), the lightweight cache automatically activates because cache-dit's BlockAdapter cannot track non-standard model architectures.


**Caching Logic**:
```python
# After warmup phase (first 3 steps)
if (current_step - warmup) % skip_interval == 0:
    # Compute new result
    result = transformer.forward(...)
    cache = result.detach()  # Save to cache
else:
    # Reuse cached result
    result = cache
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
