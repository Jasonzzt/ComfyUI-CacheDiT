# ComfyUI-CacheDiT ⚡

<div align="center">

**One-Click DiT Model Acceleration for ComfyUI**

[![cache-dit](https://img.shields.io/badge/cache--dit-v1.2.0+-blue)](https://github.com/vipshop/cache-dit)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

<a name="english"></a>

## Overview

ComfyUI-CacheDiT brings **1.4-1.6x speedup** to DiT (Diffusion Transformer) models through intelligent residual caching, with **zero configuration required**.


### Tested & Verified Models

| Model | Steps | Speedup | Status | Warmup | Skip_interval |
|-------|-------|---------|--------|---------|--------|
| **Z-Image** | 50 | 1.3x | ✅ | 12 | 5 |
| **Z-Image-Turbo** | 9 | 1.5x | ✅ | 3 | 2 |
| **Qwen-Image-2512** | 50 | 1.4-1.6x | ✅ | 5 | 3 |
| **LTX-2 T2V** | 20 | 2.0x | ✅ | 6 | 4 |
| **LTX-2 I2V** | 20 | 2.0x | ✅ | 6 | 4 |
| **WAN2.2 14B T2V** | 20 | 1.67x | ✅ | 4 | 2 |
| **WAN2.2 14B I2V** | 20 | 1.67x | ✅ | 4 | 2 |

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

**For Image Models (Z-Image, Qwen-Image):**

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | - | Input model (required) |
| `enable` | Boolean | True | Enable/disable acceleration |
| `model_type` | Combo | Auto | Auto-detect or select preset |
| `print_summary` | Boolean | True | Show performance dashboard |

**That's it!** All technical parameters (threshold, fn_blocks, warmup, etc.) are automatically configured based on your model type.

## How It Works

### Intelligent Fallback System

ComfyUI-CacheDiT uses a **two-tier acceleration approach**:

1. **Primary**: cache-dit library with DBCache algorithm
2. **Fallback**: Lightweight cache (direct forward hook replacement)

For ComfyUI models (Qwen-Image, Z-Image, etc.), the lightweight cache automatically activates because cache-dit's BlockAdapter cannot track non-standard model architectures.

### Lightweight Cache Strategy

**Model-Specific Optimization**:
- **Z-Image/Turbo**: Aggressive caching (warmup=3, skip_interval=2)
- **Qwen-Image**: Balanced approach (warmup=3, skip_interval=2-3)
- **LTX-2 (T2V/I2V)**: Conservative for temporal consistency (warmup=6, skip_interval=4)
- **WAN2.2 14B (T2V/I2V)**: Optimized for MoE architecture (warmup=4, skip_interval=2)
  - Uses dedicated `WanCacheOptimizer` node
  - Supports High-Noise + Low-Noise expert models
  - Per-transformer cache isolation (multi-instance safe)
  - Memory-efficient: detach-only caching prevents VAE OOM

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

**Memory Optimization**:
- Uses `.detach()` only (no `.clone()`)
- Saves 50% memory for cached tensors
- Prevents VAE OOM on long sequences

## Credits

This project is based on [**cache-dit**](https://github.com/vipshop/cache-dit) by Vipshop's Machine Learning Platform Team.

## FAQ

### Q: Does this work with all models?

**A:** Tested and verified for:
- ✅ Z-Image (50 steps)
- ✅ Z-Image-Turbo (9 steps)  
- ✅ Qwen-Image-2512 (50 steps)
- ✅ LTX-2 T2V (Text-to-Video, 20 steps)
- ✅ LTX-2 I2V (Image-to-Video, 20 steps)
- ✅ WAN2.2 14B T2V (Text-to-Video, 20 steps)
- ✅ WAN2.2 14B I2V (Image-to-Video, 20 steps)

**Note for LTX-2**: This audio-visual transformer uses dual latent paths (video + audio). Use the dedicated `⚡ LTX2 Cache Optimizer` node (not the standard CacheDiT node) for optimal temporal consistency and quality.

**Note for WAN2.2 14B**: This model uses a MoE (Mixture of Experts) architecture with High-Noise and Low-Noise models. Use the dedicated `⚡ Wan Cache Optimizer` node (not the standard CacheDiT node) for best results.

Other DiT models should work with auto-detection, but may need manual preset selection.

### Q: Performance Dashboard shows 0% cache hit?

**A:** This usually means:
1. Model not properly detected - try manual preset selection
2. Inference steps too short (< 10 steps) - warmup takes most steps
3. Check logs for "Lightweight cache enabled" message

### Q: Does this affect image quality?

**A:** Properly configured (default settings), quality impact is minimal:
- Cache is only used when residuals are similar between steps
- Warmup phase (3 steps) establishes stable baseline
- Conservative skip intervals prevent artifacts

---

<div align="center">

Star ⭐ this repo if you find it useful!

</div>
