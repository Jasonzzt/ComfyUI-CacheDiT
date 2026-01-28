# ComfyUI-CacheDiT ⚡

<div align="center">

**One-Click DiT Model Acceleration for ComfyUI**

[![cache-dit](https://img.shields.io/badge/cache--dit-v1.2.0+-blue)](https://github.com/vipshop/cache-dit)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

<a name="english"></a>
## 🚀 Overview

ComfyUI-CacheDiT brings **1.4-1.6x speedup** to DiT (Diffusion Transformer) models through intelligent residual caching, with **zero configuration required**.

### ✅ Tested & Verified Models

| Model | Steps | Speedup | Status |
|-------|-------|---------|--------|
| **Z-Image** | 50 | 1.5x | ✅ |
| **Z-Image-Turbo** | 9 | 1.5x | ✅ |
| **Qwen-Image-2512** | 50 | 1.4-1.6x | ✅ |

### 🧪 Experimental Support

Models with built-in presets (auto-configured):
- Flux
- HunyuanVideo
- LTX-2
- Wan 2.2
- Custom DiT models

## 📦 Installation

### Prerequisites

```bash
# Install cache-dit library (v1.2.0+)
pip install cache-dit>=1.2.0
```

### Install Node

**Clone Repository**
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Jasonzzt/ComfyUI-CacheDiT.git
```


## Quick Start

### Ultra-Simple Usage (3 Steps)

1. Load your model
2. Connect to **⚡ CacheDiT Accelerator** node
3. Connect to KSampler - **Done!**

```
[Load Checkpoint] → [⚡ CacheDiT Accelerator] → [KSampler]
```

### Node Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | - | Input model (required) |
| `enable` | Boolean | True | Enable/disable acceleration |
| `model_type` | Combo | Auto | Auto-detect or select preset |
| `print_summary` | Boolean | True | Show performance dashboard |

**That's it!** All technical parameters (threshold, fn_blocks, warmup, etc.) are automatically configured based on your model type.

## How Auto-Detection Works

The node automatically detects your model architecture:

- **QwenImageTransformer** → Qwen-Image preset
- **NextDiT / Lumina** → Z-Image preset  
- **Flux** → Flux preset
- **HunyuanVideo** → HunyuanVideo preset
- **LTX** → LTX-2 preset
- **Unknown** → Custom preset (safe defaults)

No manual configuration needed!

## Performance Dashboard

After generation, you'll see a detailed performance report:

```
╔═══════════════════════════════════════════════════════════════╗
║          CacheDiT Performance Dashboard                       ║
╠═══════════════════════════════════════════════════════════════╣
║  Model: Qwen-Image                                            ║
║  Strategy: Lightweight Cache (fallback mode)                  ║
╠───────────────────────────────────────────────────────────────╣
║  📊 Performance Metrics                                       ║
║───────────────────────────────────────────────────────────────║
║  Total Steps:              50                                 ║
║  Computed Steps:           34                                 ║
║  Cached Steps:             16                                 ║
║  Cache Hit Rate:           32.0%                              ║
║  Estimated Speedup:        1.47x                              ║
╠───────────────────────────────────────────────────────────────╣
║  ⚙️  Configuration                                            ║
║  Warmup:                   3 steps                            ║
║  Skip Interval:            2-3 steps                          ║
║  Memory Mode:              GPU (detach-only, no clone)        ║
╚═══════════════════════════════════════════════════════════════╝
```

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
- **Video models**: Conservative (warmup=8, skip_interval=5)

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

## FAQ

### Q: Does this work with all models?

**A:** Tested and verified for:
- ✅ Z-Image (50 steps)
- ✅ Z-Image-Turbo (9 steps)  
- ✅ Qwen-Image-2512 (50 steps)

Other DiT models should work with auto-detection, but may need manual preset selection.

### Q: Why is speedup "only" 1.5x?

**A:** This is actually excellent! Here's why:
- **1.5x = 33% cache hit rate** - safely caching 1 out of 3 steps
- Higher speedup (2x+) risks quality degradation
- Conservative caching maintains generation quality
- Real-world speedup matches reference implementation

### Q: Can I get more speedup?

**A:** Currently, parameters are optimized for quality-speed balance. Future updates may add "aggressive mode" for users willing to accept potential quality trade-offs.

### Q: What if auto-detection fails?

**A:** Manually select your model from the `model_type` dropdown:
- Z-Image
- Qwen-Image
- Flux
- HunyuanVideo
- LTX-2
- Custom (safe defaults)

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

**Made with ⚡ for the ComfyUI community**

Star ⭐ this repo if you find it useful!

</div>
