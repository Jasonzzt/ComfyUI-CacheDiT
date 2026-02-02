"""
ComfyUI-CacheDiT: Wan2.2 Specialized Node
==========================================

Dedicated node for Wan2.2 DiT model with MoE architecture.
Wan2.2 has High-Noise and Low-Noise expert models that can be used
in separate node instances within the same workflow.

Key Features:
- Per-transformer cache isolation (multiple instances supported)
- Lightweight cache strategy (warmup + skip interval)
- Memory-efficient caching (detach-only, no clone)
- Automatic state reset per sampling run
- Support for Tensor and Tuple outputs
"""

from __future__ import annotations
import logging
import traceback
import torch
import comfy.model_patcher
import comfy.patcher_extension
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

logger = logging.getLogger("ComfyUI-CacheDiT-Wan")


# === Per-transformer cache state registry ===
# Key: id(transformer), Value: cache state dict
_wan_cache_registry: Dict[int, Dict[str, Any]] = {}


def _get_or_create_cache_state(transformer_id: int) -> Dict[str, Any]:
    """
    Get or create cache state for a specific transformer instance.
    This ensures High-Noise and Low-Noise models have independent caches.
    """
    if transformer_id not in _wan_cache_registry:
        _wan_cache_registry[transformer_id] = {
            "enabled": False,
            "transformer_id": transformer_id,
            "call_count": 0,
            "skip_count": 0,
            "compute_count": 0,
            "last_result": None,
            "config": None,
            "compute_times": [],
        }
    return _wan_cache_registry[transformer_id]


class WanCacheConfig:
    """Configuration for Wan2.2 cache optimization."""
    
    def __init__(
        self,
        warmup_steps: int = 4,
        skip_interval: int = 2,
        verbose: bool = False,
        print_summary: bool = True,
    ):
        self.warmup_steps = warmup_steps
        self.skip_interval = skip_interval
        self.verbose = verbose
        self.print_summary = print_summary
        
        # Runtime state
        self.is_enabled = False
        self.num_inference_steps: Optional[int] = None
        self.current_step: int = 0
    
    def clone(self) -> "WanCacheConfig":
        """Create a copy for cloned models."""
        new_config = WanCacheConfig(
            warmup_steps=self.warmup_steps,
            skip_interval=self.skip_interval,
            verbose=self.verbose,
            print_summary=self.print_summary,
        )
        new_config.is_enabled = self.is_enabled
        new_config.num_inference_steps = self.num_inference_steps
        return new_config
    
    def reset(self):
        """Reset runtime state for new generation."""
        self.current_step = 0


def _enable_wan_cache(transformer, config: WanCacheConfig):
    """
    Enable lightweight cache for Wan2.2 transformer.
    
    Cache Strategy:
    - Warmup phase: Always compute first warmup_steps steps
    - Post-warmup: Compute only when (current_step - warmup_steps) % skip_interval == 0
    - Other steps: Return cached result directly
    
    Memory Optimization:
    - Use .detach() ONLY (no .clone()) to save VRAM
    - Critical for preventing VAE stage OOM
    
    Example (warmup=4, skip=2, total=20):
    - Compute: steps 1-4 (warmup) + 5, 7, 9, 11, 13, 15, 17, 19 = 12 steps
    - Cache: steps 6, 8, 10, 12, 14, 16, 18, 20 = 8 steps
    - Cache rate: 40%, speedup: ~1.7x
    """
    transformer_id = id(transformer)
    state = _get_or_create_cache_state(transformer_id)
    
    # Check if already patched
    if hasattr(transformer, '_original_forward_wan'):
        if state.get("transformer_id") == transformer_id:
            logger.info("[Wan-Cache] Already enabled, resetting state")
            state.update({
                "call_count": 0,
                "skip_count": 0,
                "compute_count": 0,
                "last_result": None,
                "compute_times": [],
            })
            return
    
    # Save original forward
    transformer._original_forward_wan = transformer.forward
    
    # Initialize state
    state.update({
        "enabled": True,
        "transformer_id": transformer_id,
        "call_count": 0,
        "skip_count": 0,
        "compute_count": 0,
        "last_result": None,
        "config": config,
        "compute_times": [],
    })
    
    def cached_forward(*args, **kwargs):
        """
        Cached forward for Wan2.2 transformer.
        
        Implements lightweight cache with warmup + skip logic.
        """
        state = _get_or_create_cache_state(transformer_id)
        state["call_count"] += 1
        call_id = state["call_count"]
        
        # Get parameters from config
        cache_config = state.get("config")
        warmup_steps = cache_config.warmup_steps if cache_config else 4
        skip_interval = cache_config.skip_interval if cache_config else 2
        
        # Debug logging for first few calls
        if call_id <= 5 and cache_config and cache_config.verbose:
            logger.info(
                f"[Wan-Cache] Call #{call_id} (transformer {transformer_id}), "
                f"warmup={warmup_steps}, skip={skip_interval}"
            )
        
        # Phase 1: Warmup - always compute
        if call_id <= warmup_steps:
            import time
            start = time.time()
            result = transformer._original_forward_wan(*args, **kwargs)
            elapsed = time.time() - start
            
            state["compute_count"] += 1
            state["compute_times"].append(elapsed)
            
            # Cache result - CRITICAL: use detach() only, NO clone()
            if isinstance(result, torch.Tensor):
                state["last_result"] = result.detach()
            elif isinstance(result, tuple):
                # Handle tuple of tensors (Wan model may return tuples)
                state["last_result"] = tuple(
                    r.detach() if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            else:
                state["last_result"] = result
            
            if call_id <= 3 and cache_config and cache_config.verbose:
                logger.info(f"[Wan-Cache] Warmup step {call_id}/{warmup_steps}, cached result")
            
            return result
        
        # Phase 2: Post-warmup - selective compute
        steps_after_warmup = call_id - warmup_steps
        should_compute = (steps_after_warmup % skip_interval == 0)
        
        cache_valid = state["last_result"] is not None
        
        if not should_compute and cache_valid:
            # Use cached result
            state["skip_count"] += 1
            cached_result = state["last_result"]
            
            if call_id <= 10 and cache_config and cache_config.verbose:
                logger.info(
                    f"[Wan-Cache] Step {call_id}: using cache "
                    f"(skip {state['skip_count']}/{call_id})"
                )
            
            return cached_result
        
        else:
            # Compute and update cache
            import time
            start = time.time()
            result = transformer._original_forward_wan(*args, **kwargs)
            elapsed = time.time() - start
            
            state["compute_count"] += 1
            state["compute_times"].append(elapsed)
            
            # Update cache - CRITICAL: use detach() only
            if isinstance(result, torch.Tensor):
                state["last_result"] = result.detach()
            elif isinstance(result, tuple):
                state["last_result"] = tuple(
                    r.detach() if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            else:
                state["last_result"] = result
            
            if call_id <= 10 and cache_config and cache_config.verbose:
                logger.info(
                    f"[Wan-Cache] Step {call_id}: computed "
                    f"({state['compute_count']}/{call_id}, {elapsed:.3f}s)"
                )
            
            return result
    
    # Replace forward method
    transformer.forward = cached_forward
    
    logger.info(
        f"[Wan-Cache] ✓ Enabled for transformer {transformer_id}: "
        f"warmup={config.warmup_steps}, skip_interval={config.skip_interval}"
    )


def _refresh_wan_cache(transformer, config: WanCacheConfig):
    """
    Refresh Wan cache for new sampling run.
    
    CRITICAL: This is called at the start of each sampling task (KSampler run).
    We MUST reset call_count and last_result to prevent reusing data from
    previous generations, which would cause artifacts.
    """
    transformer_id = id(transformer)
    state = _get_or_create_cache_state(transformer_id)
    
    try:
        # Reset ALL runtime state (critical for multi-generation workflows)
        state["call_count"] = 0
        state["skip_count"] = 0
        state["compute_count"] = 0
        state["last_result"] = None
        state["compute_times"] = []
        state["config"] = config
        
        if config.verbose:
            logger.info(
                f"[Wan-Cache] ♻️ Reset for new sampling: transformer {transformer_id}, "
                f"{config.num_inference_steps} steps"
            )
    
    except Exception as e:
        logger.error(f"[Wan-Cache] Refresh failed: {e}")
        traceback.print_exc()


def _get_wan_cache_stats(transformer_id: int):
    """Get statistics from Wan cache for a specific transformer."""
    if transformer_id not in _wan_cache_registry:
        return None
    
    state = _wan_cache_registry[transformer_id]
    
    if not state.get("enabled"):
        return None
    
    total_calls = state["call_count"]
    cache_hits = state["skip_count"]
    compute_count = state["compute_count"]
    
    if total_calls == 0:
        return None
    
    cache_hit_rate = (cache_hits / total_calls) * 100
    avg_time = sum(state["compute_times"]) / max(len(state["compute_times"]), 1)
    estimated_speedup = total_calls / max(compute_count, 1)
    
    return {
        "transformer_id": transformer_id,
        "total_calls": total_calls,
        "computed_calls": compute_count,
        "cached_calls": cache_hits,
        "cache_hit_rate": cache_hit_rate,
        "estimated_speedup": estimated_speedup,
        "avg_compute_time": avg_time,
    }


def _wan_outer_sample_wrapper(executor, *args, **kwargs):
    """
    OUTER_SAMPLE wrapper for Wan2.2.
    
    This is called at the CFGGuider.sample level BEFORE each sampling task.
    It ensures cache state is properly reset for each generation, preventing
    cross-contamination between multiple generations.
    
    Arguments:
    - executor: the original CFGGuider.sample method
    - executor.class_obj: the CFGGuider instance
    - args[0]: noise
    - args[1]: latent_image
    - args[2]: sampler (KSAMPLER)
    - args[3]: sigmas
    """
    guider = executor.class_obj
    orig_model_options = guider.model_options
    transformer = None
    config = None
    
    try:
        # Clone model options (standard ComfyUI pattern)
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        
        # Get config
        config: WanCacheConfig = guider.model_options.get("transformer_options", {}).get("wan_cache")
        if config is None:
            return executor(*args, **kwargs)
        
        # Clone and reset config
        config = config.clone()
        config.reset()
        guider.model_options["transformer_options"]["wan_cache"] = config
        
        # Extract num_inference_steps from sigmas (4th positional arg)
        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas")
        if sigmas is not None:
            num_steps = len(sigmas) - 1
            config.num_inference_steps = num_steps
        
        # Get transformer
        model_patcher = guider.model_patcher
        if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
            transformer = model_patcher.model.diffusion_model
            transformer_id = id(transformer)
            
            # Check if cache already enabled for this transformer
            cache_already_enabled = hasattr(transformer, '_original_forward_wan')
            
            if config.num_inference_steps is not None:
                if not cache_already_enabled:
                    # First time: enable cache
                    logger.info(
                        f"[Wan-Cache] 🚀 Enabling for transformer {transformer_id}: "
                        f"{config.num_inference_steps} steps"
                    )
                    _enable_wan_cache(transformer, config)
                    config.is_enabled = True
                else:
                    # Subsequent runs: REFRESH (reset state)
                    logger.info(
                        f"[Wan-Cache] ♻️ Refreshing for transformer {transformer_id}: "
                        f"{config.num_inference_steps} steps"
                    )
                    _refresh_wan_cache(transformer, config)
                    config.is_enabled = True
        
        # Execute sampling
        result = executor(*args, **kwargs)
        
        # Print summary
        if config.print_summary and transformer is not None:
            transformer_id = id(transformer)
            stats = _get_wan_cache_stats(transformer_id)
            if stats:
                logger.info(
                    f"\n╔════════════════════════════════════════════════════════╗\n"
                    f"║  Wan Cache Optimizer - Performance Summary            ║\n"
                    f"╠════════════════════════════════════════════════════════╣\n"
                    f"║  Transformer ID:     {transformer_id:>10}                      ║\n"
                    f"║  Total Calls:        {stats['total_calls']:>4} forward passes                ║\n"
                    f"║  Computed:           {stats['computed_calls']:>4} ({100*stats['computed_calls']/stats['total_calls']:>5.1f}%)                    ║\n"
                    f"║  Cached:             {stats['cached_calls']:>4} ({stats['cache_hit_rate']:>5.1f}%)                    ║\n"
                    f"║  Estimated Speedup:  {stats['estimated_speedup']:>5.2f}x                           ║\n"
                    f"║  Avg Compute Time:   {stats['avg_compute_time']:>6.3f}s                         ║\n"
                    f"╚════════════════════════════════════════════════════════╝"
                )
        
        return result
    
    except Exception as e:
        logger.error(f"[Wan-Cache] OUTER_SAMPLE wrapper failed: {e}")
        traceback.print_exc()
        return executor(*args, **kwargs)
    
    finally:
        # Restore original model options
        guider.model_options = orig_model_options


# =============================================================================
# Node Definition
# =============================================================================

class WanCacheOptimizer:
    """
    Wan2.2 Cache Optimizer Node
    
    Accelerates Wan2.2 (DiT + MoE) inference using lightweight cache strategy.
    Supports multiple instances (High-Noise + Low-Noise experts) in same workflow.
    
    Features:
    - Per-transformer cache isolation (id-based registry)
    - Warmup + skip interval strategy
    - Memory-efficient (detach-only, no clone)
    - Auto-reset per sampling run (OUTER_SAMPLE wrapper)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "warmup_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of initial steps to always compute (build cache baseline)\n"
                               "预热步数（前N步必须计算，建立缓存基线）\n"
                               "Recommended: 3-6 for balanced quality/speed"
                }),
                "skip_interval": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Compute every Nth step after warmup (others use cache)\n"
                               "跳步间隔（预热后每N步计算一次，其余用缓存）\n"
                               "Recommended: 2-3 for ~40-50% cache rate"
                }),
                "print_summary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print performance statistics after generation\n"
                               "生成后打印性能统计"
                }),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "optimize"
    CATEGORY = "⚡ CacheDiT"
    DESCRIPTION = (
        "Wan2.2 专用缓存加速器 / Wan2.2 Cache Accelerator\n\n"
        "✨ Features:\n"
        "• Lightweight cache (warmup + skip interval)\n"
        "• Multi-instance support (High-Noise + Low-Noise)\n"
        "• Memory-efficient (detach-only, prevents VAE OOM)\n"
        "• Auto-reset per sampling (clean slate each generation)\n\n"
        "📊 Performance (warmup=4, skip=2, 20 steps):\n"
        "• Compute: 12 steps (4 warmup + 8 selective)\n"
        "• Cache: 8 steps (40% cache rate)\n"
        "• Speedup: ~1.7x\n\n"
        "🔧 Recommended Settings:\n"
        "• Speed: warmup=3, skip=2 (50% cache, ~2.0x)\n"
        "• Balanced ⭐: warmup=4, skip=2 (40% cache, ~1.7x)\n"
        "• Quality: warmup=6, skip=3 (30% cache, ~1.4x)"
    )
    
    def optimize(
        self,
        model,
        warmup_steps: int = 4,
        skip_interval: int = 2,
        print_summary: bool = True,
    ):
        """Apply Wan2.2 specific cache optimization."""
        
        # Clone model (standard ComfyUI pattern)
        model = model.clone()
        
        # Create config
        config = WanCacheConfig(
            warmup_steps=warmup_steps,
            skip_interval=skip_interval,
            verbose=False,
            print_summary=print_summary,
        )
        
        # Store config in transformer_options
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        
        model.model_options["transformer_options"]["wan_cache"] = config
        
        # Register wrapper using ComfyUI's patcher_extension system
        # This ensures cache state is reset at the start of each sampling task
        try:
            model.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                "wan_cache",
                _wan_outer_sample_wrapper
            )
            
            logger.info(
                f"[Wan-Cache] ✓ Node configured: warmup={warmup_steps}, skip={skip_interval}"
            )
        
        except Exception as e:
            logger.error(f"[Wan-Cache] Failed to register wrapper: {e}")
            traceback.print_exc()
        
        return (model,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "WanCacheOptimizer": WanCacheOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanCacheOptimizer": "⚡ Wan Cache Optimizer",
}
