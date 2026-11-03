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
from .utils import detach_output

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
    """Enable lightweight cache for Wan2.2 transformer"""
    transformer_id = id(transformer)
    state = _get_or_create_cache_state(transformer_id)
    # Ensure previous wrapper is safely unwrapped to avoid recursion
    if hasattr(transformer, '_original_forward_wan'):
        transformer.forward = transformer._original_forward_wan
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
    
    # Save original forward (guard against double wrapping)
    orig_forward = transformer.forward
    if getattr(orig_forward, '__name__', None) == 'cached_forward' and hasattr(transformer, '_original_forward_wan'):
        orig_forward = transformer._original_forward_wan
    transformer._original_forward_wan = orig_forward
    
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
        state = _get_or_create_cache_state(transformer_id)
        state["call_count"] += 1
        call_id = state["call_count"]
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
            state["last_result"] = detach_output(result)
            
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
            state["last_result"] = detach_output(result)
            
            if call_id <= 10 and cache_config and cache_config.verbose:
                logger.info(
                    f"[Wan-Cache] Step {call_id}: computed "
                    f"({state['compute_count']}/{call_id}, {elapsed:.3f}s)"
                )
            
            return result
    
    # Replace forward method
    transformer.forward = cached_forward
    
    logger.info(
        f"[Wan-Cache] Enabled for transformer {transformer_id}: "
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
        try:
            guider.model_options = orig_model_options
        except:
            pass
        # CRITICAL: Clean up monkey-patched forward on transformer to prevent memory leak and recursion
        if transformer is not None and hasattr(transformer, '_original_forward_wan'):
            try:
                transformer.forward = transformer._original_forward_wan
                delattr(transformer, '_original_forward_wan')
                logger.info(f"[Wan-Cache] Cleaned up transformer {id(transformer)} on sampling exit")
            except Exception as e:
                logger.warning(f"[Wan-Cache] Cleanup on sampling exit failed: {e}")


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
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/Disable Wan cache acceleration\n启用/禁用 Wan 缓存加速"
                }),
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
    )
    
    def optimize(
        self,
        model,
        enable: bool = True,
        warmup_steps: int = 4,
        skip_interval: int = 2,
        print_summary: bool = True,
    ):
        """Apply Wan2.2 specific cache optimization."""
        
        # If disabled, return disabled model
        if not enable:
            logger.info("[Wan-Cache] Optimization disabled, model restored")
            return self.disable(model)

        # Check configuration from transformer (persistent across model clones)
        transformer = None
        existing_config = None
        
        if hasattr(model.model, 'diffusion_model'):
            transformer = model.model.diffusion_model
            existing_config = getattr(transformer, '_wan_cache_config', None)
        
        if existing_config is not None:
            # Compare parameters
            params_changed = (
                existing_config["warmup_steps"] != warmup_steps or
                existing_config["skip_interval"] != skip_interval or
                existing_config["print_summary"] != print_summary
            )
            
            if params_changed:
                logger.info(
                    f"[Wan-Cache] Parameters changed: "
                    f"warmup {existing_config['warmup_steps']}→{warmup_steps}, "
                    f"skip {existing_config['skip_interval']}→{skip_interval}"
                )
                # Disable and reconfigure
                result = self.disable(model)
                model = result[0]
            else:
                logger.info("[Wan-Cache] Configuration unchanged")
                return (model,)
        
        # Clone model for new configuration
        model = model.clone()
        
        # Create config
        config = WanCacheConfig(
            warmup_steps=warmup_steps,
            skip_interval=skip_interval,
            verbose=False,
            print_summary=print_summary,
        )
        
        # Store config in both transformer_options and transformer object
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        
        model.model_options["transformer_options"]["wan_cache"] = config
        
        # Store simplified config on transformer for persistence
        if transformer is not None:
            transformer._wan_cache_config = {
                "warmup_steps": warmup_steps,
                "skip_interval": skip_interval,
                "print_summary": print_summary,
            }
        
        # Register wrapper using ComfyUI's patcher_extension system
        try:
            model.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                "wan_cache",
                _wan_outer_sample_wrapper
            )
            
            logger.info(
                f"[Wan-Cache] Configured: warmup={warmup_steps}, skip={skip_interval}"
            )
        
        except Exception as e:
            logger.error(f"[Wan-Cache] Failed to register wrapper: {e}")
            traceback.print_exc()
        
        return (model,)
    
    def disable(self, model):
        """Disable Wan cache optimization."""
        model = model.clone()
        
        # Remove config
        if "wan_cache" in model.model_options.get("transformer_options", {}):
            del model.model_options["transformer_options"]["wan_cache"]
        
        # Remove wrapper
        if hasattr(model, "remove_wrappers_with_key"):
            model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "wan_cache")
        elif "wan_cache" in model.wrappers.get(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, {}):
            del model.wrappers[comfy.patcher_extension.WrappersMP.OUTER_SAMPLE]["wan_cache"]
        
        # Restore original forward and clear cache state
        try:
            if hasattr(model.model, 'diffusion_model'):
                transformer = model.model.diffusion_model
                
                # Restore original forward
                if hasattr(transformer, '_original_forward_wan'):
                    transformer.forward = transformer._original_forward_wan
                    delattr(transformer, '_original_forward_wan')
                    logger.info("[Wan-Cache] Restored original forward")
                
                # Clear config marker
                if hasattr(transformer, '_wan_cache_config'):
                    delattr(transformer, '_wan_cache_config')
                
                # Clear cache registry
                transformer_id = id(transformer)
                global _wan_cache_registry
                if transformer_id in _wan_cache_registry:
                    del _wan_cache_registry[transformer_id]
                    logger.info("[Wan-Cache] Cache state cleared")
        
        except Exception as e:
            logger.warning(f"[Wan-Cache] Failed to fully restore: {e}")
        
        return (model,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "WanCacheOptimizer": WanCacheOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanCacheOptimizer": "⚡ CacheDiT Wan Accelerator",
}
