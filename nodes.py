
"""
ComfyUI-CacheDiT: Node Definitions
====================================

Main node: CacheDiT_Model_Optimizer
- Accelerates DiT models via inter-step residual caching
- Smart auto-detection of inference steps via ComfyUI hooks
- Rich summary statistics dashboard
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

import comfy.model_patcher
import comfy.patcher_extension

from .utils import (
    MODEL_PRESETS,
    ModelPreset,
    PATTERN_DESCRIPTIONS,
    get_preset,
    get_all_preset_names,
    get_forward_pattern,
    build_cache_config,
    build_calibrator_config,
    print_summary_to_log,
    apply_noise_injection,
)

try:
    from cache_dit.caching import ForwardPattern
except ImportError:
    ForwardPattern = None

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

logger = logging.getLogger("ComfyUI-CacheDiT")


# === Lightweight cache state for fallback mode ===
_lightweight_cache_state = {
    "enabled": False,
    "transformer_id": None,
    "call_count": 0,
    "skip_count": 0,
    "last_result": None,
    "config": None,
}


def _enable_lightweight_cache(transformer, blocks, config, cache_config):
    """Enable lightweight cache by replacing transformer.forward"""
    global _lightweight_cache_state
    
    if hasattr(transformer, '_original_forward'):
        logger.warning("[LightweightCache] Transformer already patched, skipping")
        return
    
    transformer._original_forward = transformer.forward
    _lightweight_cache_state = {
        "enabled": True,
        "transformer_id": id(transformer),
        "call_count": 0,
        "skip_count": 0,
        "compute_count": 0,
        "last_result": None,
        "config": config,
        "cache_config": cache_config,
        "compute_times": [],
    }
    
    # === Model-specific adaptive parameters ===
    transformer_class = transformer.__class__.__name__
    total_steps = config.num_inference_steps if config.num_inference_steps else 28
    
    # Check if user provided overrides
    has_user_warmup = hasattr(config, 'user_warmup_steps') and config.user_warmup_steps > 0
    has_user_skip = hasattr(config, 'user_skip_interval') and config.user_skip_interval > 0
    
    # Debug: log what we found
    logger.info(
        f"[CacheDiT] Checking user overrides: "
        f"has_warmup_attr={hasattr(config, 'user_warmup_steps')}, "
        f"warmup_value={getattr(config, 'user_warmup_steps', 'N/A')}, "
        f"has_skip_attr={hasattr(config, 'user_skip_interval')}, "
        f"skip_value={getattr(config, 'user_skip_interval', 'N/A')}"
    )
    
    if has_user_warmup:
        # User specified warmup steps directly
        warmup_steps = config.user_warmup_steps
        logger.info(f"[CacheDiT] User override: warmup_steps={warmup_steps}")
    else:
        # Will be set from model-specific defaults
        warmup_steps = None
    
    if has_user_skip:
        # User specified skip interval
        skip_interval = config.user_skip_interval
        logger.info(f"[CacheDiT] User override: skip_interval={skip_interval}")
    else:
        # Will be set from model-specific defaults
        skip_interval = None
    
    # Model-specific configurations (set any unspecified parameters)
    if "NextDiT" in transformer_class:
        # Z-Image (NextDiT): Quality-sensitive, balanced caching for speed
        # Official cache-dit uses --scm fast (~50% cache) for Z-Image-Turbo
        # We use medium settings: warmup 50%, then skip 33% of remaining steps
        if warmup_steps is None:
            warmup_steps = max(total_steps // 2, 8)  # Warmup 50% of steps
        if skip_interval is None:
            skip_interval = 3  # Skip 33% of post-warmup steps (compute, compute, skip)
        noise_scale = 0.0  # Z-Image: NO noise injection
            
    elif "Lumina" in transformer_class:
        # Lumina2: simpler architecture, can skip more aggressively
        if warmup_steps is None:
            warmup_steps = min(3, total_steps // 3)
        if skip_interval is None:
            skip_interval = 2
        noise_scale = 0.0
    
    elif "QwenImage" in transformer_class or "Qwen" in transformer_class:
        # Qwen-Image: quality-sensitive, use conservative settings
        if warmup_steps is None:
            warmup_steps = min(3, total_steps // 10)  # Shorter warmup for speed
        if skip_interval is None:
            if total_steps <= 20:
                skip_interval = 2
            elif total_steps <= 40:
                skip_interval = 2
            else:
                skip_interval = 3  # 25% skip for very long sequences
        noise_scale = config.noise_scale if hasattr(config, 'noise_scale') else 0.0
    
    elif "Flux" in transformer_class or "FLUX" in transformer_class:
        # FLUX: well-tested, balanced approach
        if warmup_steps is None:
            warmup_steps = min(3, total_steps // 4)
        if skip_interval is None:
            skip_interval = 2
        noise_scale = config.noise_scale if hasattr(config, 'noise_scale') else 0.0
    
    else:
        # Unknown models: use safe defaults
        if warmup_steps is None:
            warmup_steps = min(config.max_warmup_steps, total_steps // 3)
        if skip_interval is None:
            if total_steps <= 15:
                skip_interval = 3  # Conservative
            elif total_steps <= 30:
                skip_interval = 2
            else:
                skip_interval = 3
        noise_scale = config.noise_scale if hasattr(config, 'noise_scale') else 0.0
    
    # Ensure all parameters are set (fallback to safe defaults)
    if warmup_steps is None:
        warmup_steps = max(total_steps // 3, 3)
        logger.warning(f"[CacheDiT] warmup_steps not set, using fallback: {warmup_steps}")
    if skip_interval is None:
        skip_interval = 3
        logger.warning(f"[CacheDiT] skip_interval not set, using fallback: {skip_interval}")
    if 'noise_scale' not in locals():
        noise_scale = 0.0
    
    def cached_forward(*args, **kwargs):
        state = _lightweight_cache_state
        state["call_count"] += 1
        call_id = state["call_count"]
        
        if call_id <= warmup_steps:
            import time
            start = time.time()
            result = transformer._original_forward(*args, **kwargs)
            elapsed = time.time() - start
            
            state["compute_count"] += 1
            state["compute_times"].append(elapsed)
            
            if isinstance(result, torch.Tensor):
                state["last_result"] = result.detach()
            elif isinstance(result, tuple):
                state["last_result"] = tuple(
                    r.detach() if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            else:
                state["last_result"] = result
            
            return result
        
        steps_after_warmup = call_id - warmup_steps
        should_skip = (steps_after_warmup % skip_interval == 0)
        
        if should_skip and state["last_result"] is not None:
            state["skip_count"] += 1
            
            cached_result = state["last_result"]
            
            if noise_scale > 0:
                if isinstance(cached_result, torch.Tensor):
                    noise = torch.randn_like(cached_result) * noise_scale
                    cached_result = cached_result + noise
                elif isinstance(cached_result, tuple):
                    # Apply noise to tensor elements in tuple
                    cached_result = tuple(
                        (r + torch.randn_like(r) * noise_scale) if isinstance(r, torch.Tensor) else r
                        for r in cached_result
                    )
            
            return cached_result
        else:
            import time
            start = time.time()
            result = transformer._original_forward(*args, **kwargs)
            elapsed = time.time() - start
            
            state["compute_count"] += 1
            state["compute_times"].append(elapsed)
            
            if isinstance(result, torch.Tensor):
                state["last_result"] = result.detach()
            elif isinstance(result, tuple):
                state["last_result"] = tuple(
                    r.detach() if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            else:
                state["last_result"] = result
            
            return result
    
    # Replace forward method
    transformer.forward = cached_forward
    
    logger.info(
        f"[CacheDiT] Lightweight cache enabled: "
        f"model={transformer_class}, steps={total_steps}, "
        f"warmup={warmup_steps}, skip_interval={skip_interval}"
    )


def _get_lightweight_cache_stats():
    """Get statistics from lightweight cache"""
    state = _lightweight_cache_state
    
    if not state["enabled"]:
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
        "total_steps": total_calls,
        "computed_steps": compute_count,
        "cached_steps": cache_hits,
        "cache_hit_rate": cache_hit_rate,
        "estimated_speedup": estimated_speedup,
        "avg_compute_time": avg_time,
    }


# =============================================================================
# Configuration Holder Class
# =============================================================================

class CacheDiTConfig:
    """
    Holds all CacheDiT configuration for a model instance.
    Stored in transformer_options for access by wrappers.
    """
    
    def __init__(
        self,
        # Basic settings
        model_type: str,
        forward_pattern: str,
        strategy: str,
        # DBCache settings
        fn_blocks: int,
        bn_blocks: int,
        threshold: float,
        max_warmup_steps: int,
        # CFG settings
        enable_separate_cfg: Optional[bool],
        cfg_compute_first: bool,
        # Advanced settings
        skip_interval: int,
        noise_scale: float,
        taylor_order: int,
        scm_policy: str,
        # Runtime settings
        verbose: bool = False,
        print_summary: bool = True,
        # User overrides
        user_warmup_steps: int = 0,
        user_skip_interval: int = 0,
    ):
        # Configuration
        self.model_type = model_type
        self.forward_pattern = forward_pattern
        self.strategy = strategy
        
        self.fn_blocks = fn_blocks
        self.bn_blocks = bn_blocks
        self.threshold = threshold
        self.max_warmup_steps = max_warmup_steps
        
        self.enable_separate_cfg = enable_separate_cfg
        self.cfg_compute_first = cfg_compute_first
        
        self.skip_interval = skip_interval
        self.noise_scale = noise_scale
        self.taylor_order = taylor_order
        self.scm_policy = scm_policy
        
        self.verbose = verbose
        self.print_summary = print_summary
        
        # User overrides for lightweight cache
        self.user_warmup_steps = user_warmup_steps
        self.user_skip_interval = user_skip_interval
        
        # Runtime state
        self.is_enabled = False
        self.num_inference_steps: Optional[int] = None
        self.current_step: int = 0
        self.first_step_done: bool = False
        
    def clone(self) -> "CacheDiTConfig":
        """Create a copy for cloned models."""
        new_config = CacheDiTConfig(
            model_type=self.model_type,
            forward_pattern=self.forward_pattern,
            strategy=self.strategy,
            fn_blocks=self.fn_blocks,
            bn_blocks=self.bn_blocks,
            threshold=self.threshold,
            max_warmup_steps=self.max_warmup_steps,
            enable_separate_cfg=self.enable_separate_cfg,
            cfg_compute_first=self.cfg_compute_first,
            skip_interval=self.skip_interval,
            noise_scale=self.noise_scale,
            taylor_order=self.taylor_order,
            scm_policy=self.scm_policy,
            verbose=self.verbose,
            print_summary=self.print_summary,
            # User overrides (FIX: these were missing!)
            user_warmup_steps=self.user_warmup_steps,
            user_skip_interval=self.user_skip_interval,
        )
        new_config.is_enabled = self.is_enabled
        new_config.num_inference_steps = self.num_inference_steps
        return new_config
    
    def reset(self):
        """Reset runtime state for new generation."""
        self.current_step = 0
        self.first_step_done = False
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get config as dict for summary display."""
        return {
            "model_type": self.model_type,
            "pattern": self.forward_pattern,
            "strategy": self.strategy,
            "fn": self.fn_blocks,
            "bn": self.bn_blocks,
            "threshold": self.threshold,
            "skip_interval": self.skip_interval,
            "noise_scale": self.noise_scale,
            "taylor_order": self.taylor_order,
        }


# =============================================================================
# ComfyUI Wrapper Functions (Smart Hooks)
# =============================================================================

def _cache_dit_outer_sample_wrapper(executor, *args, **kwargs):
    """
    OUTER_SAMPLE wrapper: The "Smart" Refresh Hook.
    
    - Auto-detects num_inference_steps from sigmas
    - Enables cache-dit at sampling start
    - Refreshes context with correct step count
    - Prints summary dashboard at end
    """
    guider = executor.class_obj
    orig_model_options = guider.model_options
    transformer = None
    config = None
    
    try:
        # Clone model options for this run
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        
        # Get CacheDiT config
        config: CacheDiTConfig = guider.model_options.get("transformer_options", {}).get("cache_dit_turbo")
        if config is None:
            return executor(*args, **kwargs)
        
        # Clone and reset config for this run
        config = config.clone()
        config.reset()
        guider.model_options["transformer_options"]["cache_dit_turbo"] = config
        
        # =====================================================================
        # SMART STEP DETECTION: Extract num_inference_steps from sigmas
        # =====================================================================
        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas")
        if sigmas is not None:
            num_steps = len(sigmas) - 1  # sigmas has N+1 elements for N steps
            config.num_inference_steps = num_steps
            
            if config.verbose:
                logger.info(f"[CacheDiT] Auto-detected {num_steps} inference steps")
        
        # Get transformer reference
        model_patcher = guider.model_patcher
        if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
            transformer = model_patcher.model.diffusion_model
            
            # Enable or refresh cache-dit
            if not config.is_enabled and config.num_inference_steps is not None:
                _enable_cache_dit(transformer, config)
                config.is_enabled = True
            elif config.is_enabled and config.num_inference_steps is not None:
                _refresh_cache_dit(transformer, config)
        
        if config.verbose:
            logger.info(
                f"[CacheDiT] Enabled: model={config.model_type}, "
                f"pattern={config.forward_pattern}, strategy={config.strategy}, "
                f"steps={config.num_inference_steps}"
            )
        
        # Execute sampling
        result = executor(*args, **kwargs)
        
        # =====================================================================
        # SUMMARY DASHBOARD: Print statistics after sampling
        # =====================================================================
        if config.print_summary and transformer is not None:
            try:
                # Check if using lightweight cache fallback
                lightweight_stats = _get_lightweight_cache_stats()
                
                if lightweight_stats is not None:
                    # Display lightweight cache statistics
                    logger.info(
                        f"\n[CacheDiT] Lightweight Cache Statistics:\n"
                        f"  Total Steps: {lightweight_stats['total_steps']}\n"
                        f"  Computed: {lightweight_stats['computed_steps']}\n"
                        f"  Cached: {lightweight_stats['cached_steps']}\n"
                        f"  Cache Hit Rate: {lightweight_stats['cache_hit_rate']:.1f}%\n"
                        f"  Estimated Speedup: {lightweight_stats['estimated_speedup']:.2f}x\n"
                        f"  Avg Compute Time: {lightweight_stats['avg_compute_time']:.3f}s"
                    )
                else:
                    # Use standard cache-dit statistics
                    dashboard = print_summary_to_log(
                        transformer=transformer,
                        model_type=config.model_type,
                        num_steps=config.num_inference_steps or 0,
                        config_info=config.get_config_info(),
                    )
            except Exception as e:
                logger.warning(f"[CacheDiT] Summary failed: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"[CacheDiT] outer_sample_wrapper error: {e}")
        import traceback
        traceback.print_exc()
        return executor(*args, **kwargs)
    finally:
        # Restore original model options
        try:
            guider.model_options = orig_model_options
        except:
            pass


def _cache_dit_diffusion_model_wrapper(executor, *args, **kwargs):
    """
    DIFFUSION_MODEL wrapper: Per-step processing.
    
    - Tracks current step
    - Applies noise injection on cached outputs (if enabled)
    """
    try:
        # Get transformer_options
        transformer_options = args[-1] if isinstance(args[-1], dict) else kwargs.get("transformer_options", {})
        config: CacheDiTConfig = transformer_options.get("cache_dit_turbo")
        
        if config is not None:
            # Track steps
            if not config.first_step_done:
                config.first_step_done = True
                config.current_step = 0
            else:
                config.current_step += 1
            
            if config.verbose and config.current_step % 10 == 0:
                logger.debug(f"[CacheDiT] Step {config.current_step}/{config.num_inference_steps}")
        
        # Execute forward pass
        output = executor(*args, **kwargs)
        
        # Apply noise injection if enabled (prevents "dead" regions in cached outputs)
        # This is CRITICAL for video models and high-resolution image generation
        if config is not None and config.noise_scale > 0:
            try:
                # Only apply to cached steps (after warmup) to preserve quality
                if config.current_step >= config.max_warmup_steps:
                    output = apply_noise_injection(
                        output=output,
                        noise_scale=config.noise_scale,
                    )
                    if config.verbose and config.current_step % 5 == 0:
                        logger.debug(
                            f"[CacheDiT] ✓ Noise injection applied at step {config.current_step}: "
                            f"scale={config.noise_scale}"
                        )
            except Exception as e:
                logger.warning(f"[CacheDiT] ✗ Noise injection failed: {e}")
        
        return output
        
    except Exception as e:
        logger.error(f"[CacheDiT] diffusion_model_wrapper error: {e}")
        return executor(*args, **kwargs)


# =============================================================================
# Cache-DiT Integration Functions
# =============================================================================

def _enable_cache_dit(transformer: torch.nn.Module, config: CacheDiTConfig):
    """
    Enable cache-dit on transformer using BlockAdapter interface.
    """
    try:
        import cache_dit
        from cache_dit import BlockAdapter
        
        # For ComfyUI models, manually extract blocks
        from .utils import _manual_extract_blocks
        
        manual_blocks = _manual_extract_blocks(transformer)
        if not manual_blocks or len(manual_blocks) == 0:
            raise RuntimeError("Failed to extract blocks from transformer")
        
        logger.info(f"[CacheDiT] Extracted {len(manual_blocks)} blocks for caching")
        
        # Build cache config
        cache_config = build_cache_config(
            num_inference_steps=config.num_inference_steps,
            fn_blocks=config.fn_blocks,
            bn_blocks=config.bn_blocks,
            threshold=config.threshold,
            max_warmup_steps=config.max_warmup_steps,
            enable_separate_cfg=config.enable_separate_cfg,
            cfg_compute_first=config.cfg_compute_first,
            skip_interval=config.skip_interval,
            strategy=config.strategy,
            scm_policy=config.scm_policy if config.scm_policy != "none" else None,
        )
        
        # Build calibrator config
        calibrator_config = build_calibrator_config(config.taylor_order)
        
        # Get forward pattern
        pattern = get_forward_pattern(config.forward_pattern)
        
        # === Check if lightweight cache should be used directly ===
        transformer_class_name = transformer.__class__.__name__
        
        # ComfyUI models typically don't work well with cache-dit BlockAdapter
        # Use lightweight cache directly for better compatibility
        use_lightweight = transformer_class_name in [
            "NextDiT",        # Z-Image
            "Lumina",         # Lumina
            "QwenImage",      # Qwen-Image
            "HunyuanVideo",   # HunyuanVideo
        ]
        
        if use_lightweight:
            logger.info(f"[CacheDiT] Using lightweight cache for {transformer_class_name}")
            _enable_lightweight_cache(
                transformer=transformer,
                blocks=manual_blocks,
                config=config,
                cache_config=cache_config,
            )
            return
        
        # === Attempt to use cache-dit's BlockAdapter ===
        cache_dit_failed = False
        try:
            # Standard models: ensure blocks are ModuleList
            blocks_module_list = manual_blocks
            if not isinstance(manual_blocks, torch.nn.ModuleList):
                blocks_module_list = torch.nn.ModuleList(manual_blocks)
                logger.info(f"[CacheDiT] Converted {len(manual_blocks)} blocks to ModuleList")
            
            # Inject blocks into transformer for cache-dit auto-detection
            if not hasattr(transformer, 'blocks'):
                transformer.blocks = blocks_module_list
                logger.info(f"[CacheDiT] Injected blocks into transformer.blocks")
            
            adapter = BlockAdapter(blocks=blocks_module_list)
            
            # Verify adapter has blocks
            if not hasattr(adapter, 'blocks') or len(adapter.blocks) == 0:
                raise RuntimeError(f"BlockAdapter created but has no blocks!")
            
            logger.info(f"[CacheDiT] BlockAdapter verified: {len(adapter.blocks)} blocks")
            
            # Enable cache with BlockAdapter
            enable_kwargs = {
                "cache_config": cache_config,
                "forward_pattern": pattern,
            }
            if calibrator_config is not None:
                enable_kwargs["calibrator_config"] = calibrator_config
            
            cache_dit.enable_cache(adapter, **enable_kwargs)
            
            # CRITICAL: Check if transformer reference is maintained
            # If transformer is None, cache-dit won't be able to track statistics
            if hasattr(adapter, 'transformer') and adapter.transformer is None:
                logger.warning(f"[CacheDiT] BlockAdapter.transformer is None - statistics won't work")
                raise RuntimeError("BlockAdapter has no transformer reference")
            
            logger.info(
                f"[CacheDiT] Cache enabled via BlockAdapter: "
                f"F{config.fn_blocks}B{config.bn_blocks}, "
                f"threshold={config.threshold}, warmup={config.max_warmup_steps}"
            )
                
        except Exception as e:
            cache_dit_failed = True
            logger.warning(f"[CacheDiT] cache-dit BlockAdapter failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info(f"[CacheDiT] Falling back to direct forward hook implementation")
        
        # === Fallback: Direct forward hook (for unsupported models) ===
        if cache_dit_failed:
            # Use simple but reliable forward replacement strategy
            _enable_lightweight_cache(
                transformer=transformer,
                blocks=manual_blocks,
                config=config,
                cache_config=cache_config,
            )
        
    except Exception as e:
        logger.error(f"[CacheDiT] ✗ Failed to enable cache: {e}")
        import traceback
        traceback.print_exc()
        raise


def _refresh_cache_dit(transformer: torch.nn.Module, config: CacheDiTConfig):
    """
    Refresh cache-dit context with updated settings.
    Called when num_inference_steps changes between requests.
    """
    try:
        # Check if using lightweight cache (always reset if enabled, regardless of transformer_id)
        if _lightweight_cache_state.get("enabled"):
            current_transformer_id = id(transformer)
            previous_transformer_id = _lightweight_cache_state.get("transformer_id")
            
            # Reset lightweight cache state for new run (required for each independent sampling)
            _lightweight_cache_state["call_count"] = 0
            _lightweight_cache_state["skip_count"] = 0
            _lightweight_cache_state["compute_count"] = 0
            _lightweight_cache_state["last_result"] = None
            _lightweight_cache_state["compute_times"] = []
            _lightweight_cache_state["config"] = config
            _lightweight_cache_state["transformer_id"] = current_transformer_id
            
            # Log only if verbose or transformer changed (different model/step in workflow)
            if config.verbose:
                if previous_transformer_id != current_transformer_id:
                    logger.info(
                        f"[CacheDiT] Lightweight cache reset for new sampling: "
                        f"{config.num_inference_steps} steps (transformer changed)"
                    )
                else:
                    logger.info(
                        f"[CacheDiT] Lightweight cache reset for new sampling: "
                        f"{config.num_inference_steps} steps"
                    )
            return
        
        # Standard cache-dit refresh
        import cache_dit
        
        # Rebuild configs with new step count
        cache_config = build_cache_config(
            num_inference_steps=config.num_inference_steps,
            fn_blocks=config.fn_blocks,
            bn_blocks=config.bn_blocks,
            threshold=config.threshold,
            max_warmup_steps=config.max_warmup_steps,
            enable_separate_cfg=config.enable_separate_cfg,
            cfg_compute_first=config.cfg_compute_first,
            skip_interval=config.skip_interval,
            strategy=config.strategy,
            scm_policy=config.scm_policy if config.scm_policy != "none" else None,
        )
        
        calibrator_config = build_calibrator_config(config.taylor_order)
        
        refresh_kwargs = {
            "cache_config": cache_config,
            "verbose": config.verbose,
        }
        if calibrator_config is not None:
            refresh_kwargs["calibrator_config"] = calibrator_config
        
        cache_dit.refresh_context(transformer, **refresh_kwargs)
        
        if config.verbose:
            logger.info(f"[CacheDiT] Context refreshed for {config.num_inference_steps} steps")
        
    except Exception as e:
        logger.warning(f"[CacheDiT] Failed to refresh context: {e}")


# =============================================================================
# Main Node: CacheDiT_Model_Optimizer
# =============================================================================

class CacheDiT_Model_Optimizer:
    """
    CacheDiT Model Optimizer for ComfyUI
    
    Accelerates DiT model inference through inter-step residual caching.
    Automatically detects inference steps and refreshes context.
    
    Supports: Qwen-Image, Z-Image, Flux, HunyuanVideo, Wan, and custom models.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_names = ["Auto"] + get_all_preset_names()
        
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/Disable CacheDiT acceleration\n启用/禁用 CacheDiT 加速"
                }),
                "model_type": (preset_names, {
                    "default": "Auto",
                    "tooltip": "Model preset (Auto=auto-detect, or select specific model)\n"
                               "模型预设 (Auto=自动检测, 或选择特定模型)"
                }),
                "warmup_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Warmup steps (0=use preset default, 8=warmup 8 steps)\n"
                               "预热步数 (0=使用预设默认值, 8=预热8步)"
                }),
                "skip_interval": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Skip interval (0=use preset default, 3=skip every 3rd step, 5=skip every 5th)\n"
                               "跳过间隔 (0=使用预设默认值, 3=每3步跳1次, 5=每5步跳1次)"
                }),
                "print_summary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print performance dashboard after sampling\n采样后打印性能仪表盘"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "optimize"
    CATEGORY = "⚡ CacheDiT"
    DESCRIPTION = (
        "Accelerate DiT models (Qwen-Image, Z-Image, etc.) via caching.\n"
        "通过缓存加速 DiT 模型 (Qwen-Image、Z-Image 等)"
    )
    
    def optimize(
        self,
        model,
        enable: bool = True,
        model_type: str = "Auto",
        warmup_steps: int = 0,
        skip_interval: int = 0,
        print_summary: bool = True,
    ):
        """Apply CacheDiT optimization to the model."""
        # If disabled, return model as-is
        if not enable:
            logger.info("[CacheDiT] Optimization disabled, model restored")
            return self.disable(model)
        
        # Check configuration from transformer (more persistent than model_options)
        transformer = None
        existing_config = None
        
        if hasattr(model.model, 'diffusion_model'):
            transformer = model.model.diffusion_model
            existing_config = getattr(transformer, '_cache_dit_config', None)
            logger.info(f"[CacheDiT] Transformer ID: {id(transformer)}, existing config: {existing_config is not None}")
        else:
            logger.info(f"[CacheDiT] No transformer found")
        
        if existing_config is not None:
            # Need to resolve "Auto" before comparison
            resolved_model_type = model_type
            logger.info(f"[CacheDiT] Existing configuration detected: model_type={existing_config['model_type']}, "
                  f"warmup_steps={existing_config['warmup_steps']}, "
                  f"skip_interval={existing_config['skip_interval']}, "
                  f"print_summary={existing_config['print_summary']}")
            if model_type == "Auto":
                if hasattr(model.model, 'diffusion_model'):
                    transformer = model.model.diffusion_model
                    class_name = transformer.__class__.__name__
                    if "Qwen" in class_name:
                        resolved_model_type = "Qwen-Image"
                    elif "NextDiT" in class_name or "Lumina" in class_name:
                        resolved_model_type = "Z-Image"
                    elif "Flux" in class_name or "FLUX" in class_name:
                        resolved_model_type = "Flux"
                    else:
                        resolved_model_type = "Custom"
                else:
                    resolved_model_type = "Custom"
            
            # Compare key parameters
            params_changed = (
                existing_config["model_type"] != resolved_model_type or
                existing_config["warmup_steps"] != warmup_steps or
                existing_config["skip_interval"] != skip_interval or
                existing_config["print_summary"] != print_summary
            )
            
            if params_changed:
                logger.info(
                    f"[CacheDiT] Parameters changed, reconfiguring... "
                    f"(model: {existing_config['model_type']}→{resolved_model_type}, "
                    f"warmup: {existing_config['warmup_steps']}→{warmup_steps}, "
                    f"skip: {existing_config['skip_interval']}→{skip_interval})"
                )
                # Disable old configuration first (disable already clones internally)
                result = self.disable(model)
                model = result[0]
            else:
                logger.info("[CacheDiT] Configuration unchanged, keeping existing setup")
                return (model,)
        else:
            # First-time configuration: need to clone
            model = model.clone()
        
        # Auto-detect model type if "Auto" is selected
        if model_type == "Auto":
            # Try to detect from model architecture
            if hasattr(model.model, 'diffusion_model'):
                transformer = model.model.diffusion_model
                class_name = transformer.__class__.__name__
                
                # Map common class names to presets
                if "Qwen" in class_name:
                    model_type = "Qwen-Image"
                elif "NextDiT" in class_name or "Lumina" in class_name:
                    model_type = "Z-Image"
                elif "Flux" in class_name or "FLUX" in class_name:
                    model_type = "Flux"
                else:
                    model_type = "Custom"
                    logger.info(f"[CacheDiT] Auto-detected unknown model: {class_name}, using Custom preset")
            else:
                model_type = "Custom"
                logger.info("[CacheDiT] Cannot auto-detect model type, using Custom preset")
        
        # Get preset
        preset = get_preset(model_type)
        
        # Use all preset defaults
        config = CacheDiTConfig(
            model_type=model_type,
            forward_pattern=preset.forward_pattern,
            strategy=preset.default_strategy,
            fn_blocks=preset.fn_blocks,
            bn_blocks=preset.bn_blocks,
            threshold=preset.threshold,
            max_warmup_steps=3,
            enable_separate_cfg=preset.enable_separate_cfg,
            cfg_compute_first=preset.cfg_compute_first,
            skip_interval=0,
            noise_scale=preset.noise_scale,
            taylor_order=1,
            scm_policy="none",
            verbose=False,
            print_summary=print_summary,
            # User overrides for lightweight cache
            user_warmup_steps=warmup_steps,
            user_skip_interval=skip_interval,
        )
        
        # Store config in both model options and transformer (for persistence)
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["cache_dit_turbo"] = config
        
        # Also store simplified config dict on transformer for persistence
        if hasattr(model.model, 'diffusion_model'):
            transformer = model.model.diffusion_model
            transformer._cache_dit_config = {
                "model_type": config.model_type,
                "warmup_steps": config.user_warmup_steps,
                "skip_interval": config.user_skip_interval,
                "print_summary": config.print_summary,
            }
            logger.info(f"[CacheDiT] Config stored on transformer: {config.model_type}, warmup={config.user_warmup_steps}, skip={config.user_skip_interval}")
        else:
            logger.info(f"[CacheDiT] Config stored: {config.model_type}, warmup={config.user_warmup_steps}, skip={config.user_skip_interval}")
        
        # Add wrappers
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "cache_dit_turbo",
            _cache_dit_outer_sample_wrapper
        )
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "cache_dit_turbo",
            _cache_dit_diffusion_model_wrapper
        )
        
        return (model,)

    def disable(self, model):
        model = model.clone()
        
        # Remove config
        if "cache_dit_turbo" in model.model_options.get("transformer_options", {}):
            del model.model_options["transformer_options"]["cache_dit_turbo"]
        
        # Remove wrappers
        for wrapper_type in [comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]:
            if "cache_dit_turbo" in model.wrappers.get(wrapper_type, {}):
                del model.wrappers[wrapper_type]["cache_dit_turbo"]
        
        # Restore original transformer state
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                transformer = model.model.diffusion_model
                
                # Restore lightweight cache forward method
                if hasattr(transformer, '_original_forward'):
                    transformer.forward = transformer._original_forward
                    delattr(transformer, '_original_forward')
                
                # Disable standard cache-dit
                try:
                    import cache_dit
                    cache_dit.disable_cache(transformer)
                except:
                    pass
                
                # Reset lightweight cache state
                global _lightweight_cache_state
                if _lightweight_cache_state.get("transformer_id") == id(transformer):
                    _lightweight_cache_state = {
                        "enabled": False,
                        "transformer_id": None,
                        "call_count": 0,
                        "skip_count": 0,
                        "last_result": None,
                        "config": None,
                    }
                    logger.info("[CacheDiT] Lightweight cache state reset")
        except Exception as e:
            logger.warning(f"[CacheDiT] Failed to fully restore model: {e}")
        
        return (model,)

# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "CacheDiT_Model_Optimizer": CacheDiT_Model_Optimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheDiT_Model_Optimizer": "⚡ CacheDiT Accelerator",
}
