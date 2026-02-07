"""
ComfyUI-CacheDiT: LTX-2 Specialized Node
=========================================

Dedicated node for LTX-2 Audio-Visual Transformer with optimized settings.
Separated from main node to avoid interference with other models.

LTX-2 Architecture:
- Dual latent paths: video (hidden_states) + audio (audio_hidden_states)
- Block forward: (h, audio_h, enc_h, audio_enc_h) -> (h, audio_h)
- Requires lightweight cache (BlockAdapter incompatible due to signature mismatch)
"""

from __future__ import annotations
import logging
import traceback
import torch
import comfy.model_patcher
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

logger = logging.getLogger("ComfyUI-CacheDiT-LTX2")


# === LTX-2 Specific Cache State ===
_ltx2_cache_state = {
    "enabled": False,
    "transformer_id": None,
    "call_count": 0,
    "skip_count": 0,
    "compute_count": 0,
    "last_result": None,
    "config": None,
    "compute_times": [],
}


class LTX2CacheConfig:
    """Configuration for LTX-2 cache optimization."""
    
    def __init__(
        self,
        warmup_steps: int = 6,
        skip_interval: int = 4,
        noise_scale: float = 0.001,
        verbose: bool = False,
        print_summary: bool = True,
    ):
        self.warmup_steps = warmup_steps
        self.skip_interval = skip_interval
        self.noise_scale = noise_scale
        self.verbose = verbose
        self.print_summary = print_summary
        
        # Runtime state
        self.is_enabled = False
        self.num_inference_steps: Optional[int] = None
        self.current_step: int = 0
    
    def clone(self) -> "LTX2CacheConfig":
        """Create a copy for cloned models."""
        new_config = LTX2CacheConfig(
            warmup_steps=self.warmup_steps,
            skip_interval=self.skip_interval,
            noise_scale=self.noise_scale,
            verbose=self.verbose,
            print_summary=self.print_summary,
        )
        new_config.is_enabled = self.is_enabled
        new_config.num_inference_steps = self.num_inference_steps
        return new_config
    
    def reset(self):
        """Reset runtime state for new generation."""
        self.current_step = 0


def _enable_ltx2_cache(transformer, config: LTX2CacheConfig):
    """Enable lightweight cache for LTX-2 transformer (timestep-based tracking)"""
    global _ltx2_cache_state
    if hasattr(transformer, '_original_forward'):
        current_id = id(transformer)
        cached_id = _ltx2_cache_state.get("transformer_id")
        
        if current_id == cached_id:
            logger.info("[LTX2-Cache] Already enabled, resetting state")
            _ltx2_cache_state.update({
                "call_count": 0,
                "skip_count": 0,
                "compute_count": 0,
                "last_result": None,
                "compute_times": [],
            })
            return
    
    transformer._original_forward = transformer.forward
    
    _ltx2_cache_state.update({
        "enabled": True,
        "transformer_id": id(transformer),
        "call_count": 0,
        "skip_count": 0,
        "compute_count": 0,
        "last_result": None,
        "config": config,
        "compute_times": [],
        "current_timestep": None,  # Track timestep instead of call count
        "timestep_count": 0,       # Number of unique timesteps seen
        "last_input_shape": None,  # Track input shape to detect resolution changes
        "calls_per_step": None,    # Estimated calls per denoising step (for I2V)
        "last_timestep_call": 0,   # Call count at last timestep change
        "i2v_mode": False,         # Whether in I2V mode
    })
    
    def cached_forward(*args, **kwargs):
        """
        Cached forward for LTX-2 transformer.
        
        CRITICAL: ComfyUI calls forward multiple times per step!
        - CFG: 2 calls (conditional + unconditional)
        - Per-block calls: 48 blocks
        - Total: ~100 calls per denoising step
        
        We must track by TIMESTEP, not call_count!
        """
        state = _ltx2_cache_state
        state["call_count"] += 1
        
        # Check input shape to detect resolution changes (e.g., upscale stage)
        # args[0] = x (input tensor/tuple)
        current_input_shape = None
        if len(args) > 0:
            x = args[0]
            if isinstance(x, torch.Tensor):
                current_input_shape = tuple(x.shape)
            elif isinstance(x, (tuple, list)) and len(x) > 0:
                # LTX-2 uses tuple input
                if isinstance(x[0], torch.Tensor):
                    current_input_shape = tuple(x[0].shape)
        
        # If shape changed, clear cache and reset counters
        last_shape = state.get("last_input_shape")
        if current_input_shape is not None and last_shape is not None:
            if current_input_shape != last_shape:
                # logger.info(
                #     f"[LTX2-Cache] Input shape changed: {last_shape} → {current_input_shape}, "
                #     "clearing cache (likely upscale stage)"
                # )
                state["last_result"] = None
                state["current_timestep"] = None
                state["timestep_count"] = 0
                state["call_count"] = 1  # Reset to 1 (current call)
                state["skip_count"] = 0
                state["compute_count"] = 0
                state["compute_times"] = []
        
        state["last_input_shape"] = current_input_shape
        
        # Extract timestep from kwargs or args
        # LTXAVModel signature: forward(self, x, timestep, context, ...)
        timestep = None
        if len(args) >= 2:
            # args[0] = x, args[1] = timestep
            timestep = args[1]
        elif 'timestep' in kwargs:
            timestep = kwargs['timestep']
        elif 'v_timestep' in kwargs:
            # Block-level forward
            timestep = kwargs['v_timestep']
        
        # DEBUG: Log what we're intercepting
        # if state["call_count"] == 1:
            # logger.info(f"[LTX2-Cache] Intercepting forward: args={len(args)}, kwargs={list(kwargs.keys())[:10]}")
            # if timestep is not None:
            #     ts_type = type(timestep)
            #     if isinstance(timestep, torch.Tensor):
            #         ts_info = f"shape={tuple(timestep.shape)}, values={timestep.flatten()[:5].tolist() if timestep.numel() <= 10 else timestep.flatten()[:5].tolist()}"
            #     elif isinstance(timestep, (tuple, list)):
            #         ts_info = f"len={len(timestep)}"
            #         if len(timestep) > 0 and isinstance(timestep[0], torch.Tensor):
            #             ts_info += f", first_elem_shape={tuple(timestep[0].shape)}, first_values={timestep[0].flatten()[:5].tolist()}"
            #     else:
            #         ts_info = f"value={timestep}"
                # logger.info(f"[LTX2-Cache] timestep type: {ts_type}, {ts_info}")
        
        # Track unique timesteps
        current_ts = None
        if timestep is not None:
            # LTX-2 uses tuple or multi-element tensor: (video_timestep, audio_timestep)
            try:
                if isinstance(timestep, (tuple, list)):
                    if len(timestep) >= 1:
                        # Use first element (video timestep) for tracking
                        ts_value = timestep[0]
                        # ts_value is still a Tensor, need to extract scalar
                        if isinstance(ts_value, torch.Tensor):
                            if ts_value.numel() > 1:
                                # Multi-element tensor: find max non-zero value for I2V
                                # I2V masks first frame to 0, but later frames have real timesteps
                                ts_flat = ts_value.flatten()
                                non_zero_ts = ts_flat[ts_flat > 0.001]
                                if non_zero_ts.numel() > 0:
                                    current_ts = float(non_zero_ts.max().item())
                                else:
                                    # All zeros - use first element
                                    current_ts = float(ts_flat[0].item())
                            else:
                                current_ts = float(ts_value.item())
                        else:
                            current_ts = float(ts_value)
                    else:
                        ts_value = None
                elif isinstance(timestep, torch.Tensor):
                    if timestep.numel() > 1:
                        # Multi-element tensor: take max non-zero
                        ts_flat = timestep.flatten()
                        non_zero_ts = ts_flat[ts_flat > 0.001]
                        if non_zero_ts.numel() > 0:
                            current_ts = float(non_zero_ts.max().item())
                        else:
                            current_ts = float(ts_flat[0].item())
                    else:
                        # Single-element tensor
                        ts_value = timestep
                        current_ts = float(ts_value.item())
                else:
                    current_ts = float(timestep)
            except Exception as e:
                # If timestep extraction fails, log and skip caching for this call
                if state["call_count"] <= 2:
                    logger.warning(f"[LTX2-Cache] Failed to extract timestep: {e}, disabling cache for safety")
                current_ts = None
        
        prev_ts = state.get("current_timestep")
        if current_ts != prev_ts and current_ts is not None:
            state["current_timestep"] = current_ts
            state["timestep_count"] += 1
            timestep_id = state["timestep_count"]
            
            # I2V detection: if first timestep is ~0, enable call-count tracking
            if state["timestep_count"] == 1 and abs(current_ts) < 0.001:
                # logger.info(f"[LTX2-Cache] Detected I2V mode (t≈0), using conservative call-based warmup")
                state["i2v_mode"] = True
            
            # Calculate calls per step for T2V detection
            if state["timestep_count"] == 2:
                calls_per_step = state["call_count"] - state["last_timestep_call"]
                state["calls_per_step"] = calls_per_step
                # logger.info(f"[LTX2-Cache] Detected ~{calls_per_step} calls per step (T2V mode)")
            
            state["last_timestep_call"] = state["call_count"]
            
            # Log timestep transitions
            # if timestep_id <= 3:
            #     logger.info(f"[LTX2-Cache] Timestep {timestep_id}: t={current_ts:.4f}")
        else:
            # Timestep unchanged - use call count for step estimation
            if state["calls_per_step"] is not None and state["calls_per_step"] > 0:
                # Estimate step from call count (T2V mode)
                estimated_step = (state["call_count"] - 1) // state["calls_per_step"] + 1
                timestep_id = max(estimated_step, state["timestep_count"])
                
                # Log progress for T2V mode
                # if state["call_count"] in [50, 100, 150] and state["timestep_count"] == 1:
                #     logger.info(f"[LTX2-Cache] Call-count tracking: call {state['call_count']}, estimated step {estimated_step}")
            else:
                # No calls_per_step yet - could be I2V or early T2V
                timestep_id = state["timestep_count"]
        
        # Get parameters from config stored in state
        cache_config = state.get("config")
        warmup_steps = cache_config.warmup_steps if cache_config else 6
        skip_interval = cache_config.skip_interval if cache_config else 4
        noise_scale = cache_config.noise_scale if cache_config else 0.001
        
        # Special handling for I2V mode: use call-based warmup
        if state.get("i2v_mode", False):
            # In I2V, warmup based on call count (more conservative)
            i2v_warmup_calls = warmup_steps
            
            if state["call_count"] <= i2v_warmup_calls:
                # Warmup phase
                import time
                start = time.time()
                result = transformer._original_forward(*args, **kwargs)
                elapsed = time.time() - start
                
                state["compute_count"] += 1
                state["compute_times"].append(elapsed)
                
                # Cache result
                if isinstance(result, tuple):
                    state["last_result"] = tuple(
                        r.detach() if isinstance(r, torch.Tensor) else r
                        for r in result
                    )
                else:
                    state["last_result"] = result.detach() if isinstance(result, torch.Tensor) else result
                
                return result
            else:
                # Post-warmup: compute periodically based on skip_interval
                calls_after_warmup = state["call_count"] - i2v_warmup_calls
                should_compute = (calls_after_warmup == 1) or ((calls_after_warmup - 1) % skip_interval == 0)
                
                cache_valid = state["last_result"] is not None
                if not should_compute and cache_valid:
                    # Use cached result
                    state["skip_count"] += 1
                    cached_result = state["last_result"]
                    
                    # Apply noise injection
                    if noise_scale > 0 and isinstance(cached_result, tuple):
                        noised = []
                        for r in cached_result:
                            if isinstance(r, torch.Tensor):
                                noise = torch.randn_like(r) * noise_scale
                                noised.append(r + noise)
                            else:
                                noised.append(r)
                        cached_result = tuple(noised)
                    
                    return cached_result
                else:
                    # Compute
                    import time
                    start = time.time()
                    result = transformer._original_forward(*args, **kwargs)
                    elapsed = time.time() - start
                    
                    state["compute_count"] += 1
                    state["compute_times"].append(elapsed)
                    
                    # Update cache
                    if isinstance(result, tuple):
                        state["last_result"] = tuple(
                            r.detach() if isinstance(r, torch.Tensor) else r
                            for r in result
                        )
                    else:
                        state["last_result"] = result.detach() if isinstance(result, torch.Tensor) else result
                    
                    return result
        
        # T2V mode: use timestep-based warmup (standard logic)
        noise_scale = cache_config.noise_scale if cache_config else 0.001
        
        # Warmup phase: always compute (based on TIMESTEP count, not call count)
        if timestep_id <= warmup_steps:
            import time
            start = time.time()
            result = transformer._original_forward(*args, **kwargs)
            elapsed = time.time() - start
            
            state["compute_count"] += 1
            state["compute_times"].append(elapsed)
            
            # Cache result - handle tuple (h, audio_h)
            if isinstance(result, tuple):
                state["last_result"] = tuple(
                    r.detach() if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            else:
                state["last_result"] = result.detach() if isinstance(result, torch.Tensor) else result
            
            return result
        
        # Post-warmup: compute periodically based on interval (timestep-based)
        steps_after_warmup = timestep_id - warmup_steps
        # Compute every skip_interval steps: 1, 1+skip, 1+2*skip, ...
        # For skip=4: compute at step 1, 5, 9, 13, 17, ...
        # For skip=6: compute at step 1, 7, 13, 19, ...
        should_compute = (steps_after_warmup == 1) or ((steps_after_warmup - 1) % skip_interval == 0)
        
        # Always compute if no cached result (safety)
        cache_valid = state["last_result"] is not None
        if not cache_valid:
            should_compute = True
        
        if not should_compute and cache_valid:
            # Use cached result
            state["skip_count"] += 1
            cached_result = state["last_result"]
            
            # Apply noise injection for temporal consistency
            if noise_scale > 0 and isinstance(cached_result, tuple):
                noised = []
                for r in cached_result:
                    if isinstance(r, torch.Tensor):
                        noise = torch.randn_like(r) * noise_scale
                        noised.append(r + noise)
                    else:
                        noised.append(r)
                cached_result = tuple(noised)
            
            return cached_result
        else:
            # Compute normally
            import time
            start = time.time()
            result = transformer._original_forward(*args, **kwargs)
            elapsed = time.time() - start
            
            state["compute_count"] += 1
            state["compute_times"].append(elapsed)
            
            # Update cache
            if isinstance(result, tuple):
                state["last_result"] = tuple(
                    r.detach() if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            else:
                state["last_result"] = result.detach() if isinstance(result, torch.Tensor) else result
            
            return result
    
    # Replace forward method
    transformer.forward = cached_forward
    
    logger.info(
        f"[LTX2-Cache] Enabled: warmup={config.warmup_steps}, "
        f"skip_interval={config.skip_interval}, noise_scale={config.noise_scale:.4f}"
    )


def _refresh_ltx2_cache(transformer, config: LTX2CacheConfig):
    """
    Refresh LTX-2 cache for new sampling run.
    Called when num_inference_steps changes between requests.
    
    CRITICAL: Reset only runtime counters, NOT enabled/transformer_id.
    This matches main node's _refresh_cache_dit behavior.
    """
    global _ltx2_cache_state
    
    try:
        current_transformer_id = id(transformer)
        previous_transformer_id = _ltx2_cache_state.get("transformer_id")
        
        # Reset ONLY runtime state (NOT enabled/transformer_id)
        _ltx2_cache_state["call_count"] = 0
        _ltx2_cache_state["skip_count"] = 0
        _ltx2_cache_state["compute_count"] = 0
        _ltx2_cache_state["last_result"] = None
        _ltx2_cache_state["compute_times"] = []
        _ltx2_cache_state["config"] = config
        _ltx2_cache_state["transformer_id"] = current_transformer_id
        _ltx2_cache_state["current_timestep"] = None
        _ltx2_cache_state["timestep_count"] = 0
        _ltx2_cache_state["last_input_shape"] = None
        
        # Log only if verbose or transformer changed
        if config.verbose:
            if previous_transformer_id != current_transformer_id:
                logger.info(
                    f"[LTX2-Cache] Lightweight cache reset for new sampling: "
                    f"{config.num_inference_steps} steps (transformer changed)"
                )
            else:
                logger.info(
                    f"[LTX2-Cache] Lightweight cache reset for new sampling: "
                    f"{config.num_inference_steps} steps"
                )
    
    except Exception as e:
        logger.error(f"[LTX2-Cache] Refresh failed: {e}")
        traceback.print_exc()


def _get_ltx2_cache_stats():
    """Get statistics from LTX-2 cache."""
    state = _ltx2_cache_state
    
    if not state.get("enabled"):
        return None
    
    # Use timestep_count (actual denoising steps), not call_count (forward calls)
    total_steps = state.get("timestep_count", 0)
    total_calls = state["call_count"]
    cache_hits = state["skip_count"]
    compute_count = state["compute_count"]
    
    if total_steps == 0:
        return None
    
    cache_hit_rate = (cache_hits / max(total_calls, 1)) * 100
    avg_time = sum(state["compute_times"]) / max(len(state["compute_times"]), 1)
    estimated_speedup = total_calls / max(compute_count, 1)
    
    return {
        "total_steps": total_steps,      # Actual denoising steps
        "total_calls": total_calls,       # Total forward calls
        "computed_calls": compute_count,  # Forward calls that computed
        "cached_calls": cache_hits,       # Forward calls that used cache
        "cache_hit_rate": cache_hit_rate,
        "estimated_speedup": estimated_speedup,
        "avg_compute_time": avg_time,
    }


def _ltx2_outer_sample_wrapper(executor, *args, **kwargs):
    """
    OUTER_SAMPLE wrapper for LTX-2.
    
    This is called at the CFGGuider.sample level, where:
    - executor: the original CFGGuider.sample method
    - executor.class_obj: the CFGGuider instance
    - args[0]: noise
    - args[1]: latent_image
    - args[2]: sampler (KSAMPLER)
    - args[3]: sigmas
    - ...
    """
    guider = executor.class_obj
    orig_model_options = guider.model_options
    transformer = None
    config = None
    
    try:
        # Clone model options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        
        # Get config
        config: LTX2CacheConfig = guider.model_options.get("transformer_options", {}).get("ltx2_cache")
        if config is None:
            return executor(*args, **kwargs)
        
        # Clone and reset config
        config = config.clone()
        config.reset()
        guider.model_options["transformer_options"]["ltx2_cache"] = config
        
        # Extract num_inference_steps from sigmas (4th positional arg)
        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas")
        if sigmas is not None:
            num_steps = len(sigmas) - 1
            config.num_inference_steps = num_steps
        
        # Get transformer
        model_patcher = guider.model_patcher
        if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
            transformer = model_patcher.model.diffusion_model
            
            # Check if cache already enabled
            cache_already_enabled = (
                hasattr(transformer, '_original_forward') or
                _ltx2_cache_state.get("enabled")
            )
            
            if config.num_inference_steps is not None:
                if not cache_already_enabled:
                    # First time: enable cache
                    logger.info(f"[LTX2-Cache] Enabling for {config.num_inference_steps} steps (first run)")
                    _enable_ltx2_cache(transformer, config)
                    config.is_enabled = True
                else:
                    # Subsequent runs: REFRESH (full reset)
                    logger.info(f"[LTX2-Cache] Refreshing for {config.num_inference_steps} steps (subsequent run)")
                    _refresh_ltx2_cache(transformer, config)
                    config.is_enabled = True
        
        # Execute sampling
        result = executor(*args, **kwargs)
        
        # Print summary
        if config.print_summary and transformer is not None:
            stats = _get_ltx2_cache_stats()
            if stats:
                logger.info(
                    f"\n[LTX2-Cache] Performance Summary:\n"
                    f"  Denoising Steps: {stats['total_steps']}\n"
                    f"  Total Forward Calls: {stats['total_calls']}\n"
                    f"  Computed: {stats['computed_calls']} ({stats['computed_calls']/stats['total_calls']*100:.1f}%)\n"
                    f"  Cached: {stats['cached_calls']} ({stats['cache_hit_rate']:.1f}%)\n"
                    f"  Estimated Speedup: {stats['estimated_speedup']:.2f}x"
                )
        
        return result
        
    except Exception as e:
        logger.error(f"[LTX2-Cache] Error: {e}")
        traceback.print_exc()
        return executor(*args, **kwargs)
    finally:
        try:
            guider.model_options = orig_model_options
        except:
            pass


class CacheDiT_LTX2_Optimizer:
    """
    LTX-2 Specialized Cache Optimizer
    
    Optimized specifically for LTX-2 Audio-Visual Transformer:
    - Conservative settings for video generation quality
    - Longer warmup for temporal consistency
    - Noise injection to prevent static artifacts
    - Handles dual-path output (video + audio latents)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "warmup_steps": ("INT", {
                    "default": 10,
                    "min": 3,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Warmup steps before caching (longer = better quality)\n"
                               "预热步数(越长质量越好)\n"
                               "Recommended: 10 for 20 steps, 15-20 for 40 steps"
                }),
                "skip_interval": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 15,
                    "step": 1,
                    "tooltip": "Cache interval (smaller = more conservative, better quality)\n"
                               "缓存间隔（越小越保守，质量越好）\n"
                               "Recommended: 6-8 for balanced, 10+ for quality-first"
                }),
                "noise_scale": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.0,
                    "max": 0.01,
                    "step": 0.0001,
                    "tooltip": "Noise injection scale for temporal consistency\n"
                               "噪声注入强度（提升时序一致性）"
                }),
                "print_summary": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "optimize"
    CATEGORY = "⚡ CacheDiT"
    DESCRIPTION = (
        "LTX-2 specialized cache optimizer for video generation.\n"
        "LTX-2 视频生成专用缓存优化器\n\n"
        "Recommended Presets for 20 steps (推荐预设 20步):\n"
        "• Speed (速度): warmup=6, skip=4 (50% cache, 2.0x speedup)\n"
        "• Balanced (平衡) ⭐: warmup=10, skip=5 (40% cache, 1.7x speedup)\n"
        "• Quality (质量): warmup=12, skip=7 (30% cache, 1.4x speedup)\n\n"
        "Math: computed = warmup + ⌈(total-warmup)/skip⌉\n"
        "For 40 steps, increase both values proportionally."
    )
    
    def optimize(
        self,
        model,
        enable: bool = True,
        warmup_steps: int = 10,
        skip_interval: int = 5,
        noise_scale: float = 0.001,
        print_summary: bool = True,
    ):
        """Apply LTX-2 specific cache optimization."""
        
        if not enable:
            logger.info("[CacheDiT] Optimization disabled, model restored")
            return self.disable(model)
        
        # Detect LTX-2 model and check existing config
        try:
            diffusion_model = model.model.diffusion_model
            model_class_name = diffusion_model.__class__.__name__
            
            if model_class_name != "LTXAVModel":
                logger.warning(
                    f"[LTX2-Cache] Not LTX-2 model (detected: {model_class_name}), "
                    "optimization skipped"
                )
                return (model,)
            
            # Check existing configuration
            existing_config = getattr(diffusion_model, '_ltx2_cache_config', None)
            
            if existing_config is not None:
                # Compare parameters
                params_changed = (
                    existing_config["warmup_steps"] != warmup_steps or
                    existing_config["skip_interval"] != skip_interval or
                    existing_config["noise_scale"] != noise_scale or
                    existing_config["print_summary"] != print_summary
                )
                
                if params_changed:
                    logger.info(
                        f"[LTX2-Cache] Parameters changed: "
                        f"warmup {existing_config['warmup_steps']}→{warmup_steps}, "
                        f"skip {existing_config['skip_interval']}→{skip_interval}"
                    )
                    # Disable and reconfigure
                    result = self.disable(model)
                    model = result[0]
                else:
                    logger.info("[LTX2-Cache] Configuration unchanged")
                    return (model,)
        
        except Exception as e:
            logger.error(f"[LTX2-Cache] Failed to detect model: {e}")
            return (model,)
        
        # Clone model for new configuration
        model = model.clone()
        
        # Create config
        config = LTX2CacheConfig(
            warmup_steps=warmup_steps,
            skip_interval=skip_interval,
            noise_scale=noise_scale,
            verbose=False,
            print_summary=print_summary,
        )
        
        # Store config in both transformer_options and transformer object
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        
        model.model_options["transformer_options"]["ltx2_cache"] = config
        
        # Store simplified config on transformer for persistence
        diffusion_model._ltx2_cache_config = {
            "warmup_steps": warmup_steps,
            "skip_interval": skip_interval,
            "noise_scale": noise_scale,
            "print_summary": print_summary,
        }
        
        # Register wrapper
        try:
            import comfy.patcher_extension
            
            model.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                "ltx2_cache",
                _ltx2_outer_sample_wrapper
            )
            
            logger.info("[LTX2-Cache] Wrapper registered")
        
        except Exception as e:
            logger.error(f"[LTX2-Cache] Failed to register wrapper: {e}")
            traceback.print_exc()
        
        logger.info(
            f"[LTX2-Cache] ✓ Configured: warmup={warmup_steps}, "
            f"skip={skip_interval}, noise={noise_scale:.4f}"
        )
        
        # Recommendations
        if warmup_steps < 8 or skip_interval < 6:
            logger.info(
                f"[LTX2-Cache] Aggressive settings (warmup={warmup_steps}, skip={skip_interval}). "
                "For better quality, try: warmup=8-10, skip=3-4"
            )
        
        return (model,)
    
    def disable(self, model):
        """Disable LTX-2 cache optimization."""
        model = model.clone()
        
        # Remove config
        if "ltx2_cache" in model.model_options.get("transformer_options", {}):
            del model.model_options["transformer_options"]["ltx2_cache"]
        
        # Remove wrapper
        if "ltx2_cache" in model.wrappers.get(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, {}):
            del model.wrappers[comfy.patcher_extension.WrappersMP.OUTER_SAMPLE]["ltx2_cache"]
        
        # Restore original forward and clear cache state
        try:
            if hasattr(model.model, 'diffusion_model'):
                transformer = model.model.diffusion_model
                
                # Restore original forward
                if hasattr(transformer, '_original_forward'):
                    transformer.forward = transformer._original_forward
                    delattr(transformer, '_original_forward')
                    logger.info("[LTX2-Cache] Restored original forward")
                
                # Clear config marker
                if hasattr(transformer, '_ltx2_cache_config'):
                    delattr(transformer, '_ltx2_cache_config')
                
                # Reset global cache state
                global _ltx2_cache_state
                _ltx2_cache_state = {
                    "enabled": False,
                    "transformer_id": None,
                    "call_count": 0,
                    "skip_count": 0,
                    "compute_count": 0,
                    "last_result": None,
                    "config": None,
                    "compute_times": [],
                }
                logger.info("[LTX2-Cache] Cache state reset")
        
        except Exception as e:
            logger.warning(f"[LTX2-Cache] Failed to fully restore: {e}")
        
        return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CacheDiT_LTX2_Optimizer": CacheDiT_LTX2_Optimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheDiT_LTX2_Optimizer": "⚡ CacheDiT LTX-2 Accelerator",
}
