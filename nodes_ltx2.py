"""
ComfyUI-CacheDiT: LTX-2 Specialized Node (Fixed Interrupt Resilience)
=========================================
"""

from __future__ import annotations
import logging
import traceback
import torch
import comfy.model_patcher
import comfy.patcher_extension
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

logger = logging.getLogger("ComfyUI-CacheDiT-LTX2")

# === LTX-2 Specific Cache State ===
def get_empty_state():
    return {
        "enabled": False,
        "transformer_id": None,
        "call_count": 0,
        "skip_count": 0,
        "compute_count": 0,
        "last_result": None,
        "config": None,
        "compute_times": [],
        "current_timestep": None,
        "timestep_count": 0,
        "last_input_shape": None,
        "calls_per_step": None,
        "last_timestep_call": 0,
        "i2v_mode": False,
    }

_ltx2_cache_state = get_empty_state()

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
        self.is_enabled = False
        self.num_inference_steps: Optional[int] = None
        self.current_step: int = 0
    
    def clone(self) -> "LTX2CacheConfig":
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
        self.current_step = 0

def _enable_ltx2_cache(transformer, config: LTX2CacheConfig):
    global _ltx2_cache_state
    
    if hasattr(transformer, '_original_forward'):
        _refresh_ltx2_cache(transformer, config)
        return

    transformer._original_forward = transformer.forward
    
    # Initialize fresh state
    new_state = get_empty_state()
    new_state.update({
        "enabled": True,
        "transformer_id": id(transformer),
        "config": config,
    })
    _ltx2_cache_state.update(new_state)
    
    def cached_forward(*args, **kwargs):
        state = _ltx2_cache_state
        state["call_count"] += 1
        
        # Detection of Resolution/Shape Changes
        current_input_shape = None
        if len(args) > 0:
            x = args[0]
            if isinstance(x, torch.Tensor):
                current_input_shape = tuple(x.shape)
            elif isinstance(x, (tuple, list)) and len(x) > 0:
                if isinstance(x[0], torch.Tensor):
                    current_input_shape = tuple(x[0].shape)
        
        last_shape = state.get("last_input_shape")
        if current_input_shape is not None and last_shape is not None:
            if current_input_shape != last_shape:
                state["last_result"] = None
                state["current_timestep"] = None
                state["timestep_count"] = 0
                state["call_count"] = 1
                state["skip_count"] = 0
                state["compute_count"] = 0
                state["compute_times"] = []
        state["last_input_shape"] = current_input_shape

        # Timestep Extraction
        timestep = None
        if len(args) >= 2:
            timestep = args[1]
        elif 'timestep' in kwargs:
            timestep = kwargs['timestep']
        elif 'v_timestep' in kwargs:
            timestep = kwargs['v_timestep']
        
        current_ts = None
        if timestep is not None:
            try:
                if isinstance(timestep, (tuple, list)):
                    ts_value = timestep[0] if len(timestep) > 0 else None
                    if isinstance(ts_value, torch.Tensor):
                        current_ts = float(ts_value.flatten()[0].item()) if ts_value.numel() > 0 else 0.0
                elif isinstance(timestep, torch.Tensor):
                    current_ts = float(timestep.flatten()[0].item()) if timestep.numel() > 0 else 0.0
                else:
                    current_ts = float(timestep)
            except:
                current_ts = None
        
        prev_ts = state.get("current_timestep")
        if current_ts != prev_ts and current_ts is not None:
            state["current_timestep"] = current_ts
            state["timestep_count"] += 1
            
            if state["timestep_count"] == 1 and abs(current_ts) < 0.001:
                state["i2v_mode"] = True
            
            if state["timestep_count"] == 2:
                state["calls_per_step"] = state["call_count"] - state["last_timestep_call"]
            
            state["last_timestep_call"] = state["call_count"]
            timestep_id = state["timestep_count"]
        else:
            if state["calls_per_step"] is not None and state["calls_per_step"] > 0:
                estimated_step = (state["call_count"] - 1) // state["calls_per_step"] + 1
                timestep_id = max(estimated_step, state["timestep_count"])
            else:
                timestep_id = state["timestep_count"]

        cache_config = state.get("config")
        warmup_steps = cache_config.warmup_steps if cache_config else 6
        skip_interval = cache_config.skip_interval if cache_config else 4
        noise_scale = cache_config.noise_scale if cache_config else 0.001
        
        # Determine if we should compute or use cache
        if state.get("i2v_mode", False):
            # I2V Call-based logic
            should_compute = (state["call_count"] <= warmup_steps) or \
                             ((state["call_count"] - warmup_steps - 1) % skip_interval == 0)
        else:
            # T2V Timestep-based logic
            should_compute = (timestep_id <= warmup_steps) or \
                             ((timestep_id - warmup_steps - 1) % skip_interval == 0)

        if not should_compute and state["last_result"] is not None:
            state["skip_count"] += 1
            cached_result = state["last_result"]
            if noise_scale > 0 and isinstance(cached_result, tuple):
                noised = []
                for r in cached_result:
                    if isinstance(r, torch.Tensor):
                        noised.append(r + torch.randn_like(r) * noise_scale)
                    else:
                        noised.append(r)
                return tuple(noised)
            return cached_result
        else:
            import time
            start = time.time()
            result = transformer._original_forward(*args, **kwargs)
            elapsed = time.time() - start
            
            state["compute_count"] += 1
            state["compute_times"].append(elapsed)
            
            if isinstance(result, tuple):
                state["last_result"] = tuple(r.detach() if isinstance(r, torch.Tensor) else r for r in result)
            else:
                state["last_result"] = result.detach() if isinstance(result, torch.Tensor) else result
            return result

    transformer.forward = cached_forward
    logger.info(f"[LTX2-Cache] Enabled: warmup={config.warmup_steps}, skip={config.skip_interval}")

def _refresh_ltx2_cache(transformer, config: LTX2CacheConfig):
    global _ltx2_cache_state
    new_state = get_empty_state()
    new_state.update({
        "enabled": True,
        "transformer_id": id(transformer),
        "config": config,
    })
    _ltx2_cache_state.update(new_state)
    if config.verbose:
        logger.info(f"[LTX2-Cache] State refreshed for new run.")

def _get_ltx2_cache_stats():
    state = _ltx2_cache_state
    if not state.get("enabled") or state["call_count"] == 0:
        return None
    
    total_calls = state["call_count"]
    cache_hits = state["skip_count"]
    compute_count = state["compute_count"]
    avg_time = sum(state["compute_times"]) / max(len(state["compute_times"]), 1)
    
    return {
        "total_calls": total_calls,
        "computed": compute_count,
        "cached": cache_hits,
        "hit_rate": (cache_hits / total_calls) * 100,
        "speedup": total_calls / max(compute_count, 1),
        "avg_time": avg_time,
    }

def _ltx2_outer_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    orig_model_options = guider.model_options
    transformer = None
    
    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        config: LTX2CacheConfig = guider.model_options.get("transformer_options", {}).get("ltx2_cache")
        
        if config is None:
            return executor(*args, **kwargs)
        
        config = config.clone()
        config.reset()
        
        model_patcher = guider.model_patcher
        if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
            transformer = model_patcher.model.diffusion_model
            # Force a fresh state on every sample start
            _refresh_ltx2_cache(transformer, config)
            _enable_ltx2_cache(transformer, config)

        result = executor(*args, **kwargs)
        
        if config.print_summary:
            stats = _get_ltx2_cache_stats()
            if stats:
                logger.info(
                    f"\n[LTX2-Cache] Summary:\n"
                    f"  Calls: {stats['total_calls']} | Cached: {stats['cached']} ({stats['hit_rate']:.1f}%)\n"
                    f"  Speedup: {stats['speedup']:.2f}x | Avg Compute: {stats['avg_time']:.4f}s"
                )
        return result
    except Exception as e:
        logger.error(f"[LTX2-Cache] Critical Error: {e}")
        traceback.print_exc()
        return executor(*args, **kwargs)
    finally:
        guider.model_options = orig_model_options

class CacheDiT_LTX2_Optimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "warmup_steps": ("INT", {"default": 10, "min": 1, "max": 20}),
                "skip_interval": ("INT", {"default": 5, "min": 1, "max": 15}),
                "noise_scale": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.01, "step": 0.0001}),
                "print_summary": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "optimize"
    CATEGORY = "⚡ CacheDiT"

    def optimize(self, model, enable=True, warmup_steps=10, skip_interval=5, noise_scale=0.001, print_summary=True):
        if not enable:
            return self.disable(model)
        
        model = model.clone()
        config = LTX2CacheConfig(warmup_steps, skip_interval, noise_scale, False, print_summary)
        
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["ltx2_cache"] = config
        
        # Register Wrapper
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "ltx2_cache",
            _ltx2_outer_sample_wrapper
        )
        return (model,)

    def disable(self, model):
        model = model.clone()
        if "ltx2_cache" in model.model_options.get("transformer_options", {}):
            del model.model_options["transformer_options"]["ltx2_cache"]
        
        # Attempt to strip wrapper and restore forward
        try:
            if hasattr(model.model, 'diffusion_model'):
                transformer = model.model.diffusion_model
                if hasattr(transformer, '_original_forward'):
                    transformer.forward = transformer._original_forward
                    delattr(transformer, '_original_forward')
            global _ltx2_cache_state
            _ltx2_cache_state.update(get_empty_state())
        except:
            pass
        return (model,)

NODE_CLASS_MAPPINGS = {"CacheDiT_LTX2_Optimizer": CacheDiT_LTX2_Optimizer}
NODE_DISPLAY_NAME_MAPPINGS = {"CacheDiT_LTX2_Optimizer": "⚡ CacheDiT LTX-2 Accelerator"}