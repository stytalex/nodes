NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .elevenlabs_voice_changer_node import (
        NODE_CLASS_MAPPINGS       as _M1,
        NODE_DISPLAY_NAME_MAPPINGS as _D1,
    )
    NODE_CLASS_MAPPINGS.update(_M1)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D1)
except Exception as e:
    print(f"[pyPTV] Failed to load elevenlabs_voice_changer_node: {e}")

try:
    from .elevenlabs_fal_voice_changer_node import (
        NODE_CLASS_MAPPINGS       as _M2,
        NODE_DISPLAY_NAME_MAPPINGS as _D2,
    )
    NODE_CLASS_MAPPINGS.update(_M2)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D2)
except Exception as e:
    print(f"[pyPTV] Failed to load elevenlabs_fal_voice_changer_node: {e}")

try:
    from .pyptv_load_video_node import (
        NODE_CLASS_MAPPINGS       as _M3,
        NODE_DISPLAY_NAME_MAPPINGS as _D3,
    )
    NODE_CLASS_MAPPINGS.update(_M3)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D3)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_load_video_node: {e}")

try:
    from .pyptv_combine_video_node import (
        NODE_CLASS_MAPPINGS       as _M4,
        NODE_DISPLAY_NAME_MAPPINGS as _D4,
    )
    NODE_CLASS_MAPPINGS.update(_M4)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D4)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_combine_video_node: {e}")

try:
    from .pyptv_rife_node import (
        NODE_CLASS_MAPPINGS       as _M5,
        NODE_DISPLAY_NAME_MAPPINGS as _D5,
    )
    NODE_CLASS_MAPPINGS.update(_M5)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D5)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_rife_node: {e}")

try:
    from .pyptv_crop_node import (
        NODE_CLASS_MAPPINGS       as _M6,
        NODE_DISPLAY_NAME_MAPPINGS as _D6,
    )
    NODE_CLASS_MAPPINGS.update(_M6)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D6)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_crop_node: {e}")

try:
    from .training_ltx23_lora import (
        NODE_CLASS_MAPPINGS       as _M7,
        NODE_DISPLAY_NAME_MAPPINGS as _D7,
    )
    NODE_CLASS_MAPPINGS.update(_M7)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D7)
except Exception as e:
    print(f"[pyPTV] Failed to load training_ltx23_lora: {e}")

try:
    from .ltx23_encode_image_latents_node import (
        NODE_CLASS_MAPPINGS       as _M8,
        NODE_DISPLAY_NAME_MAPPINGS as _D8,
    )
    NODE_CLASS_MAPPINGS.update(_M8)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D8)
except Exception as e:
    print(f"[pyPTV] Failed to load ltx23_encode_image_latents_node: {e}")

try:
    from .ltx23_encode_audio_latents_node import (
        NODE_CLASS_MAPPINGS       as _M9,
        NODE_DISPLAY_NAME_MAPPINGS as _D9,
    )
    NODE_CLASS_MAPPINGS.update(_M9)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D9)
except Exception as e:
    print(f"[pyPTV] Failed to load ltx23_encode_audio_latents_node: {e}")
    
try:
    from .ltx23_encode_caption_conditions_node import (
        NODE_CLASS_MAPPINGS       as _M10,
        NODE_DISPLAY_NAME_MAPPINGS as _D10,
    )
    NODE_CLASS_MAPPINGS.update(_M10)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D10)
except Exception as e:
    print(f"[pyPTV] Failed to load ltx23_encode_caption_conditions_node: {e}")

try:
    from .pyptv_dramabox_node import (
        NODE_CLASS_MAPPINGS       as _M11,
        NODE_DISPLAY_NAME_MAPPINGS as _D11,
    )
    NODE_CLASS_MAPPINGS.update(_M11)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D11)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_dramabox_node: {e}")
    
try:
    from .pyptv_load_dataset_node import (
        NODE_CLASS_MAPPINGS       as _M12,
        NODE_DISPLAY_NAME_MAPPINGS as _D12,
    )
    NODE_CLASS_MAPPINGS.update(_M12)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D12)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_load_dataset_node: {e}")

try:
    from .pyptv_log_viewer_node import (
        NODE_CLASS_MAPPINGS       as _M13,
        NODE_DISPLAY_NAME_MAPPINGS as _D13,
    )
    NODE_CLASS_MAPPINGS.update(_M13)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D13)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_log_viewer_node: {e}")
    
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

