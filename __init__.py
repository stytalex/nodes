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
    from .pyptv_log_viewer_node import (
        NODE_CLASS_MAPPINGS       as _M13,
        NODE_DISPLAY_NAME_MAPPINGS as _D13,
    )
    NODE_CLASS_MAPPINGS.update(_M13)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D13)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_log_viewer_node: {e}")

# ─── LTX-2.3 пайплайн (3 автономные ноды) ─────────────────────────────────
# Все три ноды автономны — без входных/выходных коннектов между собой.
#  • Train LoRA — замкнутый цикл: download → build → preprocess → train → upload
#  • Check VAE  — автономная: download → build → decode → IMAGE preview
#  • Dramabox   — автономный поток генерации голоса

try:
    from .pyptv_ltx23_check_vae_node import (
        NODE_CLASS_MAPPINGS       as _M18,
        NODE_DISPLAY_NAME_MAPPINGS as _D18,
    )
    NODE_CLASS_MAPPINGS.update(_M18)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D18)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_ltx23_check_vae_node: {e}")

try:
    from .pyptv_ltx23_train_lora_node import (
        NODE_CLASS_MAPPINGS       as _M19,
        NODE_DISPLAY_NAME_MAPPINGS as _D19,
    )
    NODE_CLASS_MAPPINGS.update(_M19)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D19)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_ltx23_train_lora_node: {e}")

try:
    from .pyptv_ltx23_dramabox_node import (
        NODE_CLASS_MAPPINGS       as _M11,
        NODE_DISPLAY_NAME_MAPPINGS as _D11,
    )
    NODE_CLASS_MAPPINGS.update(_M11)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D11)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_ltx23_dramabox_node: {e}")

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
