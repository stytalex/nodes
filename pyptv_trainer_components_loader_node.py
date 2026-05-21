"""
Trainer Components Loader (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Единая точка загрузки ВСЕХ моделей для пайплайна LTX-2.3 + DramaBox.

Что делает:
  1. Загружает .safetensors один раз в VRAM через ltx_trainer.
  2. Кэширует в глобальном dict — повторные запуски берут из кэша.
  3. Отдаёт PYPTV_MODELS — dict со всеми компонентами.

Что входит в PYPTV_MODELS:
  • video_vae_encoder    — кодирует картинки → video latents
  • video_vae_decoder    — декодирует video latents → пиксели
  • audio_vae_encoder    — кодирует аудио → audio latents
  • audio_vae_decoder    — декодирует audio latents → мел-спектр
  • vocoder              — мел-спектр → waveform
  • text_encoder         — Gemma (кодирует текст → hidden states)
  • embeddings_processor — connectors (hidden states → video/audio embeddings)
  • transformer          — LTX-2.3 DiT (для inference / тренировки)
  • dit_model            — DramaBox DiT (для генерации аудио)
  • paths                — dict с путями к .safetensors (для тренера)

Как пользоваться:
  1. Создай эту ноду на канвасе.
  2. Подключи её выход components ко всем нодам которые требуют PYPTV_MODELS.
  3. Больше нигде не надо указывать пути к .safetensors — всё здесь.
"""

import torch

from ltx_trainer.model_loader import (
    LtxModelComponents,
    load_audio_vae_encoder,
    load_audio_vae_decoder,
    load_embeddings_processor,
    load_model,
    load_text_encoder,
    load_transformer,
    load_video_vae_encoder,
    load_video_vae_decoder,
    load_vocoder,
)


# ---------------------------------------------------------------------------
# Пути (захардкожены)
# ---------------------------------------------------------------------------
_MODEL_PATH        = "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors"
_TEXT_ENCODER_PATH = "/comfyui/models/text_encoders/gemma-3-12b-it-qat"
_DIT_PATH          = "/comfyui/models/dramabox/dramabox-dit-v1.safetensors"
_AUDIO_COMP_PATH   = "/comfyui/models/dramabox/dramabox-audio-components.safetensors"
_SILENCE_LATENT    = "/comfyui/models/dramabox/assets/silence_latent_frame.pt"


# ---------------------------------------------------------------------------
# Глобальный кэш
# ---------------------------------------------------------------------------
_COMPONENTS_CACHE = {}


def _load_all_components(device_str: str = "cuda"):
    """Загружает все модели через ltx_trainer / ltx_core."""
    device = device_str
    dtype = torch.bfloat16

    print(f"[PyPTVComponentsLoader] Загрузка компонентов на {device_str}...")

    # --- Основная модель LTX-2.3 (transformer + VAE decoders + vocoder + text_encoder) ---
    print("[PyPTVComponentsLoader] Основная модель LTX-2.3...")
    main: LtxModelComponents = load_model(
        checkpoint_path=_MODEL_PATH,
        text_encoder_path=_TEXT_ENCODER_PATH,
        device=device,
        dtype=dtype,
        with_video_vae_encoder=True,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=True,
    )

    # --- Video VAE encoder ---
    print("[PyPTVComponentsLoader] Video VAE encoder...")
    video_vae_encoder = load_video_vae_encoder(_MODEL_PATH, device=device, dtype=dtype)

    # --- Audio VAE encoder ---
    print("[PyPTVComponentsLoader] Audio VAE encoder...")
    audio_vae_encoder = load_audio_vae_encoder(_AUDIO_COMP_PATH, device=device, dtype=dtype)

    # --- Embeddings processor ---
    print("[PyPTVComponentsLoader] Embeddings processor...")
    embeddings_processor = load_embeddings_processor(_MODEL_PATH, device=device, dtype=dtype)

    # --- DramaBox DiT transformer ---
    print("[PyPTVComponentsLoader] DramaBox DiT transformer...")
    dit_model = load_transformer(_DIT_PATH, device=device, dtype=dtype)

    n_params = sum(p.numel() for p in dit_model.parameters()) / 1e9
    print(f"[PyPTVComponentsLoader] DiT: {n_params:.1f}B params")

    components = {
        "video_vae_encoder":    video_vae_encoder,
        "video_vae_decoder":    main.video_vae_decoder,
        "audio_vae_encoder":    audio_vae_encoder,
        "audio_vae_decoder":    main.audio_vae_decoder,
        "vocoder":              main.vocoder,
        "text_encoder":         main.text_encoder,
        "embeddings_processor": embeddings_processor,
        "transformer":          main.transformer,
        "dit_model":            dit_model,
        "paths": {
            "model_path":        _MODEL_PATH,
            "text_encoder_path": _TEXT_ENCODER_PATH,
            "checkpoint":        _DIT_PATH,
            "audio_components":  _AUDIO_COMP_PATH,
            "silence_latent":    _SILENCE_LATENT,
        },
    }

    print("[PyPTVComponentsLoader] ✓ Все компоненты загружены")
    return components


class PyPTVTrainerComponentsLoader:
    """Загружает все модели один раз и отдаёт как PYPTV_MODELS."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("PYPTV_DATASET",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("PYPTV_MODELS", "PYPTV_DATASET")
    RETURN_NAMES = ("components", "dataset")
    FUNCTION = "load"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = False

    def load(self, dataset, device: str):
        cache_key = device
        if cache_key not in _COMPONENTS_CACHE:
            _COMPONENTS_CACHE[cache_key] = _load_all_components(device)
        else:
            print(f"[PyPTVComponentsLoader] Используем кэш")

        return (_COMPONENTS_CACHE[cache_key], dataset)


NODE_CLASS_MAPPINGS = {
    "PyPTVTrainerComponentsLoader": PyPTVTrainerComponentsLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVTrainerComponentsLoader": "Trainer Components Loader (pyPTV)",
}
