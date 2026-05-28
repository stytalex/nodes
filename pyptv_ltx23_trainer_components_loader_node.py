"""
Trainer Components Loader (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Единая точка загрузки ВСЕХ моделей для пайплайна LTX-2.3 + DramaBox.

Что делает:
  1. Загружает .safetensors один раз на CPU через ltx_trainer.
  2. Кэширует в глобальном dict — повторные запуски берут из кэша.
  3. Отдаёт PYPTV_MODELS — dict со всеми компонентами на CPU.

Каждая нода сама переносит на GPU только то, что ей нужно,
и выгружает обратно на CPU после работы.

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

# ---------------------------------------------------------------------------
# GPU / CPU  offloading helpers (используются всеми нодами пайплайна)
# ---------------------------------------------------------------------------
_ALL_MODULE_KEYS = [
    "video_vae_encoder",
    "video_vae_decoder",
    "audio_vae_encoder",
    "audio_vae_decoder",
    "vocoder",
    "text_encoder",
    "embeddings_processor",
    "transformer",
    "dit_model",
]


def load_to_gpu(components: dict, keys: list[str]) -> None:
    """Перенести указанные компоненты на GPU."""
    loaded = []
    for k in keys:
        obj = components.get(k)
        if isinstance(obj, torch.nn.Module):
            components[k] = obj.to("cuda")
            loaded.append(k)
    if loaded:
        print(f"[offload] load_to_gpu: {', '.join(loaded)}")
        torch.cuda.empty_cache()


def offload_to_cpu(components: dict, keys: list[str]) -> None:
    """Перенести указанные компоненты на CPU."""
    offloaded = []
    for k in keys:
        obj = components.get(k)
        if isinstance(obj, torch.nn.Module):
            components[k] = obj.to("cpu")
            offloaded.append(k)
    if offloaded:
        print(f"[offload] offload_to_cpu: {', '.join(offloaded)}")
        torch.cuda.empty_cache()


def _load_dramabox_dit(checkpoint_path: str, device, dtype) -> torch.nn.Module:
    """Загружает DramaBox audio-only DiT с правильным маппингом ключей."""
    from ltx_core.loader import DummyRegistry
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.model_protocol import ModelConfigurator
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.text_projection import create_caption_projection

    class _AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
        @classmethod
        def from_config(cls, cfg: dict) -> LTXModel:
            t = cfg.get("transformer", {})
            cp = None
            if not t.get("caption_proj_before_connector", False):
                with torch.device("meta"):
                    cp = create_caption_projection(t, audio=True)
            return LTXModel(
                model_type=LTXModelType.AudioOnly,
                audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                audio_in_channels=t.get("audio_in_channels", 128),
                audio_out_channels=t.get("audio_out_channels", 128),
                num_layers=t.get("num_layers", 48),
                audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                norm_eps=t.get("norm_eps", 1e-6),
                attention_type=AttentionFunction(t.get("attention_type", "default")),
                positional_embedding_theta=10000.0,
                audio_positional_embedding_max_pos=[20.0],
                timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                double_precision_rope=t.get("frequencies_precision", False) == "float64",
                apply_gated_attention=t.get("apply_gated_attention", False),
                audio_caption_projection=cp,
                cross_attention_adaln=t.get("cross_attention_adaln", False),
            )

    sd_ops = (
        SDOps("AO")
        .with_matching(prefix="model.diffusion_model.")
        .with_replacement("model.diffusion_model.", "")
    )

    return Builder(
        model_path=checkpoint_path,
        model_class_configurator=_AudioOnlyConfigurator,
        model_sd_ops=sd_ops,
        registry=DummyRegistry(),
    ).build(device=device, dtype=dtype).eval()


def _load_all_components():
    """Загружает все модели на CPU через ltx_trainer / ltx_core.
    Каждая нода сама грузит нужное на GPU через load_to_gpu()."""
    device = "cpu"
    dtype = torch.bfloat16

    print(f"[PyPTVComponentsLoader] Загрузка компонентов на CPU...")

    # --- Основная модель LTX-2.3 (transformer + VAE decoders + vocoder + text_encoder) ---
    print("[PyPTVComponentsLoader] Основная модель LTX-2.3...")
    main: LtxModelComponents = load_model(
        checkpoint_path=_MODEL_PATH,
        text_encoder_path=_TEXT_ENCODER_PATH,
        device=device,
        dtype=dtype,
        with_video_vae_encoder=False,
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

    # --- DramaBox DiT transformer (audio-only, своя схема ключей) ---
    print("[PyPTVComponentsLoader] DramaBox DiT transformer...")
    dit_model = _load_dramabox_dit(_DIT_PATH, device=device, dtype=dtype)

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
            }
        }

    RETURN_TYPES = ("PYPTV_MODELS", "PYPTV_DATASET")
    RETURN_NAMES = ("components", "dataset")
    FUNCTION = "load"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = False

    def load(self, dataset):
        if "all" not in _COMPONENTS_CACHE:
            _COMPONENTS_CACHE["all"] = _load_all_components()
        else:
            print(f"[PyPTVComponentsLoader] Используем кэш")

        return (_COMPONENTS_CACHE["all"], dataset)


NODE_CLASS_MAPPINGS = {
    "PyPTVTrainerComponentsLoader": PyPTVTrainerComponentsLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVTrainerComponentsLoader": "Trainer Components Loader (pyPTV)",
}
