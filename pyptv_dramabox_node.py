"""
Драмабокс batch — читает prompts.json, генерирует аудио для каждого промпта,
сохраняет 0000.wav, 0001.wav ... в папку датасета.

prompts.json формат:
[
    "A woman speaks warmly, \"Hello, how are you today?\"",
    "A woman says confidently, \"Let me show you something interesting.\"",
    ...
]

Структура папок:
  /comfyui/models/dramabox/
  ├── dramabox-dit-v1.safetensors          ← checkpoint (DiT трансформер)
  ├── dramabox-audio-components.safetensors ← audio_components (Audio VAE + Vocoder)
  └── assets/silence_latent_frame.pt        ← silence_latent (для IC-LoRA voice cloning)

  /comfyui/models/checkpoints/
  └── ltx-2.3-22b-dev.safetensors          ← full_checkpoint (Gemma text encoder + embeddings)

  /comfyui/models/text_encoders/gemma-3-12b-it-qat/
  └── (Gemma folder)                        ← gemma_root
"""
import json
import os
import random
import re
import uuid
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from comfy.utils import ProgressBar

# ltx_core — устанавливается из https://github.com/Lightricks/LTX-2.git packages/ltx-core
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import DummyRegistry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import create_caption_projection
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.tools import AudioLatentTools
from safetensors import safe_open


# =====================================================================
# Кэш моделей (глобальный — живёт пока ComfyUI запущен)
# =====================================================================
_loader_cache = {}


# =====================================================================
# AudioConditionByReferenceLatent
# Взято из их audio_conditioning.py (src/audio_conditioning.py в DramaBox репо)
# =====================================================================
class AudioConditionByReferenceLatent(ConditioningItem):
    """Conditions audio generation on a reference audio latent for voice cloning.
    
    - Patchifies reference latent [B, C, T, F] -> [B, ref_T, 128]
    - Computes 1D temporal positions via AudioPatchifier
    - Sets denoise_mask = 1.0 - strength (strength=1.0 -> mask=0 -> frozen)
    - Builds ASYMMETRIC attention mask: target->ref=1 (attend), ref->target=0 (read-only)
    - APPENDS ref tokens to END of latent sequence (IC-LoRA pattern)
    """

    def __init__(self, latent: torch.Tensor, strength: float = 1.0):
        self.latent = latent
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: AudioLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)

        ref_shape = AudioLatentShape(
            batch=self.latent.shape[0],
            channels=self.latent.shape[1],
            frames=self.latent.shape[2],
            mel_bins=self.latent.shape[3],
        )
        positions = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=ref_shape,
            device=self.latent.device,
        )
        # Небольшой сдвиг чтобы не было точного t=0 коллизии с target
        positions = positions + 0.5

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=torch.float32,
        )

        # Асимметричная маска внимания:
        # target -> ref : 1.0  (target читает reference — голос)
        # ref -> target : 0.0  (ref не видит шумный target — read-only)
        # ref -> ref    : 1.0  (self-attention внутри reference)
        batch_size = tokens.shape[0]
        num_target = latent_state.latent.shape[1]
        num_ref = tokens.shape[1]
        total = num_target + num_ref

        mask = torch.zeros(
            (batch_size, total, total),
            device=self.latent.device,
            dtype=torch.float32,
        )

        if latent_state.attention_mask is not None:
            mask[:, :num_target, :num_target] = latent_state.attention_mask
        else:
            mask[:, :num_target, :num_target] = 1.0

        mask[:, :num_target, num_target:] = 1.0   # target -> ref
        # mask[:, num_target:, :num_target] = 0.0  # ref -> target (уже 0)
        mask[:, num_target:, num_target:] = 1.0   # ref -> ref

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=mask,
        )


# =====================================================================
# Helpers
# =====================================================================
def _auto_rescale_for_cfg(cfg: float) -> float:
    """CFG-aware std-rescale schedule that prevents output clipping at high cfg."""
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0:
        return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


def _estimate_duration(prompt: str, multiplier: float = 1.1) -> float:
    """Простая оценка длительности из промпта."""
    quotes = re.findall(r'"([^"]+)"', prompt)
    if not quotes:
        quotes = re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", prompt)
        quotes = [q for q in quotes if len(q.split()) > 3]

    if quotes:
        spoken = " ".join(quotes)
    elif ":" in prompt:
        spoken = prompt.split(":", 1)[1].strip()
    else:
        spoken = prompt

    CHARS_PER_SEC = 14.0
    text_len = len(spoken)

    if text_len < 40:
        chars_per_sec = CHARS_PER_SEC * 0.6
    elif text_len < 80:
        chars_per_sec = CHARS_PER_SEC * 0.8
    else:
        chars_per_sec = CHARS_PER_SEC

    duration = text_len / chars_per_sec
    sentence_count = spoken.count(".") + spoken.count("!") + spoken.count("?")
    duration += sentence_count * 0.3
    duration += sum([
        0.5 * len(re.findall(r"\blaugh", prompt, re.IGNORECASE)),
        0.5 * len(re.findall(r"\bpause", prompt, re.IGNORECASE)),
        0.3 * len(re.findall(r"\bsigh", prompt, re.IGNORECASE)),
    ])
    duration = max(3.0, round(duration + 2.0, 1))
    return duration * multiplier


def _resolve_seed(seed: int, idx: int, seed_mode: str) -> int:
    if seed_mode == "fixed":
        return seed
    elif seed_mode == "increment":
        return seed + idx
    else:
        return random.randint(0, 0xffffffff)


# =====================================================================
# Loader: загружает и кэширует все модели
# =====================================================================
class DramaboxTTSLoader:
    """Загружает модели с кэшированием.
    
    checkpoint          — dramabox-dit-v1.safetensors       (DiT трансформер)
    audio_components    — dramabox-audio-components.safetensors (Audio VAE + Vocoder)
    full_checkpoint     — ltx-2.3-22b-dev.safetensors       (Gemma text encoder)
    gemma_root          — папка с Gemma весами
    silence_latent_path — assets/silence_latent_frame.pt    (для IC-LoRA)
    """

    def __init__(
        self,
        checkpoint: str,
        audio_components: str,
        full_checkpoint: str,
        gemma_root: str,
        silence_latent_path: str,
        device: str = "cuda",
        dtype_str: str = "bf16",
    ):
        self.checkpoint = checkpoint
        self.audio_components = audio_components
        self.full_checkpoint = full_checkpoint
        self.gemma_root = gemma_root
        self.silence_latent_path = silence_latent_path
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

        self._prompt_encoder = None
        self._velocity_model = None
        self._audio_conditioner = None
        self._audio_decoder = None
        self._patchifier = AudioPatchifier(patch_size=1)

        print(f"[DramaboxTTSLoader] Инициализация на {device} ({dtype_str})")
        self._load_all()

    def _load_all(self):
        # 1. PromptEncoder — Gemma text encoder.
        # Грузится из full_checkpoint (ltx-2.3-22b-dev.safetensors) потому что там
        # лежат веса embeddings processor / connectors поверх Gemma LLM.
        print("[DramaboxTTSLoader] Загрузка PromptEncoder (Gemma)...")
        self._prompt_encoder = PromptEncoder(
            checkpoint_path=self.full_checkpoint,
            gemma_root=self.gemma_root,
            dtype=self.dtype,
            device=self.device,
            warm=True,
            use_bnb_4bit=True,
            audio_only=True,
        )

        # 2. AudioConditioner (Audio VAE encoder) и AudioDecoder (Audio VAE decoder + Vocoder).
        # У DramaBox эти компоненты вынесены в отдельный файл dramabox-audio-components.safetensors,
        # поэтому передаём audio_components, а НЕ full_checkpoint.
        print("[DramaboxTTSLoader] Загрузка AudioConditioner (Audio VAE encoder)...")
        self._audio_conditioner = AudioConditioner(
            checkpoint_path=self.audio_components,
            dtype=self.dtype,
            device=self.device,
            warm=True,
        )

        print("[DramaboxTTSLoader] Загрузка AudioDecoder (Audio VAE decoder + Vocoder)...")
        self._audio_decoder = AudioDecoder(
            checkpoint_path=self.audio_components,
            dtype=self.dtype,
            device=self.device,
            warm=True,
        )

        # 3. DiT трансформер — читаем конфиг из safetensors metadata.
        print("[DramaboxTTSLoader] Загрузка DramaBox DiT transformer...")
        with safe_open(self.checkpoint, framework="pt") as f:
            config = json.loads(f.metadata()["config"])

        class AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
            @classmethod
            def from_config(cls, cfg):
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

        audio_sd_ops = SDOps("AO").with_matching(
            prefix="model.diffusion_model."
        ).with_replacement("model.diffusion_model.", "")

        builder = Builder(
            model_path=self.checkpoint,
            model_class_configurator=AudioOnlyConfigurator,
            model_sd_ops=audio_sd_ops,
            registry=DummyRegistry(),
        )

        self._velocity_model = builder.build(
            device=self.device, dtype=self.dtype
        ).to(self.device).eval()

        n_params = sum(p.numel() for p in self._velocity_model.parameters()) / 1e9
        print(f"[DramaboxTTSLoader] DiT: {n_params:.1f}B params")
        print("[DramaboxTTSLoader] ✓ Все модели загружены")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        voice_ref_path: Optional[str] = None,
        cfg_scale: float = 2.5,
        stg_scale: float = 1.5,
        duration_multiplier: float = 1.1,
        gen_duration: float = 0.0,
        ref_duration: float = 10.0,
        seed: int = 42,
        rescale_scale: Optional[float] = None,
    ) -> tuple:
        """Генерировать аудио. Returns: (waveform [C, N], sample_rate)."""

        # Целевая длительность
        gen_dur = float(gen_duration) if gen_duration > 0 else _estimate_duration(prompt, duration_multiplier)

        fps = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1

        pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(patchifier=self._patchifier, target_shape=target_shape)

        # Initial state
        state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)

        # Voice reference conditioning через IC-LoRA (AudioConditionByReferenceLatent)
        if voice_ref_path and os.path.exists(voice_ref_path):
            print(f"  [Voice ref] Загрузка {voice_ref_path}...")
            voice = decode_audio_from_file(voice_ref_path, self.device, 0.0, ref_duration)
            if voice is not None:
                w = voice.waveform
                if w.dim() == 2:
                    if w.shape[0] == 1:
                        w = w.repeat(2, 1)
                    w = w.unsqueeze(0)
                elif w.dim() == 3 and w.shape[1] == 1:
                    w = w.repeat(1, 2, 1)

                target_samples = int(ref_duration * voice.sampling_rate)
                if w.shape[-1] < target_samples:
                    w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
                w = w[..., :target_samples]

                # Peak normalize до -4 dBFS
                peak = w.abs().max()
                if peak > 0:
                    w = w * (10 ** (-4.0 / 20) / peak)

                voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)

                # Кодируем через Audio VAE (из audio_components)
                ref_latent = self._audio_conditioner(
                    lambda enc: vae_encode_audio(voice, enc, None)
                )
                print(f"  [Voice ref] Encoded latent: {ref_latent.shape}")

                # Применяем IC-LoRA conditioning — добавляем ref токены в конец sequence
                conditioning = AudioConditionByReferenceLatent(
                    latent=ref_latent.to(self.device, self.dtype),
                    strength=1.0,
                )
                state = conditioning.apply_to(latent_state=state, latent_tools=audio_tools)
                print(f"  [Voice ref] State after conditioning: latent={state.latent.shape}")

        # Noise
        gen = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)
        state = noiser(state, noise_scale=1.0)

        # Encode prompt
        print(f"  [Prompt] Кодирование текста...")
        prompts_list = [prompt]
        use_cfg = cfg_scale > 1.0
        if use_cfg:
            prompts_list.append(
                "worst quality, inconsistent, robotic, distorted, noise, static, "
                "muffled, unclear, unnatural, monotone"
            )

        ctx = self._prompt_encoder(prompts_list, streaming_prefetch_count=None)
        a_ctx = ctx[0].audio_encoding
        a_ctx_neg = ctx[1].audio_encoding if use_cfg else None

        # Denoiser
        resc = _auto_rescale_for_cfg(cfg_scale) if rescale_scale is None else rescale_scale

        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                stg_blocks=[29],
                rescale_scale=resc,
                modality_scale=1.0,
            ),
            negative_context=a_ctx_neg,
        )
        denoiser = GuidedDenoiser(
            v_context=None,
            a_context=a_ctx,
            video_guider=None,
            audio_guider=guider,
        )

        # Sigmas
        sigmas = LTX2Scheduler().execute(steps=30, latent=state.latent).to(self.device)

        # Denoise loop
        print(f"  [Denoise] 30 шагов...")
        x0 = X0Model(self._velocity_model)
        _, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=None,
            audio_state=state,
            stepper=EulerDiffusionStep(),
            transformer=x0,
            denoiser=denoiser,
        )

        # Strip ref tokens + unpatchify
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        # End-of-clip silence-prior fix (граница ~20s, frame 513)
        latent = audio_state.latent
        if latent.shape[2] > 513:
            f0, f1 = 511, 514
            n = f1 - f0
            patched = latent.clone()
            for f in (512, 513):
                t = (f - f0) / n
                patched[:, :, f, :] = (1.0 - t) * latent[:, :, f0, :] + t * latent[:, :, f1, :]
            latent = patched

        # Decode (Audio VAE decoder + Vocoder из audio_components)
        print(f"  [Decode] Декодирование...")
        decoded = self._audio_decoder(latent)
        wav = decoded.waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        return wav.cpu(), decoded.sampling_rate


# =====================================================================
# ComfyUI Node
# =====================================================================
class Dramabox_pyPTV:
    """Генерирует озвучку для batch промптов из prompts.json."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg_scale": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 20.0, "step": 0.1}),
                "stg_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "duration_multiplier": ("FLOAT", {
                    "default": 1.1, "min": 0.5, "max": 3.0, "step": 0.05,
                    "tooltip": "Ignored when gen_duration > 0",
                }),
                "gen_duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5,
                    "tooltip": "0 = auto-estimate from prompt",
                }),
                "ref_duration": ("FLOAT", {
                    "default": 10.0, "min": 3.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Seconds of voice ref to use (3-30)",
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
                "seed_mode": (["fixed", "increment", "random"], {"default": "fixed"}),
                "rescale_scale": ("STRING", {
                    "default": "auto",
                    "tooltip": "auto | 0 to disable | float 0-1",
                }),
                "device": ("STRING", {"default": "cuda"}),
            },
            "optional": {
                "voice_ref": ("AUDIO", {}),
            },
        }

    CATEGORY = "pyPTV"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("processed_count",)
    FUNCTION = "generate_batch"
    OUTPUT_NODE = True

    def generate_batch(
        self,
        cfg_scale: float,
        stg_scale: float,
        duration_multiplier: float,
        gen_duration: float,
        ref_duration: float,
        seed: int,
        seed_mode: str,
        rescale_scale: str,
        device: str,
        voice_ref=None,
    ):
        # --- Жёстко заданные пути (не меняются) ---
        checkpoint = "/comfyui/models/dramabox/dramabox-dit-v1.safetensors"
        audio_components = "/comfyui/models/dramabox/dramabox-audio-components.safetensors"
        silence_latent = "/comfyui/models/dramabox/assets/silence_latent_frame.pt"
        full_checkpoint = "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors"
        gemma_root = "/comfyui/models/text_encoders/gemma-3-12b-it-qat"
        prompts_json = "/tmp/dataset/prompts.json"
        output_folder = "/tmp/dataset"

        # --- Проверить пути ---
        for path, name in [
            (checkpoint, "checkpoint"),
            (audio_components, "audio_components"),
            (silence_latent, "silence_latent"),
            (full_checkpoint, "full_checkpoint"),
            (gemma_root, "gemma_root"),
            (prompts_json, "prompts_json"),
        ]:
            if not os.path.exists(path):
                raise ValueError(f"[Dramabox_pyPTV] {name} не найден: {path}")

        # --- Читаем промпты ---
        with open(prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            raise ValueError("prompts.json должен содержать список строк (JSON array)")
        print(f"[Dramabox_pyPTV] Найдено {len(prompts)} промптов, seed_mode={seed_mode}")

        # --- Выходная папка ---
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        # --- Сохранить voice_ref на диск ---
        ref_path = None
        if voice_ref is not None:
            ref_wav = f"/tmp/dramabox_ref_{uuid.uuid4().hex[:8]}.wav"
            wf = voice_ref["waveform"]
            if wf.dim() == 3:
                wf = wf.squeeze(0)
            torchaudio.save(ref_wav, wf.cpu(), voice_ref["sample_rate"])
            ref_path = ref_wav
            print(f"[Dramabox_pyPTV] Voice ref сохранён: {ref_wav}")

        # --- Парсить rescale_scale ---
        rs_str = rescale_scale.strip().lower()
        rs = None if rs_str == "auto" else 0.0 if rs_str == "0" else float(rs_str)

        # --- Загрузить модели (с кэшем по всем 4 путям) ---
        cache_key = (checkpoint, audio_components, full_checkpoint, gemma_root, device)
        if cache_key not in _loader_cache:
            print(f"[Dramabox_pyPTV] Первый запуск — загружаем модели...")
            _loader_cache[cache_key] = DramaboxTTSLoader(
                checkpoint=checkpoint,
                audio_components=audio_components,
                full_checkpoint=full_checkpoint,
                gemma_root=gemma_root,
                silence_latent_path=silence_latent,
                device=device,
                dtype_str="bf16",
            )
        else:
            print(f"[Dramabox_pyPTV] Используем кэшированные модели")
        loader = _loader_cache[cache_key]

        # --- Batch генерация ---
        pbar = ProgressBar(len(prompts))
        processed = 0

        for idx, prompt in enumerate(prompts):
            out_file = out_path / f"{idx:04d}.wav"

            if out_file.exists():
                print(f"[Dramabox_pyPTV] Пропуск {idx:04d}.wav (уже существует)")
                processed += 1
                pbar.update(1)
                continue

            current_seed = _resolve_seed(seed, idx, seed_mode)
            print(f"[Dramabox_pyPTV] [{idx+1}/{len(prompts)}] seed={current_seed} | {prompt[:60]}...")

            try:
                wav, sr = loader.generate(
                    prompt=prompt,
                    voice_ref_path=ref_path,
                    cfg_scale=cfg_scale,
                    stg_scale=stg_scale,
                    duration_multiplier=duration_multiplier,
                    gen_duration=gen_duration,
                    ref_duration=ref_duration,
                    seed=current_seed,
                    rescale_scale=rs,
                )
                torchaudio.save(str(out_file), wav, sr)
                processed += 1
                print(f"  → сохранено: {out_file.name} ({wav.shape[-1] / sr:.1f}s)")
            except Exception as e:
                print(f"[Dramabox_pyPTV] ОШИБКА [{idx}]: {e}")
                import traceback
                traceback.print_exc()

            pbar.update(1)

        print(f"[Dramabox_pyPTV] Готово: {processed}/{len(prompts)} → {output_folder}")
        return (processed,)


# =====================================================================
# Registration
# =====================================================================
NODE_CLASS_MAPPINGS = {"Dramabox_pyPTV": Dramabox_pyPTV}
NODE_DISPLAY_NAME_MAPPINGS = {"Dramabox_pyPTV": "Dramabox (pyPTV)"}
