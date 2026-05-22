"""
DramaBox Batch TTS
═══════════════════════════════════════════════════════════════════════════════
Генерирует озвучку для каждого промпта из /tmp/dataset/prompts.json.
Сохраняет 0000.wav, 0001.wav ... в /tmp/dataset/

Как работает:
  1. Читает /tmp/dataset/prompts.json — массив строк (промптов).
  2. Для каждого промпта:
     a. Оценивает длительность из текста (или берёт фиксированную).
     b. Кодирует текст через Gemma + embeddings processor (из PYPTV_MODELS).
     c. Если подан voice_ref — кодирует reference аудио через Audio VAE
        и применяет IC-LoRA conditioning (клонирование голоса).
     d. Запускает denoising loop через DramaBox DiT (30 шагов Euler).
     e. Декодирует латенты → waveform через Audio VAE decoder + Vocoder.
     f. Сохраняет .wav.
  3. Пропускает уже существующие файлы — можно перезапускать safely.

Входы:
  • components          — PYPTV_MODELS из Trainer Components Loader
  • cfg_scale           — насколько строго следовать промпту (2.5)
  • stg_scale           — Skip-Token Guidance (1.5)
  • duration_multiplier — запас к авто-оценке длительности (1.1)
  • gen_duration        — фиксированная длительность в сек (0 = авто)
  • ref_duration        — сколько секунд voice ref использовать (10)
  • seed / seed_mode    — fixed / increment / random
  • rescale_scale       — авто-коррекция латентов при CFG (auto)
  • voice_ref           — опциональное референсное аудио (AUDIO)

Выход:
  • processed_count — сколько wav сгенерировано
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
from safetensors import safe_open

# ltx_core
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.loader import DummyRegistry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import create_caption_projection
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape

# ---------------------------------------------------------------------------
# AudioConditionByReferenceLatent (из DramaBox src/audio_conditioning.py)
# ---------------------------------------------------------------------------
class AudioConditionByReferenceLatent(ConditioningItem):
    """IC-LoRA voice cloning — appends reference audio tokens to target sequence."""

    def __init__(self, latent: torch.Tensor, strength: float = 1.0):
        self.latent = latent
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: AudioLatentTools) -> LatentState:
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
        positions = positions + 0.5  # небольшой сдвиг чтобы избежать t=0 коллизии

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=torch.float32,
        )

        batch_size = tokens.shape[0]
        num_target = latent_state.latent.shape[1]
        num_ref    = tokens.shape[1]
        total      = num_target + num_ref

        mask = torch.zeros((batch_size, total, total), device=self.latent.device, dtype=torch.float32)
        if latent_state.attention_mask is not None:
            mask[:, :num_target, :num_target] = latent_state.attention_mask
        else:
            mask[:, :num_target, :num_target] = 1.0
        mask[:, :num_target, num_target:] = 1.0  # target -> ref
        mask[:, num_target:, num_target:] = 1.0  # ref -> ref

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=mask,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _auto_rescale_for_cfg(cfg: float) -> float:
    if cfg <= 2.0: return 0.0
    if cfg <= 3.0: return 0.6 * (cfg - 2.0)
    if cfg <= 4.0: return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0: return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


def _estimate_duration(prompt: str, multiplier: float = 1.1) -> float:
    quotes = re.findall(r'"([^"]+)"', prompt)
    if not quotes:
        quotes = re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", prompt)
        quotes = [q for q in quotes if len(q.split()) > 3]
    spoken = " ".join(quotes) if quotes else (prompt.split(":", 1)[1].strip() if ":" in prompt else prompt)
    CHARS_PER_SEC = 14.0
    text_len = len(spoken)
    cps = CHARS_PER_SEC * (0.6 if text_len < 40 else 0.8 if text_len < 80 else 1.0)
    duration  = text_len / cps
    duration += (spoken.count(".") + spoken.count("!") + spoken.count("?")) * 0.3
    duration += 0.5 * len(re.findall(r"\blaugh", prompt, re.IGNORECASE))
    duration += 0.5 * len(re.findall(r"\bpause", prompt, re.IGNORECASE))
    duration += 0.3 * len(re.findall(r"\bsigh",  prompt, re.IGNORECASE))
    return max(3.0, round(duration + 2.0, 1)) * multiplier


def _resolve_seed(seed: int, idx: int, mode: str) -> int:
    if mode == "fixed":     return seed
    if mode == "increment": return seed + idx
    return random.randint(0, 0xffffffff)


def _load_waveform_for_ref(path: str, device: torch.device, max_duration: float) -> Optional[Audio]:
    """Загрузить аудио файл для voice reference."""
    try:
        waveform, sr = torchaudio.load(path)       # [C, N]
        max_samples = int(max_duration * sr)
        waveform = waveform[..., :max_samples]
        return Audio(waveform=waveform.to(device), sampling_rate=sr)
    except Exception as e:
        print(f"  [Voice ref] Ошибка загрузки {path}: {e}")
        return None


def _euler_loop(
    sigmas: torch.Tensor,
    state: LatentState,
    x0_model: X0Model,
    denoiser,
    stepper: EulerDiffusionStep,
) -> LatentState:
    """Простой Euler denoising loop без ltx_pipelines."""
    audio_state = state
    for i in range(len(sigmas) - 1):
        sigma     = sigmas[i]
        sigma_next = sigmas[i + 1]
        _, audio_state = stepper.step(
            sigma=sigma,
            sigma_next=sigma_next,
            video_state=None,
            audio_state=audio_state,
            model=x0_model,
            denoiser=denoiser,
        )
    return audio_state


# ---------------------------------------------------------------------------
# Модель-конфигуратор для DramaBox DiT (AudioOnly)
# ---------------------------------------------------------------------------
class _AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
    @classmethod
    def from_config(cls, cfg: dict) -> LTXModel:
        t  = cfg.get("transformer", {})
        cp = None
        if not t.get("caption_proj_before_connector", False):
            with torch.device("meta"):
                cp = create_caption_projection(t, audio=True)
        return LTXModel(
            model_type=LTXModelType.AudioOnly,
            audio_num_attention_heads    = t.get("audio_num_attention_heads",    32),
            audio_attention_head_dim     = t.get("audio_attention_head_dim",     64),
            audio_in_channels            = t.get("audio_in_channels",           128),
            audio_out_channels           = t.get("audio_out_channels",          128),
            num_layers                   = t.get("num_layers",                   48),
            audio_cross_attention_dim    = t.get("audio_cross_attention_dim",  2048),
            norm_eps                     = t.get("norm_eps",                    1e-6),
            attention_type               = AttentionFunction(t.get("attention_type", "default")),
            positional_embedding_theta   = 10000.0,
            audio_positional_embedding_max_pos = [20.0],
            timestep_scale_multiplier    = t.get("timestep_scale_multiplier",  1000),
            use_middle_indices_grid      = t.get("use_middle_indices_grid",     True),
            rope_type                    = LTXRopeType(t.get("rope_type", "interleaved")),
            double_precision_rope        = t.get("frequencies_precision", False) == "float64",
            apply_gated_attention        = t.get("apply_gated_attention",      False),
            audio_caption_projection     = cp,
            cross_attention_adaln        = t.get("cross_attention_adaln",      False),
        )


# ---------------------------------------------------------------------------
# Загрузчик и генератор
# ---------------------------------------------------------------------------
class DramaboxTTSLoader:
    """Принимает готовые компоненты из PyPTVTrainerComponentsLoader."""

    def __init__(self, components: dict, device: str = "cuda"):
        self.device = torch.device(device)
        self.dtype  = torch.bfloat16
        self._patchifier = AudioPatchifier(patch_size=1)

        self._text_encoder         = components["text_encoder"]
        self._embeddings_processor = components["embeddings_processor"]
        self._audio_encoder        = components["audio_vae_encoder"]
        self._audio_decoder        = components["audio_vae_decoder"]
        self._vocoder              = components["vocoder"]
        self._velocity_model       = components["dit_model"]

        print(f"[DramaboxTTSLoader] Компоненты получены — {device}")

    def _encode_prompt(self, prompts: list[str]) -> list:
        """Кодировать промпты через Gemma + embeddings processor."""
        results = []
        with torch.inference_mode():
            for prompt in prompts:
                hs, mask = self._text_encoder.encode(prompt)
                out = self._embeddings_processor.process_hidden_states(hs, mask)
                results.append(out)
        return results

    @torch.inference_mode()
    def generate(
        self,
        prompt:             str,
        voice_ref_path:     Optional[str] = None,
        cfg_scale:          float = 2.5,
        stg_scale:          float = 1.5,
        duration_multiplier: float = 1.1,
        gen_duration:       float = 0.0,
        ref_duration:       float = 10.0,
        seed:               int   = 42,
        rescale_scale:      Optional[float] = None,
    ) -> tuple[torch.Tensor, int]:
        """Вернуть (waveform [C, N], sample_rate)."""

        # Целевая длительность
        gen_dur = float(gen_duration) if gen_duration > 0 else _estimate_duration(prompt, duration_multiplier)

        fps      = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1

        pixel_shape  = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools  = AudioLatentTools(patchifier=self._patchifier, target_shape=target_shape)

        # Initial latent state
        state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)

        # Voice reference IC-LoRA conditioning
        if voice_ref_path and os.path.exists(voice_ref_path):
            voice = _load_waveform_for_ref(voice_ref_path, self.device, ref_duration)
            if voice is not None:
                w = voice.waveform
                # Stereo если нужно
                if w.dim() == 2:
                    w = w.repeat(2, 1) if w.shape[0] == 1 else w
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

                voice_audio = Audio(waveform=w, sampling_rate=voice.sampling_rate)

                # Кодируем через Audio VAE encoder
                ref_latent = vae_encode_audio(voice_audio, self._audio_encoder)
                print(f"  [Voice ref] latent: {ref_latent.shape}")

                cond  = AudioConditionByReferenceLatent(latent=ref_latent.to(self.device, self.dtype))
                state = cond.apply_to(state, audio_tools)

        # Gaussian noise
        gen   = torch.Generator(device=self.device).manual_seed(seed)
        state = GaussianNoiser(generator=gen)(state, noise_scale=1.0)

        # Encode prompts
        neg_prompt = (
            "worst quality, inconsistent, robotic, distorted, noise, "
            "static, muffled, unclear, unnatural, monotone"
        )
        use_cfg = cfg_scale > 1.0
        encoded = self._encode_prompt([prompt] + ([neg_prompt] if use_cfg else []))
        a_ctx     = encoded[0].audio_encoding
        a_ctx_neg = encoded[1].audio_encoding if use_cfg else None

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

        # Простой denoiser — передаём контекст напрямую в x0_model через замыкание
        class _SimpleDenoiser:
            def __init__(self, a_ctx, guider):
                self.a_ctx  = a_ctx
                self.guider = guider
            def __call__(self, x0_fn, video_state, audio_state, sigma):
                return self.guider(
                    x0_fn=x0_fn,
                    video_state=video_state,
                    audio_state=audio_state,
                    sigma=sigma,
                    video_context=None,
                    audio_context=self.a_ctx,
                )

        denoiser = _SimpleDenoiser(a_ctx, guider)

        # Sigmas + denoise loop
        sigmas = LTX2Scheduler().execute(steps=30, latent=state.latent).to(self.device)
        x0     = X0Model(self._velocity_model)
        audio_state = _euler_loop(sigmas, state, x0, denoiser, EulerDiffusionStep())

        # Strip ref tokens + unpatchify
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        # End-of-clip silence-prior fix (frame 512-513 ~ 20s boundary)
        latent = audio_state.latent
        if latent.shape[2] > 513:
            patched = latent.clone()
            for f in (512, 513):
                t = (f - 511) / 3
                patched[:, :, f, :] = (1.0 - t) * latent[:, :, 511, :] + t * latent[:, :, 514, :]
            latent = patched

        # Decode: audio VAE decoder + vocoder
        decoded = vae_decode_audio(latent, self._audio_decoder, self._vocoder)
        wav = decoded.waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        return wav.cpu(), decoded.sampling_rate


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------
class Dramabox_pyPTV:
    """Генерирует batch озвучки из /tmp/dataset/prompts.json → /tmp/dataset/000N.wav"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("PYPTV_MODELS",),
                "dataset": ("PYPTV_DATASET",),
                "cfg_scale": ("FLOAT", {
                    "default": 2.5, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "Classifier-Free Guidance. Выше = точнее следует промпту. Ниже = естественнее звучит.",
                }),
                "stg_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Skip-Token Guidance. Улучшает качество генерации. 0 = отключить.",
                }),
                "duration_multiplier": ("FLOAT", {
                    "default": 1.1, "min": 0.5, "max": 3.0, "step": 0.05,
                    "tooltip": "Множитель авто-оценки длительности. 1.1 = +10% запас. Игнорируется если gen_duration > 0.",
                }),
                "gen_duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Явная длительность генерации в секундах. 0 = автооценка из промпта.",
                }),
                "ref_duration": ("FLOAT", {
                    "default": 10.0, "min": 3.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Сколько секунд voice reference использовать для клонирования тембра (3-30 сек).",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffff,
                    "tooltip": "Сид генерации. Влияет на вариативность результата.",
                }),
                "seed_mode": (["fixed", "increment", "random"], {
                    "default": "fixed",
                    "tooltip": "fixed = один сид для всех. increment = seed+idx для каждого. random = случайный.",
                }),
                "rescale_scale": ("STRING", {
                    "default": "auto",
                    "tooltip": "Rescale латентов при CFG чтобы избежать клиппинга. auto = авто по cfg_scale. 0 = отключить. float 0-1 = явное значение.",
                }),
            },
            "optional": {
                "voice_ref": ("AUDIO", {
                    "tooltip": "Референсное аудио для клонирования голоса (10+ сек). Подключи Load Audio ноду.",
                }),
            },
        }

    CATEGORY     = "pyPTV"
    RETURN_TYPES = ("INT", "PYPTV_DATASET")
    RETURN_NAMES = ("processed_count", "dataset")
    FUNCTION     = "generate_batch"
    OUTPUT_NODE  = True

    def generate_batch(
        self,
        components,
        dataset,
        cfg_scale:           float,
        stg_scale:           float,
        duration_multiplier: float,
        gen_duration:        float,
        ref_duration:        float,
        seed:                int,
        seed_mode:           str,
        rescale_scale:       str,
        voice_ref=None,
    ):
        device = "cuda"
        root = dataset["root"]
        prompts_json = f"{root}/prompts.json"
        output_folder = root

        # --- Проверить пути ---
        if not os.path.exists(prompts_json):
            raise ValueError(f"[Dramabox_pyPTV] Не найден {prompts_json}")

        # --- Читаем промпты ---
        with open(prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            raise ValueError("prompts.json должен быть JSON array строк")
        if not prompts:
            raise ValueError("prompts.json пуст — нет промптов для генерации аудио")
        if not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("prompts.json должен содержать только непустые строки")

        print(f"[Dramabox_pyPTV] {len(prompts)} промптов, seed_mode={seed_mode}")

        # --- Выходная папка ---
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        # --- Voice ref на диск ---
        ref_path = None
        if voice_ref is not None:
            ref_wav = f"/tmp/dramabox_ref_{uuid.uuid4().hex[:8]}.wav"
            wf = voice_ref["waveform"]
            if wf.dim() == 3:
                wf = wf.squeeze(0)
            torchaudio.save(ref_wav, wf.cpu(), voice_ref["sample_rate"])
            ref_path = ref_wav
            print(f"[Dramabox_pyPTV] Voice ref: {ref_wav}")

        # --- rescale_scale ---
        rs_str = rescale_scale.strip().lower()
        rs = None if rs_str == "auto" else float(rs_str)

        # --- Loader из components ---
        loader = DramaboxTTSLoader(components, device=device)

        # --- Batch генерация ---
        pbar      = ProgressBar(len(prompts))
        processed = 0

        for idx, prompt in enumerate(prompts):
            out_file = out_path / f"{idx:04d}.wav"

            if out_file.exists():
                print(f"[Dramabox_pyPTV] Пропуск {idx:04d}.wav")
                processed += 1
                pbar.update(1)
                continue

            current_seed = _resolve_seed(seed, idx, seed_mode)
            print(f"[Dramabox_pyPTV] [{idx+1}/{len(prompts)}] seed={current_seed} | {prompt[:60]}...")

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
            print(f"  → {out_file.name} ({wav.shape[-1] / sr:.1f}s)")
            pbar.update(1)

        if processed != len(prompts):
            raise RuntimeError(
                f"[Dramabox_pyPTV] Генерация не завершена: {processed}/{len(prompts)}. "
                f"Проверьте логи выше — возможна ошибка при генерации wav."
            )

        dataset["has_audio"] = True
        print(f"[Dramabox_pyPTV] Готово: {processed}/{len(prompts)}")
        return (processed, dataset)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS        = {"Dramabox_pyPTV": Dramabox_pyPTV}
NODE_DISPLAY_NAME_MAPPINGS = {"Dramabox_pyPTV": "Dramabox (pyPTV)"}

