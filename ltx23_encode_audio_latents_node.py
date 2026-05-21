"""
Аудио файлы → Audio Latents (.pt)
═══════════════════════════════════════════════════════════════════════════════
Берёт все аудио файлы из /tmp/dataset и прогоняет их через Audio VAE Encoder.
Результат — аудио-латенты для тренера, сохраняются в
/tmp/dataset/.precomputed/audio_latents/

Как работает:
  1. Собирает все аудио из /tmp/dataset (wav, mp3, flac, ogg, m4a, aac).
  2. Сортирует по имени — порядок должен совпадать с картинками.
  3. Загружает waveform [1, 1, N], микширует в моно если стерео.
  4. Через audio_vae_encoder (из PYPTV_MODELS) кодируется в latent.
     VAE сам делает ресемплинг → mel-спектр → autoencoder → normalize.
  5. Сохраняет как 0000.pt, 0001.pt, ...

Формат .pt:
  {
      "latents": Tensor [8, T, 16],
  }

Входы:
  • components — PYPTV_MODELS из Trainer Components Loader
  • dtype      — bfloat16 (быстрее) или float32 (точнее)

Выход:
  • processed_count — сколько аудио закодировано
"""

from pathlib import Path

import torch
import torchaudio


SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _collect_audio_files(folder: str) -> list[Path]:
    """Собрать все аудио файлы из папки, отсортировать по имени."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )


def _load_waveform(path: Path) -> tuple[torch.Tensor, int]:
    """
    Загрузить аудио файл.
    Возвращает (waveform [1, N], sample_rate) — моно, оригинальный sr.
    AudioVAE.encode() сам делает ресемплинг внутри через AudioPreprocessor.
    """
    waveform, sr = torchaudio.load(str(path))  # [C, N]

    # Микшируем в моно если стерео
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, N]

    # Добавляем batch dimension: [1, N] → [1, 1, N]
    waveform = waveform.unsqueeze(0)

    return waveform, sr


# ---------------------------------------------------------------------------
# Нода ComfyUI
# ---------------------------------------------------------------------------

class LTX23EncodeAudioLatents:
    """
    Batch-кодирование аудио файлов в аудио-латенты LTX-2.

    Использует Audio VAE Encoder из ltx_trainer.
    AudioVAE.encode() делает всё сам: ресемплинг → mel-спектрограмма →
    кодирование → нормализация латентов.

    Нумерация .pt файлов (0000, 0001, ...) совпадает с нодой картинок —
    порядок файлов в папке должен соответствовать порядку картинок.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("PYPTV_MODELS",),
                "dataset": ("PYPTV_DATASET",),
                "dtype": (["bfloat16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Точность вычислений при кодировании. bfloat16 — быстрее, меньше VRAM. float32 — выше точность но медленнее и требует больше памяти.",
                }),
            }
        }

    RETURN_TYPES = ("INT", "PYPTV_DATASET")
    RETURN_NAMES = ("processed_count", "dataset")
    FUNCTION = "encode"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def encode(
        self,
        components,
        dataset,
        dtype: str,
    ):
        audio_vae = components.get("audio_vae_encoder")
        if audio_vae is None:
            raise RuntimeError("Audio VAE encoder не загружен. Подключите Trainer Components Loader.")
        device = "cuda"
        root = dataset["root"]
        audio_folder = root
        output_folder = f"{root}/.precomputed/audio_latents"
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        audio_files = _collect_audio_files(audio_folder)
        if not audio_files:
            raise ValueError(f"Аудио файлы не найдены в папке: {audio_folder}")

        print(f"[LTX23EncodeAudioLatents] Найдено {len(audio_files)} аудио файлов")

        audio_vae = audio_vae.to(device)
        audio_vae.eval()

        sample_rate = getattr(audio_vae, "sample_rate", 44100)
        print(f"[LTX23EncodeAudioLatents] AudioVAE sample_rate={sample_rate}")

        processed = 0
        for idx, audio_path in enumerate(audio_files):
            out_file = out_path / f"{idx:04d}.pt"

            if out_file.exists():
                print(f"[LTX23EncodeAudioLatents] Пропуск {audio_path.name} (уже существует)")
                processed += 1
                continue

            print(f"[LTX23EncodeAudioLatents] [{idx+1}/{len(audio_files)}] {audio_path.name}")

            try:
                # Загружаем waveform [1, 1, N] и оригинальный sr
                waveform, sr = _load_waveform(audio_path)
                waveform = waveform.to(device=device, dtype=torch_dtype)

                with torch.no_grad():
                    # AudioVAE.encode() делает всё сам:
                    # ресемплинг → mel → autoencoder → normalize
                    # принимает waveform [B, C, N] и sample_rate
                    latent = audio_vae.encode(waveform, sample_rate=sr)  # [B, 8, T, 16]

                latent = latent.squeeze(0).cpu()   # [8, T, 16]

                latent_data = {"latents": latent}
                torch.save(latent_data, out_file)
                processed += 1

                duration = waveform.shape[-1] / sr
                print(f"  → latent shape: {latent.shape}, duration: {duration:.2f}s")

            except Exception as e:
                print(f"[LTX23EncodeAudioLatents] ОШИБКА {audio_path.name}: {e}")

        print(f"[LTX23EncodeAudioLatents] Готово: {processed}/{len(audio_files)} → {output_folder}")
        return (processed, dataset)


# ---------------------------------------------------------------------------
# Регистрация
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LTX23EncodeAudioLatents": LTX23EncodeAudioLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23EncodeAudioLatents": "LTX-2.3 Encode Audio Latents (pyPTV)",
}
