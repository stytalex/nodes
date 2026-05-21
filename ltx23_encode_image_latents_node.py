"""
Картинки → Video Latents (.pt)
═══════════════════════════════════════════════════════════════════════════════
Берёт все изображения из /tmp/dataset и прогоняет их через Video VAE Encoder.
Результат — латенты для тренера, сохраняются в /tmp/dataset/.precomputed/latents/

Как работает:
  1. Собирает все картинки из /tmp/dataset (jpg, png, webp, bmp).
  2. Сортирует по имени — порядок важен, должен совпадать с аудио.
  3. Каждая картинка ресайзится до оригинального размера, округлённого до ×32.
  4. Через video_vae_encoder (из PYPTV_MODELS) кодируется в latent tensor.
  5. Сохраняет как 0000.pt, 0001.pt, ...

Формат .pt:
  {
      "latents":    Tensor [128, 1, H//32, W//32],
      "num_frames": 1,
      "height":     H // 32,
      "width":      W // 32,
      "fps":        25.0,
  }

Входы:
  • components — PYPTV_MODELS из Trainer Components Loader
  • dtype      — bfloat16 (быстрее) или float32 (точнее)

Выход:
  • processed_count — сколько картинок закодировано
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_images(folder: str) -> list[Path]:
    """Собрать все изображения из папки, отсортировать по имени."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )


def _load_image_tensor(path: Path) -> torch.Tensor:
    """
    Загрузить изображение, ресайзнуть до ближайшего кратного 32
    (берём оригинальный размер, округляем вниз до ×32),
    вернуть tensor [1, 3, H, W] в диапазоне [-1, 1].
    """
    img = Image.open(path).convert("RGB")

    w = (img.width  // 32) * 32
    h = (img.height // 32) * 32
    w = max(w, 32)
    h = max(h, 32)

    img = img.resize((w, h), Image.LANCZOS)

    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)        # [3, H, W]  в [0, 1]
    tensor = tensor * 2.0 - 1.0   # [-1, 1]
    tensor = tensor.unsqueeze(0)   # [1, 3, H, W]
    return tensor


def _encode_image(image_tensor: torch.Tensor, vae_encoder, device: str, dtype: torch.dtype) -> dict:
    """
    Прогнать один кадр через LTX Video VAE Encoder.

    image_tensor: [1, 3, H, W] в [-1, 1]
    VAE ожидает [B, 3, F, H, W] → добавляем временное измерение F=1.
    """
    # [1, 3, H, W] → [1, 3, 1, H, W]
    video_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=dtype)

    with torch.no_grad():
        latent = vae_encoder.encode(video_tensor)   # [1, 128, 1, H//32, W//32]

    latent = latent.squeeze(0).cpu()         # [128, 1, H//32, W//32]

    _, f, h, w = latent.shape

    return {
        "latents":    latent,
        "num_frames": f,
        "height":     h,
        "width":      w,
        "fps":        25.0,
    }


class LTX23EncodeImageLatents:
    """
    Batch-кодирование изображений в видео-латенты LTX-2.

    Входы:
        video_vae_encoder — из Trainer Components Loader (PYPTV_MODELS)
        device — cuda / cpu
        dtype  — bfloat16 / float32

    Выход:
        processed_count — сколько файлов обработано
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("PYPTV_MODELS",),
                "dataset": ("PYPTV_DATASET",),
                "dtype": (["bfloat16", "float32"], {"default": "bfloat16"}),
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
        vae_encoder = components.get("video_vae_encoder")
        if vae_encoder is None:
            raise RuntimeError("Video VAE encoder не загружен. Подключите Trainer Components Loader.")
        device = "cuda"
        root = dataset["root"]
        images_folder = root
        output_folder = f"{root}/.precomputed/latents"
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        images = _collect_images(images_folder)
        if not images:
            raise ValueError(f"Изображения не найдены в папке: {images_folder}")

        print(f"[LTX23EncodeImageLatents] Найдено {len(images)} изображений")

        vae_encoder = vae_encoder.to(device)
        vae_encoder.eval()

        processed = 0
        for idx, img_path in enumerate(images):
            out_file = out_path / f"{idx:04d}.pt"

            if out_file.exists():
                print(f"[LTX23EncodeImageLatents] Пропуск {img_path.name} (уже существует)")
                processed += 1
                continue

            print(f"[LTX23EncodeImageLatents] [{idx+1}/{len(images)}] {img_path.name}")

            try:
                image_tensor = _load_image_tensor(img_path)
                latent_data  = _encode_image(image_tensor, vae_encoder, device, torch_dtype)
                torch.save(latent_data, out_file)
                processed += 1
            except Exception as e:
                print(f"[LTX23EncodeImageLatents] ОШИБКА {img_path.name}: {e}")

        print(f"[LTX23EncodeImageLatents] Готово: {processed}/{len(images)} → {output_folder}")
        return (processed, dataset)


NODE_CLASS_MAPPINGS = {
    "LTX23EncodeImageLatents": LTX23EncodeImageLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23EncodeImageLatents": "LTX-2.3 Encode Image Latents (pyPTV)",
}
