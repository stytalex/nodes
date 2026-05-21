"""
Нода 1: Картинки → Video Latents (.pt)
Прогоняет папку с изображениями через LTX-2 Video VAE Encoder
и сохраняет латенты в формате, который ожидает тренер.

Формат выходного .pt:
{
    "latents": Tensor [128, 1, H//32, W//32],
    "num_frames": 1,
    "height": H // 32,
    "width":  W // 32,
    "fps": 25.0,
}
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


def _load_image_tensor(path: Path, target_width: int, target_height: int) -> torch.Tensor:
    """
    Загрузить изображение, ресайзнуть до target_width x target_height,
    вернуть tensor [1, 3, H, W] в диапазоне [-1, 1].
    """
    img = Image.open(path).convert("RGB")

    w = (target_width  // 32) * 32
    h = (target_height // 32) * 32

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
        vae            — стандартный ComfyUI VAE (LTX-2 чекпоинт)
        images_folder  — папка с изображениями
        output_folder  — куда сохранять .pt файлы (папка latents/)
        width          — целевая ширина (кратная 32)
        height         — целевая высота (кратная 32)
        device         — cuda / cpu
        dtype          — bfloat16 / float32

    Выход:
        processed_count — сколько файлов обработано
        output_folder   — путь к папке с .pt файлами
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "images_folder": ("STRING", {
                    "default": "/tmp/dataset",
                    "multiline": False,
                }),
                "output_folder": ("STRING", {
                    "default": "/tmp/dataset/.precomputed/latents",
                    "multiline": False,
                }),
                "width": ("INT", {
                    "default": 768,
                    "min": 32,
                    "max": 2048,
                    "step": 32,
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 32,
                    "max": 2048,
                    "step": 32,
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bfloat16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("processed_count", "output_folder")
    FUNCTION = "encode"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def encode(
        self,
        vae,
        images_folder: str,
        output_folder: str,
        width: int,
        height: int,
        device: str,
        dtype: str,
    ):
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        images = _collect_images(images_folder)
        if not images:
            raise ValueError(f"Изображения не найдены в папке: {images_folder}")

        print(f"[LTX23EncodeImageLatents] Найдено {len(images)} изображений")

        # Достаём VideoEncoder из ComfyUI VAE объекта
        vae_encoder = vae.first_stage_model
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
                image_tensor = _load_image_tensor(img_path, width, height)
                latent_data  = _encode_image(image_tensor, vae_encoder, device, torch_dtype)
                torch.save(latent_data, out_file)
                processed += 1
            except Exception as e:
                print(f"[LTX23EncodeImageLatents] ОШИБКА {img_path.name}: {e}")

        print(f"[LTX23EncodeImageLatents] Готово: {processed}/{len(images)} → {output_folder}")
        return (processed, str(out_path))


NODE_CLASS_MAPPINGS = {
    "LTX23EncodeImageLatents": LTX23EncodeImageLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23EncodeImageLatents": "LTX-2.3 Encode Image Latents (pyPTV)",
}
