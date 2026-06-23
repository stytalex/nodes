"""
LTX-2.3 Check VAE Quality (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Читает декодированные .png из .precomputed/decoded_videos/, отдаёт батч
картинок в ComfyUI для визуальной проверки качества VAE до тренировки.

Если видны артефакты — поменяйте resolution buckets и перезапустите
PyPTVLtx23Preprocess.

Входы:
  • decoded_preview_dir — путь от PyPTVLtx23Preprocess

Выходы:
  • images_preview         — IMAGE тензор для UI
  • preprocessed_data_root — пробрасывает дальше в Trainer
"""

import os
import numpy as np
import torch
from PIL import Image


def _load_pngs(paths, max_count=64, target=512):
    if not paths:
        return torch.zeros(1, target, target, 3, dtype=torch.float32)
    imgs = []
    for p in paths[:max_count]:
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            scale = target / max(w, h)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGB", (target, target), (0, 0, 0))
            canvas.paste(img, ((target - nw) // 2, (target - nh) // 2))
            imgs.append(np.array(canvas).astype(np.float32) / 255.0)
        except Exception as e:
            print(f"[CheckVAE] fail {p}: {e}")
    if not imgs:
        return torch.zeros(1, target, target, 3, dtype=torch.float32)
    return torch.from_numpy(np.stack(imgs, axis=0))


class PyPTVLtx23CheckVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoded_preview_dir": ("STRING", {
                    "default": "/home/dataset/.precomputed/decoded_videos",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_preview", "preprocessed_data_root")
    FUNCTION = "run"
    CATEGORY = "pyPTV"

    @classmethod
    def IS_CHANGED(cls, decoded_preview_dir, **kwargs):
        try:
            return os.path.getmtime(decoded_preview_dir)
        except Exception:
            return 0

    def run(self, decoded_preview_dir):
        if not os.path.isdir(decoded_preview_dir):
            raise RuntimeError(f"Нет папки декода: {decoded_preview_dir}")

        pngs = sorted(
            os.path.join(decoded_preview_dir, f)
            for f in os.listdir(decoded_preview_dir)
            if f.lower().endswith(".png")
        )
        if not pngs:
            raise RuntimeError(f"В {decoded_preview_dir} нет .png")

        preview = _load_pngs(pngs)
        preprocessed_root = os.path.dirname(decoded_preview_dir.rstrip("/"))

        return (preview, preprocessed_root)


NODE_CLASS_MAPPINGS = {
    "PyPTVLtx23CheckVAE": PyPTVLtx23CheckVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVLtx23CheckVAE": "LTX-2.3 Check VAE Quality (pyPTV)",
}
