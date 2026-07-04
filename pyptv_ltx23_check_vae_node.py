"""
LTX-2.3 Check VAE Quality — автономная нода (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Отдельная нода для визуальной проверки качества VAE-кодирования ДО тренировки.

Автономна: сама скачивает датасет с HF, строит dataset.json, запускает
process_dataset.py с флагом --decode и показывает декодированные PNG в UI.
Пользователь смотрит — если артефакты или размытие, меняет resolution bucket
(через смену исходных картинок) и перезапускает ноду.

Никаких входных коннектов от пайплайна. Выход IMAGE — для превью в ComfyUI.
Использует свою папку /home/dataset_vae_check, чтобы не мешать train-ноде.

Модели НЕ скачиваются нодой — только пути.
"""

import os

import numpy as np
import torch
from PIL import Image

from . import pyptv_ltx23_common as cm

TAG = "CheckVAE"
DATASET_DIR = "/home/dataset_vae_check"
DOWNLOAD_TMP = "/home/hf_repo_download_vae"


def _load_pngs(paths, max_count=64, target=512):
    """Загрузить список PNG в батч-тензор [N, target, target, 3] для превью."""
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
    """Автономная проверка качества VAE: HF download → build → decode → IMAGE preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Датасет (загрузка с HF) ──
                "dataset_repo_id": ("STRING", {
                    "default": "username/datasets",
                    "multiline": False,
                    "tooltip": "HF dataset repo, откуда качать датасет",
                }),
                "dataset_subfolder": ("STRING", {
                    "default": "mydataset",
                    "multiline": False,
                    "tooltip": "Подпапка внутри репо с картинками/аудио",
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace access token",
                }),
                "trigger_word": ("STRING", {
                    "default": "JSRv1rpd",
                    "multiline": False,
                    "tooltip": "Триггер-слово LoRA (для --lora-trigger)",
                }),

                # ── Модели / пути ──
                "model_path": ("STRING", {
                    "default": "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors",
                    "multiline": False,
                }),
                "text_encoder_path": ("STRING", {
                    "default": "/comfyui/models/text_encoders/gemma-3-12b-it-qat",
                    "multiline": False,
                }),
                "ltx_repo_path": ("STRING", {
                    "default": "/home/LTX-2",
                    "multiline": False,
                    "tooltip": "Корень репозитория LTX-2",
                }),
                "log_file": ("STRING", {
                    "default": "/home/ltx_check_vae.log",
                    "multiline": False,
                }),
            }
        }

    # Выход IMAGE — для отображения превью декодированных картинок в ComfyUI.
    # Входных коннектов нет — нода полностью автономна.
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_preview",)
    FUNCTION = "run"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Всегда пересчитывать при запуске — нода автономная, качает свежий датасет.
        return float("nan")

    def run(self, dataset_repo_id, dataset_subfolder, hf_token, trigger_word,
            model_path, text_encoder_path, ltx_repo_path, log_file):

        cm.reset_log(log_file)
        log = lambda m: cm.log(log_file, TAG, m)

        trainer_cwd = os.path.join(ltx_repo_path, "packages", "ltx-trainer")
        if not os.path.isdir(trainer_cwd):
            raise RuntimeError(f"Не найден ltx-trainer: {trainer_cwd}")

        # === ШАГ 1: загрузка датасета с HF ===
        log("ШАГ 1/3: загрузка датасета с HuggingFace")
        cm.hf_download_dataset(
            dataset_repo_id, dataset_subfolder, hf_token,
            DATASET_DIR, DOWNLOAD_TMP, log_file, TAG,
        )

        # === ШАГ 2: построение dataset.json + buckets ===
        log("ШАГ 2/3: построение dataset.json")
        dataset_json, buckets_str, count, _ = cm.build_dataset(
            DATASET_DIR, trigger_word, log_file, TAG,
        )
        log(f"Датасет: {count} пар, buckets={buckets_str}")

        # === ШАГ 3: препроцессинг с --decode ===
        log("ШАГ 3/3: препроцессинг с --decode (process_dataset.py)")
        preprocessed_root = os.path.join(DATASET_DIR, ".precomputed")
        decoded_dir = os.path.join(preprocessed_root, "decoded_videos")
        cmd = [
            "python", "scripts/process_dataset.py",
            dataset_json,
            "--resolution-buckets", buckets_str,
            "--model-path", model_path,
            "--text-encoder-path", text_encoder_path,
            "--lora-trigger", trigger_word,
            "--decode",
            "--overwrite",
        ]
        result = cm.run_logged(
            cmd, trainer_cwd, log_file,
            "stdout/stderr process_dataset.py --decode", TAG,
        )
        if result.returncode != 0:
            log(f"process_dataset.py упал, returncode={result.returncode}")
            raise RuntimeError(
                f"Decode failed: returncode={result.returncode}, см. {log_file}"
            )

        # === Читаем декодированные PNG → IMAGE ===
        if not os.path.isdir(decoded_dir):
            raise RuntimeError(f"decoded_videos не создан: {decoded_dir}")
        pngs = sorted(
            os.path.join(decoded_dir, f)
            for f in os.listdir(decoded_dir)
            if f.lower().endswith(".png")
        )
        if not pngs:
            raise RuntimeError(f"В {decoded_dir} нет .png после декода")
        log(f"Декодировано PNG: {len(pngs)}")

        preview = _load_pngs(pngs)
        log("Готово — проверьте превью. Если есть артефакты, смените картинки/buckets.")
        return (preview,)


NODE_CLASS_MAPPINGS = {
    "PyPTVLtx23CheckVAE": PyPTVLtx23CheckVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVLtx23CheckVAE": "LTX-2.3 Check VAE Quality (pyPTV)",
}
