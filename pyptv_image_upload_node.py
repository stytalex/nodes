"""
Image Batch Upload (pyPTV)
Принимает картинки загруженные через браузер (кастомный JS виджет),
сохраняет в папку датасета с нумерацией 0000.jpg, 0001.jpg ...

Использует стандартный ComfyUI upload endpoint (/upload/image)
для получения файлов, затем копирует их в output_folder.
"""

import os
import shutil
from pathlib import Path

import folder_paths
from server import PromptServer
from aiohttp import web


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# REST endpoint — вызывается из JS после загрузки файлов
# ---------------------------------------------------------------------------

@PromptServer.instance.routes.post("/pyptv/image_batch_finalize")
async def image_batch_finalize(request):
    """
    JS сначала загружает файлы через стандартный /upload/image,
    затем вызывает этот endpoint с именами загруженных файлов
    и путём output_folder.

    Body JSON:
    {
        "filenames": ["file1.jpg", "file2.png", ...],
        "output_folder": "/tmp/dataset"
    }

    Копирует файлы из ComfyUI input/ в output_folder
    с нумерацией 0000, 0001 ...
    """
    data = await request.json()
    filenames = data.get("filenames", [])
    output_folder = data.get("output_folder", "/tmp/dataset")

    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    input_dir = Path(folder_paths.get_input_directory())

    copied = 0
    errors = []

    # Сортируем по имени для консистентной нумерации
    filenames_sorted = sorted(filenames)

    for idx, filename in enumerate(filenames_sorted):
        src = input_dir / filename
        if not src.exists():
            errors.append(f"Файл не найден: {filename}")
            continue

        ext = src.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            errors.append(f"Неподдерживаемый формат: {filename}")
            continue

        dst = out_path / f"{idx:04d}{ext}"
        shutil.copy2(str(src), str(dst))
        copied += 1

    return web.json_response({
        "copied": copied,
        "total": len(filenames),
        "output_folder": str(out_path),
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# Нода ComfyUI
# ---------------------------------------------------------------------------

class LTX23ImageBatchUpload:
    """
    Загрузка картинок с ПК в папку датасета.

    Кнопка "Upload Images" в ноде открывает системный
    файловый диалог — выбираешь несколько файлов сразу.
    Прогресс отображается прямо в ноде.
    После загрузки файлы сохраняются в output_folder
    с нумерацией 0000.jpg, 0001.jpg ...

    Выход:
        file_count    — сколько файлов загружено
        output_folder — папка с файлами
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {
                    "default": "/tmp/dataset",
                    "multiline": False,
                    "tooltip": "Папка датасета — картинки сохраняются как 0000.jpg, 0001.jpg ...",
                }),
            },
            "hidden": {
                # Список загруженных файлов передаётся из JS виджета
                "uploaded_files": ("STRING", {"default": "[]"}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("file_count", "output_folder")
    FUNCTION = "process"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def process(self, output_folder: str, uploaded_files: str = "[]"):
        import json

        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            filenames = json.loads(uploaded_files)
        except Exception:
            filenames = []

        input_dir = Path(folder_paths.get_input_directory())
        copied = 0

        filenames_sorted = sorted(filenames)

        for idx, filename in enumerate(filenames_sorted):
            src = input_dir / filename
            if not src.exists():
                print(f"[LTX23ImageBatchUpload] Файл не найден: {filename}")
                continue

            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                print(f"[LTX23ImageBatchUpload] Пропуск (формат): {filename}")
                continue

            dst = out_path / f"{idx:04d}{ext}"
            shutil.copy2(str(src), str(dst))
            copied += 1
            print(f"[LTX23ImageBatchUpload] {filename} → {dst.name}")

        print(f"[LTX23ImageBatchUpload] Готово: {copied} файлов → {output_folder}")
        return (copied, str(out_path))


# ---------------------------------------------------------------------------
# Регистрация
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LTX23ImageBatchUpload": LTX23ImageBatchUpload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23ImageBatchUpload": "Image Batch Upload (pyPTV)",
}
