"""
Dataset Loader from HuggingFace
═══════════════════════════════════════════════════════════════════════════════
Скачивает датасет с HuggingFace Hub в /home/dataset.

Как работает:
  1. Полностью очищает /home/dataset (включая проверку что всё удалено).
  2. Запускает hf download repo_id --include "subfolder/*" в /home/hf_repo_download.
  3. Проверяет что подпапка существует и не пуста — иначе ошибка.
  4. Переименовывает файлы в 001.ext, 002.ext ... (по алфавиту).
  5. Перемещает их в /home/dataset.
  6. Удаляет временную папку.
  7. Возвращает статистику: количество файлов + размер.

Авторизация:
  • hf_token — токен для приватных репо. Вводится руками в ноду.

Входы:
  • repo_id   — репозиторий HF (default: username/datasets)
  • subfolder — подпапка внутри репо (default: mydataset)
  • hf_token  — HuggingFace access token

Выходы:
  • downloaded — сколько файлов скачано (INT)
  • log        — статистика по файлам (STRING) — цепляй к Log Viewer
"""

import shutil
import subprocess
from pathlib import Path


class LTX23LoadDataset:
    """Скачивает датасет с HuggingFace Hub в /home/dataset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {
                    "default": "username/datasets",
                    "multiline": False,
                    "tooltip": "HuggingFace dataset repo ID, например: username/datasets",
                }),
                "subfolder": ("STRING", {
                    "default": "mydataset",
                    "multiline": False,
                    "tooltip": "Подпапка внутри репо — её содержимое скачивается в /home/dataset",
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace access token (для приватных репо)",
                }),
            }
        }

    RETURN_TYPES = ("PYPTV_DATASET", "STRING")
    RETURN_NAMES = ("dataset", "log")
    FUNCTION = "download"
    CATEGORY = "pyPTV"

    def download(
        self,
        repo_id: str,
        subfolder: str,
        hf_token: str,
    ):
        dest = Path("/home/dataset")
        tmp_dir = Path("/home/hf_repo_download")
        token = hf_token.strip()
        sf = subfolder.strip()

        # --- Очистка старого датасета ---
        if dest.exists():
            print(f"[LTX23LoadDataset] Очистка {dest} ...")
            shutil.rmtree(dest)
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(f"Не удалось очистить {dest}")
        dest.mkdir(parents=True, exist_ok=True)

        # --- Очистка временной папки ---
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            if tmp_dir.exists() and any(tmp_dir.iterdir()):
                raise RuntimeError(f"Не удалось очистить {tmp_dir}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # --- Формируем команду hf download ---
        cmd = [
            "hf", "download", repo_id.strip(),
            "--repo-type", "dataset",
            "--include", f"{sf}/*",
            "--local-dir", str(tmp_dir),
        ]
        if token:
            cmd += ["--token", token]

        print(f"[LTX23LoadDataset] Скачивание {repo_id}/{sf} ...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            err = result.stderr.strip() if result.stderr else "unknown error"
            print(f"[LTX23LoadDataset] ОШИБКА:\n{err}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"hf download failed: {err}")

        # --- Переносим содержимое подпапки в dest ---
        src_folder = tmp_dir / sf

        if not src_folder.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ValueError(f"Подпапка '{sf}' не найдена в репозитории {repo_id}")

        if not any(src_folder.iterdir()):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ValueError(f"Подпапка '{sf}' пуста в репозитории {repo_id}")

        # Переносим картинки, аудио, подписи и служебные файлы.
        #
        # Два класса файлов:
        #   • нумеруемые медиа — картинки персонажа и аудио-реплики.
        #     Нумеруются раздельно но единообразно: 001.jpg/001.wav, 002.jpg/002.wav, ...
        #     Порядок сортировки внутри каждого типа должен совпадать (стандартное соглашение).
        #   • опорные/служебные — сохраняются по исходному имени без нумерации.
        #     Это reference.wav (voice ref для Dramabox), prompts.json (список промптов
        #     для Dramabox), silence_latent_frame.pt (ассет Dramabox), общий caption.txt.
        #     Builder позже удалит служебные файлы, не нужные тренеру.
        IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        KEEP_EXTS  = {".txt", ".json", ".pt"}
        # Имена файлов (без расширения), которые НЕ нумеруются — опорные/служебные.
        PROTECTED_STEMS = {"reference", "prompts", "caption", "silence_latent_frame"}

        files = sorted(p for p in src_folder.iterdir() if p.is_file())

        img_files   = [f for f in files if f.suffix.lower() in IMG_EXTS]
        audio_files = [f for f in files if f.suffix.lower() in AUDIO_EXTS]
        keep_files  = [f for f in files if f.suffix.lower() in KEEP_EXTS]

        def _move_kept(f):
            """Сохранить служебный файл по исходному имени (не нумеровать)."""
            dst = dest / f.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(f), str(dst))

        for f in keep_files:
            _move_kept(f)

        # Из медиа убираем опорные (reference.*) — их тоже переносим по имени.
        for f in list(img_files) + list(audio_files):
            if f.stem.lower() in PROTECTED_STEMS:
                _move_kept(f)
        img_files   = [f for f in img_files if f.stem.lower() not in PROTECTED_STEMS]
        audio_files = [f for f in audio_files if f.stem.lower() not in PROTECTED_STEMS]

        for idx, f in enumerate(img_files, start=1):
            dst = dest / f"{idx:03d}{f.suffix.lower()}"
            if dst.exists():
                dst.unlink()
            shutil.move(str(f), str(dst))

        for idx, f in enumerate(audio_files, start=1):
            dst = dest / f"{idx:03d}{f.suffix.lower()}"
            if dst.exists():
                dst.unlink()
            shutil.move(str(f), str(dst))

        # --- Удаляем временную папку ---
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # --- Статистика ---
        file_list = sorted(
            p for p in dest.rglob("*")
            if p.is_file() and p.name not in (".gitattributes", ".gitignore")
        )
        downloaded = len(file_list)

        if downloaded == 0:
            raise RuntimeError(f"Файлы не скачались — папка датасета пуста")

        total_bytes = sum(f.stat().st_size for f in file_list)
        lines = [f"{downloaded} files, {total_bytes / 1024 / 1024:.2f} MB"]
        for f in file_list:
            size = f.stat().st_size
            size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
            lines.append(f"{f.name}  ({size_str})")

        status = "\n".join(lines)
        print(f"[LTX23LoadDataset] {downloaded} files → {dest}")

        dataset = {"root": str(dest)}
        return (dataset, status)


NODE_CLASS_MAPPINGS = {
    "LTX23LoadDataset": LTX23LoadDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23LoadDataset": "Load Dataset from HF (pyPTV)",
}
