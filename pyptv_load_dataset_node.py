"""
Dataset Loader (pyPTV)
Скачивает содержимое подпапки из приватного репозитория HuggingFace Hub
и помещает файлы в /tmp/dataset (предварительно очищая её).

Использует `hf download` CLI (huggingface_hub >= 1.11).
"""

import shutil
import subprocess
from pathlib import Path


class LTX23LoadDataset:
    """Скачивает датасет с HuggingFace Hub в /tmp/dataset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {
                    "default": "avidscreator/datasets",
                    "multiline": False,
                    "tooltip": "HuggingFace dataset repo ID, например: avidscreator/datasets",
                }),
                "subfolder": ("STRING", {
                    "default": "mydataset",
                    "multiline": False,
                    "tooltip": "Подпапка внутри репо — её содержимое скачивается в /tmp/dataset",
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace access token (для приватных репо)",
                }),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("downloaded", "log")
    FUNCTION = "download"
    CATEGORY = "pyPTV"

    def download(
        self,
        repo_id: str,
        subfolder: str,
        hf_token: str,
    ):
        dest = Path("/tmp/dataset")
        tmp_dir = Path("/tmp/hf_repo_download")
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

        # Сортируем файлы по имени и переименовываем в 001.ext, 002.ext ...
        files = sorted(p for p in src_folder.iterdir() if p.is_file())
        for idx, f in enumerate(files, start=1):
            ext = f.suffix.lower()
            dst = dest / f"{idx:03d}{ext}"
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

        return (downloaded, status)


NODE_CLASS_MAPPINGS = {
    "LTX23LoadDataset": LTX23LoadDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23LoadDataset": "Load Dataset from HF (pyPTV)",
}
