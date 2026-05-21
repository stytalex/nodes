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
                    "default": "mylora",
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
    RETURN_NAMES = ("downloaded", "status")
    FUNCTION = "download"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

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
        dest.mkdir(parents=True, exist_ok=True)

        # --- Очистка временной папки ---
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
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
            return (0, f"ERROR: {err}")

        # --- Переносим содержимое подпапки в dest ---
        src_folder = tmp_dir / sf

        if src_folder.exists():
            for f in src_folder.iterdir():
                dst = dest / f.name
                if dst.exists():
                    shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
                shutil.move(str(f), str(dst))
        else:
            print(f"[LTX23LoadDataset] Предупреждение: подпапка {sf} не найдена, переношу всё")
            for f in tmp_dir.iterdir():
                if f.name in (".gitattributes", ".gitignore"):
                    continue
                dst = dest / f.name
                if dst.exists():
                    shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
                shutil.move(str(f), str(dst))

        # --- Удаляем временную папку ---
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # --- Подсчёт скачанных файлов ---
        downloaded = sum(1 for p in dest.rglob("*") if p.is_file())
        status = f"OK: {downloaded} files"
        print(f"[LTX23LoadDataset] {status} → {dest}")

        return (downloaded, status)


NODE_CLASS_MAPPINGS = {
    "LTX23LoadDataset": LTX23LoadDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23LoadDataset": "Load Dataset from HF (pyPTV)",
}
