"""
Upload Checkpoints to HuggingFace (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Загружает .safetensors из папки тренера (output_dir) на HuggingFace Hub.

Как работает:
  1. Сканирует output_dir, ищет все .safetensors файлы.
  2. Добавляет префикс к имени каждого файла (например mylora_ → mylora_001.safetensors).
  3. Копирует во временную папку с новыми именами.
  4. Заливает на HF через hf upload в указанную подпапку.
  5. Удаляет временную папку.

Входы:
  • output_dir   — папка с чекпоинтами (от Train LoRA)
  • repo_id      — репозиторий HF (default: username/loras)
  • subfolder    — подпапка внутри репо (default: test)
  • hf_token     — HuggingFace access token
  • lora_prefix  — префикс имени файлов (default: "", например "mylora_")

Выходы:
  • uploaded_count — сколько файлов залито
  • log            — список залитых файлов (STRING) → цепляй к Log Viewer
"""

import shutil
import subprocess
from pathlib import Path


class PyPTVUploadCheckpoints:
    """Загружает .safetensors чекпоинты на HuggingFace Hub."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {
                    "default": "/home/lora_output",
                    "multiline": False,
                    "tooltip": "Папка с чекпоинтами от тренера",
                }),
                "repo_id": ("STRING", {
                    "default": "username/loras",
                    "multiline": False,
                    "tooltip": "HuggingFace repo ID, например: username/loras",
                }),
                "subfolder": ("STRING", {
                    "default": "test",
                    "multiline": False,
                    "tooltip": "Подпапка внутри репо — куда заливать чекпоинты",
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace access token",
                }),
                "lora_prefix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Префикс для имени файлов, например: mylora_",
                }),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("uploaded_count", "log")
    FUNCTION = "upload"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def upload(
        self,
        output_dir: str,
        repo_id: str,
        subfolder: str,
        hf_token: str,
        lora_prefix: str,
    ):
        src = Path(output_dir)
        tmp_dir = Path("/home/hf_upload")
        token = hf_token.strip()
        prefix = lora_prefix.strip()
        sf = subfolder.strip()

        if not src.exists():
            raise ValueError(f"Папка не найдена: {output_dir}")

        # Собираем .safetensors
        files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix == ".safetensors")
        if not files:
            raise ValueError(f"В папке {output_dir} нет .safetensors файлов")

        # Очищаем временную папку
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Копируем с переименованием
        uploaded_names = []
        for f in files:
            new_name = f"{prefix}{f.name}"
            shutil.copy2(str(f), str(tmp_dir / new_name))
            uploaded_names.append(new_name)

        print(f"[PyPTVUploadCheckpoints] Загрузка {len(uploaded_names)} файлов...")

        # hf upload
        cmd = [
            "hf", "upload", repo_id.strip(),
            str(tmp_dir),
            sf,
            "--include", "*.safetensors",
        ]
        if token:
            cmd += ["--token", token]

        result = subprocess.run(cmd, capture_output=True, text=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if result.returncode != 0:
            err = result.stderr.strip() if result.stderr else "unknown error"
            raise RuntimeError(f"hf upload failed: {err}")

        # Статистика
        lines = [f"Uploaded {len(uploaded_names)} files to {repo_id}/{sf}:"]
        for name in uploaded_names:
            lines.append(f"  {name}")
        log = "\n".join(lines)
        print(f"[PyPTVUploadCheckpoints] {log}")

        return (len(uploaded_names), log)


NODE_CLASS_MAPPINGS = {
    "PyPTVUploadCheckpoints": PyPTVUploadCheckpoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVUploadCheckpoints": "Upload Checkpoints to HF (pyPTV)",
}
