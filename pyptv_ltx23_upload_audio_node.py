"""
Upload Audio to HuggingFace Dataset (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Загружает сгенерированные аудиофайлы (.wav) в HuggingFace dataset-репозиторий.

Удобно использовать после Dramabox_pyPTV: его выход audio_dir подаётся сюда,
и WAV (001.wav, 002.wav, ...) заливаются в нужную подпапку датасета, который
потом подхватывается LTX23LoadDataset.

Как работает:
  1. Сканирует audio_dir на .wav (опционально любые аудио-расширения).
  2. Опционально переименовывает в NNN.wav (по порядку) во временной папке.
  3. Заливает в репо через hf upload в указанную подпапку.
  4. Удаляет временную папку.

Входы:
  • audio_dir  — папка с WAV (например от Dramabox_pyPTV)
  • repo_id    — HF dataset repo, например username/datasets
  • subfolder  — подпапка внутри репо, куда заливать
  • hf_token   — HuggingFace access token
  • rename     — переименовать ли в NNN.wav перед заливкой
  • audio_ext  — какие расширения заливать

Выходы:
  • uploaded_count — сколько файлов залито (INT)
  • log            — список залитых файлов (STRING) → цеплять к Log Viewer
"""

import shutil
import subprocess
from pathlib import Path


class PyPTVLtx23UploadAudio:
    """Загружает аудиофайлы (.wav) в HuggingFace dataset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_dir": ("STRING", {
                    "default": "/home/dramabox_out",
                    "multiline": False,
                    "tooltip": "Папка с аудиофайлами (например от Dramabox)",
                }),
                "repo_id": ("STRING", {
                    "default": "username/datasets",
                    "multiline": False,
                    "tooltip": "HuggingFace dataset repo, например: username/datasets",
                }),
                "subfolder": ("STRING", {
                    "default": "mydataset",
                    "multiline": False,
                    "tooltip": "Подпапка внутри репо — куда заливать",
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace access token",
                }),
                "rename": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Переименовать файлы в NNN.wav по порядку перед заливкой",
                }),
                "audio_ext": ([".wav", ".wav+.mp3+.flac"], {
                    "default": ".wav",
                    "tooltip": "Какие расширения заливать",
                }),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("uploaded_count", "log")
    FUNCTION = "upload"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def upload(self, audio_dir, repo_id, subfolder, hf_token, rename, audio_ext):
        src = Path(audio_dir)
        token = hf_token.strip()
        sf = subfolder.strip()

        if not src.is_dir():
            raise ValueError(f"Папка не найдена: {audio_dir}")

        # Набор расширений
        exts = {".wav"}
        if audio_ext == ".wav+.mp3+.flac":
            exts = {".wav", ".mp3", ".flac"}

        files = sorted(
            p for p in src.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
        if not files:
            raise ValueError(f"В {audio_dir} нет файлов ({'/'.join(sorted(exts))})")

        # Временная папка для заливки
        tmp_dir = Path("/home/hf_audio_upload")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        uploaded = []
        if rename:
            for idx, f in enumerate(files, start=1):
                dst = tmp_dir / f"{idx:03d}{f.suffix.lower()}"
                shutil.copy2(str(f), str(dst))
                uploaded.append(dst.name)
        else:
            for f in files:
                dst = tmp_dir / f.name
                shutil.copy2(str(f), str(dst))
                uploaded.append(f.name)

        # hf upload: локальная папка → подпапка репо
        cmd = [
            "hf", "upload", repo_id.strip(),
            str(tmp_dir),
        ]
        if sf:
            cmd += [sf]
        cmd += ["--include", "*.wav", "*.mp3", "*.flac"]
        if token:
            cmd += ["--token", token]

        print(f"[UploadAudio] Заливка {len(uploaded)} файлов в {repo_id}/{sf}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if result.returncode != 0:
            err = (result.stderr or "unknown error").strip()
            raise RuntimeError(f"hf upload failed: {err}")

        lines = [f"Uploaded {len(uploaded)} files to {repo_id}/{sf}:"]
        for name in uploaded:
            lines.append(f"  {name}")
        log = "\n".join(lines)
        print(f"[UploadAudio] {log}")

        return (len(uploaded), log)


NODE_CLASS_MAPPINGS = {
    "PyPTVLtx23UploadAudio": PyPTVLtx23UploadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVLtx23UploadAudio": "Upload Audio to HF (pyPTV)",
}
