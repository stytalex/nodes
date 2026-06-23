"""
LTX-2.3 Preprocess Dataset (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Запускает scripts/process_dataset.py из ltx-trainer с флагом --decode.
Создаёт .precomputed/{latents,audio_latents,conditions,decoded_videos}.

Subprocess блокирующий — следующая нода (Check VAE) должна получить готовые
декодированные превью.

Входы:
  • dataset_json       — от PyPTVLtx23DatasetBuilder
  • resolution_buckets — от PyPTVLtx23DatasetBuilder
  • model_path         — путь к ltx-2.3 .safetensors
  • text_encoder_path  — путь к Gemma
  • trigger_word       — для --lora-trigger
  • ltx_repo_path      — корень LTX-2 репо
  • log_file           — путь к лог-файлу

Выходы:
  • preprocessed_data_root — путь к .precomputed
  • decoded_preview_dir    — путь к .precomputed/decoded_videos
  • log_file               — путь к лог-файлу
  • log                    — содержимое лога
"""

import os
import datetime
import subprocess


TAG = "Preprocess"


def _log(log_path: str, msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{TAG}] {msg}\n"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")


def _reset_log(log_path: str):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    open(log_path, "w").close()


class PyPTVLtx23Preprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_json": ("STRING", {
                    "default": "/home/dataset/dataset.json",
                    "multiline": False,
                }),
                "resolution_buckets": ("STRING", {
                    "default": "960x544x1;768x512x1",
                    "multiline": False,
                }),
                "model_path": ("STRING", {
                    "default": "/models/ltx-2.3-22b-dev.safetensors",
                    "multiline": False,
                }),
                "text_encoder_path": ("STRING", {
                    "default": "/models/gemma-3-12b-it-qat-q4_0-unquantized",
                    "multiline": False,
                }),
                "trigger_word": ("STRING", {
                    "default": "JSRv1rpd",
                    "multiline": False,
                }),
                "ltx_repo_path": ("STRING", {
                    "default": "/home/LTX-2",
                    "multiline": False,
                }),
                "log_file": ("STRING", {
                    "default": "/home/ltx_preprocess.log",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("preprocessed_data_root", "decoded_preview_dir", "log_file", "log")
    FUNCTION = "run"
    CATEGORY = "pyPTV"

    def run(self, dataset_json, resolution_buckets, model_path, text_encoder_path,
            trigger_word, ltx_repo_path, log_file):
        _reset_log(log_file)

        trainer_cwd = os.path.join(ltx_repo_path, "packages", "ltx-trainer")
        if not os.path.isdir(trainer_cwd):
            raise RuntimeError(f"Не найден ltx-trainer: {trainer_cwd}")
        if not os.path.isfile(dataset_json):
            raise RuntimeError(f"dataset.json не найден: {dataset_json}")

        dataset_dir = os.path.dirname(dataset_json)
        preprocessed_root = os.path.join(dataset_dir, ".precomputed")
        decoded_dir = os.path.join(preprocessed_root, "decoded_videos")

        cmd = [
            "python", "scripts/process_dataset.py",
            dataset_json,
            "--resolution-buckets", resolution_buckets,
            "--model-path", model_path,
            "--text-encoder-path", text_encoder_path,
            "--lora-trigger", trigger_word,
            "--decode",
        ]

        _log(log_file, f"cwd: {trainer_cwd}")
        _log(log_file, f"cmd: {' '.join(cmd)}")
        _log(log_file, "Запуск process_dataset.py...")

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write("\n--- stdout/stderr process_dataset.py ---\n")
            lf.flush()
            result = subprocess.run(
                cmd,
                cwd=trainer_cwd,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )

        if result.returncode != 0:
            _log(log_file, f"process_dataset.py упал, returncode={result.returncode}")
            raise RuntimeError(f"Preprocess failed: returncode={result.returncode}, см. {log_file}")

        latents_dir = os.path.join(preprocessed_root, "latents")
        n_latents = 0
        if os.path.isdir(latents_dir):
            n_latents = len([f for f in os.listdir(latents_dir) if not f.startswith(".")])
        _log(log_file, f"Завершён. Латентов: {n_latents}")

        if not os.path.isdir(decoded_dir):
            _log(log_file, f"WARN: decoded_videos не создан в {decoded_dir}")

        with open(log_file, "r", encoding="utf-8") as f:
            log_text = f.read()

        return (preprocessed_root, decoded_dir, log_file, log_text)


NODE_CLASS_MAPPINGS = {
    "PyPTVLtx23Preprocess": PyPTVLtx23Preprocess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVLtx23Preprocess": "LTX-2.3 Preprocess Dataset (pyPTV)",
}
