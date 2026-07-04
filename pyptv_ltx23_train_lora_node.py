"""
LTX-2.3 Train LoRA — единая замкнутая нода (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Объединяет весь пайплайн обучения LoRA в ОДНУ автономную ноду:
  1. Загрузка датасета с HF (load_dataset)
  2. Построение dataset.json + выбор buckets (dataset_builder)
  3. Препроцессинг latents (preprocess → process_dataset.py)
  4. Запуск тренировки (train.py)
  5. Заливка чекпоинтов на HF (upload_checkpoints)

Нода ЗАМКНУТА: нет входных коннектов и нет выходных сокетов.
Запускаешь — она делает весь цикл сама, прогресс пишет в лог-файл и показывает
его в себе. Разделять на 5 нод смысла нет — данные ходят последовательно.

Тренер запускается через subprocess (subprocess.Popen, не блокируя ComfyUI):
чистый AcceleratorState каждый запуск, никаких конфликтов с памятью ComfyUI,
тренер работает ровно как Lightricks задумали.

Модели НЕ скачиваются нодой — только пути (ставятся отдельно на RunPod).
"""

import os
import subprocess

from . import pyptv_ltx23_common as cm

TAG = "Trainer"
CONFIG_PATH = "/home/ltx_train_config.yaml"
DATASET_DIR = "/home/dataset"
DOWNLOAD_TMP = "/home/hf_repo_download"

# target_modules для character LoRA с голосом
BASE_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
FF_VIDEO     = ["ff.net.0.proj", "ff.net.2"]
FF_AUDIO     = ["audio_ff.net.0.proj", "audio_ff.net.2"]

CONFIG_TEMPLATE = """\
seed: {seed}
output_dir: "{output_dir}"

model:
  model_path: "{model_path}"
  text_encoder_path: "{text_encoder_path}"
  training_mode: "lora"

training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: true
    latents_dir: "audio_latents"

lora:
  rank: {lora_rank}
  alpha: {lora_alpha}
  dropout: {lora_dropout}
  target_modules:
{target_modules_yaml}

data:
  preprocessed_data_root: "{preprocessed_data_root}"
  num_dataloader_workers: {num_workers}

optimization:
  learning_rate: {learning_rate}
  steps: {training_steps}
  batch_size: {batch_size}
  gradient_accumulation_steps: {grad_accum}
  max_grad_norm: {max_grad_norm}
  optimizer_type: "{optimizer_type}"
  scheduler_type: "{scheduler_type}"
  enable_gradient_checkpointing: {grad_checkpoint}

acceleration:
  mixed_precision_mode: "{mixed_precision}"
  quantization: null
  load_text_encoder_in_8bit: {te_8bit}

validation:
  interval: {validation_interval}
  video_dims: [{validation_width}, {validation_height}, {validation_frames}]
  frame_rate: {validation_fps}
  seed: {validation_seed}
  inference_steps: {validation_inference_steps}
  guidance_scale: {validation_guidance}
  stg_scale: {validation_stg_scale}
  stg_blocks: [29]
  stg_mode: "stg_av"
  generate_audio: {generate_audio}
  generate_video: {generate_video}
  samples:
    - prompt: "{validation_prompt}"

checkpoints:
  interval: {checkpoint_interval}
  keep_last_n: {keep_last_n}
  precision: bfloat16
  save_training_state: "minimal"
"""


def _py_bool(b):
    return "true" if b else "false"


def _yaml_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _format_modules(modules):
    return "\n".join(f'    - "{m}"' for m in modules)


class PyPTVLtx23TrainLora:
    """Единая замкнутая нода: загрузка → датасет → препроцесс → тренировка → аплоад."""

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
                    "tooltip": "Триггер-слово LoRA (prepended автоматически через --lora-trigger)",
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
                "output_dir": ("STRING", {
                    "default": "/home/lora_output",
                    "multiline": False,
                    "tooltip": "Папка для результатов обучения",
                }),
                "log_file": ("STRING", {
                    "default": "/home/ltx_train.log",
                    "multiline": False,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),

                # ── LoRA ──
                "lora_rank": ("INT", {
                    "default": 64, "min": 1, "max": 256,
                    "tooltip": "Ранг LoRA. Больше = больше ёмкость + VRAM",
                }),
                "lora_alpha": ("INT", {
                    "default": 64, "min": 1, "max": 256,
                    "tooltip": "Scaling factor, обычно равен rank",
                }),
                "lora_dropout": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "enable_ff_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Добавить ff.net.* — ёмкость для видео",
                }),
                "enable_ff_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Добавить audio_ff.* — для тренировки аудио ветки",
                }),

                # ── Optimization ──
                "learning_rate": ("FLOAT", {
                    "default": 1.5e-4, "min": 1e-7, "max": 1.0, "step": 1e-6,
                }),
                "training_steps": ("INT", {"default": 1000, "min": 1, "max": 1000000}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "grad_accum": ("INT", {"default": 1, "min": 1, "max": 64}),
                "max_grad_norm": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                }),
                "optimizer_type": (["adamw", "adamw8bit", "lion", "prodigy"], {
                    "default": "adamw",
                }),
                "scheduler_type": (["cosine", "linear", "constant", "constant_with_warmup"], {
                    "default": "cosine",
                }),
                "grad_checkpoint": ("BOOLEAN", {"default": True}),
                "num_workers": ("INT", {"default": 2, "min": 0, "max": 16}),

                # ── Acceleration ──
                "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16"}),
                "te_8bit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Загружать text encoder в 8-bit (экономия VRAM)",
                }),

                # ── Validation ──
                "validation_prompt": ("STRING", {
                    "default": "JSRv1rpd young woman speaking",
                    "multiline": True,
                }),
                "validation_interval": ("INT", {"default": 200, "min": 1, "max": 100000}),
                "validation_width": ("INT", {
                    "default": 576, "min": 64, "max": 4096, "step": 32,
                }),
                "validation_height": ("INT", {
                    "default": 576, "min": 64, "max": 4096, "step": 32,
                }),
                "validation_frames": ("INT", {
                    "default": 1, "min": 1, "max": 257,
                    "tooltip": "1 для картинки, иначе frames%8==1: 9,17,25,33...",
                }),
                "validation_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                }),
                "validation_seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "validation_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "validation_guidance": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1,
                }),
                "validation_stg_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                }),
                "generate_video": ("BOOLEAN", {"default": True}),
                "generate_audio": ("BOOLEAN", {"default": True}),

                # ── Checkpoints ──
                "checkpoint_interval": ("INT", {"default": 250, "min": 1, "max": 100000}),
                "keep_last_n": ("INT", {"default": 3, "min": 1, "max": 100}),

                # ── Upload ──
                "upload_repo_id": ("STRING", {
                    "default": "avidscreator/loras",
                    "multiline": False,
                    "tooltip": "HF repo куда заливать чекпоинты",
                }),
                "upload_subfolder": ("STRING", {
                    "default": "test",
                    "multiline": False,
                    "tooltip": "Подпапка для чекпоинтов",
                }),
                "lora_prefix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Префикс имени файлов, например: mylora_",
                }),
                "upload_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Залить чекпоинты на HF после тренировки",
                }),
            }
        }

    # Замкнутая нода: нет входных коннектов и нет выходных сокетов.
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def run(self,
            # dataset
            dataset_repo_id, dataset_subfolder, hf_token, trigger_word,
            # paths / models
            model_path, text_encoder_path, ltx_repo_path, output_dir, log_file, seed,
            # lora
            lora_rank, lora_alpha, lora_dropout, enable_ff_video, enable_ff_audio,
            # optimization
            learning_rate, training_steps, batch_size, grad_accum, max_grad_norm,
            optimizer_type, scheduler_type, grad_checkpoint, num_workers,
            # acceleration
            mixed_precision, te_8bit,
            # validation
            validation_prompt, validation_interval,
            validation_width, validation_height, validation_frames, validation_fps,
            validation_seed, validation_inference_steps, validation_guidance,
            validation_stg_scale, generate_video, generate_audio,
            # checkpoints
            checkpoint_interval, keep_last_n,
            # upload
            upload_repo_id, upload_subfolder, lora_prefix, upload_after):

        cm.reset_log(log_file)
        log = lambda m: cm.log(log_file, TAG, m)

        # ── валидации входов ──
        trainer_cwd = os.path.join(ltx_repo_path, "packages", "ltx-trainer")
        if not os.path.isdir(trainer_cwd):
            raise RuntimeError(f"Не найден ltx-trainer: {trainer_cwd}")
        cm.validate_resolution(validation_width, validation_height)
        cm.validate_frames(validation_frames)

        # === ШАГ 1: загрузка датасета с HF ===
        log("=" * 60)
        log("ШАГ 1/5: загрузка датасета с HuggingFace")
        cm.hf_download_dataset(
            dataset_repo_id, dataset_subfolder, hf_token,
            DATASET_DIR, DOWNLOAD_TMP, log_file, TAG,
        )

        # === ШАГ 2: построение dataset.json + buckets ===
        log("=" * 60)
        log("ШАГ 2/5: построение dataset.json")
        dataset_json, buckets_str, count, _ = cm.build_dataset(
            DATASET_DIR, trigger_word, log_file, TAG,
        )
        log(f"Датасет: {count} пар, buckets={buckets_str}")

        # === ШАГ 3: препроцессинг latents ===
        log("=" * 60)
        log("ШАГ 3/5: препроцессинг (process_dataset.py)")
        preprocessed_root = os.path.join(DATASET_DIR, ".precomputed")
        preprocess_cmd = [
            "python", "scripts/process_dataset.py",
            dataset_json,
            "--resolution-buckets", buckets_str,
            "--model-path", model_path,
            "--text-encoder-path", text_encoder_path,
            "--lora-trigger", trigger_word,
            "--overwrite",
        ]
        result = cm.run_logged(
            preprocess_cmd, trainer_cwd, log_file,
            "stdout/stderr process_dataset.py", TAG,
        )
        if result.returncode != 0:
            log(f"process_dataset.py упал, returncode={result.returncode}")
            raise RuntimeError(
                f"Preprocess failed: returncode={result.returncode}, см. {log_file}"
            )
        latents_dir = os.path.join(preprocessed_root, "latents")
        n = len(os.listdir(latents_dir)) if os.path.isdir(latents_dir) else 0
        log(f"Препроцессинг завершён. Латентов: {n}")

        # === ШАГ 4: генерация config.yaml + запуск train.py ===
        log("=" * 60)
        log("ШАГ 4/5: запуск тренировки")
        if lora_alpha != lora_rank:
            log(f"WARN: lora_alpha ({lora_alpha}) != lora_rank ({lora_rank})")
        os.makedirs(output_dir, exist_ok=True)

        # сборка target_modules
        modules = list(BASE_MODULES)
        if enable_ff_video:
            modules += FF_VIDEO
        if enable_ff_audio:
            modules += FF_AUDIO

        config_yaml = CONFIG_TEMPLATE.format(
            seed=seed,
            output_dir=output_dir,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            preprocessed_data_root=preprocessed_root,
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules_yaml=_format_modules(modules),
            num_workers=num_workers,
            learning_rate=f"{learning_rate:.2e}",
            training_steps=training_steps, batch_size=batch_size,
            grad_accum=grad_accum, max_grad_norm=max_grad_norm,
            optimizer_type=optimizer_type, scheduler_type=scheduler_type,
            grad_checkpoint=_py_bool(grad_checkpoint),
            mixed_precision=mixed_precision, te_8bit=_py_bool(te_8bit),
            validation_interval=validation_interval,
            validation_width=validation_width, validation_height=validation_height,
            validation_frames=validation_frames, validation_fps=validation_fps,
            validation_seed=validation_seed,
            validation_inference_steps=validation_inference_steps,
            validation_guidance=validation_guidance,
            validation_stg_scale=validation_stg_scale,
            generate_audio=_py_bool(generate_audio),
            generate_video=_py_bool(generate_video),
            validation_prompt=_yaml_escape(validation_prompt.strip()),
            checkpoint_interval=checkpoint_interval, keep_last_n=keep_last_n,
        )
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(config_yaml)
        log(f"Config записан: {CONFIG_PATH}")
        log(f"  lora: rank={lora_rank}, alpha={lora_alpha}, modules={len(modules)}")
        log(f"  opt:  lr={learning_rate}, steps={training_steps}, "
            f"bs={batch_size}x{grad_accum}, {optimizer_type}/{scheduler_type}")
        log(f"  val:  every {validation_interval}, "
            f"{validation_width}x{validation_height}x{validation_frames}")
        log(f"  ckpt: every {checkpoint_interval}, keep {keep_last_n}")

        train_cmd = ["python", "scripts/train.py", CONFIG_PATH]
        # train.py запускаем через Popen (НЕ блокируем ComfyUI) — мониторинг через лог.
        lf = open(log_file, "a", encoding="utf-8")
        lf.write("\n--- stdout/stderr train.py ---\n")
        lf.flush()
        try:
            process = subprocess.Popen(
                train_cmd, cwd=trainer_cwd, stdout=lf, stderr=subprocess.STDOUT,
            )
        except Exception as e:
            lf.close()
            log(f"Ошибка Popen: {e}")
            raise
        log(f"Запущен train.py, PID: {process.pid}")

        # ждём завершения в замкнутой ноде (до аплоада нужен финальный чекпоинт)
        process.wait()
        lf.close()
        if process.returncode != 0:
            log(f"train.py упал, returncode={process.returncode}")
            raise RuntimeError(
                f"Training failed: returncode={process.returncode}, см. {log_file}"
            )
        log("Тренировка завершена успешно.")

        # === ШАГ 5: заливка чекпоинтов на HF ===
        log("=" * 60)
        log("ШАГ 5/5: загрузка чекпоинтов на HuggingFace")
        if upload_after:
            self._upload_checkpoints(
                output_dir, upload_repo_id, upload_subfolder,
                hf_token, lora_prefix, log_file,
            )
        else:
            log("Заливка отключена (upload_after=False)")

        # ── финальный лог для UI ──
        log("=" * 60)
        log(f"ГОТОВО. Чекпоинты: {output_dir}")
        log_text = ""
        if os.path.isfile(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                log_text = f.read()
        return {"ui": {"text": [log_text]}, "result": ()}

    def _upload_checkpoints(self, output_dir, repo_id, subfolder,
                            hf_token, lora_prefix, log_file):
        """Копирует .safetensors с префиксом во временную папку и льёт на HF."""
        import shutil
        from pathlib import Path

        src = Path(output_dir)
        files = sorted(p for p in src.rglob("*")
                       if p.is_file() and p.suffix == ".safetensors")
        if not files:
            cm.log(log_file, TAG, f"WARN: в {output_dir} нет .safetensors")
            return 0

        tmp_dir = Path("/home/hf_upload")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        prefix = (lora_prefix or "").strip()
        names = []
        for f in files:
            new_name = f"{prefix}{f.name}"
            shutil.copy2(str(f), str(tmp_dir / new_name))
            names.append(new_name)

        cm.log(log_file, TAG, f"Заливка {len(names)} чекпоинтов в {repo_id}/{subfolder} ...")
        ok = cm.hf_upload(
            repo_id, subfolder, hf_token, tmp_dir,
            ["*.safetensors"], log_file, TAG,
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if ok:
            cm.log(log_file, TAG, "Залито: " + ", ".join(names))
        return len(names)


NODE_CLASS_MAPPINGS = {
    "PyPTVLtx23TrainLora": PyPTVLtx23TrainLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVLtx23TrainLora": "LTX-2.3 Train LoRA (pyPTV)",
}
