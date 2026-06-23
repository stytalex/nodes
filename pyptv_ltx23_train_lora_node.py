"""
LTX-2.3 Train LoRA (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Генерирует /home/ltx_train_config.yaml из UI-параметров и запускает
scripts/train.py через subprocess.Popen — НЕ блокирует ComfyUI.
За прогрессом смотрите через PyPTVLogViewer на log_file.

Все ключевые параметры обучения вынесены в UI:
  • LoRA:       rank, alpha, dropout
  • Optimizer:  learning_rate, steps, batch_size, grad_accum, optimizer, scheduler
  • Validation: prompt, interval, dims (W×H×frames), inference_steps, guidance,
                generate_video/audio
  • Checkpoints: interval, keep_last_n

Выход output_dir можно цеплять в PyPTVUploadCheckpoints.
"""

import os
import datetime
import subprocess


TAG = "Trainer"
CONFIG_PATH = "/home/ltx_train_config.yaml"


# ── target_modules для character LoRA с голосом ──────────────────────────
# Включаются опционально через флаги в UI:
#   enable_ff_video  → ff.net.0.proj, ff.net.2          (ёмкость для видео)
#   enable_ff_audio  → audio_ff.net.0.proj, audio_ff.net.2  (аудио feed-forward)
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


def _py_bool(b: bool) -> str:
    """Python bool → YAML bool ('true'/'false')"""
    return "true" if b else "false"


def _yaml_escape(s: str) -> str:
    """Экранирование строки для YAML (защита от " внутри prompt)"""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _format_modules(modules: list) -> str:
    """Список модулей → отступ YAML"""
    return "\n".join(f'    - "{m}"' for m in modules)


class PyPTVLtx23TrainLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── пути ──
                "preprocessed_data_root": ("STRING", {
                    "default": "/home/dataset/.precomputed",
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
                "output_dir": ("STRING", {
                    "default": "/home/lora_output",
                    "multiline": False,
                }),
                "ltx_repo_path": ("STRING", {
                    "default": "/home/LTX-2",
                    "multiline": False,
                }),
                "log_file": ("STRING", {
                    "default": "/home/ltx_train.log",
                    "multiline": False,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),

                # ── LoRA ──
                "lora_rank": ("INT", {
                    "default": 64, "min": 1, "max": 256,
                    "tooltip": "Ранг LoRA матриц. Больше = больше ёмкость + VRAM",
                }),
                "lora_alpha": ("INT", {
                    "default": 64, "min": 1, "max": 256,
                    "tooltip": "Scaling factor. Обычно равен rank",
                }),
                "lora_dropout": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "enable_ff_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Добавить ff.net.0.proj/ff.net.2 — ёмкость для видео",
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output_dir", "log_file", "process_pid")
    FUNCTION = "run"
    CATEGORY = "pyPTV"

    def run(self,
            # пути
            preprocessed_data_root, model_path, text_encoder_path, output_dir,
            ltx_repo_path, log_file, seed,
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
            checkpoint_interval, keep_last_n):

        _reset_log(log_file)

        # ── валидация ──
        trainer_cwd = os.path.join(ltx_repo_path, "packages", "ltx-trainer")
        if not os.path.isdir(trainer_cwd):
            raise RuntimeError(f"Не найден ltx-trainer: {trainer_cwd}")
        if not os.path.isdir(preprocessed_data_root):
            raise RuntimeError(f"Нет preprocessed_data_root: {preprocessed_data_root}")

        # LTX-2.3 ограничения
        if validation_width % 32 != 0 or validation_height % 32 != 0:
            raise RuntimeError(
                f"validation_width/height должны быть кратны 32 "
                f"(получено {validation_width}x{validation_height})"
            )
        if validation_frames != 1 and validation_frames % 8 != 1:
            raise RuntimeError(
                f"validation_frames должно быть 1 (картинка) или frames%8==1 "
                f"(9, 17, 25, 33...). Получено: {validation_frames}"
            )
        if lora_alpha != lora_rank:
            _log(log_file, f"WARN: lora_alpha ({lora_alpha}) != lora_rank ({lora_rank})")

        os.makedirs(output_dir, exist_ok=True)

        # ── сборка target_modules ──
        modules = list(BASE_MODULES)
        if enable_ff_video:
            modules += FF_VIDEO
        if enable_ff_audio:
            modules += FF_AUDIO

        # ── формирование config.yaml ──
        config_yaml = CONFIG_TEMPLATE.format(
            seed=seed,
            output_dir=output_dir,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            preprocessed_data_root=preprocessed_data_root,
            # lora
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules_yaml=_format_modules(modules),
            # data
            num_workers=num_workers,
            # optimization
            learning_rate=f"{learning_rate:.2e}",
            training_steps=training_steps,
            batch_size=batch_size,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            grad_checkpoint=_py_bool(grad_checkpoint),
            # acceleration
            mixed_precision=mixed_precision,
            te_8bit=_py_bool(te_8bit),
            # validation
            validation_interval=validation_interval,
            validation_width=validation_width,
            validation_height=validation_height,
            validation_frames=validation_frames,
            validation_fps=validation_fps,
            validation_seed=validation_seed,
            validation_inference_steps=validation_inference_steps,
            validation_guidance=validation_guidance,
            validation_stg_scale=validation_stg_scale,
            generate_audio=_py_bool(generate_audio),
            generate_video=_py_bool(generate_video),
            validation_prompt=_yaml_escape(validation_prompt.strip()),
            # checkpoints
            checkpoint_interval=checkpoint_interval,
            keep_last_n=keep_last_n,
        )

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(config_yaml)
        _log(log_file, f"Config записан: {CONFIG_PATH}")
        _log(log_file, f"  lora: rank={lora_rank}, alpha={lora_alpha}, "
                       f"modules={len(modules)}")
        _log(log_file, f"  opt:  lr={learning_rate}, steps={training_steps}, "
                       f"bs={batch_size}x{grad_accum}, {optimizer_type}/{scheduler_type}")
        _log(log_file, f"  val:  every {validation_interval}, "
                       f"{validation_width}x{validation_height}x{validation_frames}, "
                       f"video={generate_video}, audio={generate_audio}")
        _log(log_file, f"  ckpt: every {checkpoint_interval}, keep {keep_last_n}")

        # ── запуск train.py ──
        cmd = ["python", "scripts/train.py", CONFIG_PATH]
        _log(log_file, f"cwd: {trainer_cwd}")
        _log(log_file, f"cmd: {' '.join(cmd)}")

        lf = open(log_file, "a", encoding="utf-8")
        lf.write("\n--- stdout/stderr train.py ---\n")
        lf.flush()

        try:
            process = subprocess.Popen(
                cmd,
                cwd=trainer_cwd,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            lf.close()
            _log(log_file, f"Ошибка Popen: {e}")
            raise

        _log(log_file, f"Запущен train.py, PID: {process.pid}")

        return (output_dir, log_file, process.pid)


NODE_CLASS_MAPPINGS = {
    "PyPTVLtx23TrainLora": PyPTVLtx23TrainLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVLtx23TrainLora": "LTX-2.3 Train LoRA (pyPTV)",
}
