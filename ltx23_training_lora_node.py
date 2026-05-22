"""
LTX-2.3 Train LoRA
═══════════════════════════════════════════════════════════════════════════════
Запускает обучение LoRA на преподготовленных латентах.

Что нужно ДО запуска:
  1. /tmp/dataset/.precomputed/latents/       — video latents
  2. /tmp/dataset/.precomputed/audio_latents/ — audio latents
  3. /tmp/dataset/.precomputed/conditions/    — caption conditions

Как работает:
  • Берёт пути к моделям из components (PYPTV_MODELS).
  • Собирает LtxTrainerConfig без yaml, без subprocess.
  • Создаёт LtxvTrainer и запускает обучение.
  • Сохраняет только LoRA чекпоинты. Валидация отключена, resume отключён.

Чекпоинты:
  • interval=250  — сохраняет каждые 250 шагов.
  • keep_last_n=2 — хранит только 2 последних (старые авто-удаляются).
  • При steps=2000 получаешь чекпоинты ~1750 и 2000.
  • Финальная LoRA сохраняется всегда независимо от keep_last_n.

Входы:
  • components — PYPTV_MODELS из Trainer Components Loader
  • dataset    — PYPTV_DATASET из Encode Caption Conditions

Выход:
  • output_dir — папка с весами LoRA
"""

from pathlib import Path

import torch

from ltx_trainer.config import (
    AccelerationConfig,
    CheckpointsConfig,
    DataConfig,
    FlowMatchingConfig,
    HubConfig,
    LoraConfig,
    LtxTrainerConfig,
    ModelConfig,
    OptimizationConfig,
    ValidationConfig,
    WandbConfig,
)
from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig


class LTX23TrainingLora:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("PYPTV_MODELS",),
                "dataset": ("PYPTV_DATASET",),

                # --- Модель ---
                "with_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Обучать аудио ветку модели вместе с видео. Требует папку audio_latents/ в датасете.",
                }),
                "num_dataloader_workers": ("INT", {
                    "default": 2, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Количество фоновых процессов для загрузки данных. 0 = синхронная загрузка.",
                }),

                # --- LoRA ---
                "lora_rank": ("INT", {
                    "default": 32, "min": 2, "max": 256, "step": 2,
                    "tooltip": "Ранг LoRA матриц. Рекомендуется 16–64. Для персонажа достаточно 32.",
                }),
                "lora_alpha": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Коэффициент масштабирования LoRA. Обычно ставят равным rank.",
                }),
                "target_modules": ("STRING", {
                    "default": "to_k,to_q,to_v,to_out.0",
                    "multiline": False,
                    "tooltip": "Модули трансформера куда применяется LoRA. Через запятую.",
                }),
                "lora_dropout": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Dropout для слоёв LoRA. 0.05 — лёгкая регуляризация.",
                }),

                # --- Оптимизация ---
                "learning_rate": ("FLOAT", {
                    "default": 1e-4, "min": 1e-7, "max": 1e-2,
                    "step": 1e-6, "round": False,
                    "tooltip": "Скорость обучения. Для LoRA рекомендуется 1e-4 до 1e-5.",
                }),
                "steps": ("INT", {
                    "default": 2000, "min": 1, "max": 100000, "step": 100,
                    "tooltip": "Количество шагов обучения. Для 20-25 сэмплов достаточно 1500-2500.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Размер батча на один GPU.",
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Накопление градиентов. Эффективный батч = batch_size × этот параметр.",
                }),
                "scheduler_type": ([
                    "constant", "linear", "cosine",
                    "cosine_with_restarts", "polynomial",
                ], {
                    "default": "linear",
                    "tooltip": "Тип расписания learning rate. linear — плавно снижает LR до 0.",
                }),
                "enable_gradient_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Экономия VRAM за счёт скорости (~20-30% медленнее). Рекомендуется включить.",
                }),

                # --- Ускорение ---
                "mixed_precision": (["bf16", "fp16", "no"], {
                    "default": "bf16",
                    "tooltip": "bf16 — рекомендуется для A100/H100. no — fp32, вдвое больше VRAM.",
                }),
                "quantization": ([
                    "none", "int8-quanto", "int4-quanto",
                    "int2-quanto", "fp8-quanto",
                ], {
                    "default": "none",
                    "tooltip": "Квантизация модели для экономии VRAM. none — лучшее качество.",
                }),
                "load_text_encoder_in_8bit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Загрузить Gemma в 8-bit для экономии VRAM (~12GB вместо ~24GB).",
                }),

                # --- Чекпоинты ---
                "output_dir": ("STRING", {
                    "default": "/tmp/lora_output",
                    "multiline": False,
                    "tooltip": "Папка куда сохраняются веса LoRA.",
                }),
                "checkpoint_interval": ("INT", {
                    "default": 250, "min": 50, "max": 10000, "step": 50,
                    "tooltip": "Сохранять чекпоинт каждые N шагов. При keep_last_n=2 старые авто-удаляются.",
                }),
                "keep_last_n": ("INT", {
                    "default": 5, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Сколько последних чекпоинтов хранить. 5 = последние 5 (например 1500, 1750, 2000, 2250, 2500 при steps=2500).",
                }),

                # --- Прочее ---
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**31,
                    "tooltip": "Зерно случайности для воспроизводимости.",
                }),
                "first_frame_conditioning_p": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Вероятность кондиционирования на первый кадр. 0.5 = половина батчей в I2V режиме.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    FUNCTION = "train"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def train(
        self,
        components,
        dataset,
        with_audio: bool,
        num_dataloader_workers: int,
        lora_rank: int,
        lora_alpha: int,
        target_modules: str,
        lora_dropout: float,
        learning_rate: float,
        steps: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        scheduler_type: str,
        enable_gradient_checkpointing: bool,
        mixed_precision: str,
        quantization: str,
        load_text_encoder_in_8bit: bool,
        output_dir: str,
        checkpoint_interval: int,
        keep_last_n: int,
        seed: int,
        first_frame_conditioning_p: float,
    ):
        root = dataset["root"]
        preprocessed_data_root = root
        paths = components.get("paths", {})
        model_path = paths.get("model_path", "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors")
        text_encoder_path = paths.get("text_encoder_path", "/comfyui/models/text_encoders/gemma-3-12b-it-qat")

        modules = [m.strip() for m in target_modules.split(",") if m.strip()]
        quant = None if quantization == "none" else quantization

        config = LtxTrainerConfig(
            seed=seed,
            output_dir=output_dir,

            model=ModelConfig(
                model_path=model_path,
                text_encoder_path=text_encoder_path,
                training_mode="lora",
                load_checkpoint=None,          # resume отключён всегда
            ),

            lora=LoraConfig(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=modules,
            ),

            training_strategy=TextToVideoConfig(
                name="text_to_video",
                first_frame_conditioning_p=first_frame_conditioning_p,
                with_audio=with_audio,
                audio_latents_dir="audio_latents",
            ),

            optimization=OptimizationConfig(
                learning_rate=learning_rate,
                steps=steps,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=1.0,
                optimizer_type="adamw",
                scheduler_type=scheduler_type,
                scheduler_params={},
                enable_gradient_checkpointing=enable_gradient_checkpointing,
            ),

            acceleration=AccelerationConfig(
                mixed_precision_mode=mixed_precision,
                quantization=quant,
                load_text_encoder_in_8bit=load_text_encoder_in_8bit,
                offload_optimizer_during_validation=False,
            ),

            data=DataConfig(
                preprocessed_data_root=preprocessed_data_root,
                num_dataloader_workers=num_dataloader_workers,
            ),

            validation=ValidationConfig(
                prompts=[],                    # валидация отключена
                negative_prompt="",
                images=None,
                video_dims=(576, 576, 89),
                frame_rate=25.0,
                seed=seed,
                inference_steps=30,
                interval=None,                 # отключена
                guidance_scale=4.0,
                stg_scale=1.0,
                stg_blocks=[29],
                stg_mode="stg_av",
                generate_audio=False,
                skip_initial_validation=True,
            ),

            checkpoints=CheckpointsConfig(
                interval=checkpoint_interval if checkpoint_interval > 0 else None,
                keep_last_n=keep_last_n,
                precision="bfloat16",
                save_training_state="minimal",
                no_resume=True,                # всегда начинаем с нуля
            ),

            hub=HubConfig(push_to_hub=False),
            flow_matching=FlowMatchingConfig(
                timestep_sampling_mode="shifted_logit_normal",
            ),
            wandb=WandbConfig(enabled=False),
        )

        print(f"[LTX23TrainingLora] Запуск тренировки: {steps} шагов → {output_dir}")
        print(f"[LTX23TrainingLora] LoRA rank={lora_rank}, lr={learning_rate}, with_audio={with_audio}")
        print(f"[LTX23TrainingLora] Чекпоинты каждые {checkpoint_interval} шагов, храним последние {keep_last_n}")
        print(f"[LTX23TrainingLora] Валидация ОТКЛЮЧЕНА, resume ОТКЛЮЧЕН")

        trainer = LtxvTrainer(config)
        saved_path, stats = trainer.train(disable_progress_bars=False)

        print(f"[LTX23TrainingLora] Готово за {stats.total_time_seconds/60:.1f} мин")
        print(f"[LTX23TrainingLora] Скорость: {stats.steps_per_second:.2f} шагов/сек")
        print(f"[LTX23TrainingLora] Пик VRAM: {stats.peak_gpu_memory_gb:.1f} GB")
        print(f"[LTX23TrainingLora] Веса сохранены: {saved_path}")

        return (str(Path(output_dir)),)


NODE_CLASS_MAPPINGS = {
    "LTX23TrainingLora": LTX23TrainingLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23TrainingLora": "LTX-2.3 Train LoRA (pyPTV)",
}
