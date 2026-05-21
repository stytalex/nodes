"""
LTX-2.3 Train LoRA
═══════════════════════════════════════════════════════════════════════════════
Запускает обучение LoRA на преподготовленных латентах.

Что нужно ДО запуска:
  1. /tmp/dataset/.precomputed/latents/       — video latents ( Encode Image Latents )
  2. /tmp/dataset/.precomputed/audio_latents/ — audio latents ( Encode Audio Latents )
  3. /tmp/dataset/.precomputed/conditions/    — conditions     ( Encode Caption Conditions )

Как работает:
  1. Берёт пути к моделям из components (PYPTV_MODELS).
  2. Собирает LtxTrainerConfig в памяти — без yaml, без subprocess.
  3. Создаёт LtxvTrainer и запускает обучение.
  4. Сохраняет чекпоинты LoRA и валидационные видео.

Основные параметры:
  • lora_rank / lora_alpha    — размер LoRA матриц (32/32)
  • learning_rate             — скорость обучения (1e-4)
  • steps                     — сколько шагов обучать (2000)
  • batch_size                — размер батча (1)
  • mixed_precision           — bf16 / fp16 / no
  • with_audio                — обучать аудио-ветку тоже
  • validation_prompt         — промпт для тестовых видео
  • validation_steps          — как часто генерировать валидацию
  • checkpoint_interval       — как часто сохранять чекпоинты

Входы:
  • components         — PYPTV_MODELS из Trainer Components Loader
  • Все остальные — гиперпараметры обучения

Выход:
  • output_dir — папка с весами LoRA и валидациями
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
                # --- Модель ---
                "load_checkpoint": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Путь к чекпоинту для продолжения обучения. Оставьте пустым чтобы начать заново. Можно указать папку — тренер сам найдёт последний чекпоинт.",
                }),

                # --- Датасет ---
                "preprocessed_data_root": ("STRING", {
                    "default": "/tmp/dataset",
                    "multiline": False,
                    "tooltip": "Папка датасета. Тренер автоматически ищет подпапку .precomputed/ внутри неё с готовыми латентами.",
                }),
                "with_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Обучать аудио ветку модели вместе с видео. Требует папку audio_latents/ в датасете. Если выключить — обучается только видео.",
                }),
                "num_dataloader_workers": ("INT", {
                    "default": 2, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Количество фоновых процессов для загрузки данных. 0 = синхронная загрузка (удобно для отладки ошибок датасета). 2-4 = быстрее для реального обучения.",
                }),

                # --- LoRA ---
                "lora_rank": ("INT", {
                    "default": 32, "min": 2, "max": 256, "step": 2,
                    "tooltip": "Ранг LoRA матриц. Чем выше — тем больше параметров и выразительности, но больше памяти и риск переобучения. Рекомендуется: 16-64. Для персонажа достаточно 32.",
                }),
                "lora_alpha": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Коэффициент масштабирования LoRA. Эффективный масштаб = alpha/rank. При alpha=rank масштаб равен 1.0. Обычно ставят равным rank.",
                }),
                "target_modules": ("STRING", {
                    "default": "to_k,to_q,to_v,to_out.0",
                    "multiline": False,
                    "tooltip": "Модули трансформера куда применяется LoRA. Через запятую. 'to_k,to_q,to_v,to_out.0' — покрывает все слои внимания включая аудио и кросс-модальные. Для видео-only используй 'attn1.to_k,attn1.to_q,attn1.to_v,attn1.to_out.0'.",
                }),
                "lora_dropout": ("FLOAT", {
                "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                "tooltip": "Dropout для слоёв LoRA. Помогает при переобучении на малых датасетах (20-25 сэмплов). 0.05 — лёгкая регуляризация. 0.0 — отключить.",
                }),
                # --- Оптимизация ---
                "learning_rate": ("FLOAT", {
                    "default": 1e-4, "min": 1e-7, "max": 1e-2,
                    "step": 1e-6, "round": False,
                    "tooltip": "Скорость обучения. Слишком высокая — модель расходится. Слишком низкая — обучается медленно. Для LoRA рекомендуется 1e-4 до 1e-5.",
                }),
                "steps": ("INT", {
                    "default": 2000, "min": 1, "max": 100000, "step": 100,
                    "tooltip": "Количество шагов обучения. Для 20-25 сэмплов обычно достаточно 1500-2500 шагов. Больше шагов = риск переобучения.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Размер батча на один GPU. При нехватке VRAM оставьте 1 и увеличьте gradient_accumulation_steps.",
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Накопление градиентов. Эффективный батч = batch_size × этот параметр. Позволяет имитировать большой батч при малом VRAM.",
                }),
                "scheduler_type": ([
                    "constant", "linear", "cosine",
                    "cosine_with_restarts", "polynomial",
                ], {
                    "default": "linear",
                    "tooltip": "Тип расписания learning rate. linear — плавно снижает LR до 0. cosine — снижает по косинусу. constant — не меняет LR. Рекомендуется linear.",
                }),
                "enable_gradient_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Экономия VRAM за счёт скорости (~20-30% медленнее). Рекомендуется включить — позволяет обучать при меньшем VRAM.",
                }),

                # --- Ускорение ---
                "mixed_precision": (["bf16", "fp16", "no"], {
                    "default": "bf16",
                    "tooltip": "Точность вычислений. bf16 — рекомендуется для современных GPU (A100, H100). fp16 — для старых GPU. no — полная точность fp32, требует вдвое больше VRAM.",
                }),
                "quantization": ([
                    "none", "int8-quanto", "int4-quanto",
                    "int2-quanto", "fp8-quanto",
                ], {
                    "default": "none",
                    "tooltip": "Квантизация модели для экономии VRAM. none — без квантизации (лучшее качество). int8-quanto — умеренная экономия. int4-quanto — сильная экономия но хуже качество. Несовместимо с full fine-tuning.",
                }),
                "load_text_encoder_in_8bit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Загрузить Gemma в 8-bit для экономии VRAM (~12GB вместо ~24GB). Требует bitsandbytes. Небольшая потеря качества эмбеддингов.",
                }),
                "offload_optimizer_during_validation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Выгружать состояние оптимизатора на CPU во время генерации валидационных видео. Помогает если не хватает VRAM для одновременного нахождения VAE + трансформер + оптимизатор на GPU. Замедляет валидацию.",
                }),

                # --- Чекпоинты ---
                "output_dir": ("STRING", {
                    "default": "/tmp/lora_output",
                    "multiline": False,
                    "tooltip": "Папка куда сохраняются веса LoRA и валидационные видео.",
                }),
                "checkpoint_interval": ("INT", {
                    "default": 250, "min": 0, "max": 10000, "step": 50,
                    "tooltip": "Сохранять чекпоинт каждые N шагов. 0 = сохранять только в конце. Полезно для восстановления после сбоя.",
                }),
                "keep_last_n": ("INT", {
                    "default": 3, "min": -1, "max": 100, "step": 1,
                    "tooltip": "Сколько последних чекпоинтов хранить. -1 = хранить все. 3 = только 3 последних (экономит место на диске).",
                }),
                "no_resume": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Начать обучение заново игнорируя сохранённое состояние оптимизатора. Веса из load_checkpoint всё равно загрузятся, но счётчик шагов и оптимизатор сбросятся.",
                }),

                # --- Валидация ---
                "validation_prompt": ("STRING", {
                    "default": "TRIGGER young woman with clear smooth skin, ash-brown hair, grey-blue eyes, dark-blue suit. The audio captures ambient room tone.",
                    "multiline": True,
                    "tooltip": "Промпт для генерации валидационного видео во время обучения. Используй тот же триггер и описание что и в датасете.",
                }),
                "validation_steps": ("INT", {
                    "default": 250, "min": 0, "max": 10000, "step": 50,
                    "tooltip": "Генерировать валидационное видео каждые N шагов. 0 = отключить валидацию (быстрее обучение).",
                }),
                "validation_inference_steps": ("INT", {
                    "default": 30, "min": 10, "max": 100, "step": 5,
                    "tooltip": "Количество шагов денойзинга при генерации валидационного видео. 30 — быстро и достаточно для мониторинга. Больше = лучше качество но медленнее.",
                }),
                "validation_width": ("INT", {
                    "default": 576, "min": 32, "max": 2048, "step": 32,
                    "tooltip": "Ширина валидационного видео в пикселях. Должна быть кратна 32. Меньше = быстрее генерация.",
                }),
                "validation_height": ("INT", {
                    "default": 576, "min": 32, "max": 2048, "step": 32,
                    "tooltip": "Высота валидационного видео в пикселях. Должна быть кратна 32.",
                }),
                "validation_frames": ("INT", {
                    "default": 89, "min": 1, "max": 257, "step": 8,
                    "tooltip": "Количество кадров валидационного видео. Должно быть frames % 8 == 1 (1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89...). Нода скорректирует автоматически.",
                }),

                # --- Прочее ---
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**31,
                    "tooltip": "Зерно случайности для воспроизводимости результатов обучения и валидации.",
                }),
                "first_frame_conditioning_p": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Вероятность кондиционирования на первый кадр во время обучения. 0.5 = половина батчей обучается в режиме image-to-video. Увеличь если хочешь лучший I2V режим.",
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
        load_checkpoint: str,
        preprocessed_data_root: str,
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
        offload_optimizer_during_validation: bool,
        output_dir: str,
        checkpoint_interval: int,
        keep_last_n: int,
        no_resume: bool,
        validation_prompt: str,
        validation_steps: int,
        validation_inference_steps: int,
        validation_width: int,
        validation_height: int,
        validation_frames: int,
        seed: int,
        first_frame_conditioning_p: float,
    ):
        paths = components.get("paths", {})
        model_path = paths.get("model_path", "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors")
        text_encoder_path = paths.get("text_encoder_path", "/comfyui/models/text_encoders/gemma-3-12b-it-qat")

        modules = [m.strip() for m in target_modules.split(",") if m.strip()]
        quant = None if quantization == "none" else quantization
        checkpoint = load_checkpoint.strip() if load_checkpoint.strip() else None

        if validation_frames % 8 != 1:
            validation_frames = (validation_frames // 8) * 8 + 1
            print(f"[training_ltx23_lora] validation_frames скорректировано до {validation_frames}")

        config = LtxTrainerConfig(
            seed=seed,
            output_dir=output_dir,

            model=ModelConfig(
                model_path=model_path,
                text_encoder_path=text_encoder_path,
                training_mode="lora",
                load_checkpoint=checkpoint,
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
                offload_optimizer_during_validation=offload_optimizer_during_validation,
            ),

            data=DataConfig(
                preprocessed_data_root=preprocessed_data_root,
                num_dataloader_workers=num_dataloader_workers,
            ),

            validation=ValidationConfig(
                prompts=[validation_prompt] if validation_prompt else [],
                negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                images=None,
                video_dims=(validation_width, validation_height, validation_frames),
                frame_rate=25.0,
                seed=seed,
                inference_steps=validation_inference_steps,
                interval=validation_steps if validation_steps > 0 else None,
                guidance_scale=4.0,
                stg_scale=1.0,
                stg_blocks=[29],
                stg_mode="stg_av",
                generate_audio=with_audio,
                skip_initial_validation=True,
            ),

            checkpoints=CheckpointsConfig(
                interval=checkpoint_interval if checkpoint_interval > 0 else None,
                keep_last_n=keep_last_n,
                precision="bfloat16",
                save_training_state="minimal",
                no_resume=no_resume,
            ),

            hub=HubConfig(push_to_hub=False),
            flow_matching=FlowMatchingConfig(
                timestep_sampling_mode="shifted_logit_normal",
            ),
            wandb=WandbConfig(enabled=False),
        )

        print(f"[training_ltx23_lora] Запуск тренировки: {steps} шагов → {output_dir}")
        print(f"[training_ltx23_lora] LoRA rank={lora_rank}, lr={learning_rate}, with_audio={with_audio}")
        if checkpoint:
            print(f"[training_ltx23_lora] Продолжение с чекпоинта: {checkpoint}")

        trainer = LtxvTrainer(config)
        saved_path, stats = trainer.train(disable_progress_bars=False)

        print(f"[training_ltx23_lora] Готово за {stats.total_time_seconds/60:.1f} мин")
        print(f"[training_ltx23_lora] Скорость: {stats.steps_per_second:.2f} шагов/сек")
        print(f"[training_ltx23_lora] Пик VRAM: {stats.peak_gpu_memory_gb:.1f} GB")
        print(f"[training_ltx23_lora] Веса сохранены: {saved_path}")

        return (str(Path(output_dir)),)


NODE_CLASS_MAPPINGS = {
    "LTX23TrainingLora": LTX23TrainingLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23TrainingLora": "LTX-2.3 Train LoRA (pyPTV)",
}
