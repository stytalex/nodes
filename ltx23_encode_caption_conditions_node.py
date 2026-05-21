"""
Нода 3: Caption текст → Conditions (.pt)
Прогоняет один caption через Gemma + embeddings processor
и сохраняет N одинаковых .pt файлов (по количеству сэмплов в датасете).

Формат выходного .pt:
{
    "video_prompt_embeds": Tensor [seq_len, 4096],
    "audio_prompt_embeds": Tensor [seq_len, 4096],
    "prompt_attention_mask": Tensor [seq_len] bool,
}

Путь идентичен _load_text_encoder_and_cache_embeddings() из trainer.py:
    text_encoder.encode(prompt) → hidden_states, mask
    embeddings_processor.process_hidden_states(hs, mask) → video_encoding, audio_encoding
"""

from pathlib import Path

import torch

from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder


class LTX23EncodeCaptionConditions:
    """
    Кодирование одного caption в conditions латенты для всего датасета.

    Один и тот же caption (например триггер + описание персонажа)
    сохраняется N раз — по одному файлу на каждый сэмпл датасета.

    Входы:
        model_path         — путь к .safetensors LTX-2 (нужен для embeddings processor)
        text_encoder_path  — путь к папке с Gemma моделью
        caption            — текст caption (одинаковый для всех сэмплов)
        num_samples        — количество сэмплов (= количество картинок/аудио)
        output_folder      — куда сохранять .pt (папка conditions/)
        lora_trigger       — опциональный триггер-токен (prepend к caption)
        device             — cuda / cpu
        load_in_8bit       — загрузить Gemma в 8bit для экономии памяти

    Выход:
        processed_count — сколько файлов сохранено
        output_folder   — путь к папке с .pt файлами
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {
                    "default": "young woman with clear smooth skin ash-brown hair grey-blue eyes dark-blue suit",
                    "multiline": True,
                }),
                "num_samples": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                }),
            },
            "optional": {
                "lora_trigger": ("STRING", {
                    "default": "JSRv1rpd",
                    "multiline": False,
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "load_in_8bit": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("processed_count",)
    FUNCTION = "encode"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def encode(
        self,
        caption: str,
        num_samples: int,
        lora_trigger: str = "",
        device: str = "cuda",
        load_in_8bit: bool = False,
    ):
        model_path = "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors"
        text_encoder_path = "/comfyui/models/text_encoders/gemma-3-12b-it-qat"
        output_folder = "/tmp/dataset/.precomputed/conditions"
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        # Препендить триггер если задан
        full_caption = f"{lora_trigger} {caption}".strip() if lora_trigger else caption
        print(f"[LTX23EncodeCaptionConditions] Caption: {full_caption[:80]}...")
        print(f"[LTX23EncodeCaptionConditions] Сэмплов: {num_samples}")

        # --- Загрузка Gemma ---
        print("[LTX23EncodeCaptionConditions] Загрузка Gemma text encoder...")
        text_encoder = load_text_encoder(
            gemma_model_path=text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
        )

        # --- Загрузка embeddings processor ---
        print("[LTX23EncodeCaptionConditions] Загрузка embeddings processor...")
        embeddings_processor = load_embeddings_processor(
            checkpoint_path=model_path,
            device=device,
            dtype=torch.bfloat16,
        )

        # --- Кодируем caption ---
        print("[LTX23EncodeCaptionConditions] Кодирование caption...")
        with torch.inference_mode():
            hidden_states, mask = text_encoder.encode(full_caption)
            out = embeddings_processor.process_hidden_states(hidden_states, mask)

        # Результат — то что ожидает _training_step() тренера
        condition_data = {
            "video_prompt_embeds":  out.video_encoding.cpu(),
            "audio_prompt_embeds":  out.audio_encoding.cpu(),
            "prompt_attention_mask": mask.cpu(),
        }

        shape = condition_data["video_prompt_embeds"].shape
        print(f"[LTX23EncodeCaptionConditions] Embedding shape: {shape}")

        # Выгрузить Gemma — она больше не нужна
        del text_encoder
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Сохраняем N одинаковых файлов ---
        processed = 0
        for idx in range(num_samples):
            out_file = out_path / f"{idx:04d}.pt"

            if out_file.exists():
                print(f"[LTX23EncodeCaptionConditions] Пропуск {idx:04d}.pt (уже существует)")
                processed += 1
                continue

            torch.save(condition_data, out_file)
            processed += 1

        print(f"[LTX23EncodeCaptionConditions] Готово: {processed}/{num_samples} → {output_folder}")
        return (processed,)


NODE_CLASS_MAPPINGS = {
    "LTX23EncodeCaptionConditions": LTX23EncodeCaptionConditions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23EncodeCaptionConditions": "LTX-2.3 Encode Caption Conditions (pyPTV)",
}
