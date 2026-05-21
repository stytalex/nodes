"""
Нода 3: Caption → Conditions (.pt)
Читает caption из /tmp/dataset/caption.txt, кодирует через Gemma + embeddings processor
и сохраняет N одинаковых .pt файлов + N .txt файлов (001.txt, 002.txt ...).

Формат выходного .pt:
{
    "video_prompt_embeds": Tensor [seq_len, 4096],
    "audio_prompt_embeds": Tensor [seq_len, 4096],
    "prompt_attention_mask": Tensor [seq_len] bool,
}
"""

from pathlib import Path

import torch

from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder


class LTX23EncodeCaptionConditions:
    """
    Кодирование caption из caption.txt в conditions латенты для всего датасета.
    Сохраняет .pt в .precomputed/conditions/ и .txt в корень датасета.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_samples": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("processed_count",)
    FUNCTION = "encode"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def encode(self, num_samples: int):
        caption_path = Path("/tmp/dataset/caption.txt")
        model_path = "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors"
        text_encoder_path = "/comfyui/models/text_encoders/gemma-3-12b-it-qat"
        output_folder = "/tmp/dataset/.precomputed/conditions"
        device = "cuda"

        # --- Читаем caption ---
        if not caption_path.exists():
            raise FileNotFoundError(f"caption.txt не найден: {caption_path}")

        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption:
            raise ValueError("caption.txt пуст")

        print(f"[LTX23EncodeCaptionConditions] Caption: {caption[:80]}...")
        print(f"[LTX23EncodeCaptionConditions] Сэмплов: {num_samples}")

        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        # --- Загрузка Gemma ---
        print("[LTX23EncodeCaptionConditions] Загрузка Gemma text encoder...")
        text_encoder = load_text_encoder(
            gemma_model_path=text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=False,
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
            hidden_states, mask = text_encoder.encode(caption)
            out = embeddings_processor.process_hidden_states(hidden_states, mask)

        condition_data = {
            "video_prompt_embeds":  out.video_encoding.cpu(),
            "audio_prompt_embeds":  out.audio_encoding.cpu(),
            "prompt_attention_mask": mask.cpu(),
        }

        shape = condition_data["video_prompt_embeds"].shape
        print(f"[LTX23EncodeCaptionConditions] Embedding shape: {shape}")

        # Выгрузить Gemma
        del text_encoder
        torch.cuda.empty_cache()

        # --- Сохраняем .pt ---
        processed = 0
        for idx in range(1, num_samples + 1):
            pt_file = out_path / f"{idx:03d}.pt"
            if not pt_file.exists():
                torch.save(condition_data, pt_file)
            processed += 1

        print(f"[LTX23EncodeCaptionConditions] Готово: {processed}/{num_samples}")
        return (processed,)


NODE_CLASS_MAPPINGS = {
    "LTX23EncodeCaptionConditions": LTX23EncodeCaptionConditions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23EncodeCaptionConditions": "LTX-2.3 Encode Caption Conditions (pyPTV)",
}
