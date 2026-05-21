"""
Caption → Conditions (.pt)
═══════════════════════════════════════════════════════════════════════════════
Читает единый caption из /tmp/dataset/caption.txt, кодирует его через
Gemma text encoder + embeddings processor (ltx_trainer) и сохраняет N
одинаковых conditions-файлов.

Как работает:
  1. Читает /tmp/dataset/caption.txt — должен содержать один текст.
  2. Gemma (text_encoder из PYPTV_MODELS) кодирует текст → hidden states.
  3. Embeddings processor превращает hidden states → video + audio embeddings.
  4. Сохраняет N файлов: /tmp/dataset/.precomputed/conditions/001.pt ... 00N.pt
     Все файлы одинаковые — один caption на весь датасет.

Формат .pt:
  {
      "video_prompt_embeds":   Tensor [seq_len, 4096],
      "audio_prompt_embeds":   Tensor [seq_len, 4096],
      "prompt_attention_mask": Tensor [seq_len] bool,
  }

Входы:
  • components  — PYPTV_MODELS из Trainer Components Loader
  • num_samples — сколько conditions сделать (default 25, = размер датасета)

Выход:
  • processed_count — сколько .pt файлов сохранено
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
                "components": ("PYPTV_MODELS",),
                "dataset": ("PYPTV_DATASET",),
                "num_samples": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("INT", "PYPTV_DATASET")
    RETURN_NAMES = ("processed_count", "dataset")
    FUNCTION = "encode"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def encode(self, components, dataset, num_samples: int):
        root = dataset["root"]
        caption_path = Path(f"{root}/caption.txt")
        output_folder = f"{root}/.precomputed/conditions"
        device = "cuda"

        text_encoder = components.get("text_encoder")
        embeddings_processor = components.get("embeddings_processor")
        if text_encoder is None or embeddings_processor is None:
            raise RuntimeError("Компоненты не загружены. Подключите Trainer Components Loader.")

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
        return (processed, dataset)


NODE_CLASS_MAPPINGS = {
    "LTX23EncodeCaptionConditions": LTX23EncodeCaptionConditions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23EncodeCaptionConditions": "LTX-2.3 Encode Caption Conditions (pyPTV)",
}
