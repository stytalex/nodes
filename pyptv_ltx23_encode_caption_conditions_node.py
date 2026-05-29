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
  4. Считает количество латентов в .precomputed/latents/ и создаёт столько же
     conditions файлов: 0000.pt, 0001.pt, ... (совпадает с именами латентов).
     Все файлы одинаковые — один caption на весь датасет.

Формат .pt:
  {
      "video_prompt_embeds":   Tensor [seq_len, 4096],
      "audio_prompt_embeds":   Tensor [seq_len, 4096],
      "prompt_attention_mask": Tensor [seq_len] bool,
  }

Входы:
  • components — PYPTV_MODELS из Trainer Components Loader
  • dataset    — PYPTV_DATASET из Encode Image/Audio Latents

Выход:
  • processed_count — сколько .pt файлов сохранено
"""

from pathlib import Path

import torch

from .pyptv_ltx23_trainer_components_loader_node import load_to_gpu, offload_to_cpu


class LTX23EncodeCaptionConditions:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("PYPTV_MODELS",),
                "dataset": ("PYPTV_DATASET",),
            }
        }

    RETURN_TYPES = ("INT", "PYPTV_DATASET")
    RETURN_NAMES = ("processed_count", "dataset")
    FUNCTION = "encode"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def encode(self, components, dataset):
        root = dataset["root"]
        caption_path = Path(f"{root}/caption.txt")
        output_folder = f"{root}/.precomputed/conditions"

        text_encoder = components.get("text_encoder")
        embeddings_processor = components.get("embeddings_processor")
        if text_encoder is None or embeddings_processor is None:
            raise RuntimeError("Компоненты не загружены. Подключите Trainer Components Loader.")

        if not caption_path.exists():
            raise FileNotFoundError(f"caption.txt не найден: {caption_path}")

        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption:
            raise ValueError("caption.txt пуст")

        # Количество conditions = количество латентов (берём из любой готовой папки)
        for latents_dir in [
            Path(f"{root}/.precomputed/latents"),
            Path(f"{root}/.precomputed/audio_latents"),
        ]:
            if latents_dir.exists():
                num_samples = len(list(latents_dir.glob("*.pt")))
                if num_samples > 0:
                    print(f"[LTX23EncodeCaptionConditions] Считаем по {latents_dir.name}: {num_samples} файлов")
                    break
        else:
            raise RuntimeError("Нет готовых латентов — сначала запусти Encode Image Latents или Encode Audio Latents.")

        print(f"[LTX23EncodeCaptionConditions] Caption: {caption[:80]}...")
        print(f"[LTX23EncodeCaptionConditions] Сэмплов (по латентам): {num_samples}")

        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        load_to_gpu(components, ["text_encoder", "embeddings_processor"])
        text_encoder = components["text_encoder"]
        embeddings_processor = components["embeddings_processor"]

        # Сохраняем вывод feature_extractor (до коннекторов).
        # Тренер сам применяет коннекторы в _training_step через create_embeddings.
        print("[LTX23EncodeCaptionConditions] Кодирование caption...")
        with torch.inference_mode():
            hidden_states, mask = text_encoder.encode(caption, padding_side="left")
            video_feats, audio_feats = embeddings_processor.feature_extractor(
                hidden_states, mask, "left"
            )

        condition_data = {
            "video_prompt_embeds":   video_feats[0].cpu().contiguous(),   # [seq_len, feat_dim]
            "prompt_attention_mask": mask[0].cpu().contiguous(),           # [seq_len]
        }
        if audio_feats is not None:
            condition_data["audio_prompt_embeds"] = audio_feats[0].cpu().contiguous()

        print(f"[LTX23EncodeCaptionConditions] Embedding shape: {condition_data['video_prompt_embeds'].shape}")

        processed = 0
        for idx in range(num_samples):
            pt_file = out_path / f"{idx:04d}.pt"
            if not pt_file.exists():
                torch.save(condition_data, pt_file)
            processed += 1

        print(f"[LTX23EncodeCaptionConditions] Готово: {processed}/{num_samples}")

        offload_to_cpu(components, ["text_encoder", "embeddings_processor"])
        return (processed, dataset)


NODE_CLASS_MAPPINGS = {
    "LTX23EncodeCaptionConditions": LTX23EncodeCaptionConditions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23EncodeCaptionConditions": "LTX-2.3 Encode Caption Conditions (pyPTV)",
}
