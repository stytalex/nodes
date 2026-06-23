"""
LTX-2.3 Dataset Builder (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Принимает PYPTV_DATASET (от LTX23LoadDataset), приводит файлы к виду NNNN.ext,
выравнивает количество картинок и аудио, выбирает resolution buckets,
собирает dataset.json для ltx-trainer.

Что делает:
  1. Инвентаризирует dataset_dir, удаляет мусор (caption.txt, prompts.json,
     reference.txt, *.mp3).
  2. Переименовывает картинки и аудио в формат 0001.ext, сохраняя парность
     с .txt подписями.
  3. Выравнивает количество (min(images, audios)), лишнее удаляет.
  4. Анализирует разрешения картинок → выбирает buckets из допустимых LTX-2.3.
  5. Собирает dataset.json (caption из .txt с удалением триггер-слова).
  6. Возвращает превью картинок для UI.

Входы:
  • dataset      — PYPTV_DATASET от LTX23LoadDataset
  • trigger_word — триггер LoRA (например JSRv1rpd)
  • log_file     — путь к лог-файлу

Выходы:
  • dataset_json       — путь к собранному dataset.json
  • resolution_buckets — строка вида "960x544x1;768x512x1"
  • images_preview     — IMAGE тензор для превью в ComfyUI
  • dataset_count      — итоговое количество пар
  • log                — содержимое лога
"""

import os
import re
import json
import datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image


ALLOWED_BUCKETS = [
    (1536, 1024), (1024, 1536),
    (1280, 960),  (960, 1280),
    (1024, 768),  (768, 1024),
    (960, 544),   (544, 960),
    (768, 512),   (512, 768),
    (640, 480),   (480, 640),
    (512, 512),
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
AUDIO_EXTS = (".wav",)
JUNK_FILES = {"caption.txt", "prompts.json", "reference.txt"}
JUNK_EXTS  = {".mp3", ".flac", ".ogg", ".m4a", ".aac"}

TAG = "DatasetBuilder"


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


def _pick_bucket(width: int, height: int):
    ratio = width / height
    area = width * height

    def score(b):
        bw, bh = b
        br = bw / bh
        ba = bw * bh
        return abs(ratio - br) / max(ratio, br) * 2.0 + abs(area - ba) / max(area, ba)

    return min(ALLOWED_BUCKETS, key=score)


def _load_preview(paths, max_count=64, target=512):
    if not paths:
        return torch.zeros(1, target, target, 3, dtype=torch.float32)
    imgs = []
    for p in paths[:max_count]:
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            scale = target / max(w, h)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGB", (target, target), (0, 0, 0))
            canvas.paste(img, ((target - nw) // 2, (target - nh) // 2))
            imgs.append(np.array(canvas).astype(np.float32) / 255.0)
        except Exception as e:
            print(f"[DatasetBuilder] preview load fail {p}: {e}")
    if not imgs:
        return torch.zeros(1, target, target, 3, dtype=torch.float32)
    return torch.from_numpy(np.stack(imgs, axis=0))


class PyPTVLtx23DatasetBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("PYPTV_DATASET", {"tooltip": "От LTX23LoadDataset"}),
                "trigger_word": ("STRING", {
                    "default": "JSRv1rpd",
                    "multiline": False,
                    "tooltip": "Триггер LoRA — будет prepend-иться process_dataset.py",
                }),
                "log_file": ("STRING", {
                    "default": "/home/ltx_dataset.log",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("dataset_json", "resolution_buckets", "images_preview", "dataset_count", "log")
    FUNCTION = "build"
    CATEGORY = "pyPTV"

    def build(self, dataset, trigger_word, log_file):
        _reset_log(log_file)

        dataset_dir = dataset.get("root") if isinstance(dataset, dict) else None
        if not dataset_dir or not os.path.isdir(dataset_dir):
            raise RuntimeError(f"Невалидный PYPTV_DATASET: {dataset}")

        _log(log_file, f"Начало обработки {dataset_dir}")

        # === Шаг 1: чистка мусора ===
        removed = []
        for name in os.listdir(dataset_dir):
            full = os.path.join(dataset_dir, name)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(name)[1].lower()
            if name in JUNK_FILES or ext in JUNK_EXTS:
                try:
                    os.remove(full)
                    removed.append(name)
                except Exception as e:
                    _log(log_file, f"не удалось удалить {name}: {e}")
        if removed:
            _log(log_file, f"Удалены лишние файлы: {', '.join(removed)}")

        # === Шаг 2: инвентаризация ===
        files = sorted(os.listdir(dataset_dir))
        images = sorted([f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
        audios = sorted([f for f in files if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
        _log(log_file, f"Найдено картинок: {len(images)}, аудио: {len(audios)}")

        # === Шаг 3: переименование в NNNN.ext (идемпотентно) ===
        already_normalized = bool(images or audios) and all(
            re.match(r"^\d{4}\.", f) for f in images + audios
        )

        if not already_normalized:
            _log(log_file, "Переименование в формат NNNN")
            tmp_prefix = "__tmp_ltx_rename__"

            def rename_group(file_list):
                # фаза 1 → tmp
                stage = []
                for i, fn in enumerate(file_list):
                    src = os.path.join(dataset_dir, fn)
                    tmp = os.path.join(dataset_dir, f"{tmp_prefix}{i}_{fn}")
                    os.rename(src, tmp)
                    stage.append((tmp, fn))
                # фаза 2 → финальное имя + парный .txt
                for i, (tmp_path, original) in enumerate(stage, start=1):
                    base, ext = os.path.splitext(original)
                    new_name = f"{i:04d}{ext.lower()}"
                    os.rename(tmp_path, os.path.join(dataset_dir, new_name))
                    txt_src = os.path.join(dataset_dir, base + ".txt")
                    if os.path.exists(txt_src):
                        txt_dst = os.path.join(dataset_dir, f"{i:04d}.txt")
                        if txt_src != txt_dst:
                            # если коллизия — сначала удаляем существующий
                            if os.path.exists(txt_dst):
                                os.remove(txt_dst)
                            os.rename(txt_src, txt_dst)

            rename_group(images)
            rename_group(audios)

            files = sorted(os.listdir(dataset_dir))
            images = sorted([f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
            audios = sorted([f for f in files if os.path.splitext(f)[1].lower() in AUDIO_EXTS])

        # === Шаг 4: выравнивание ===
        min_count = min(len(images), len(audios))
        if min_count == 0:
            raise RuntimeError(f"Пустой датасет: images={len(images)}, audios={len(audios)}")

        extra = []
        for name in images[min_count:] + audios[min_count:]:
            full = os.path.join(dataset_dir, name)
            base = os.path.splitext(name)[0]
            try:
                os.remove(full)
                extra.append(name)
                txt = os.path.join(dataset_dir, base + ".txt")
                if os.path.exists(txt):
                    os.remove(txt)
            except Exception as e:
                _log(log_file, f"не удалось удалить {name}: {e}")
        if extra:
            _log(log_file, f"Удалены лишние пары: {', '.join(extra)}")

        images = images[:min_count]
        audios = audios[:min_count]

        # === Шаг 5: buckets ===
        chosen = set()
        for img_name in images:
            try:
                with Image.open(os.path.join(dataset_dir, img_name)) as im:
                    w, h = im.size
                chosen.add(_pick_bucket(w, h))
            except Exception as e:
                _log(log_file, f"не удалось прочитать {img_name}: {e}")
        if not chosen:
            chosen.add((512, 512))

        buckets_str = ";".join(f"{w}x{h}x1" for w, h in sorted(chosen))
        _log(log_file, f"Выбраны buckets: {buckets_str}")

        # === Шаг 6: dataset.json ===
        entries = []
        trig = trigger_word.strip()
        trig_re = re.compile(rf"\b{re.escape(trig)}\b", re.IGNORECASE) if trig else None

        for img_name, audio_name in zip(images, audios):
            base = os.path.splitext(img_name)[0]
            txt_path = os.path.join(dataset_dir, base + ".txt")
            caption = ""
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            else:
                _log(log_file, f"нет .txt для {img_name}")

            if trig_re and trig_re.search(caption):
                caption = trig_re.sub("", caption)
                caption = re.sub(r"\s+", " ", caption).strip(" ,.;:-")

            entries.append({
                "caption": caption,
                "video": os.path.join(dataset_dir, img_name),
                "audio": os.path.join(dataset_dir, audio_name),
            })

        output_json = os.path.join(dataset_dir, "dataset.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        _log(log_file, f"Датасет готов: {len(entries)} пар → {output_json}")

        # === Превью ===
        preview = _load_preview([os.path.join(dataset_dir, n) for n in images])

        # читаем лог чтобы вернуть в выход
        with open(log_file, "r", encoding="utf-8") as f:
            log_text = f.read()

        return (output_json, buckets_str, preview, len(entries), log_text)


NODE_CLASS_MAPPINGS = {
    "PyPTVLtx23DatasetBuilder": PyPTVLtx23DatasetBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVLtx23DatasetBuilder": "LTX-2.3 Build Dataset JSON (pyPTV)",
}