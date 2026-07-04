"""
Общие хелперы для LTX-2.3 нод (pyPTV).
═══════════════════════════════════════════════════════════════════════════════
Переиспользуется train и check_vae нодами — сам по себе не является ComfyUI-нодой.
Ноды остаются автономными (нет коннектов между ними в графе), просто делят утилиты:
  • логирование в едином формате
  • скачивание/заливка датасета с HuggingFace
  • построение dataset.json + выбор resolution buckets (логика DatasetBuilder)
  • запуск subprocess тренера с перенаправлением вывода в лог
"""

import datetime
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Допустимые resolution buckets LTX-2.3 (ширина и высота кратны 32, F=1 для картинок)
# ---------------------------------------------------------------------------
ALLOWED_BUCKETS = [
    (1536, 1024), (1024, 1536),
    (1280, 960),  (960, 1280),
    (1024, 768),  (768, 1024),
    (960, 544),   (544, 960),
    (768, 512),   (512, 768),
    (640, 480),   (480, 640),
    (512, 512),
]

IMAGE_EXTS    = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
AUDIO_EXTS    = (".wav",)
# Служебные файлы, которые builder удаляет перед построением dataset.json
JUNK_FILES    = {"caption.txt", "prompts.json", "reference.txt"}
JUNK_EXTS     = {".mp3", ".flac", ".ogg", ".m4a", ".aac"}
# Опорные файлы Dramabox — builder их тоже удаляет (они не нужны тренеру)
PROTECTED_STEMS = {"reference", "prompts", "caption", "silence_latent_frame"}


# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

def reset_log(log_path: str):
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        open(log_path, "w").close()


def log(log_path: str, tag: str, msg: str):
    """Единый формат: [2024-01-15 10:23:45] [TAG] message"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    print(line, flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

def hf_download_dataset(repo_id, subfolder, hf_token, dest, tmp_dir, log_path, tag):
    """
    Скачать датасет из HF в dest. Файлы переименовываются:
      • картинки/аудио-реплики → NNN.ext (раздельно, единый порядок)
      • reference.*/prompts.json/silence_latent_frame.pt → по имени (опорные для Dramabox)
    Возвращает список путей к картинкам в dest (для превью, если нужно).
    """
    token = (hf_token or "").strip()
    sf = (subfolder or "").strip()
    dest = Path(dest)
    tmp_dir = Path(tmp_dir)

    # очистка
    for d in (dest, tmp_dir):
        if d.exists():
            shutil.rmtree(d)
            if d.exists() and any(d.iterdir()):
                raise RuntimeError(f"Не удалось очистить {d}")
        d.mkdir(parents=True, exist_ok=True)

    cmd = ["hf", "download", repo_id.strip(), "--repo-type", "dataset",
           "--local-dir", str(tmp_dir)]
    if sf:
        cmd += ["--include", f"{sf}/*"]
    if token:
        cmd += ["--token", token]

    log(log_path, tag, f"Скачивание {repo_id}/{sf or '(root)'} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or "unknown error").strip()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"hf download failed: {err}")

    search_root = tmp_dir / sf if sf and (tmp_dir / sf).is_dir() else tmp_dir
    if not search_root.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(f"Подпапка '{sf}' не найдена в {repo_id}")

    files = sorted(p for p in search_root.iterdir() if p.is_file())
    if not files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(f"Подпапка '{sf}' пуста в {repo_id}")

    IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    AUD = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    KEEP = {".txt", ".json", ".pt"}

    def _keep_by_name(f):
        dst = dest / f.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(f), str(dst))

    # служебные .txt/.json/.pt — по имени
    keep_files = [f for f in files if f.suffix.lower() in KEEP]
    for f in keep_files:
        _keep_by_name(f)

    # медиа
    img_files = [f for f in files if f.suffix.lower() in IMG]
    aud_files = [f for f in files if f.suffix.lower() in AUD]

    # опорные (reference.*) из медиа — по имени, не нумеровать
    for f in list(img_files) + list(aud_files):
        if f.stem.lower() in PROTECTED_STEMS:
            _keep_by_name(f)
    img_files = [f for f in img_files if f.stem.lower() not in PROTECTED_STEMS]
    aud_files = [f for f in aud_files if f.stem.lower() not in PROTECTED_STEMS]

    for idx, f in enumerate(img_files, start=1):
        dst = dest / f"{idx:03d}{f.suffix.lower()}"
        if dst.exists():
            dst.unlink()
        shutil.move(str(f), str(dst))
    for idx, f in enumerate(aud_files, start=1):
        dst = dest / f"{idx:03d}{f.suffix.lower()}"
        if dst.exists():
            dst.unlink()
        shutil.move(str(f), str(dst))

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # статистика
    file_list = [p for p in dest.rglob("*") if p.is_file()
                 and p.name not in (".gitattributes", ".gitignore")]
    if not file_list:
        raise RuntimeError("Файлы не скачались — папка датасета пуста")
    total_mb = sum(f.stat().st_size for f in file_list) / 1024 / 1024
    log(log_path, tag, f"Скачано {len(file_list)} файлов ({total_mb:.1f} MB) → {dest}")
    return [dest / f"{i:03d}{Path(f).suffix.lower()}" for i, f in enumerate(img_files, 1)]


def hf_upload(repo_id, subfolder, hf_token, src_dir, include_patterns, log_path, tag):
    """Залить файлы из src_dir в HF-репо (dataset или model) в подпапку."""
    token = (hf_token or "").strip()
    sf = (subfolder or "").strip()
    src = Path(src_dir)

    cmd = ["hf", "upload", repo_id.strip(), str(src)]
    if sf:
        cmd += [sf]
    cmd += ["--include", *include_patterns]
    if token:
        cmd += ["--token", token]

    log(log_path, tag, f"Заливка в {repo_id}/{sf or '(root)'} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or "unknown error").strip()
        log(log_path, tag, f"WARN: hf upload failed: {err}")
        return False
    return True


# ---------------------------------------------------------------------------
# Dataset Builder: инвентаризация → переименование → выравнивание → buckets → dataset.json
# ---------------------------------------------------------------------------

def _pick_bucket(width, height):
    ratio = width / height
    area = width * height

    def score(b):
        bw, bh = b
        br = bw / bh
        ba = bw * bh
        return abs(ratio - br) / max(ratio, br) * 2.0 + abs(area - ba) / max(area, ba)

    return min(ALLOWED_BUCKETS, key=score)


def build_dataset(dataset_dir, trigger_word, log_path, tag):
    """
    Полная логика DatasetBuilder:
      1. Чистит мусор (caption.txt, prompts.json, reference.txt, *.mp3).
      2. Переименовывает картинки/аудио в NNNN.ext (идёмпотентно).
      3. Выравнивает количество (min(images, audios)), лишнее удаляет.
      4. Выбирает resolution buckets по разрешениям картинок.
      5. Собирает dataset.json (caption из .txt, триггер-слово удаляется из подписи).

    Возвращает (dataset_json_path, buckets_str, count, image_paths).
    """
    dataset_dir = Path(dataset_dir)
    log(log_path, tag, f"Начало обработки {dataset_dir}")

    # === чистка мусора ===
    removed = []
    for name in os.listdir(dataset_dir):
        full = dataset_dir / name
        if not full.is_file():
            continue
        ext = full.suffix.lower()
        if name in JUNK_FILES or ext in JUNK_EXTS or full.stem.lower() in PROTECTED_STEMS:
            try:
                os.remove(full)
                removed.append(name)
            except OSError as e:
                log(log_path, tag, f"не удалось удалить {name}: {e}")
    if removed:
        log(log_path, tag, f"Удалены лишние файлы: {', '.join(removed)}")

    # === инвентаризация ===
    files = sorted(os.listdir(dataset_dir))
    images = sorted(f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    audios = sorted(f for f in files if os.path.splitext(f)[1].lower() in AUDIO_EXTS)
    log(log_path, tag, f"Найдено картинок: {len(images)}, аудио: {len(audios)}")

    # === переименование в NNNN.ext (идемпотентно) ===
    already = bool(images or audios) and all(
        re.match(r"^\d{3,4}\.", f) for f in images + audios
    )
    if not already:
        log(log_path, tag, "Переименование в формат NNNN")
        tmp_prefix = "__tmp_ltx_rename__"

        def rename_group(file_list):
            # фаза 1 → tmp
            stage = []
            for i, fn in enumerate(file_list):
                src = dataset_dir / fn
                tmp = dataset_dir / f"{tmp_prefix}{i}_{fn}"
                os.rename(src, tmp)
                stage.append((tmp, fn))
            # фаза 2 → финальное имя + парный .txt
            for i, (tmp_path, original) in enumerate(stage, start=1):
                base, ext = os.path.splitext(original)
                new_name = f"{i:04d}{ext.lower()}"
                os.rename(tmp_path, dataset_dir / new_name)
                txt_src = dataset_dir / (base + ".txt")
                if txt_src.exists():
                    txt_dst = dataset_dir / f"{i:04d}.txt"
                    if txt_src != txt_dst:
                        if txt_dst.exists():
                            os.remove(txt_dst)
                        os.rename(txt_src, txt_dst)

        rename_group(images)
        rename_group(audios)

        files = sorted(os.listdir(dataset_dir))
        images = sorted(f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        audios = sorted(f for f in files if os.path.splitext(f)[1].lower() in AUDIO_EXTS)

    # === выравнивание ===
    min_count = min(len(images), len(audios))
    if min_count == 0:
        raise RuntimeError(f"Пустой датасет: images={len(images)}, audios={len(audios)}")

    extra = []
    for name in images[min_count:] + audios[min_count:]:
        full = dataset_dir / name
        base = os.path.splitext(name)[0]
        try:
            os.remove(full)
            extra.append(name)
            txt = dataset_dir / (base + ".txt")
            if txt.exists():
                os.remove(txt)
        except OSError as e:
            log(log_path, tag, f"не удалось удалить {name}: {e}")
    if extra:
        log(log_path, tag, f"Удалены лишние пары: {', '.join(extra)}")

    images = images[:min_count]
    audios = audios[:min_count]

    # === buckets ===
    from PIL import Image as _PILImage  # локальный импорт — нужен только здесь
    chosen = set()
    for img_name in images:
        try:
            with _PILImage.open(dataset_dir / img_name) as im:
                w, h = im.size
            chosen.add(_pick_bucket(w, h))
        except Exception as e:
            log(log_path, tag, f"не удалось прочитать {img_name}: {e}")
    if not chosen:
        chosen.add((512, 512))
    buckets_str = ";".join(f"{w}x{h}x1" for w, h in sorted(chosen))
    log(log_path, tag, f"Выбраны buckets: {buckets_str}")

    # === dataset.json ===
    entries = []
    trig = (trigger_word or "").strip()
    trig_re = re.compile(rf"\b{re.escape(trig)}\b", re.IGNORECASE) if trig else None

    for img_name, audio_name in zip(images, audios):
        base = os.path.splitext(img_name)[0]
        txt_path = dataset_dir / (base + ".txt")
        caption = ""
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            log(log_path, tag, f"нет .txt для {img_name}")

        if trig_re and trig_re.search(caption):
            caption = trig_re.sub("", caption)
            caption = re.sub(r"\s+", " ", caption).strip(" ,.;:-")

        entries.append({
            "caption": caption,
            "video": str(dataset_dir / img_name),
            "audio": str(dataset_dir / audio_name),
        })

    output_json = str(dataset_dir / "dataset.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    log(log_path, tag, f"Датасет готов: {len(entries)} пар → {output_json}")

    image_paths = [str(dataset_dir / n) for n in images]
    return output_json, buckets_str, len(entries), image_paths


# ---------------------------------------------------------------------------
# subprocess с логированием
# ---------------------------------------------------------------------------

def run_logged(cmd, cwd, log_path, header, tag):
    """Запустить subprocess, перенаправив stdout+stderr в лог-файл. Возвращает CompletedProcess."""
    log(log_path, tag, f"cwd: {cwd}")
    log(log_path, tag, f"cmd: {' '.join(cmd)}")
    with open(log_path, "a", encoding="utf-8") as lf:
        if header:
            lf.write(f"\n--- {header} ---\n")
        lf.flush()
        result = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    return result


# ---------------------------------------------------------------------------
# LTX-2.3 валидации
# ---------------------------------------------------------------------------

def validate_resolution(width, height):
    if width % 32 != 0 or height % 32 != 0:
        raise RuntimeError(
            f"width/height должны быть кратны 32 (получено {width}x{height})"
        )


def validate_frames(frames):
    if frames != 1 and (frames - 1) % 8 != 0:
        raise RuntimeError(
            f"frames должно быть 1 (картинка) или frames%8==1 (9, 17, 25, 33...). "
            f"Получено: {frames}"
        )
