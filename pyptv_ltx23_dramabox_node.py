"""
DramaBox Voice Generation (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Автономный поток генерации голоса через DramaBox (ResembleAI/LTX-2.3 TTS).

Полный цикл замкнут внутри ноды — никаких входных коннектов от пайплайна
тренировки и никаких выходных сокетов. Запускаешь ноду отдельно, она делает всё
сама, а результат (лог генерации) показывает в себе.

Что делает (по ТЗ):
  1. Скачивает prompts.json + reference.wav из HF-датасета (hf download).
  2. Прогоняет каждый промпт через DramaBox (src/inference.py) с референсом голоса.
  3. Получает WAV-файлы по числу промптов (001.wav, 002.wav, ...).
  4. Заливает эти аудиофайлы обратно в HF-датасет.

DramaBox запускается через subprocess — нода тонкая обёртка, никакого импорта
её внутренностей напрямую. Модели НЕ скачиваются нодой — только пути.

Формат промптов — DramaBox TTS, диалоги в "двойных кавычках", ремарки снаружи.
Пример: 'A woman speaks warmly, "Hello, how are you today?"'

UI-параметры:
  • repo_id          — HF dataset repo, откуда брать prompts.json + reference.wav
  • subfolder        — подпапка внутри репо (или "" для корня)
  • hf_token         — токен HF (для приватных репо)
  • dramabox_repo_path — путь к клонированному репо DramaBox (с ltx2/, src/)
  • full_checkpoint  — путь к ltx-2.3-22b-dev.safetensors (вес для Dramabox)
  • gemma_root       — путь к каталогу Gemma-энкодера
  • reference_wav    — имя файла референса голоса в репо (default: reference.wav)
  • output_dir       — куда складывать сгенерированные WAV (default: /home/dramabox_out)
  • cfg_scale, stg_scale, ref_duration, seed — параметры генерации

Результат:
  • лог процесса отображается в самой ноде (ui.text), выходных сокетов нет.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path


TAG = "Dramabox"


def _log(log_path: str, msg: str):
    line = f"[{TAG}] {msg}"
    print(line)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _load_prompts(prompts_json_path: str):
    """
    prompts.json может быть либо списком строк, либо списком объектов
    с ключом prompt/text, либо {prompt: str} / {prompts: [..]}. Поддерживаем всё.
    Возвращаем список строк-промптов.
    """
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Объект-обёртка
    if isinstance(data, dict):
        if isinstance(data.get("prompts"), list):
            data = data["prompts"]
        elif "prompt" in data:
            data = [data["prompt"]]
        else:
            # fallback: значения-строки
            data = list(data.values())

    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            txt = item.get("prompt") or item.get("text") or item.get("caption")
            if txt:
                prompts.append(txt)
    return prompts


class Dramabox_pyPTV:
    """Генерация голоса через DramaBox (subprocess-обёртка над src/inference.py)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── HF датасет ──
                "repo_id": ("STRING", {
                    "default": "username/datasets",
                    "multiline": False,
                    "tooltip": "HF dataset repo, откуда брать prompts.json + reference.wav",
                }),
                "subfolder": ("STRING", {
                    "default": "mydataset",
                    "multiline": False,
                    "tooltip": "Подпапка внутри репо (или пусто для корня)",
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace access token (для приватных репо)",
                }),
                # ── DramaBox ──
                "dramabox_repo_path": ("STRING", {
                    "default": "/home/DramaBox",
                    "multiline": False,
                    "tooltip": "Путь к клонированному репо DramaBox (с src/ и ltx2/)",
                }),
                "full_checkpoint": ("STRING", {
                    "default": "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors",
                    "multiline": False,
                    "tooltip": "Путь к ltx-2.3-22b-dev.safetensors",
                }),
                "gemma_root": ("STRING", {
                    "default": "/comfyui/models/text_encoders/gemma-3-12b-it-qat",
                    "multiline": False,
                    "tooltip": "Путь к каталогу Gemma-энкодера",
                }),
                "reference_wav": ("STRING", {
                    "default": "reference.wav",
                    "multiline": False,
                    "tooltip": "Имя файла референса голоса в датасете",
                }),
                "output_dir": ("STRING", {
                    "default": "/home/dramabox_out",
                    "multiline": False,
                    "tooltip": "Куда складывать сгенерированные WAV",
                }),
                "log_file": ("STRING", {
                    "default": "/home/dramabox.log",
                    "multiline": False,
                }),
                # ── Параметры генерации ──
                "cfg_scale": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG scale: ниже = естественнее, выше = точнее по тексту",
                }),
                "stg_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Skip-token guidance (блок 29) — выразительность",
                }),
                "ref_duration": ("FLOAT", {
                    "default": 10.0, "min": 3.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Сколько секунд референса использовать (3–30с)",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**31 - 1,
                }),
            }
        }

    # Автономная нода: полный цикл (скачать → сгенерить → залить) замкнут внутри неё.
    # Нет выходных сокетов и нет входных коннектов от пайплайна — только UI-параметры.
    # Результат (сгенерировано/залито/лог) показывается в самой ноде через ui.
    RETURN_TYPES = ()
    FUNCTION = "generate"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def generate(self, repo_id, subfolder, hf_token, dramabox_repo_path,
                 full_checkpoint, gemma_root, reference_wav, output_dir, log_file,
                 cfg_scale, stg_scale, ref_duration, seed):

        # сброс лога
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            open(log_file, "w").close()

        db_repo = Path(dramabox_repo_path)
        inference_script = db_repo / "src" / "inference.py"
        if not inference_script.is_file():
            raise RuntimeError(f"Не найден {inference_script}. Проверьте dramabox_repo_path.")
        if not os.path.isfile(full_checkpoint):
            raise RuntimeError(f"full_checkpoint не найден: {full_checkpoint}")
        if not os.path.isdir(gemma_root):
            raise RuntimeError(f"gemma_root не найден: {gemma_root}")

        out_dir = Path(output_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Шаг 1: скачать prompts.json + reference.wav из HF ──
        tmp_dir = Path("/home/dramabox_download")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        sf = subfolder.strip()
        token = hf_token.strip()
        # --include работает относительно репо; подтягиваем только нужные файлы.
        # Кладём всё в tmp_dir и далее ищем по подпапке.
        dl_cmd = [
            "hf", "download", repo_id.strip(),
            "--repo-type", "dataset",
            "--local-dir", str(tmp_dir),
        ]
        if sf:
            dl_cmd += ["--include", f"{sf}/*"]
        if token:
            dl_cmd += ["--token", token]

        _log(log_file, f"Скачивание prompts.json + {reference_wav} из {repo_id}/{sf} ...")
        result = subprocess.run(dl_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or "unknown error").strip()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"hf download failed: {err}")

        search_root = tmp_dir / sf if sf and (tmp_dir / sf).is_dir() else tmp_dir
        prompts_path = None
        for p in search_root.rglob("prompts.json"):
            prompts_path = p
            break
        if prompts_path is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError("prompts.json не найден в скачанном датасете")

        ref_path = None
        for p in search_root.rglob("*"):
            if p.is_file() and p.name.lower() == reference_wav.lower():
                ref_path = p
                break
        # референс опционален — Dramabox умеет без него (raw base model)
        if ref_path:
            _log(log_file, f"Референс голоса: {ref_path}")
        else:
            _log(log_file, f"WARN: {reference_wav} не найден — генерация без voice ref")

        # ── Шаг 2: прогоны промптов ──
        prompts = _load_prompts(str(prompts_path))
        if not prompts:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError("В prompts.json нет ни одного промпта")
        _log(log_file, f"Промптов к генерации: {len(prompts)}")

        generated = 0
        for idx, prompt in enumerate(prompts, start=1):
            wav_name = f"{idx:03d}.wav"
            wav_path = out_dir / wav_name

            cmd = [
                "python", "src/inference.py",
                "--prompt", prompt,
                "--output", str(wav_path),
                "--full-checkpoint", full_checkpoint,
                "--gemma-root", gemma_root,
                "--cfg-scale", str(cfg_scale),
                "--stg-scale", str(stg_scale),
                "--stg-block", "29",
                "--ref-duration", str(ref_duration),
                "--seed", str(seed),
            ]
            if ref_path:
                cmd += ["--voice-sample", str(ref_path)]
            else:
                cmd += ["--no-ref"]

            _log(log_file, f"[{idx}/{len(prompts)}] {wav_name}  ←  {prompt[:80]}")

            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"\n--- inference.py [{wav_name}] ---\n")
                lf.flush()
                r = subprocess.run(
                    cmd,
                    cwd=str(db_repo),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )

            if r.returncode != 0:
                _log(log_file, f"ОШИБКА генерации {wav_name} (returncode={r.returncode})")
                raise RuntimeError(
                    f"DramaBox inference failed для {wav_name}, см. {log_file}"
                )
            if not wav_path.is_file():
                raise RuntimeError(f"WAV не создан: {wav_path}, см. {log_file}")
            generated += 1

        _log(log_file, f"Сгенерировано {generated} WAV → {out_dir}")

        # ── Шаг 3: залить обратно в HF-датасет ──
        up_cmd = [
            "hf", "upload", repo_id.strip(),
            str(out_dir),
        ]
        # подпапка как директория назначения
        if sf:
            up_cmd += [sf]
        up_cmd += ["--include", "*.wav"]
        if token:
            up_cmd += ["--token", token]

        _log(log_file, f"Заливка {generated} WAV в {repo_id}/{sf} ...")
        up_result = subprocess.run(up_cmd, capture_output=True, text=True)
        if up_result.returncode != 0:
            err = (up_result.stderr or "unknown error").strip()
            _log(log_file, f"WARN: hf upload failed: {err}")
            # не падаем — аудио уже сгенерированы локально, юзер может залить вручную

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # ── лог для возврата в UI ──
        log_text = ""
        if log_file and os.path.isfile(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                log_text = f.read()

        summary = f"Сгенерировано и залито: {generated} WAV → {repo_id.strip()}/{sf or '(root)'}"
        return {"ui": {"text": [log_text], "summary": [summary]}, "result": ()}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS        = {"Dramabox_pyPTV": Dramabox_pyPTV}
NODE_DISPLAY_NAME_MAPPINGS = {"Dramabox_pyPTV": "Dramabox Voice Gen (pyPTV)"}
