нода DramaBox лучше всего вызывать через subprocess
https://huggingface.co/ResembleAI/Dramabox
Dramabox is a prompt-driven TTS where the prompt itself controls everything — speaker identity, emotion, delivery, laughs, sighs, breaths, pauses, transitions. An optional 10-second voice reference clones the target timbre. It is an IC-LoRA fine-tune of the LTX-2.3 3.3B audio-only model (Diffusion Transformer + flow matching), conditioned on Gemma 3 12B text embeddings.
	
🤗 Model 	ResembleAI/Dramabox
🎭 Demo Space 	ResembleAI/Dramabox (ZeroGPU)
💻 Code 	resemble-ai/DramaBox
🏗️ Base model 	Lightricks/LTX-2.3
📜 License 	LTX-2 Community License — see LICENSE
Quick start
Python (warm server — recommended, ~2.5 s / generation)

from src.inference_server import TTSServer

server = TTSServer(device="cuda")              # downloads weights on first run

server.generate_to_file(
    prompt='A woman speaks warmly, "Hello, how are you today?" '
           'She laughs, "Hahaha, it is so good to see you!"',
    output="output.wav",
    voice_ref="reference.wav",                  # optional, 10+ seconds of target voice
    cfg_scale=2.5,
    stg_scale=1.5,
    duration_multiplier=1.1,
    seed=42,
)

CLI

python src/inference.py \
    --prompt 'A woman speaks warmly, "Hello, how are you today?"' \
    --voice-sample reference.wav \
    --output output.wav \
    --cfg-scale 2.5 --stg-scale 1.5

Inference parameters
Parameter 	Default 	What it does
prompt 	— 	The scene description. Dialogue inside "double quotes", stage directions outside. See "Prompt format" below.
voice_ref (--voice-sample) 	None 	Optional 10+ s audio clip whose timbre the model clones. Without it, the model picks a voice that fits the description.
cfg_scale 	2.5 	Classifier-free guidance — how strictly the output follows the prompt. Lower = more natural, higher = more text-faithful but more dramatic. Auto-rescaled internally to prevent clipping at high cfg (see Auto rescale below).
stg_scale 	1.5 	Skip-token guidance — applied through the perturbed transformer block path (block 29). Increases expressive emphasis without saturating like cfg.
duration_multiplier (--duration-multiplier) 	1.1 	Multiplier on the auto-estimated speech length (10 % breathing-room headroom). Only used when gen_duration (or --gen-duration) is 0.
gen_duration (--gen-duration, "Target duration" slider) 	0 (auto) 	Explicit output duration in seconds. Set to 20–60 s for music or long scenes. Overrides the prompt-based estimate when > 0.
ref_duration (--ref-duration, "Reference duration" slider) 	10.0 	How many seconds of the voice reference the model conditions on (3–30 s). Longer ref → richer timbre capture, shorter ref → faster encode.
seed 	42 	Reproducibility.
rescale_scale (--rescale-scale) 	"auto" 	Latent-side CFG std-rescale. The default is a cfg-aware schedule (0 below cfg=2, ramping to 1.0 by cfg=10) that keeps the output peak below 0 dBFS at every cfg. Pass any float in [0, 1] to override or 0 to disable.
watermark (--no-watermark to disable) 	True 	Apply Resemble Perth imperceptible neural watermark to the output. Survives MP3/AAC, common edits; ≈ 100 % detection accuracy.


и надо сделать ноду pyptv_ltx23_upload_audio_node.py грузиим аудиофайлы в репо huggingface 
в нужный датасет - типа 001.wav 002.wav 


референс аудио будет как опция


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS        = {"Dramabox_pyPTV": Dramabox_pyPTV}
NODE_DISPLAY_NAME_MAPPINGS = {"Dramabox_pyPTV": "Dramabox (pyPTV)"}

