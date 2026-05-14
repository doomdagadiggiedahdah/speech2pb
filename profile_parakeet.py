#!/usr/bin/env python3
"""
Profile each step of the Parakeet pipeline to find bottlenecks.
Generates a 5-second sine-wave audio clip for transcription timing.
Run with: .venv/bin/python profile_parakeet.py
"""
import time
import numpy as np

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
SAMPLE_RATE = 16000

def t(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"  {elapsed:6.2f}s  {label}")
    return result

def make_audio(seconds=5):
    """Sine wave — gives the model something non-silent to chew on."""
    t_arr = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), dtype=np.float32)
    return 0.1 * np.sin(2 * np.pi * 440 * t_arr)

def transcribe_array(model, audio):
    import tempfile, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        output = model.transcribe([f.name])
    result = output[0]
    return (result.text if hasattr(result, "text") else str(result)).strip()

print("\n=== Parakeet pipeline profiler ===\n")

print("[1] Model load")
import nemo.collections.asr as nemo_asr

model = t("from_pretrained (CPU, float32)",
          lambda: nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu"))

model = t(".half()  (CPU float32 -> float16)",
          lambda: model.half())

model = t(".to('cuda')  (CPU -> GPU transfer)",
          lambda: model.to("cuda"))

t(".eval()",
  lambda: model.eval())

print()
print("[2] Transcription latency")

audio_5s  = make_audio(5)
audio_8s  = make_audio(8)
audio_30s = make_audio(30)

# Warm-up run (first CUDA kernel launch is always slow)
t("warm-up transcribe (5s audio, first call)",
  lambda: transcribe_array(model, audio_5s))

t("transcribe 5s audio",
  lambda: transcribe_array(model, audio_5s))

t("transcribe 8s audio  (CHUNK_SEC default)",
  lambda: transcribe_array(model, audio_8s))

t("transcribe 30s audio (long clip)",
  lambda: transcribe_array(model, audio_30s))

print()
print("[3] tempfile overhead")
import tempfile, soundfile as sf

def write_wav(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, SAMPLE_RATE)

t("write 8s wav to tempfile",
  lambda: write_wav(audio_8s))

print()
print("[4] VRAM usage")
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
used, total = result.stdout.strip().split(", ")
print(f"  {used} / {total} MiB used")

print("\n=== done ===\n")
