#!/usr/bin/env python3
"""
Parakeet TDT 0.6B v2 transcription backend.
Drop-in replacement for local_transcribe.py — same CLI interface.

Usage:
    parakeet_transcribe.py audiofile
    parakeet_transcribe.py audiofile --wait   # preload model, wait for file
"""
import os
import sys
import tempfile
import subprocess
import shutil

import numpy as np
import soundfile as sf

MODEL_NAME = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
DEVICE     = os.environ.get("PARAKEET_DEVICE", "cuda")

# Energy VAD: skip chunks that are mostly silence (RMS below threshold)
VAD_ENERGY_THRESHOLD = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.005"))


def load_audio_mono_16k(path):
    """Load audio as float32 mono at 16kHz. Uses ffmpeg if available."""
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-i", path,
            "-ar", "16000", "-ac", "1", "-f", "f32le", "-",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        audio = np.frombuffer(result.stdout, dtype="float32")
        return audio, 16000
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import math
        n = math.floor(len(audio) * (16000 / sr))
        audio = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(audio)), audio).astype("float32")
    return audio, 16000


def load_model():
    import nemo.collections.asr as nemo_asr
    # Load to CPU first — float32 model is ~2.5GB, exceeds 4GB VRAM if loaded directly
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu")
    # float16 halves VRAM to ~1.2GB, fits comfortably on 4GB
    model = model.half()
    model = model.to(DEVICE)
    model.eval()
    return model


def transcribe(path, model=None):
    """Transcribe audio file, return text string."""
    if model is None:
        model = load_model()

    audio, sr = load_audio_mono_16k(path)
    if len(audio) == 0:
        return ""

    # Energy VAD: bail early on silent clips
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < VAD_ENERGY_THRESHOLD:
        return ""

    # Parakeet wants a wav file at 16kHz mono
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, sr)
        output = model.transcribe([f.name])

    # NeMo returns a list of Hypothesis objects or strings depending on version
    result = output[0]
    if hasattr(result, "text"):
        return result.text.strip()
    return str(result).strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: parakeet_transcribe.py audiofile [--wait]", file=sys.stderr)
        sys.exit(2)

    audio_path = sys.argv[1]
    wait_mode = "--wait" in sys.argv

    if wait_mode:
        import time
        model = load_model()
        print("[model loaded]", file=sys.stderr, flush=True)
        while not os.path.exists(audio_path):
            time.sleep(0.1)
        print(transcribe(audio_path, model=model))
    else:
        print(transcribe(audio_path))
