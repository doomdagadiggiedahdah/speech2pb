#!/usr/bin/env python3
import os, sys, tempfile
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

# ---------------- config ----------------
MODEL_SIZE   = os.environ.get("FW_MODEL", "small")
DEVICE       = os.environ.get("FW_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("FW_COMPUTE", "int8_float16")

CHUNK_SEC    = float(os.environ.get("CHUNK_SEC", "12.0"))   # slightly longer default
OVERLAP_SEC  = float(os.environ.get("OVERLAP_SEC", "1.0"))
BEAM_SIZE    = int(os.environ.get("BEAM_SIZE", "3"))

# Less aggressive VAD defaults
VAD_MIN_SILENCE_MS = int(os.environ.get("VAD_MIN_SILENCE_MS", "150"))
VAD_SPEECH_PAD_MS  = int(os.environ.get("VAD_SPEECH_PAD_MS", "400"))
DISABLE_VAD        = os.environ.get("DISABLE_VAD", "0") == "1"

def load_audio_mono_16k(path):
    # Use ffmpeg to decode any format to raw pcm, then fall back to soundfile for proper wavs
    import subprocess, shutil
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-i", path,
            "-ar", "16000", "-ac", "1", "-f", "f32le", "-",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        audio = np.frombuffer(result.stdout, dtype="float32")
        return audio, 16000
    # fallback: soundfile (only works for standard wav/flac/ogg)
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import math
        ratio = 16000 / sr
        n = math.floor(len(audio) * ratio)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, n)
        audio = np.interp(x_new, x_old, audio).astype("float32")
        sr = 16000
    return audio, 16000

def transcribe_array(model, audio, sr, use_vad=True):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, sr)
        segments, _ = model.transcribe(
            f.name,
            beam_size=BEAM_SIZE,
            vad_filter=use_vad and not DISABLE_VAD,
            vad_parameters={
                "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
                "speech_pad_ms": VAD_SPEECH_PAD_MS,
            } if (use_vad and not DISABLE_VAD) else None,
            temperature=0.0,
        )
        return "".join(s.text for s in segments).strip()

def main(path):
    audio, sr = load_audio_mono_16k(path)
    if len(audio) == 0:
        return ""

    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    total_sec = len(audio) / sr

    # If it's short, don't chunk. Just one pass with VAD.
    if total_sec <= CHUNK_SEC * 1.2:
        return transcribe_array(model, audio, sr, use_vad=True)

    # For long speech: chunk *after* VAD has already helped on each chunk.
    texts = []
    start = 0.0
    while start < total_sec:
        end = min(total_sec, start + CHUNK_SEC)
        s = int(start * sr)
        e = int(end * sr)
        chunk = audio[s:e]

        txt = transcribe_array(model, chunk, sr, use_vad=True)
        if txt:
            texts.append(txt)

        if end >= total_sec:
            break
        start = max(0.0, end - OVERLAP_SEC)

    return " ".join(texts).strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: local_transcribe.py audiofile", file=sys.stderr)
        sys.exit(2)
    print(main(sys.argv[1]))
