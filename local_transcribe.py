#!/usr/bin/env python3
import os, sys, tempfile
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

# ---------------- config ----------------
MODEL_SIZE   = os.environ.get("FW_MODEL", "small")
DEVICE       = os.environ.get("FW_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("FW_COMPUTE", "int8_float16")

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

def _transcribe_segments(model, audio, sr, use_vad=True):
    """Yield segment texts lazily from a numpy audio array."""
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
        for segment in segments:
            yield segment.text

def transcribe_stream(path, model=None):
    """Yield transcript text pieces as they are decoded (streaming)."""
    audio, sr = load_audio_mono_16k(path)
    if len(audio) == 0:
        return

    if model is None:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    # One call for the whole file: faster-whisper windows it into 30s pieces itself.
    # Manual 12s slicing cost ~4.4s per slice regardless of content (every call
    # encodes a padded 30s window) and made long recordings ~4x slower.
    yield from _transcribe_segments(model, audio, sr, use_vad=True)

def main(path):
    """Return full transcript as a single string (non-streaming)."""
    return "".join(transcribe_stream(path)).strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: local_transcribe.py audiofile [--wait]", file=sys.stderr)
        sys.exit(2)

    audio_path = sys.argv[1]
    wait_mode = "--wait" in sys.argv

    if wait_mode:
        # Preload model immediately, then wait for the audio file to appear.
        # This lets main.sh start us in the background during recording so
        # model load time overlaps with the user speaking.
        import time
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("[model loaded]", file=sys.stderr, flush=True)
        while not os.path.exists(audio_path):
            time.sleep(0.1)
        for text in transcribe_stream(audio_path, model=model):
            print(text, end="", flush=True)
    else:
        for text in transcribe_stream(audio_path):
            print(text, end="", flush=True)

    print()  # final newline
