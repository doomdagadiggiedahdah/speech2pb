#!/usr/bin/env python3
"""
Parakeet eager transcription.
Reads raw PCM float32 16kHz mono from stdin. Transcribes chunks in the
background while recording continues; only the tail needs processing after
the hotkey is released.

Usage (from local_transcribe.sh):
    ffmpeg -f alsa -i default -ar 16000 -ac 1 -f f32le FIFO &
    parakeet_eager.py < FIFO

Latency model:
  - For clips < CHUNK_SEC: equivalent to parakeet_transcribe.py --wait
  - For clips > CHUNK_SEC: each chunk is transcribed while recording continues;
    post-recording work is just the last partial chunk (~OVERLAP_SEC of audio)
"""
import sys
import os
import threading
import numpy as np
import tempfile
import soundfile as sf

SAMPLE_RATE       = 16000
MODEL_NAME        = os.environ.get("PARAKEET_MODEL",  "nvidia/parakeet-tdt-0.6b-v2")
DEVICE            = os.environ.get("PARAKEET_DEVICE", "cuda")
CHUNK_SEC         = float(os.environ.get("CHUNK_SEC",  "8.0"))
OVERLAP_SEC       = float(os.environ.get("OVERLAP_SEC", "1.0"))
VAD_THRESHOLD     = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.005"))

CHUNK_SAMPLES   = int(CHUNK_SEC   * SAMPLE_RATE)
OVERLAP_SAMPLES = int(OVERLAP_SEC * SAMPLE_RATE)


def load_model():
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu")
    model = model.half().to(DEVICE).eval()
    return model


def transcribe_array(model, audio):
    if float(np.sqrt(np.mean(audio ** 2))) < VAD_THRESHOLD:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        output = model.transcribe([f.name])
    result = output[0]
    return (result.text if hasattr(result, "text") else str(result)).strip()


def _norm(word):
    """Lowercase + strip punctuation for boundary comparison."""
    return word.lower().strip(".,!?;:'\"")

def dedup_join(text_a, text_b):
    """
    Join two transcripts by finding the longest common contiguous word substring
    between the tail of text_a and anywhere in text_b, then splicing.

    Uses the same LCS-suffix algorithm as NeMo's longest_common_subsequence_merge,
    but at the word level with punctuation-normalized comparison. This handles
    Parakeet's context-dependent punctuation and capitalisation at chunk boundaries.
    """
    if not text_a:
        return text_b
    if not text_b:
        return text_a

    words_a = text_a.split()
    words_b = text_b.split()
    norm_a = [_norm(w) for w in words_a]
    norm_b = [_norm(w) for w in words_b]

    # Only search the tail of A (corresponding to the overlap window)
    search_a = norm_a[-30:]
    offset_a = len(norm_a) - len(search_a)

    m, n = len(search_a), len(norm_b)
    # LCSuff[i][j] = length of common substring ending at search_a[i-1], norm_b[j-1]
    LCSuff = [[0] * (n + 1) for _ in range(m + 1)]
    best_len = 0
    best_i = 0  # end of match in search_a (exclusive, 1-indexed)
    best_j = 0  # end of match in norm_b (exclusive, 1-indexed)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if search_a[i - 1] == norm_b[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                if LCSuff[i][j] > best_len:
                    best_len = LCSuff[i][j]
                    best_i = i
                    best_j = j
            else:
                LCSuff[i][j] = 0

    if best_len < 1:
        return text_a + " " + text_b

    # Keep words_a up through end of match, then words_b after end of match
    keep_a = offset_a + best_i          # absolute index into words_a (exclusive)
    keep_b = best_j                     # start of non-overlapping tail in words_b
    return " ".join(words_a[:keep_a] + words_b[keep_b:])


def main():
    # Buffer stdin immediately so ffmpeg never blocks on the pipe
    audio_buffer = np.array([], dtype=np.float32)
    buffer_lock  = threading.Lock()
    stdin_done   = threading.Event()

    def read_stdin():
        nonlocal audio_buffer
        while True:
            data = sys.stdin.buffer.read(SAMPLE_RATE // 10 * 4)  # 100ms at float32
            if not data:
                break
            chunk = np.frombuffer(data, dtype=np.float32).copy()
            with buffer_lock:
                audio_buffer = np.concatenate([audio_buffer, chunk])
        stdin_done.set()

    threading.Thread(target=read_stdin, daemon=True).start()

    model = load_model()
    print("[model loaded]", file=sys.stderr, flush=True)

    result_parts   = []
    processed_upto = 0  # samples already handed off for transcription

    while True:
        with buffer_lock:
            buf_len = len(audio_buffer)

        if buf_len - processed_upto >= CHUNK_SAMPLES:
            # Grab chunk with overlap from previous boundary
            chunk_start = max(0, processed_upto - OVERLAP_SAMPLES)
            with buffer_lock:
                chunk = audio_buffer[chunk_start : processed_upto + CHUNK_SAMPLES].copy()

            text = transcribe_array(model, chunk)
            result_parts.append(text)
            processed_upto += CHUNK_SAMPLES
            print(f"[eager] {text[:80]}", file=sys.stderr, flush=True)

        elif stdin_done.is_set():
            break
        else:
            stdin_done.wait(timeout=0.1)

    # Final chunk: audio after the last completed chunk boundary
    with buffer_lock:
        remaining = audio_buffer[max(0, processed_upto - OVERLAP_SAMPLES):].copy()

    if len(remaining) > SAMPLE_RATE // 4:  # skip if < 0.25s
        text = transcribe_array(model, remaining)
        result_parts.append(text)

    # Dedup-join all parts into final transcript
    result = ""
    for part in result_parts:
        result = dedup_join(result, part)

    print(result.strip())


if __name__ == "__main__":
    main()
