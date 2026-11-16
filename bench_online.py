#!/usr/bin/env python3
"""Replay benchmark: transcribe *while recording* with committed-prefix seeking.

Unlike bench_chunked.py (independent fixed cuts, no context), this mirrors what
faster-whisper does internally on a whole file: decode a ~30s window, keep every
segment except the last (possibly cut-off) one, seek to the end of the last kept
segment, and prompt the next window with the text committed so far. The only
difference online is that a window is decoded as soon as its audio *exists*.

For each archived recording:
  BASELINE  one model.transcribe() on the whole atempo'd file (what the hotkey does now)
  ONLINE    the seek loop above, run as a discrete-event simulation: a window is
            started at max(its audio has been recorded, GPU free), using measured
            compute times. wait = finish of the final tail call - recording length.

Prints metrics only (no transcript text). Writes .output/bench_online/results.jsonl.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / ".output" / "audio"
OUT = ROOT / ".output" / "bench_online"
SR = 16000
ATEMPO = 1.5
WINDOW = 30.0      # stretched seconds per online call
PROMPT_CHARS = 200
OPTS = dict(beam_size=3, temperature=0.0, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 150, "speech_pad_ms": 400})


def load(path):
    out = subprocess.run(["ffmpeg", "-nostdin", "-i", str(path), "-af", f"atempo={ATEMPO}",
                          "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def run(model, audio, prompt=None):
    t0 = time.monotonic()
    segs = list(model.transcribe(audio, initial_prompt=prompt or None, **OPTS)[0])
    return segs, time.monotonic() - t0


def norm(text):
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1] / max(len(r), 1)


def online(model, audio):
    n = len(audio) / SR                  # stretched length
    L = n * ATEMPO                       # real recording length
    c, gpu_free, text, calls = 0.0, 0.0, [], 0
    while c + WINDOW < n:                # a full window exists before Stop
        arrive = (c + WINDOW) * ATEMPO
        chunk = audio[int(c * SR):int((c + WINDOW) * SR)]
        segs, dt = run(model, chunk, "".join(text)[-PROMPT_CHARS:])
        calls += 1
        gpu_free = max(arrive, gpu_free) + dt
        keep = segs[:-1] if len(segs) > 1 else []
        if keep:
            text += [s.text for s in keep]
            c += keep[-1].end
        else:                            # one long segment / silence: take it whole
            text += [s.text for s in segs]
            c += WINDOW
    segs, dt = run(model, audio[int(c * SR):], "".join(text)[-PROMPT_CHARS:])
    calls += 1
    text += [s.text for s in segs]
    done = max(L, gpu_free) + dt
    return "".join(text).strip(), done - L, calls, n - c, gpu_free - L


def main():
    files = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel("distil-large-v3.5", device="cuda", compute_type="int8_float16")
    run(model, load(AUDIO_DIR / files[0])[:SR * 10])  # warm-up
    print(f"{'len':>6} {'base_wait':>9} {'online_wait':>11} {'calls':>5} {'tail_s':>6} {'lag@stop':>8} {'wer':>6}")
    with open(OUT / "results.jsonl", "a") as log:
        for f in files:
            audio = load(AUDIO_DIR / f)
            segs, base_t = run(model, audio)
            base = "".join(s.text for s in segs).strip()
            text, wait, calls, tail, lag = online(model, audio)
            w = wer(base, text)
            L = len(audio) / SR * ATEMPO
            print(f"{L:6.0f} {base_t:9.2f} {wait:11.2f} {calls:5d} {tail:6.1f} {max(lag, 0):8.2f} {w:6.3f}", flush=True)
            log.write(json.dumps(dict(file=f, real_secs=L, base_wait=base_t, online_wait=wait, calls=calls,
                                      tail_secs=tail, lag_at_stop=lag, wer_vs_base=w, window=WINDOW)) + "\n")


if __name__ == "__main__":
    main()
