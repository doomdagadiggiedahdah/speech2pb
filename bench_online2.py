#!/usr/bin/env python3
"""Replay benchmark v2: transcribe-while-recording with seam fixes + silence guard.

Same discrete-event simulation as bench_online.py (a 30s stretched window is decoded as
soon as its audio exists and the GPU is free; wait = end of the final tail call - Stop),
with these changes:

  * word_timestamps=True; a window commits every segment except the last, and the next
    window starts in the *gap* after the last committed word (midpoint between that word's
    end and the next word's start, capped at +0.5s) instead of at the segment end stamp.
  * if a window yields a single segment, commit only its words ending >= HOLD s before the
    window edge and cut after the last of those (old code took the whole window, which can
    cut a word in half).
  * seam dedupe: drop the longest prefix of the new window that repeats the last <=8
    committed words (>=2 words, or 1 word of >=4 letters).
  * silence guard, using the VAD of the *whole recording so far* (silero is a causal
    streaming model; run live it costs ~nothing per 32ms frame, so here it is computed once
    up front and not charged to the wait). A cold-started VAD on an isolated 30s window
    reports phantom speech in its first ~1s, which is what let windows of pure silence
    reach Whisper and hallucinate. Skip the GPU call when the window overlaps < MIN_SPEECH s
    of global speech; drop segments overlapping < 0.2s of it, or with no_speech_prob >
    NS_MAX, avg_logprob < LP_MIN or compression_ratio > 2.4.

Usage: bench_online2.py [--ref F|T] [--prompt N] [--variant NAME] file.m4a ...
Prints metrics only (no transcript text). Writes .output/bench_online/results2.jsonl.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / ".output" / "audio"
OUT = ROOT / ".output" / "bench_online"
SR = 16000
ATEMPO = 1.5
WINDOW = 30.0       # stretched seconds per online call
HOLD = 3.0          # single-segment window: don't commit words this close to the edge
MIN_SPEECH = 0.5    # stretched seconds of VAD speech needed to decode a window
NS_MAX, LP_MIN = 0.6, -1.0
VADP = {"min_silence_duration_ms": 150, "speech_pad_ms": 400}
OPTS = dict(beam_size=3, temperature=0.0, vad_filter=True, vad_parameters=VADP)


def load(path):
    out = subprocess.run(["ffmpeg", "-nostdin", "-i", str(path), "-af", f"atempo={ATEMPO}",
                          "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


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


def run(model, audio, prompt=None, **kw):
    o = dict(OPTS); o.update(kw)
    t0 = time.monotonic()
    segs = list(model.transcribe(audio, initial_prompt=prompt or None, **o)[0])
    return segs, time.monotonic() - t0


def speech_ts(audio):
    return [(t["start"] / SR, t["end"] / SR) for t in get_speech_timestamps(audio, VadOptions(**VADP))]


def overlap(ts, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in ts)


def good(s):
    return not (s.no_speech_prob > NS_MAX or s.avg_logprob < LP_MIN or s.compression_ratio > 2.4)


def dedupe(committed, words):
    """Drop the longest prefix of `words` (word objects) repeating the tail of `committed` (norm tokens)."""
    new = [norm(w.word) for w in words]
    flat_idx = [(i, t) for i, ts in enumerate(new) for t in ts]
    toks = [t for _, t in flat_idx]
    for k in range(min(8, len(committed), len(toks)), 0, -1):
        if toks[:k] == committed[-k:] and (k >= 2 or len(toks[0]) >= 4):
            last_word = flat_idx[k - 1][0]
            return words[last_word + 1:], k
    return words, 0


class _W:
    """Stand-in for a faster-whisper Word when the call ran without word timestamps."""
    def __init__(self, word):
        self.word = word


def online(model, audio, prompt_chars=200, guard=True, cut="words", dd=True, fast_tail=False):
    n = len(audio) / SR
    L = n * ATEMPO
    c, gpu_free, text, toks, calls, skipped, dup = 0.0, 0.0, [], [], 0, 0, 0
    dts = []

    gv = speech_ts(audio) if guard else None

    def decode(chunk, arrive, final):
        nonlocal gpu_free, calls, skipped, dup
        t0 = time.monotonic()
        if guard and overlap(gv, c, c + len(chunk) / SR) < MIN_SPEECH:
            dt = time.monotonic() - t0
            gpu_free = max(arrive, gpu_free) + dt
            skipped += 1
            return None, dt
        segs, dt = run(model, chunk, "".join(text)[-prompt_chars:] if prompt_chars else None,
                       word_timestamps=(cut == "words" and not (final and fast_tail)))
        dt = time.monotonic() - t0
        dts.append(round(dt, 2))
        calls += 1
        gpu_free = max(arrive, gpu_free) + dt
        if guard:
            segs = [s for s in segs if good(s) and overlap(gv, c + s.start, c + s.end) >= 0.2]
        return segs, dt

    def commit(words):
        nonlocal dup
        if dd:
            words, k = dedupe(toks, words)
            dup += k
        if words:
            text.append("".join(w.word for w in words))
            toks.extend(t for w in words for t in norm(w.word))

    while c + WINDOW < n:
        arrive = (c + WINDOW) * ATEMPO
        segs, _ = decode(audio[int(c * SR):int((c + WINDOW) * SR)], arrive, False)
        if segs is None or not segs:            # silence: move on, keep a 1s overlap
            c += WINDOW - 1.0
            continue
        if cut != "words":                      # old behaviour (segment-end cut)
            keep = segs[:-1] if len(segs) > 1 else segs
            text.extend(s.text for s in keep)
            toks.extend(t for s in keep for t in norm(s.text))
            c += keep[-1].end if len(segs) > 1 else WINDOW
            continue
        allw = [w for s in segs for w in s.words]
        if len(segs) > 1:
            kw = [w for s in segs[:-1] for w in s.words]
        else:
            kw = [w for w in allw if w.end <= WINDOW - HOLD]
        if not kw:                              # nothing safely committable: a 1s step at worst
            c += max(1.0, WINDOW - HOLD - 1.0) if not allw else max(1.0, allw[0].start - 0.2)
            continue
        nxt = allw[len(kw)] if len(kw) < len(allw) else None
        cut_at = kw[-1].end if nxt is None else min((kw[-1].end + nxt.start) / 2, kw[-1].end + 0.5)
        commit(kw)
        c += max(cut_at, 1.0)
    gpu_free_before_tail = gpu_free
    segs, dt = decode(audio[int(c * SR):], L, True)
    if segs:
        if cut == "words":
            commit([w for s in segs for w in s.words] if segs[0].words else
                   [_W(t) for s in segs for t in re.findall(r" ?\S+", s.text)])
        else:
            text.extend(s.text for s in segs)
    return "".join(text).strip(), gpu_free - L, dict(calls=calls, skipped=skipped, dup_words=dup, tail_secs=(n - c) * ATEMPO,
                                                 lag_at_stop=gpu_free_before_tail - L, dts=dts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default="F", help="F: whole file cond_on_prev=False, T: =True")
    ap.add_argument("--prompt", type=int, default=200)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--fast-tail", action="store_true", help="final call after Stop without word timestamps")
    ap.add_argument("--cut", default="words", choices=["words", "segment"])
    ap.add_argument("--variant", default="v2")
    ap.add_argument("--dump", help="directory to write hypothesis text per file (for offline scoring)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel("distil-large-v3.5", device="cuda", compute_type="int8_float16")
    warm = load(AUDIO_DIR / a.files[0])[:SR * 20]
    run(model, warm); run(model, warm, word_timestamps=True)  # warm-up (word alignment has its own first-call cost)
    print(f"{'len':>6} {'ref_wait':>8} {'wait':>6} {'calls':>5} {'skip':>4} {'dup':>4} {'tail_s':>6} {'wer':>6} {'ref_w':>5} {'hyp_w':>5}")
    with open(OUT / "results2.jsonl", "a") as log:
        for f in a.files:
            audio = load(AUDIO_DIR / f)
            ref, ref_t = "", 0.0
            if a.ref != "none":
                segs, ref_t = run(model, audio, condition_on_previous_text=(a.ref == "T"))
                ref = "".join(s.text for s in segs).strip()
            text, wait, st = online(model, audio, a.prompt, not a.no_guard, a.cut, not a.no_dedupe, a.fast_tail)
            w = wer(ref, text) if norm(ref) else float(len(norm(text)))
            if a.dump:
                Path(a.dump).mkdir(parents=True, exist_ok=True)
                (Path(a.dump) / f"{f}.{a.variant}.txt").write_text(text)
            L = len(audio) / SR * ATEMPO
            print(f"{L:6.0f} {ref_t:8.2f} {wait:6.2f} {st['calls']:5d} {st['skipped']:4d} {st['dup_words']:4d} "
                  f"{st['tail_secs']:6.1f} {w:6.3f} {len(norm(ref)):5d} {len(norm(text)):5d}  dts={st['dts']}", flush=True)
            log.write(json.dumps(dict(file=f, variant=a.variant, ref=a.ref, prompt=a.prompt, real_secs=L,
                                      ref_wait=ref_t, online_wait=wait, wer_vs_ref=w,
                                      ref_words=len(norm(ref)), hyp_words=len(norm(text)), **st)) + "\n")


if __name__ == "__main__":
    main()
