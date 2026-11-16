#!/usr/bin/env python3
"""Replay archived recordings through the live online code path (online.OnlineTranscriber)
against a running gpu-queued, with the audio arriving in 0.1s blocks at --speed x real time,
then compare with today's single call after Stop (atempo whole file -> one interactive job).

    GPUQ_SOCKET=/path/test.sock replay_online.py [--speed 3] [--close-mic] file.m4a ...

--close-mic mimics the Sennheiser path: whisper without VAD + energy guard.
Prints and logs metrics only (no transcript text) to .output/bench_online/replay_live.jsonl.
wait = Stop (last block delivered) -> online transcript ready. single_wait = atempo + call.
"""
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import online, stt_client  # noqa: E402

ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / ".output" / "audio"
OUT = ROOT / ".output" / "bench_online" / "replay_live.jsonl"
SR_IN, ATEMPO, BLOCK = 44100, 1.5, 4410


def load_raw(path: Path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ar", str(SR_IN),
                          "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


class Calls:
    def __init__(self):
        self.tail_queued = 0.0

    def __call__(self, path, options, final):
        resp = stt_client.send_transcribe(path, options=options)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        if final:
            self.tail_queued += resp.get("queued", 0.0)
        return resp["segments"]


def gaps(ref: str, hyp: str) -> tuple[int, int]:
    """Longest run of reference words missing from hyp, and of hyp words not in ref
    (Levenshtein alignment): a dropped or doubled stretch at a seam shows up here."""
    r, h = online.norm(ref), online.norm(hyp)
    D = np.zeros((len(r) + 1, len(h) + 1), np.int32)
    D[:, 0], D[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    i, j, run_d, run_i, best_d, best_i = len(r), len(h), 0, 0, 0, 0
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            i, j, run_d, run_i = i - 1, j - 1, 0, 0
        elif i and D[i, j] == D[i - 1, j] + 1:
            i, run_d, run_i = i - 1, run_d + 1, 0
        else:
            j, run_i, run_d = j - 1, run_i + 1, 0
        best_d, best_i = max(best_d, run_d), max(best_i, run_i)
    return best_d, best_i


def single_call(raw: np.ndarray, extra: dict | None = None) -> tuple[str, float]:
    """Today's path: write wav, ffmpeg atempo, one interactive job."""
    import soundfile as sf
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as d:
        src, fast = Path(d) / "in.wav", Path(d) / "fast.wav"
        sf.write(src, raw, SR_IN)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-filter:a",
                        f"atempo={ATEMPO}", str(fast)], check=True)
        resp = stt_client.send_transcribe(str(fast), options={**stt_client.TRANSCRIBE_OPTIONS, **(extra or {})})
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["text"], time.monotonic() - t0 - resp.get("queued", 0.0)


def replay(raw: np.ndarray, speed: float, close_mic: bool, fast_tail: bool = False):
    chunks: list[bytes] = []
    calls = Calls()
    tr = online.OnlineTranscriber(chunks, calls, base_options=lambda: stt_client.TRANSCRIBE_OPTIONS,
                                  in_rate=SR_IN, tempo=ATEMPO, energy_guard=close_mic,
                                  keepalive=stt_client.send_preload, fast_tail=fast_tail)
    t0 = time.monotonic()
    for k, i in enumerate(range(0, len(raw), BLOCK)):
        chunks.append(raw[i:i + BLOCK].tobytes())
        delay = t0 + (k + 1) * BLOCK / SR_IN / speed - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    t_stop = time.monotonic()
    try:
        text = tr.finish()
    finally:
        tr.close()
    return text, time.monotonic() - t_stop - calls.tail_queued, tr.stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--close-mic", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fast-tail", action="store_true")
    a = ap.parse_args()
    if a.close_mic:
        stt_client.TRANSCRIBE_OPTIONS["vad_filter"] = False
        stt_client.TRANSCRIBE_OPTIONS.pop("vad_parameters", None)
    stt_client.send_preload()
    while stt_client.send_status().get("status") != "ready":
        time.sleep(0.5)
    print(f"{'file':>30} {'real_s':>6} {'sent':>4} {'skip':>4} {'infl':>4} {'wait':>5} {'1call':>5} "
          f"{'tail_s':>6} {'diff':>6} {'floor':>6} {'miss':>4} {'xtra':>4} {'ref_w':>5} {'hyp_w':>5}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for f in a.files:
        p = Path(f) if Path(f).exists() else AUDIO_DIR / f
        raw = load_raw(p)
        real = len(raw) / SR_IN
        text, wait, st = replay(raw, a.speed, a.close_mic, a.fast_tail)
        ref, ref_wait = ("", 0.0) if a.no_ref else single_call(raw)
        nr, nh = len(online.norm(ref)), len(online.norm(text))
        diff = online.word_diff(ref, text) if nr else float(nh)
        alt = single_call(raw, {"condition_on_previous_text": False})[0] if nr else ""
        floor = online.word_diff(ref, alt) if nr else 0.0
        fmiss = gaps(ref, alt)[0] if nr else 0
        miss, extra = gaps(ref, text) if nr else (0, nh)
        print(f"{p.name:>30} {real:6.0f} {st['windows_sent']:4d} {st['windows_skipped']:4d} "
              f"{int(st['inflight_at_stop']):4d} {wait:5.2f} {ref_wait:5.2f} {st['tail_audio_secs'] * ATEMPO:6.1f} "
              f"{diff:6.3f} {floor:6.3f}/{fmiss:<3d} {miss:4d} {extra:4d} {nr:5d} {nh:5d}  drop={st['words_dropped']}w q{st['segs_dropped_quality']}{st['drop_why']} v{st['segs_dropped_nospeech']} dup={st['dup_words']}", flush=True)
        with open(OUT, "a") as log:
            log.write(json.dumps(dict(file=p.name, tag=a.tag, speed=a.speed, close_mic=a.close_mic, fast_tail=a.fast_tail,
                                      real_secs=round(real, 1), wait=round(wait, 3), single_wait=round(ref_wait, 3),
                                      word_diff=round(diff, 4), single_vs_single_nocond=round(floor, 4), max_missing_run=miss, floor_max_missing_run=fmiss, max_extra_run=extra, ref_words=nr, hyp_words=nh, **st)) + "\n")


if __name__ == "__main__":
    main()
