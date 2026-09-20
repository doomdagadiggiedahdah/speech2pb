#!/usr/bin/env python3
"""Stage 1 benchmark: does chunking transcription during recording cut wait-after-Stop?

Replays archived recordings (read-only on .output/audio/original_*.m4a) and measures,
for each file:

  BASELINE  — exactly mirrors main.py's post-Stop path: decode -> atempo 1.5x -> one
              daemon transcribe call on the whole recording.
  CHUNKED   — splits the real-time (pre-atempo) audio into C-second pieces (C in
              {20, 30, 45}s), atempo+transcribes each piece individually (as if the
              daemon processed it the moment it was recorded), then runs a small
              discrete-event simulation of wait-after-Stop: pieces arrive at their
              real-time boundary, the daemon serializes them, and wait = the last
              piece's finish time minus the recording length L.
  SILENCE   — one variant at C=30 where interior cut points are nudged to the
              quietest 200ms window within +/-3s of the fixed boundary (lowest RMS).

Writes only under .output/bench_chunked/ (raw jsonl rows + tmp wavs, cleaned up after).
Does not touch main.py / stt_daemon.py / stt_client.py / local_transcribe.py.
"""
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import stt_client  # noqa: E402

AUDIO_DIR = ROOT / ".output" / "audio"
OUT_DIR = ROOT / ".output" / "bench_chunked"
TMP_DIR = OUT_DIR / "tmp"
RESULTS_JSONL = OUT_DIR / "results.jsonl"
SUMMARY_MD = OUT_DIR / "summary.md"

ATEMPO_RATE = 1.5
C_VALUES = [20, 30, 45]
SILENCE_C = 30
SILENCE_SEARCH_RADIUS = 3.0
SILENCE_WINDOW_MS = 200

WALL_CLOCK_BUDGET_S = 22 * 60  # soft cap; skip remaining files (not measurements) if exceeded

WARMUP_FILE = "original_20260916_134356.m4a"  # ~13s, used only to warm the daemon

# ~3 each around 30s / 60s / 120s / 180s+, picked from the 80 most-recently-modified
# archived recordings via ffprobe durations (see bench log). Trimmed from 12 to 9 to
# fit the ~25 min budget (measurements-per-file kept full; file count cut instead).
FILES = [
    "original_20260917_001914.m4a",   # 27.9s
    "original_20260916_140130.m4a",   # 29.8s
    "original_20260919_104458.m4a",   # 28.3s
    "original_20260916_135316.m4a",   # 58.8s
    "original_20260919_001157.m4a",   # 62.2s
    "original_20260916_120123.m4a",   # 121.2s
    "original_20260916_105616.m4a",   # 117.2s
    "original_20260918_211732.m4a",   # 169.3s
    "original_20260918_234847.m4a",   # 207.2s
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def run_ffmpeg_decode(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", "44100", "-ac", "1", "-c:a", "pcm_f32le", str(dst)],
        check=True,
    )


def run_atempo(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-filter:a", f"atempo={ATEMPO_RATE}", str(dst)],
        check=True,
    )


def timed_atempo(src: Path, dst: Path) -> float:
    t0 = time.perf_counter()
    run_atempo(src, dst)
    return time.perf_counter() - t0


def timed_transcribe(path: Path) -> tuple[float, dict]:
    t0 = time.perf_counter()
    resp = stt_client.send_transcribe(str(path))
    return time.perf_counter() - t0, resp


# ---------------- silence-aligned cut ----------------
def find_quiet_boundary(audio: np.ndarray, sr: int, boundary_s: float, total_s: float) -> float:
    lo = max(0.0, boundary_s - SILENCE_SEARCH_RADIUS)
    hi = min(total_s, boundary_s + SILENCE_SEARCH_RADIUS)
    win = max(1, int(SILENCE_WINDOW_MS / 1000 * sr))
    hop = max(1, int(0.02 * sr))
    lo_i, hi_i = int(lo * sr), int(hi * sr)
    best_pos, best_rms = int(boundary_s * sr), None
    pos = lo_i
    while pos + win <= hi_i:
        seg = audio[pos:pos + win]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) else 1e9
        if best_rms is None or rms < best_rms:
            best_rms, best_pos = rms, pos
        pos += hop
    return (best_pos + win / 2) / sr


# ---------------- WER (word-level Levenshtein, self-contained) ----------------
def normalize_words(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def wer(ref_text: str, hyp_text: str) -> float:
    ref, hyp = normalize_words(ref_text), normalize_words(hyp_text)
    if not ref:
        return 0.0 if not hyp else 1.0
    return word_edit_distance(ref, hyp) / len(ref)


# ---------------- per-file processing ----------------
def process_file(name: str, rows: list) -> dict:
    m4a_path = AUDIO_DIR / name
    decoded = TMP_DIR / f"decoded_{name}.wav"
    run_ffmpeg_decode(m4a_path, decoded)
    audio, sr = sf.read(str(decoded), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total_s = len(audio) / sr
    log(f"{name}: decoded, {total_s:.1f}s")

    result = {"file": name, "length_s": total_s, "variants": {}}

    # ---- baseline ----
    base_fast = TMP_DIR / f"base_{uuid.uuid4().hex[:8]}.wav"
    atempo_t = timed_atempo(decoded, base_fast)
    daemon_t, resp = timed_transcribe(base_fast)
    base_fast.unlink(missing_ok=True)
    if "error" in resp:
        raise RuntimeError(f"baseline daemon error for {name}: {resp['error']}")
    baseline_text = resp["text"]
    rows.append({"file": name, "kind": "baseline", "piece_len_s": total_s,
                 "atempo_s": atempo_t, "daemon_s": daemon_t, "total_s": atempo_t + daemon_t})
    result["baseline_wait_s"] = atempo_t + daemon_t
    result["baseline_atempo_s"] = atempo_t
    result["baseline_daemon_s"] = daemon_t
    result["baseline_text"] = baseline_text
    log(f"{name}: baseline wait={atempo_t + daemon_t:.2f}s (atempo={atempo_t:.2f} daemon={daemon_t:.2f})")

    def run_variant(cvalue: int, silence: bool):
        edges = [0.0]
        e = float(cvalue)
        while e < total_s:
            edges.append(e)
            e += cvalue
        edges.append(total_s)
        edges = sorted(set(edges))

        if silence and len(edges) > 2:
            new_edges = [edges[0]]
            for b in edges[1:-1]:
                nb = find_quiet_boundary(audio, sr, b, total_s)
                nb = max(new_edges[-1] + 0.5, min(nb, total_s - 0.5))
                new_edges.append(nb)
            new_edges.append(edges[-1])
            edges = new_edges

        pieces = []
        label = f"silence_c{cvalue}" if silence else f"c{cvalue}"
        for i in range(len(edges) - 1):
            s, e = edges[i], edges[i + 1]
            if e - s <= 0.05:
                continue
            s_i, e_i = int(s * sr), int(e * sr)
            piece_path = TMP_DIR / f"p_{uuid.uuid4().hex[:8]}.wav"
            fast_path = TMP_DIR / f"pf_{uuid.uuid4().hex[:8]}.wav"
            sf.write(str(piece_path), audio[s_i:e_i], sr)
            a_t = timed_atempo(piece_path, fast_path)
            d_t, resp = timed_transcribe(fast_path)
            piece_path.unlink(missing_ok=True)
            fast_path.unlink(missing_ok=True)
            text = resp.get("text", "") if "error" not in resp else ""
            if "error" in resp:
                log(f"  WARNING piece error {name} {label} idx={i}: {resp['error']}")
            pieces.append({"start": s, "end": e, "len": e - s, "atempo_s": a_t,
                            "daemon_s": d_t, "total_s": a_t + d_t, "text": text})
            rows.append({"file": name, "kind": label, "piece_idx": i, "piece_len_s": e - s,
                         "atempo_s": a_t, "daemon_s": d_t, "total_s": a_t + d_t})

        # discrete-event sim: daemon serializes pieces, each ready at its real-time boundary
        daemon_free = 0.0
        finish_last = 0.0
        for p in pieces:
            start_proc = max(daemon_free, p["end"])
            finish_last = start_proc + p["total_s"]
            daemon_free = finish_last
        sim_wait = finish_last - total_s

        joined_text = "".join(p["text"] for p in pieces)
        w = wer(baseline_text, joined_text)

        full_costs = [p["total_s"] for p in pieces[:-1]] or [p["total_s"] for p in pieces]
        avg_full_cost = sum(full_costs) / len(full_costs)
        mean_random_wait = 0.5 * avg_full_cost
        worst_random_wait = avg_full_cost

        log(f"  {name} {label}: n_pieces={len(pieces)} sim_wait={sim_wait:.2f}s wer={w:.3f}")
        return {"pieces": pieces, "sim_wait_s": sim_wait, "wer": w, "joined_text": joined_text,
                "mean_random_wait_s": mean_random_wait, "worst_random_wait_s": worst_random_wait,
                "n_pieces": len(pieces)}

    for c in C_VALUES:
        result["variants"][f"c{c}"] = run_variant(c, silence=False)
    result["variants"][f"silence_c{SILENCE_C}"] = run_variant(SILENCE_C, silence=True)

    decoded.unlink(missing_ok=True)
    return result


def fit_linear(xs, ys):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    r, b = np.polyfit(xs, ys, 1)
    return float(r), float(b)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if not stt_client.is_running():
        log("ERROR: stt-daemon is not running / not reachable via socket. Aborting.")
        sys.exit(1)

    log("daemon reachable, preloading model...")
    print(stt_client.send_preload())

    # warm-up (throwaway, not recorded)
    warm_decoded = TMP_DIR / "warmup_decoded.wav"
    warm_fast = TMP_DIR / "warmup_fast.wav"
    run_ffmpeg_decode(AUDIO_DIR / WARMUP_FILE, warm_decoded)
    run_atempo(warm_decoded, warm_fast)
    t0 = time.perf_counter()
    resp = stt_client.send_transcribe(str(warm_fast))
    log(f"warm-up transcribe done in {time.perf_counter() - t0:.2f}s: {resp.get('text', resp)[:60]!r}")
    warm_decoded.unlink(missing_ok=True)
    warm_fast.unlink(missing_ok=True)

    rows = []
    file_results = []
    start_time = time.perf_counter()
    skipped = []
    for name in FILES:
        if time.perf_counter() - start_time > WALL_CLOCK_BUDGET_S:
            skipped.append(name)
            log(f"wall-clock budget exceeded, skipping remaining file: {name}")
            continue
        try:
            fr = process_file(name, rows)
            file_results.append(fr)
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR processing {name}: {exc!r} -- skipping this file")
            rows.append({"file": name, "kind": "error", "error": repr(exc)})

    with open(RESULTS_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- global cost model fit: total_s (atempo+daemon) vs piece_len_s, across all rows ----
    lens = [r["piece_len_s"] for r in rows if "piece_len_s" in r and "total_s" in r]
    totals = [r["total_s"] for r in rows if "piece_len_s" in r and "total_s" in r]
    r_slope, b_call = fit_linear(lens, totals) if len(lens) >= 2 else (float("nan"), float("nan"))

    atempo_times = [r["atempo_s"] for r in rows if "atempo_s" in r]
    atempo_lens = [r["piece_len_s"] for r in rows if "atempo_s" in r]
    atempo_mean = float(np.mean(atempo_times)) if atempo_times else float("nan")
    if len(atempo_lens) >= 2:
        atempo_r, atempo_b = fit_linear(atempo_lens, atempo_times)
    else:
        atempo_r, atempo_b = float("nan"), float("nan")

    # ---- summary.md ----
    lines = []
    lines.append("# bench_chunked results\n")
    lines.append(f"Files processed: {len(file_results)}/{len(FILES)}"
                  + (f" (skipped: {skipped})" if skipped else "") + "\n")
    lines.append(f"Total wall clock: {time.perf_counter() - start_time:.1f}s\n")
    lines.append("## Fitted cost model (total_s = atempo + daemon call, across all measured pieces + baselines)\n")
    lines.append(f"- `time = {b_call:.3f} + {r_slope:.4f} * piece_audio_secs`  "
                 f"(predicted: 1.9 + 0.27*L)\n")
    lines.append(f"- atempo alone: mean = {atempo_mean:.3f}s, fit = {atempo_b:.3f} + {atempo_r:.4f}*L\n")

    lines.append("\n## Per-file results\n")
    lines.append("| file | length_s | baseline_wait_s | c20_sim_wait | c30_sim_wait | c45_sim_wait | "
                  "silence_c30_sim_wait | c20_wer | c30_wer | c45_wer | silence_c30_wer |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for fr in file_results:
        v = fr["variants"]
        lines.append(
            f"| {fr['file']} | {fr['length_s']:.1f} | {fr['baseline_wait_s']:.2f} | "
            f"{v['c20']['sim_wait_s']:.2f} | {v['c30']['sim_wait_s']:.2f} | {v['c45']['sim_wait_s']:.2f} | "
            f"{v['silence_c30']['sim_wait_s']:.2f} | "
            f"{v['c20']['wer']:.3f} | {v['c30']['wer']:.3f} | {v['c45']['wer']:.3f} | "
            f"{v['silence_c30']['wer']:.3f} |"
        )

    if file_results:
        mean_baseline = float(np.mean([fr["baseline_wait_s"] for fr in file_results]))
        worst_baseline = float(np.max([fr["baseline_wait_s"] for fr in file_results]))
        lines.append(f"\nMean baseline wait: {mean_baseline:.2f}s, worst: {worst_baseline:.2f}s\n")
        for c_key in ["c20", "c30", "c45", "silence_c30"]:
            sims = [fr["variants"][c_key]["sim_wait_s"] for fr in file_results]
            wers = [fr["variants"][c_key]["wer"] for fr in file_results]
            mean_rand = float(np.mean([fr["variants"][c_key]["mean_random_wait_s"] for fr in file_results]))
            worst_rand = float(np.max([fr["variants"][c_key]["worst_random_wait_s"] for fr in file_results]))
            lines.append(
                f"- {c_key}: mean sim_wait={np.mean(sims):.2f}s worst sim_wait={np.max(sims):.2f}s, "
                f"mean WER={np.mean(wers):.3f}, "
                f"random-stop approx: mean={mean_rand:.2f}s worst={worst_rand:.2f}s"
            )

    lines.append("\n## Transcripts (baseline vs joined chunks) -- see results.jsonl for per-piece text\n")
    for fr in file_results:
        lines.append(f"\n### {fr['file']}\n")
        lines.append(f"**baseline:** {fr['baseline_text']}\n")
        for c_key in ["c20", "c30", "c45", "silence_c30"]:
            lines.append(f"**{c_key}:** {fr['variants'][c_key]['joined_text']}\n")

    SUMMARY_MD.write_text("\n".join(lines))
    log(f"wrote {RESULTS_JSONL} and {SUMMARY_MD}")

    try:
        TMP_DIR.rmdir()
    except OSError:
        pass  # leftovers from an error path; harmless, left for inspection


if __name__ == "__main__":
    main()
