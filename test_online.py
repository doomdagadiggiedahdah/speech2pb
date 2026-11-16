"""CPU-only checks for online.py: VAD post-processing parity, streaming stretch, seam logic.
Run: .venv/bin/python -m pytest -q test_online.py   (uses one archived recording; prints no text)"""
import time
from pathlib import Path
import subprocess

import numpy as np
import pytest

import online

AUDIO = sorted((Path(__file__).parent / ".output" / "audio").glob("original_*.m4a"))


def _load(path, rate, tempo=None):
    af = ["-af", f"atempo={tempo}"] if tempo else []
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), *af, "-ar", str(rate),
                          "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


@pytest.fixture(scope="module")
def sample():
    big = [p for p in AUDIO[-200:] if 3_000_000 < p.stat().st_size]   # >= ~2 min: tests slice 100s
    if not big:
        pytest.skip("no archived recordings")
    return big[-1]


def test_speech_intervals_match_faster_whisper(sample):
    from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model
    a = _load(sample, 16000, 1.5)[:16000 * 90]
    ref = [(t["start"], t["end"]) for t in get_speech_timestamps(a, VadOptions(min_silence_duration_ms=150, speech_pad_ms=400))]
    padded = np.pad(a, (0, 512 - len(a) % 512))
    probs = np.asarray(get_vad_model()(padded)).reshape(-1)
    assert online.speech_intervals(probs, len(a)) == ref


def test_streaming_silero_close_to_one_shot(sample):
    a = _load(sample, 16000, 1.5)[:16000 * 60]
    one = online.SpeechTracker(); one.update(a)
    inc = online.SpeechTracker()
    for k in range(1, 61, 7):          # uneven increments
        inc.update(a[:16000 * k])
    inc.update(a)
    assert abs(one.speech(0, 60) - inc.speech(0, 60)) < 1.0


def test_stretcher_streams_and_matches_whole_file(sample):
    raw = _load(sample, 44100)[:44100 * 20]
    got, first = [], []
    st = online.Stretcher(44100, 1.5, lambda pcm: (got.append(pcm), first or first.append(time.monotonic())))
    t0 = time.monotonic()
    for i in range(0, len(raw), 4410 * 10):          # 1s blocks
        st.write(raw[i:i + 4410 * 10].tobytes())
    t_written = time.monotonic()
    time.sleep(0.5)
    n_before_close = sum(map(len, got))
    st.close()
    out = np.concatenate(got)
    expect = 20 / 1.5 * 16000
    assert abs(len(out) - expect) < 16000 * 0.2
    assert n_before_close > expect * 0.8, "ffmpeg should emit audio before stdin closes"
    assert first[0] - t0 < 1.0


def test_dedupe_and_good():
    committed = online.norm("one two three four")
    words = [(0, 1, " three"), (1, 2, " four"), (2, 3, " five")]
    assert online.dedupe(committed, words) == ([(2, 3, " five")], 2)
    assert online.dedupe(committed, [(0, 1, " four"), (1, 2, " x")]) == ([(1, 2, " x")], 1)
    assert online.dedupe(online.norm("a b it"), [(0, 1, " it"), (1, 2, " x")])[1] == 0   # 1 short word: kept
    assert online.good({"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.5})
    assert not online.good({"no_speech_prob": 0.9})
    assert not online.good({"avg_logprob": -1.5})
    assert not online.good({"compression_ratio": 2.6})


def test_online_end_to_end_with_fake_model(sample):
    """Fake transcribe: one segment per call with words every 2s. Checks the loop terminates,
    windows advance, the tail runs, and finish() returns text."""
    raw = _load(sample, 44100)[:44100 * 100]
    calls = []

    def fake(path, opts, final):
        import soundfile as sf
        n = sf.info(path).duration
        calls.append((round(n, 1), final, opts["word_timestamps"]))
        words = [[t, t + 1.5, f" w{len(calls)}_{int(t)}", 0.9] for t in np.arange(0.0, n - 1.5, 2.0)]
        return [{"start": 0.0, "end": n, "text": "".join(w[2] for w in words), "words": words,
                 "no_speech_prob": 0.0, "avg_logprob": -0.2, "compression_ratio": 1.2}]

    chunks = []
    o = online.OnlineTranscriber(chunks, fake, lambda: {"beam_size": 3, "initial_prompt": "x"},
                                 energy_guard=True)
    for i in range(0, len(raw), 4410):
        chunks.append(raw[i:i + 4410].tobytes())
    time.sleep(1.0)
    text = o.finish()
    o.close()
    assert calls[-1][1] is True and all(not f for _, f, _ in calls[:-1])
    assert len(text.split()) > 10
    assert o.stats["windows_sent"] >= 1


def test_nonfinite_and_unaligned_blocks_are_sanitized(sample):
    """pw-record without --raw prepended an AU header (one NaN 'sample'); a NaN anywhere made
    whisper return nothing for its window (live 2026-11-16). The feeder must zero non-finite
    samples and cope with blocks that split a float."""
    raw = _load(sample, 44100)[:44100 * 8]
    hdr = bytes.fromhex("646e732e18000000ffffffff0600000044ac000001000000")
    data = hdr + raw.tobytes()
    seen = []

    def fake(path, opts, final):
        import soundfile as sf
        seen.append(sf.read(path, dtype="float32")[0])
        return []

    chunks = []
    o = online.OnlineTranscriber(chunks, fake, lambda: {}, energy_guard=True)
    for i in range(0, len(data), 4407):               # odd block size: floats split across blocks
        chunks.append(data[i:i + 4407])
    text = o.finish()
    lv = o.levels()
    o.close()
    assert text == ""
    assert o.stats["nonfinite_samples"] == 1 and lv["nonfinite"] == 0
    assert seen and all(np.isfinite(a).all() for a in seen)
    assert abs(lv["secs"] - (len(raw) + 6) / 44100 / 1.5) < 0.1
