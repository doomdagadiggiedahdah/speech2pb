#!/usr/bin/env python3
"""
online — transcribe while recording (runs.jsonl pipeline "online-v1").

The hotkey used to send the whole recording to gpu-queued after Stop, so the wait grew with
length (~3.8s + 0.048s per recorded second). Here 30s windows of the *stretched* audio
(atempo 1.5, same as the single-call path) are decoded while Mat is still talking; after Stop
only the tail is left. Algorithm = bench_online2.py variant "v2p0" (validated in replay:
pooled word diff vs whole-file decode 4.2%, no hallucinations on silent recordings):

  * every window: same options as the single call (stt_client.TRANSCRIBE_OPTIONS), no
    initial_prompt, word_timestamps=True.
  * commit every segment but the last; a single-segment window commits only words ending
    >= HOLD s before the window edge. The next window starts in the gap after the last
    committed word (midpoint to the next word's start, capped at +0.5s).
  * seam dedupe: drop up to 8 leading words repeating the committed tail
    (>= 2 words, or 1 word of >= 4 letters).
  * silence guard on speech evidence for the WHOLE stream so far (never a cold per-window
    VAD: that reports phantom speech in each window's first ~1s, which is what made silent
    windows hallucinate): skip windows with < MIN_SPEECH s of speech; drop segments
    overlapping < SEG_MIN_SPEECH s of it, or with no_speech_prob > 0.6, avg_logprob < -1,
    compression_ratio > 2.4.
    Speech evidence = streaming Silero VAD; when the hotkey runs whisper without VAD (the
    close Sennheiser mic, for whispering — Silero calls whispered speech silence, 0-7s of
    "speech" in recordings that hold 400-670 transcribed words) it is Silero OR an energy
    detector (frames > ENERGY_DB above the stream's noise floor), so whispers are never
    skipped. In that close-mic mode the per-segment speech check is off (it still cut real
    whispered segments: replay 2026-11-16, 4-9 segments per file); window skip + quality
    filters remain. STT_ONLINE_SEG_GUARD=1/0 forces it on/off.

Audio path: the Recorder keeps appending raw 44.1 kHz float32 blocks to a list; a feeder
thread here copies new blocks into ONE long-running ffmpeg (atempo -> 16 kHz), so the
stretched stream is the same audio the old whole-file atempo would make (no per-window
stretch seams). Windows are written as 16 kHz float wavs and sent to gpu-queued as
ordinary interactive jobs.

Every failure raises OnlineError from finish(); main.py then runs the old single-call path
on the recorder's own copy of the audio, so a dictation is never lost.
"""
import logging, os, queue, re, subprocess, tempfile, threading, time
from pathlib import Path

import numpy as np

log = logging.getLogger("stt_hk")

SR = 16000
FRAME = 512                 # Silero frame: 32 ms at 16 kHz; all speech masks use this grid
FRAME_S = FRAME / SR
WINDOW = 30.0               # stretched seconds per online call
HOLD = 3.0                  # single-segment window: keep words this close to the edge for later
MIN_SPEECH = 0.5            # stretched s of speech a window (or the tail) needs to be decoded
SEG_MIN_SPEECH = 0.2        # stretched s of speech a decoded segment must overlap to be kept
NS_MAX, LP_MIN, CR_MAX = 0.6, -1.0, 2.4
DEDUPE_MAX = 8
VAD = dict(threshold=0.5, min_silence_ms=150, speech_pad_ms=400)   # = TRANSCRIBE_OPTIONS' vad_parameters
ENERGY_DB = 8.0             # energy detector: frame level above the noise floor (10th pct of the stream)
ENERGY_MIN_RUN = 3          # ... for at least this many consecutive frames (~0.1 s)
KEEPALIVE_EVERY = 20.0      # s; while recording, re-preload so gpu-queued's 120s idle unload can't
                            # leave a cold model for the tail after a long pause


class OnlineError(RuntimeError):
    pass


# ── text helpers (identical to bench_online2.py) ─────────────────────────────
def norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def word_diff(ref: str, hyp: str) -> float:
    """Word-level edit distance / reference words (WER-style), on normalised tokens."""
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1] / max(len(r), 1)


def dedupe(committed: list[str], words: list) -> tuple[list, int]:
    """Drop the longest prefix of `words` ((start, end, text) tuples) that repeats the tail of
    `committed` (normalised tokens): <= DEDUPE_MAX tokens, >= 2, or 1 token of >= 4 letters."""
    flat = [(i, t) for i, w in enumerate(words) for t in norm(w[2])]
    toks = [t for _, t in flat]
    for k in range(min(DEDUPE_MAX, len(committed), len(toks)), 0, -1):
        if toks[:k] == committed[-k:] and (k >= 2 or len(toks[0]) >= 4):
            return words[flat[k - 1][0] + 1:], k
    return words, 0


def good(seg: dict) -> bool:
    return not (seg.get("no_speech_prob", 0.0) > NS_MAX or seg.get("avg_logprob", 0.0) < LP_MIN
                or seg.get("compression_ratio", 0.0) > CR_MAX)


# ── speech evidence ──────────────────────────────────────────────────────────
def speech_intervals(probs: np.ndarray, n_samples: int, threshold=VAD["threshold"],
                     min_silence_ms=VAD["min_silence_ms"], speech_pad_ms=VAD["speech_pad_ms"]):
    """faster_whisper.vad.get_speech_timestamps' post-processing (v1.2.1, defaults:
    min_speech 0 ms, max_speech inf) applied to precomputed frame probabilities.
    Returns [(start_sample, end_sample)]. test_online.py checks it matches the original."""
    neg = max(threshold - 0.15, 0.01)
    min_sil = SR * min_silence_ms / 1000
    pad = SR * speech_pad_ms / 1000
    speeches, triggered, start, temp_end = [], False, 0, 0
    for i, p in enumerate(probs):
        pos = FRAME * i
        if p >= threshold and temp_end:
            temp_end = 0
        if p >= threshold and not triggered:
            triggered, start = True, pos
            continue
        if p < neg and triggered:
            if not temp_end:
                temp_end = pos
            if pos - temp_end < min_sil:
                continue
            if temp_end - start > 0:
                speeches.append([start, temp_end])
            triggered, temp_end = False, 0
    if triggered and n_samples - start > 0:
        speeches.append([start, n_samples])
    for i, sp in enumerate(speeches):
        if i == 0:
            sp[0] = int(max(0, sp[0] - pad))
        if i != len(speeches) - 1:
            gap = speeches[i + 1][0] - sp[1]
            if gap < 2 * pad:
                sp[1] += int(gap // 2)
                speeches[i + 1][0] = int(max(0, speeches[i + 1][0] - gap // 2))
            else:
                sp[1] = int(min(n_samples, sp[1] + pad))
                speeches[i + 1][0] = int(max(0, speeches[i + 1][0] - pad))
        else:
            sp[1] = int(min(n_samples, sp[1] + pad))
    return [tuple(s) for s in speeches]


def _silero_session():
    """faster-whisper's Silero v6 ONNX, opened the way faster_whisper.vad does it, but
    without importing faster_whisper itself (~7s cold: ctranslate2, tokenizers, av ...)."""
    import importlib.util
    import onnxruntime
    pkg = importlib.util.find_spec("faster_whisper").submodule_search_locations[0]
    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = opts.intra_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.log_severity_level = 4
    return onnxruntime.InferenceSession(os.path.join(pkg, "assets", "silero_vad_v6.onnx"),
                                        providers=["CPUExecutionProvider"], sess_options=opts)


class SpeechTracker:
    """Speech evidence for the stretched stream so far, on the 32 ms frame grid.

    Silero runs causally, only on frames it hasn't seen (LSTM state and the 64-sample
    context carried over), so keeping it current costs ~nothing per window. Its timestamp
    post-processing is re-run over the whole stream on each query, like a whole-file VAD."""

    def __init__(self, silero: bool = True, energy: bool = False):
        self.use_silero, self.use_energy = silero, energy
        self.nf = 0
        self.probs = np.zeros(0, np.float32)
        self.db = np.zeros(0, np.float32)
        self._mask = None
        if silero:
            self.session = _silero_session()
            self.h = np.zeros((1, 1, 128), np.float32)
            self.c = np.zeros((1, 1, 128), np.float32)
            self.ctx = np.zeros(64, np.float32)

    def update(self, audio: np.ndarray):
        """`audio` = the whole stretched stream so far (only new whole frames are processed)."""
        nf = len(audio) // FRAME
        if nf <= self.nf:
            return
        fr = audio[self.nf * FRAME:nf * FRAME].reshape(-1, FRAME)
        self.db = np.concatenate([self.db, 10 * np.log10((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
                                  .astype(np.float32)])
        if self.use_silero:
            ctx = np.concatenate([self.ctx[None, :], fr[:-1, -64:]], 0)
            batch = np.concatenate([ctx, fr], 1).astype(np.float32)
            outs = []
            for i in range(0, len(batch), 10000):
                out, self.h, self.c = self.session.run(
                    None, {"input": batch[i:i + 10000], "h": self.h, "c": self.c})
                outs.append(np.asarray(out).reshape(-1))
            self.probs = np.concatenate([self.probs, *outs])
            self.ctx = fr[-1, -64:].copy()
        self.nf = nf
        self._mask = None

    def mask(self) -> np.ndarray:
        if self._mask is None:
            m = np.zeros(self.nf, bool)
            if self.use_silero:
                for s, e in speech_intervals(self.probs, self.nf * FRAME):
                    m[s // FRAME:-(-e // FRAME)] = True
            if self.use_energy and self.nf:
                active = self.db > np.percentile(self.db, 10) + ENERGY_DB
                # keep only runs of >= ENERGY_MIN_RUN frames (clicks/pops are shorter)
                run = np.convolve(active.astype(np.int32), np.ones(ENERGY_MIN_RUN, np.int32), "valid") == ENERGY_MIN_RUN
                keep = np.zeros(self.nf, bool)
                for k in range(ENERGY_MIN_RUN):
                    keep[k:k + len(run)] |= run
                m |= keep
            self._mask = m
        return self._mask

    def speech(self, a: float, b: float) -> float:
        """Seconds of speech between stretched times a and b."""
        m = self.mask()
        return float(m[max(0, int(a / FRAME_S)):max(0, int(np.ceil(b / FRAME_S)))].sum()) * FRAME_S


# ── streaming atempo ─────────────────────────────────────────────────────────
class Stretcher:
    """One ffmpeg process: raw f32 mono at `in_rate` on stdin -> atempo -> 16 kHz f32 on stdout."""

    def __init__(self, in_rate: int, tempo: float, on_pcm):
        filt = f"atempo={tempo}" if tempo <= 2.0 else f"atempo=2.0,atempo={tempo / 2.0}"
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(in_rate), "-ac", "1",
             "-i", "pipe:0", "-af", filt, "-ar", str(SR), "-ac", "1", "-f", "f32le", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err)
        self.on_pcm = on_pcm
        self.eof = threading.Event()
        self.reader = threading.Thread(target=self._read, name="online-stretch-read", daemon=True)
        self.reader.start()

    def _read(self):
        rest = b""
        try:
            while True:
                b = self.proc.stdout.read1(65536)
                if not b:
                    break
                b = rest + b
                cut = len(b) - len(b) % 4
                rest = b[cut:]
                if cut:
                    self.on_pcm(np.frombuffer(b[:cut], np.float32))
        finally:
            self.eof.set()

    def write(self, data: bytes):
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def close(self, timeout: float = 10.0):
        """Close stdin and wait for ffmpeg to flush everything out."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        if not self.eof.wait(timeout):
            self.proc.kill()
            raise OnlineError("ffmpeg stretch did not flush")
        rc = self.proc.wait(timeout=5)
        if rc != 0:
            self.err.seek(0)
            raise OnlineError(f"ffmpeg stretch exit {rc}: {self.err.read()[-200:].decode(errors='replace')}")

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()


# ── the online transcriber ───────────────────────────────────────────────────
class OnlineTranscriber:
    """Follows a growing list of raw float32 blocks (the Recorder's `_chunks`) and decodes
    30s stretched windows as they become available.

    transcribe(wav_path, options, final) -> list of gpu-queued segment dicts (raises on error).
    base_options() -> the single-call options at the time of the call (main.py adjusts them
    for the mic after recording starts). energy_guard: add the energy detector (no-VAD mic).
    keepalive(): called every KEEPALIVE_EVERY s with no GPU call while recording."""

    def __init__(self, chunks: list, transcribe, base_options, in_rate: int = 44100, tempo: float = 1.5,
                 energy_guard: bool = False, keepalive=None, workdir: str | None = None,
                 fast_tail: bool = False):
        self.chunks, self.transcribe, self.base_options = chunks, transcribe, base_options
        self.tempo, self.keepalive, self.fast_tail = tempo, keepalive, fast_tail
        self.energy_guard = energy_guard
        # Per-segment speech check only where the detector can see the speech: whispers into the
        # close mic sit too near its noise floor (replay 2026-11-16: it dropped 4-9 real segments
        # per file). There the window-level skip + quality filters remain, as the single call has none.
        self.seg_guard = os.environ.get("STT_ONLINE_SEG_GUARD", "auto") == "1" or (
            os.environ.get("STT_ONLINE_SEG_GUARD", "auto") == "auto" and not energy_guard)
        self.tmp = tempfile.TemporaryDirectory(prefix="stt_online_", dir=workdir)
        self.cv = threading.Condition()
        self.buf = np.zeros(SR * 120, np.float32)
        self.n = 0                       # stretched samples available
        self.stopping = threading.Event()
        self.eof = False                 # ffmpeg flushed after Stop: self.n is final
        self.error: BaseException | None = None
        self.done = threading.Event()
        self.text_parts: list[str] = []
        self.toks: list[str] = []
        self.tracker = None
        self.busy_since = None           # monotonic start of the GPU call in flight
        self.call_ends: list[float] = [] # monotonic end of every GPU call, in order
        self.last_call = time.monotonic()
        self.stats = dict(windows_sent=0, windows_skipped=0, dup_words=0, window_secs=[],
                          tail_audio_secs=0.0, tail_call_secs=0.0, tail_skipped=False,
                          inflight_at_stop=False, inflight_wait_secs=0.0, flush_secs=0.0,
                          backlog_windows_after_stop=0, segs_dropped_quality=0,
                          segs_dropped_nospeech=0, words_dropped=0,
                          drop_why=dict(ns=0, lp=0, cr=0), nonfinite_samples=0,
                          windows_empty=0)
        self.tail_start = None           # stretched second where the tail call started (diagnostics)
        self.stretcher = Stretcher(in_rate, tempo, self._on_pcm)
        self.feeder = threading.Thread(target=self._feed, name="online-feed", daemon=True)
        self.worker = threading.Thread(target=self._run, name="online-decode", daemon=True)
        self.feeder.start()
        self.worker.start()

    # -- audio in ------------------------------------------------------------
    def _feed(self):
        """Copy new recorder blocks into ffmpeg every 50 ms; after stop(), drain and close."""
        i, rest = 0, b""
        try:
            while True:
                stopping = self.stopping.is_set()      # read before draining: no block can be missed
                new = self.chunks[i:]
                if new:
                    i += len(new)
                    data = rest + b"".join(new)
                    cut = len(data) - len(data) % 4
                    data, rest = data[:cut], data[cut:]
                    # A single NaN/inf (e.g. a stray header) makes whisper return nothing for the
                    # whole window it lands in: zero it before it reaches ffmpeg.
                    pcm = np.frombuffer(data, np.float32)
                    bad = ~np.isfinite(pcm)
                    if bad.any():
                        self.stats["nonfinite_samples"] += int(bad.sum())
                        pcm = np.where(bad, np.float32(0), pcm)
                        data = pcm.tobytes()
                    self.stretcher.write(data)
                if stopping:
                    break
                time.sleep(0.05)
        except Exception as e:
            self._fail(e)
            self.stretcher.kill()

    def _on_pcm(self, pcm: np.ndarray):
        with self.cv:
            if self.n + len(pcm) > len(self.buf):
                self.buf = np.concatenate([self.buf[:self.n],
                                           np.zeros(max(len(self.buf), len(pcm)), np.float32)])
            self.buf[self.n:self.n + len(pcm)] = pcm
            self.n += len(pcm)
            self.cv.notify_all()

    def audio(self, lo: float = 0.0, hi: float | None = None) -> np.ndarray:
        """Copy of stretched audio between stretched seconds lo and hi (None = end)."""
        with self.cv:
            a, b = int(lo * SR), self.n if hi is None else min(self.n, int(hi * SR))
            return self.buf[a:max(a, b)].copy()

    def _fail(self, e: BaseException):
        if self.error is None:
            self.error = e
            log.warning("online: %s: %s", type(e).__name__, e)
        with self.cv:
            self.cv.notify_all()

    # -- decoding --------------------------------------------------------------
    def _speech(self, a: float, b: float) -> float:
        with self.cv:
            view = self.buf[:self.n]
        self.tracker.update(view)
        return self.tracker.speech(a, b)

    def _decode(self, c: float, audio: np.ndarray, final: bool):
        """One gpu-queued call for stretched audio starting at c. None = skipped (no speech)."""
        if self._speech(c, c + len(audio) / SR) < MIN_SPEECH:
            return None
        import soundfile as sf
        path = Path(self.tmp.name) / f"w{c:08.2f}.wav"
        sf.write(path, audio, SR, subtype="FLOAT")
        opts = dict(self.base_options())
        opts.pop("initial_prompt", None)
        opts["word_timestamps"] = not (final and self.fast_tail)
        t0 = time.monotonic()
        self.busy_since = t0
        try:
            segs = self.transcribe(str(path), opts, final)
        finally:
            self.busy_since = None
            self.last_call = time.monotonic()
            self.call_ends.append(self.last_call)
            path.unlink(missing_ok=True)
        dt = time.monotonic() - t0
        if final:
            self.stats["tail_call_secs"] = round(dt, 3)
        else:
            self.stats["windows_sent"] += 1
            self.stats["window_secs"].append(round(dt, 2))
        if opts["word_timestamps"] and any(s.get("text", "").strip() and not s.get("words") for s in segs):
            raise OnlineError("gpu-queued returned no word timestamps (restart it to pick up word_timestamps)")
        kept = []
        for s in segs:
            why = ("quality" if not good(s) else
                   "nospeech" if self.seg_guard and self._speech(c + s["start"], c + s["end"]) < SEG_MIN_SPEECH
                   else None)
            if why is None:
                kept.append(s)
                continue
            self.stats[f"segs_dropped_{why}"] += 1
            self.stats["words_dropped"] += len(norm(s.get("text", "")))
            if why == "quality":
                for k, bad in (("ns", s.get("no_speech_prob", 0) > NS_MAX), ("lp", s.get("avg_logprob", 0) < LP_MIN),
                               ("cr", s.get("compression_ratio", 0) > CR_MAX)):
                    self.stats["drop_why"][k] += bad
        return kept

    def _commit(self, words: list):
        words, k = dedupe(self.toks, words)
        self.stats["dup_words"] += k
        if words:
            self.text_parts.append("".join(w[2] for w in words))
            self.toks.extend(t for w in words for t in norm(w[2]))

    def _window(self, c: float) -> float:
        """Decode the full window starting at stretched second c; return the next start."""
        segs = self._decode(c, self.audio(c, c + WINDOW), final=False)
        if segs is None:
            self.stats["windows_skipped"] += 1
        elif not segs:                                # speech evidence but nothing kept: worth a look
            self.stats["windows_empty"] += 1
        if not segs:                                  # silence: move on, keep a 1s overlap
            return c + WINDOW - 1.0
        allw = [tuple(w[:3]) for s in segs for w in s.get("words", [])]
        if len(segs) > 1:
            kw = [tuple(w[:3]) for s in segs[:-1] for w in s.get("words", [])]
        else:
            kw = [w for w in allw if w[1] <= WINDOW - HOLD]
        if not kw:                                    # nothing safely committable: a 1s step at worst
            return c + (max(1.0, WINDOW - HOLD - 1.0) if not allw else max(1.0, allw[0][0] - 0.2))
        nxt = allw[len(kw)] if len(kw) < len(allw) else None
        cut = kw[-1][1] if nxt is None else min((kw[-1][1] + nxt[0]) / 2, kw[-1][1] + 0.5)
        self._commit(kw)
        return c + max(cut, 1.0)

    def _run(self):
        try:
            self.tracker = SpeechTracker(silero=True, energy=self.energy_guard)
            c = 0.0
            while True:
                with self.cv:
                    while not (self.error or self.eof or self.n >= (c + WINDOW) * SR):
                        self.cv.wait(timeout=1.0)
                        if (self.keepalive and not self.stopping.is_set()
                                and time.monotonic() - self.last_call > KEEPALIVE_EVERY):
                            self.last_call = time.monotonic()
                            threading.Thread(target=self._keepalive, daemon=True).start()
                    if self.error:
                        return
                    if self.eof and self.n <= (c + WINDOW) * SR:
                        break
                    if self.stopping.is_set():
                        self.stats["backlog_windows_after_stop"] += 1
                c = self._window(c)
            tail = self.audio(c)
            self.tail_start = c
            self.stats["tail_audio_secs"] = round(len(tail) / SR, 2)
            segs = self._decode(c, tail, final=True) if len(tail) else None
            self.stats["tail_skipped"] = segs is None
            if segs:
                self._commit([tuple(w[:3]) for s in segs for w in s.get("words", [])] if not self.fast_tail else
                             [(0, 0, t) for s in segs for t in re.findall(r" ?\S+", s["text"])])
        except BaseException as e:
            self._fail(e)
        finally:
            self.done.set()

    def _keepalive(self):
        try:
            self.keepalive()
        except Exception as e:
            log.debug("online keepalive failed: %s", e)

    # -- the end -----------------------------------------------------------------
    def finish(self, timeout: float = 300.0) -> str:
        """Call after the recorder stopped (its blocks are all in `chunks`). Blocks until the
        tail is decoded; returns the transcript. Raises OnlineError on any failure."""
        t0 = time.monotonic()
        self.stopping.set()
        calls_before = len(self.call_ends)
        self.stats["inflight_at_stop"] = inflight = self.busy_since is not None
        try:
            self.feeder.join(timeout=10)
            if self.error:
                raise OnlineError(f"online failed during recording: {self.error}")
            self.stretcher.close()
            self.stats["flush_secs"] = round(time.monotonic() - t0, 3)
            with self.cv:
                self.eof = True
                self.cv.notify_all()
            if not self.done.wait(timeout):
                raise OnlineError(f"online decode not done after {timeout:.0f}s")
            if inflight and len(self.call_ends) > calls_before:
                self.stats["inflight_wait_secs"] = round(self.call_ends[calls_before] - t0, 3)
            if self.error:
                raise OnlineError(f"online failed: {self.error}")
            return "".join(self.text_parts).strip()
        finally:
            self.stretcher.kill()

    def levels(self, lo: float = 0.0, hi: float | None = None) -> dict:
        """RMS/peak (dBFS) and non-finite sample count of the stretched audio in [lo, hi)."""
        a = self.audio(lo, hi)
        fin = a[np.isfinite(a)].astype(np.float64)
        db = lambda x: round(20 * np.log10(x + 1e-12), 1)
        return dict(secs=round(len(a) / SR, 2), nonfinite=int(len(a) - len(fin)),
                    rms_db=db(np.sqrt((fin ** 2).mean())) if len(fin) else None,
                    peak_db=db(np.abs(fin).max()) if len(fin) else None)

    def stretched_secs(self) -> float:
        return self.n / SR

    def close(self):
        self.stopping.set()
        self.stretcher.kill()
        self.tmp.cleanup()
