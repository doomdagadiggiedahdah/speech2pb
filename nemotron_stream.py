#!/usr/bin/env python3
"""
Real-time streaming STT using nvidia/nemotron-speech-streaming-en-0.6b.

Opens the microphone via sounddevice, feeds audio chunks through NeMo's
cache-aware streaming encoder, and prints partial transcripts in real-time.

VRAM stays flat (~1.7GB) regardless of how long you talk — the encoder
maintains a fixed-size rolling cache.

Usage:
    .venv/bin/python nemotron_stream.py
    .venv/bin/python nemotron_stream.py --chunk-ms 160   # larger chunks = lower CPU, same accuracy
    .venv/bin/python nemotron_stream.py --device cpu      # run on CPU

Ctrl+C to stop. Press Enter to finalize a block (copies cleaned text to clipboard).
"""
import argparse
import math
import os
import re
import subprocess
import sys
import threading
import time
from datetime import date, timedelta

import numpy as np
import torch

SAMPLE_RATE = 16000


# ── text cleanup ──────────────────────────────────────────────────────────────

# Filler words/phrases to strip (matched as whole words, case-insensitive)
_FILLERS = [
    r"\buh\b", r"\bum\b", r"\bumm\b", r"\buhh\b",
    r"\boh\b(?!\s+(?:my|no|yes|god|wow|okay|man|well|really|great|right|dear|so|that|this|it|I|he|she|we|they|the|a))",
    r"\bahh?\b", r"\bhmm\b", r"\bhuh\b",
    r"\byeah\s+(?:um|uh)\b", r"\bokay\s+(?:so|and)\b",
    r"\blike\b(?=\s*,?\s*(?:um|uh|I|you|we|they|he|she|it|the|a|this|that|so|but|and|if|when|what|how|why|where)\b)",
]
_FILLER_RE = re.compile("|".join(_FILLERS), re.IGNORECASE)


def clean_transcript(text: str) -> str:
    """Light cleanup: strip fillers, fix whitespace/punctuation."""
    text = _FILLER_RE.sub("", text)
    text = re.sub(r"\s*,\s*,+", ",", text)       # collapse repeated commas
    text = re.sub(r"\s{2,}", " ", text)            # collapse whitespace
    text = re.sub(r"\s+([.,!?])", r"\1", text)     # space before punctuation
    text = re.sub(r"^\s*[.,]\s*", "", text)         # leading punctuation
    return text.strip()


def copy_to_clipboard(text: str):
    """Copy text to X11 clipboard via xclip."""
    try:
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        proc.communicate(text.encode(), timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[clipboard copy failed — is xclip installed?]", file=sys.stderr)


def _sunday_of_week(week_num: int, year: int) -> date:
    """Return the Sunday starting ISO week `week_num`."""
    jan4 = date(year, 1, 4)  # always in ISO week 1
    monday_w1 = jan4 - timedelta(days=jan4.isoweekday() - 1)
    monday = monday_w1 + timedelta(weeks=week_num - 1)
    return monday - timedelta(days=1)  # Sunday before Monday


def weeklist(raw: str | None = None) -> str:
    """Generate a Sun-Sat markdown date list.

    Accepts 'W20', 'W20 2026', '2026-05-10', or blank (= previous week).
    """
    year = date.today().year

    if not raw or not raw.strip():
        # Default: previous week
        today = date.today()
        # Current week's Sunday (Sun before or on today)
        this_sunday = today - timedelta(days=(today.weekday() + 1) % 7)
        sunday = this_sunday - timedelta(weeks=1)
    elif raw.strip().upper().startswith("W"):
        parts = raw.strip().split()
        week_num = int(parts[0][1:])
        if len(parts) > 1:
            year = int(parts[1])
        sunday = _sunday_of_week(week_num, year)
    else:
        d = date.fromisoformat(raw.strip())
        sunday = d - timedelta(days=(d.weekday() + 1) % 7)

    lines = [f"- ({(sunday + timedelta(days=i)).isoformat()})" for i in range(7)]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Nemotron streaming STT")
    p.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--chunk-ms", type=int, default=640, help="audio chunk size in ms (80-1040)")
    p.add_argument("--file", type=str, default=None,
                   help="transcribe a file instead of live mic (for testing)")
    p.add_argument("--weeklist", nargs="?", const="", default=None, metavar="WEEK",
                   help="print week date list and exit (e.g. W20, 2026-05-10, or blank for prev week)")
    p.add_argument("--verbose", action="store_true", help="show NeMo startup logs")
    return p.parse_args()


def load_model(model_name, device, verbose=False):
    import logging as _logging

    print(f"Loading {model_name}...", file=sys.stderr, flush=True)
    t0 = time.monotonic()

    if not verbose:
        _logging.disable(_logging.WARNING)

    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    # Load to CPU first in float32, then half() to fit in 4GB VRAM
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name, map_location="cpu"
    )

    if not verbose:
        _logging.disable(_logging.INFO)
    else:
        _logging.disable(_logging.NOTSET)

    model = model.half().to(device)
    model.eval()

    # Configure greedy decoding
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.compute_timestamps = False
        decoding_cfg.preserve_alignments = False
        if hasattr(model, 'joint'):
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
            # Disable CUDA graph decoder — known to crash on some GPUs
            if hasattr(decoding_cfg.greedy, 'use_cuda_graph_decoder'):
                decoding_cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(decoding_cfg)

    _logging.disable(_logging.NOTSET)

    elapsed = time.monotonic() - t0
    print(f"Model loaded in {elapsed:.1f}s", file=sys.stderr, flush=True)
    return model


class FeatureBuffer:
    """Mel-spectrogram feature buffer for cache-aware streaming.

    Inlined from NeMo's CacheFeatureBufferer to avoid the pipecat dependency.
    Maintains a rolling sample buffer and a rolling feature (mel) buffer.
    """

    LOG_MEL_ZERO = -16.635

    def __init__(self, sample_rate, buffer_size_in_secs, chunk_size_in_secs,
                 preprocessor_cfg, device):
        import nemo.collections.asr as nemo_asr

        self.device = device
        self.n_feat = preprocessor_cfg.features
        self.window_stride = preprocessor_cfg.window_stride
        self.chunk_size_in_secs = chunk_size_in_secs

        # Sample buffer (rolling)
        self._buf_samples = int(buffer_size_in_secs * sample_rate)
        self._sample_buf = torch.zeros(self._buf_samples, dtype=torch.float32)

        # Chunk sizes in samples
        self._chunk_samples = int(chunk_size_in_secs * sample_rate)
        self._lookback_samples = int(self.window_stride * sample_rate)

        # Feature buffer (rolling mel frames)
        self._feat_buf_len = int(buffer_size_in_secs / self.window_stride)
        self._feat_chunk_len = int(chunk_size_in_secs / self.window_stride)
        self._feat_buf = torch.full(
            [self.n_feat, self._feat_buf_len],
            self.LOG_MEL_ZERO, dtype=torch.float32, device=device,
        )

        # Preprocessor (mel extractor)
        self._preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self._preprocessor.to(device)

    def reset(self):
        self._sample_buf.zero_()
        self._feat_buf.fill_(self.LOG_MEL_ZERO)

    def update(self, audio: np.ndarray):
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        n = audio.shape[0]
        # Shift sample buffer and append new audio
        self._sample_buf[:-n] = self._sample_buf[n:].clone()
        self._sample_buf[-n:] = audio

        # Extract mel features from the last chunk + lookback
        samples = self._sample_buf[-(self._lookback_samples + self._chunk_samples):]
        sig = samples.unsqueeze(0).to(self.device)
        sig_len = torch.tensor([sig.shape[1]], device=self.device)
        with torch.no_grad():
            features, _ = self._preprocessor(input_signal=sig, length=sig_len)
        features = features.squeeze(0)

        # Trim to expected chunk length
        if (diff := features.shape[1] - self._feat_chunk_len - 1) > 0:
            features = features[:, :-diff]

        # Shift feature buffer and append
        fc = features[:, -self._feat_chunk_len:]
        self._feat_buf[:, :-self._feat_chunk_len] = self._feat_buf[:, self._feat_chunk_len:].clone()
        self._feat_buf[:, -self._feat_chunk_len:] = fc

    def get_feature_buffer(self) -> torch.Tensor:
        return self._feat_buf.clone()


class StreamingTranscriber:
    """Wraps NeMo's cache-aware streaming for live audio input."""

    def __init__(self, model, chunk_size_in_secs=0.08, device="cuda"):
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        self.model = model
        self.device = device
        self.Hypothesis = Hypothesis

        # Extract model params
        window_stride = model.cfg.preprocessor.window_stride
        model_stride = model.cfg.encoder.subsampling_factor

        model_chunk_size = model.encoder.streaming_cfg.chunk_size
        if isinstance(model_chunk_size, list):
            model_chunk_size = model_chunk_size[1]

        pre_encode_cache_size = model.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(pre_encode_cache_size, list):
            pre_encode_cache_size = pre_encode_cache_size[1]

        # Compute tokens per frame from requested chunk size
        tokens_per_frame = math.ceil(
            np.trunc(chunk_size_in_secs / window_stride) / model_stride
        )
        model.encoder.setup_streaming_params(
            chunk_size=model_chunk_size // model_stride,
            shift_size=tokens_per_frame,
        )

        # Feature buffer
        pre_encode_secs = pre_encode_cache_size * window_stride
        buffer_secs = pre_encode_secs + (model_chunk_size * window_stride)

        self._audio_buffer = FeatureBuffer(
            sample_rate=SAMPLE_RATE,
            buffer_size_in_secs=buffer_secs,
            chunk_size_in_secs=chunk_size_in_secs,
            preprocessor_cfg=model.cfg.preprocessor,
            device=device,
        )

        self._blank_id = len(model.tokenizer.vocab)
        self._sep = "\u2581"  # sentencepiece separator

        self.reset()

    def reset(self):
        """Reset all streaming state — call periodically for long sessions."""
        self._cache_ch, self._cache_t, self._cache_ch_len = \
            self.model.encoder.get_initial_cache_state(
                batch_size=1, dtype=torch.float16, device=self.device
            )
        self._prev_hyp = [self.Hypothesis(
            score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None
        )]
        self._audio_buffer.reset()
        self._step = 0

    @property
    def drop_extra_pre_encoded(self):
        return self.model.encoder.streaming_cfg.drop_extra_pre_encoded

    def process_chunk(self, audio_float32: np.ndarray) -> str:
        """Feed audio chunk (float32, 16kHz mono), return accumulated transcript."""
        self._audio_buffer.update(audio_float32)

        features = self._audio_buffer.get_feature_buffer()
        feat_len = torch.tensor([features.shape[1]], device=self.device)
        features = features.unsqueeze(0).half()  # match model dtype

        with torch.no_grad():
            encoded, enc_len, self._cache_ch, self._cache_t, self._cache_ch_len = \
                self.model.encoder.cache_aware_stream_step(
                    processed_signal=features,
                    processed_signal_length=feat_len,
                    cache_last_channel=self._cache_ch,
                    cache_last_time=self._cache_t,
                    cache_last_channel_len=self._cache_ch_len,
                    keep_all_outputs=False,
                    drop_extra_pre_encoded=self.drop_extra_pre_encoded,
                )

            best_hyp = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoded, enc_len,
                return_hypotheses=True,
                partial_hypotheses=self._prev_hyp,
            )

        self._prev_hyp = best_hyp
        self._step += 1

        # Decode tokens to text
        tokens = [int(t) for t in best_hyp[0].y_sequence if int(t) != self._blank_id]
        if not tokens:
            return ""
        pieces = self.model.tokenizer.ids_to_tokens(tokens)
        text = "".join(
            p.replace(self._sep, ' ') if p.startswith(self._sep) else p
            for p in pieces
        )
        return text


def _finalize_block(text, prev_lines, cols):
    """Finalize current transcript block: clean, print, copy to clipboard."""
    cleaned = clean_transcript(text)
    if not cleaned:
        return

    # Clear the live-updating text
    if prev_lines > 1:
        sys.stdout.write(f"\033[{prev_lines - 1}A")
    sys.stdout.write(f"\r\033[J")

    # Print the cleaned version as a permanent block
    sys.stdout.write(f"{cleaned}\n\n")
    sys.stdout.flush()

    copy_to_clipboard(cleaned)
    print(f"[copied {len(cleaned)} chars to clipboard]", file=sys.stderr, flush=True)


def stream_from_mic(transcriber, chunk_ms):
    """Stream from microphone, print partial transcripts.

    Press Enter to finalize a block (cleans fillers, copies to clipboard, resets).
    Press Ctrl+C to stop.
    """
    import select
    import sounddevice as sd
    import tty
    import termios

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)

    prev_text = ""
    prev_lines = 0
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    # Set terminal to raw mode so we can detect Enter without blocking
    stdin_fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(stdin_fd)
    enter_pressed = threading.Event()

    def watch_keys():
        """Background thread: detect Enter key in raw mode."""
        try:
            tty.setcbreak(stdin_fd)
            while True:
                if select.select([stdin_fd], [], [], 0.1)[0]:
                    ch = os.read(stdin_fd, 1)
                    if ch in (b'\n', b'\r'):
                        enter_pressed.set()
                    elif ch == b'\x03':  # Ctrl+C
                        break
        except (OSError, ValueError):
            pass

    key_thread = threading.Thread(target=watch_keys, daemon=True)
    key_thread.start()

    print(f"\nListening (chunk={chunk_ms}ms) — Enter to finalize block, Ctrl+C to stop\n",
          file=sys.stderr, flush=True)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=chunk_samples, dtype="float32") as stream:
            while True:
                audio, _ = stream.read(chunk_samples)
                audio = audio.squeeze()

                # Check for Enter
                if enter_pressed.is_set():
                    enter_pressed.clear()
                    if prev_text.strip():
                        _finalize_block(prev_text, prev_lines, cols)
                    transcriber.reset()
                    prev_text = ""
                    prev_lines = 0
                    continue

                text = transcriber.process_chunk(audio)
                if text != prev_text:
                    if prev_lines > 1:
                        sys.stdout.write(f"\033[{prev_lines - 1}A")
                    sys.stdout.write(f"\r\033[J{text}")
                    sys.stdout.flush()
                    prev_lines = max(1, (len(text) + cols - 1) // cols)
                    prev_text = text

    except KeyboardInterrupt:
        if prev_text.strip():
            _finalize_block(prev_text, prev_lines, cols)
        print("\nStopped.", file=sys.stderr, flush=True)
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)


def stream_from_file(transcriber, filepath, chunk_ms):
    """Stream from an audio file, simulating real-time mic input."""
    import subprocess

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 4  # float32

    # Use ffmpeg to convert to raw PCM
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-i", filepath,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    prev_text = ""
    t0 = time.monotonic()
    chunk_count = 0

    print(f"\nStreaming from {filepath} (chunk={chunk_ms}ms)...\n", file=sys.stderr, flush=True)

    try:
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break

            audio = np.frombuffer(data, dtype=np.float32)
            if len(audio) < chunk_samples:
                audio = np.pad(audio, (0, chunk_samples - len(audio)))

            text = transcriber.process_chunk(audio)
            chunk_count += 1

            if text != prev_text:
                elapsed = time.monotonic() - t0
                audio_time = chunk_count * chunk_ms / 1000
                sys.stdout.write(f"\r\033[K[{audio_time:.1f}s] {text}")
                sys.stdout.flush()
                prev_text = text

    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()

    if prev_text.strip():
        sys.stdout.write("\n")
    elapsed = time.monotonic() - t0
    audio_secs = chunk_count * chunk_ms / 1000
    print(f"\nDone: {audio_secs:.1f}s audio in {elapsed:.1f}s "
          f"({audio_secs/elapsed:.1f}x realtime)", file=sys.stderr, flush=True)


def main():
    args = parse_args()

    if args.weeklist is not None:
        raw = args.weeklist if args.weeklist else None
        result = weeklist(raw)
        print(result)
        copy_to_clipboard(result)
        return

    model = load_model(args.model, args.device, verbose=args.verbose)
    transcriber = StreamingTranscriber(
        model,
        chunk_size_in_secs=args.chunk_ms / 1000.0,
        device=args.device,
    )

    if args.file:
        stream_from_file(transcriber, args.file, args.chunk_ms)
    else:
        stream_from_mic(transcriber, args.chunk_ms)


if __name__ == "__main__":
    main()
