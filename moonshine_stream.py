#!/usr/bin/env python3
"""
Real-time streaming STT using UsefulSensors/moonshine-streaming-medium.

Lightweight alternative to Nemotron — runs in ~500MB VRAM (float16) or on CPU.
Uses chunked processing: accumulates audio, transcribes each chunk independently,
prints partial results.

Note: transformers doesn't expose encoder cache yet, so each chunk re-encodes.
This is fine for ~1-2s chunks on GPU (Moonshine is very fast).

Usage:
    .venv/bin/python moonshine_stream.py
    .venv/bin/python moonshine_stream.py --chunk-sec 2.0
    .venv/bin/python moonshine_stream.py --device cpu
    .venv/bin/python moonshine_stream.py --file test.wav

Ctrl+C to stop.
"""
import argparse
import sys
import time

import numpy as np
import torch

SAMPLE_RATE = 16000


def parse_args():
    p = argparse.ArgumentParser(description="Moonshine streaming STT")
    p.add_argument("--model", default="UsefulSensors/moonshine-streaming-medium")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--chunk-sec", type=float, default=1.5,
                   help="process audio in chunks of this many seconds")
    p.add_argument("--overlap-sec", type=float, default=0.3,
                   help="overlap between chunks for boundary handling")
    p.add_argument("--file", type=str, default=None,
                   help="transcribe a file instead of live mic (for testing)")
    return p.parse_args()


def load_model(model_name, device):
    print(f"Loading {model_name}...", file=sys.stderr, flush=True)
    t0 = time.monotonic()

    from transformers import MoonshineStreamingForConditionalGeneration, AutoProcessor

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)

    elapsed = time.monotonic() - t0
    print(f"Model loaded in {elapsed:.1f}s", file=sys.stderr, flush=True)
    return model, processor


class ChunkedTranscriber:
    """Accumulates audio, transcribes in chunks, emits full running transcript."""

    def __init__(self, model, processor, chunk_sec=1.5, overlap_sec=0.3, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = next(model.parameters()).dtype

        self.chunk_samples = int(chunk_sec * SAMPLE_RATE)
        self.overlap_samples = int(overlap_sec * SAMPLE_RATE)
        self.stride_samples = self.chunk_samples - self.overlap_samples

        self.audio_buffer = np.array([], dtype=np.float32)
        self.completed_texts = []
        self.current_text = ""

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        if len(audio) == 0:
            return ""

        inputs = self.processor(
            audio, return_tensors="pt", sampling_rate=SAMPLE_RATE
        )
        inputs = {k: v.to(self.device, self.dtype) if v.dtype.is_floating_point
                  else v.to(self.device)
                  for k, v in inputs.items()}

        # Prevent hallucination on short audio
        max_length = max(4, int(len(audio) * 6.5 / SAMPLE_RATE))

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=max_length)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def add_audio(self, audio_float32: np.ndarray) -> str:
        """Add audio samples, return current full transcript."""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float32])

        # Process complete chunks
        while len(self.audio_buffer) >= self.chunk_samples:
            chunk = self.audio_buffer[:self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.stride_samples:]

            text = self._transcribe_chunk(chunk)
            if text:
                self.completed_texts.append(text)

        # Transcribe whatever is buffered (partial chunk) for display
        if len(self.audio_buffer) > SAMPLE_RATE * 0.3:  # at least 300ms
            self.current_text = self._transcribe_chunk(self.audio_buffer)
        else:
            self.current_text = ""

        return self.get_full_text()

    def flush(self) -> str:
        """Process remaining audio buffer."""
        if len(self.audio_buffer) > 0:
            text = self._transcribe_chunk(self.audio_buffer)
            if text:
                self.completed_texts.append(text)
            self.audio_buffer = np.array([], dtype=np.float32)
            self.current_text = ""
        return self.get_full_text()

    def get_full_text(self) -> str:
        parts = self.completed_texts[:]
        if self.current_text:
            parts.append(self.current_text)
        return " ".join(parts)


def stream_from_mic(transcriber, chunk_ms=100):
    """Stream from microphone, print partial transcripts."""
    import sounddevice as sd

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    prev_text = ""

    print(f"\nListening...\n", file=sys.stderr, flush=True)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=chunk_samples, dtype="float32") as stream:
            while True:
                audio, _ = stream.read(chunk_samples)
                audio = audio.squeeze()

                text = transcriber.add_audio(audio)
                if text != prev_text:
                    sys.stdout.write(f"\r\033[K{text}")
                    sys.stdout.flush()
                    prev_text = text

    except KeyboardInterrupt:
        final = transcriber.flush()
        if final.strip():
            sys.stdout.write(f"\r\033[K{final}\n")
        print("\nStopped.", file=sys.stderr, flush=True)


def stream_from_file(transcriber, filepath, chunk_ms=100):
    """Stream from an audio file, simulating real-time mic input."""
    import subprocess

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 4

    cmd = [
        "ffmpeg", "-nostdin", "-y", "-i", filepath,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    prev_text = ""
    t0 = time.monotonic()
    chunk_count = 0

    print(f"\nStreaming from {filepath}...\n", file=sys.stderr, flush=True)

    try:
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break

            audio = np.frombuffer(data, dtype=np.float32)
            chunk_count += 1

            text = transcriber.add_audio(audio)
            if text != prev_text:
                audio_time = chunk_count * chunk_ms / 1000
                sys.stdout.write(f"\r\033[K[{audio_time:.1f}s] {text}")
                sys.stdout.flush()
                prev_text = text

    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()

    final = transcriber.flush()
    if final.strip():
        sys.stdout.write(f"\r\033[K{final}\n")

    elapsed = time.monotonic() - t0
    audio_secs = chunk_count * chunk_ms / 1000
    print(f"\nDone: {audio_secs:.1f}s audio in {elapsed:.1f}s "
          f"({audio_secs/elapsed:.1f}x realtime)", file=sys.stderr, flush=True)


def main():
    args = parse_args()
    model, processor = load_model(args.model, args.device)
    transcriber = ChunkedTranscriber(
        model, processor,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        device=args.device,
    )

    if args.file:
        stream_from_file(transcriber, args.file)
    else:
        stream_from_mic(transcriber)


if __name__ == "__main__":
    main()
