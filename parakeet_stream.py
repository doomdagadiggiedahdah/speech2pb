#!/usr/bin/env python3
"""
Parakeet cache-aware streaming transcription.
Reads raw PCM float32 16kHz mono from stdin, transcribes incrementally
using NeMo's cache-aware streaming encoder — GPU runs during recording.

Usage (called from local_transcribe.sh):
    ffmpeg -f alsa -i default -ar 16000 -ac 1 -f f32le - | parakeet_stream.py

Streaming behaviour:
  - First output after ~5.8s of audio (model's context window)
  - Updates every ~320ms after that while recording continues
  - Final output when stdin closes (recording ends)
"""
import sys
import os
import threading
import numpy as np
import torch

SAMPLE_RATE = 16000
MODEL_NAME  = os.environ.get("PARAKEET_MODEL",  "nvidia/parakeet-tdt-0.6b-v2")
DEVICE      = os.environ.get("PARAKEET_DEVICE", "cuda")
# Read stdin in 100ms chunks so ffmpeg never blocks on the pipe
STDIN_READ_SAMPLES = SAMPLE_RATE // 10  # 1600 float32 samples


def load_model():
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu")
    model = model.half()
    model = model.to(DEVICE)
    model.eval()
    return model


def main():
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    # ── Start buffering stdin immediately so ffmpeg never blocks ──────────────
    raw_chunks = []
    stdin_done = threading.Event()

    def read_stdin():
        while True:
            data = sys.stdin.buffer.read(STDIN_READ_SAMPLES * 4)  # float32 = 4 bytes
            if not data:
                break
            raw_chunks.append(np.frombuffer(data, dtype=np.float32).copy())
        stdin_done.set()

    threading.Thread(target=read_stdin, daemon=True).start()

    # ── Load model (audio buffers in background) ───────────────────────────────
    model = load_model()
    print("[model loaded]", file=sys.stderr, flush=True)

    # ── Set up cache-aware streaming ──────────────────────────────────────────
    streaming_buf = CacheAwareStreamingAudioBuffer(model)

    cache_ch, cache_t, cache_ch_len = model.encoder.get_initial_cache_state(
        batch_size=1, dtype=torch.float16, device=model.device
    )
    previous_hyps = None
    step = 0
    appended_idx = 0   # how many raw_chunks have been appended to streaming_buf
    last_text = ""

    # ── Main streaming loop ───────────────────────────────────────────────────
    while not stdin_done.is_set() or appended_idx < len(raw_chunks):
        # Append any newly buffered audio into the streaming buffer
        while appended_idx < len(raw_chunks):
            streaming_buf.append_audio(raw_chunks[appended_idx])
            appended_idx += 1

        # Iterate over however many complete streaming chunks are ready
        any_processed = False
        for chunk_audio, chunk_lengths in streaming_buf:
            drop = model.encoder.streaming_cfg.drop_extra_pre_encoded if step > 0 else 0
            chunk_audio  = chunk_audio.to(model.device, dtype=torch.float16)
            chunk_lengths = chunk_lengths.to(model.device)

            with torch.inference_mode():
                (pred_out, texts,
                 cache_ch, cache_t, cache_ch_len,
                 previous_hyps) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_ch,
                    cache_last_time=cache_t,
                    cache_last_channel_len=cache_ch_len,
                    keep_all_outputs=streaming_buf.is_buffer_empty(),
                    previous_hypotheses=previous_hyps,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )

            step += 1
            any_processed = True
            if texts:
                last_text = texts[0]
                print(f"[stream] {last_text}", file=sys.stderr, flush=True)

        if not any_processed:
            stdin_done.wait(timeout=0.05)

    # ── Output final result ───────────────────────────────────────────────────
    print(last_text)


if __name__ == "__main__":
    main()
