#!/usr/bin/env python3
"""
Live microphone transcription using whisper_streaming + faster-whisper.

Usage:
    python live_transcribe.py

Press Ctrl+C to stop.
"""
import sys, os, numpy as np, sounddevice as sd, queue

WHISPER_STREAMING_DIR = os.environ.get("WHISPER_STREAMING_DIR", "/tmp/whisper_streaming")
sys.path.insert(0, WHISPER_STREAMING_DIR)

from whisper_online import FasterWhisperASR, OnlineASRProcessor  # type: ignore

MODEL_SIZE   = os.environ.get("FW_MODEL", "small")
COMPUTE_TYPE = os.environ.get("FW_COMPUTE", "float16")
LANGUAGE     = os.environ.get("LANGUAGE", "en")
CHUNK_SEC    = float(os.environ.get("CHUNK_SEC", "3.0"))
SAMPLE_RATE  = 16000

def main():
    from faster_whisper import WhisperModel
    print(f"Loading model '{MODEL_SIZE}' ({COMPUTE_TYPE})...", file=sys.stderr)
    asr = FasterWhisperASR(LANGUAGE, MODEL_SIZE)
    # Override with our compute_type (whisper_online hardcodes float16)
    asr.model = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE_TYPE)
    online = OnlineASRProcessor(asr)

    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())  # mono

    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    print("Listening... (Ctrl+C to stop)\n", file=sys.stderr)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=chunk_samples, callback=callback):
        try:
            while True:
                chunk = audio_queue.get()
                online.insert_audio_chunk(chunk)
                result = online.process_iter()
                if result[2].strip():
                    print(result[2], end=" ", flush=True)
        except KeyboardInterrupt:
            print("\n\n[stopped]", file=sys.stderr)
            final = online.finish()
            if final[2].strip():
                print(final[2], flush=True)

if __name__ == "__main__":
    main()
