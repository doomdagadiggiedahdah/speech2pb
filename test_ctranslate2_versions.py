"""
CTranslate2 hallucination fix validation.
Tests whether ctranslate2 4.6.3 fixes Conv1D hallucination on CUDA 12.0 + TU117.

Run with ctranslate2==4.4.0 (baseline), then reinstall 4.6.3 and run again:
    uv pip install "ctranslate2==4.6.3"
    .venv/bin/python test_ctranslate2_versions.py

Related issue: https://github.com/OpenNMT/CTranslate2/issues/1780
"""

import ctranslate2
from faster_whisper import WhisperModel

AUDIO_FILE = "recording.wav"
TESTS = [
    ("small",    "int8"),
    ("large-v3", "int8"),
    ("large-v3", "float32"),
]

print(f"ctranslate2 version: {ctranslate2.__version__}")
print(f"audio file: {AUDIO_FILE}")
print("=" * 60)

for model_size, compute_type in TESTS:
    print(f"\n--- {model_size} / {compute_type} ---")
    try:
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        segments, info = model.transcribe(AUDIO_FILE, language="en")
        text = " ".join(s.text for s in segments)
        print(f"language detected: {info.language} ({info.language_probability:.2f})")
        print(f"output: {text[:300]}")
        # hallucination checks
        if "ご視聴" in text or text.count("Thank you") > 3 or text.count("!") > 10:
            print("RESULT: HALLUCINATION DETECTED")
        else:
            print("RESULT: looks clean")
    except Exception as e:
        print(f"RESULT: ERROR -- {e}")
    print()
