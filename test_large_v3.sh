#!/usr/bin/env bash
# Test whisper large-v3 on GPU (INT8) using the existing local_transcribe.py
# Usage: ./test_large_v3.sh [audio_file]
#        defaults to recording.wav if no file given

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO="${1:-$SCRIPT_DIR/recording.wav}"

if [[ ! -f "$AUDIO" ]]; then
    echo "Audio file not found: $AUDIO"
    exit 1
fi

echo "=== Whisper large-v3 GPU test ==="
echo "Audio: $AUDIO"
echo "Device: cuda | Compute: int8_float16"
echo ""

time FW_MODEL=large-v3 \
     FW_DEVICE=cuda \
     FW_COMPUTE=int8_float16 \
     "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/local_transcribe.py" "$AUDIO"
