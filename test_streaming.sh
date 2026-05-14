#!/bin/bash
# Test streaming transcription by playing audio through speakers
# while recording with the microphone.
#
# Usage:
#   ./test_streaming.sh nemotron [audio_file]
#   ./test_streaming.sh moonshine [audio_file]
#   ./test_streaming.sh nemotron --file-mode [audio_file]   # pipe file directly (no speaker/mic)
#
# If no audio_file is given, generates a TTS test clip using espeak-ng.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
BACKEND="${1:-nemotron}"
FILE_MODE=false

shift || true

if [[ "${1:-}" == "--file-mode" ]]; then
    FILE_MODE=true
    shift || true
fi

AUDIO_FILE="${1:-}"

# Generate test audio if none provided
if [[ -z "$AUDIO_FILE" ]]; then
    AUDIO_FILE="/tmp/stt_test_speech.wav"
    echo "Generating test speech with espeak-ng..."
    if command -v espeak-ng &>/dev/null; then
        espeak-ng -v en-us -s 140 -w "$AUDIO_FILE" \
            "This is a test of the real-time streaming speech to text system. \
             The quick brown fox jumps over the lazy dog. \
             One two three four five six seven eight nine ten."
    elif command -v espeak &>/dev/null; then
        espeak -v en-us -s 140 -w "$AUDIO_FILE" \
            "This is a test of the real-time streaming speech to text system. \
             The quick brown fox jumps over the lazy dog. \
             One two three four five six seven eight nine ten."
    else
        echo "ERROR: espeak-ng not found. Install it or provide an audio file."
        echo "  sudo apt install espeak-ng"
        exit 1
    fi
    echo "Test audio: $AUDIO_FILE"
    echo ""
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "ERROR: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Show audio duration
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$AUDIO_FILE" 2>/dev/null || echo "?")
echo "Audio duration: ${DURATION}s"
echo "Backend: $BACKEND"
echo ""

if [[ "$BACKEND" == "nemotron" ]]; then
    SCRIPT="$SCRIPT_DIR/nemotron_stream.py"
elif [[ "$BACKEND" == "moonshine" ]]; then
    SCRIPT="$SCRIPT_DIR/moonshine_stream.py"
else
    echo "Unknown backend: $BACKEND (use 'nemotron' or 'moonshine')"
    exit 1
fi

if $FILE_MODE; then
    # Direct file mode — pipe audio through ffmpeg, no speaker/mic needed
    echo "=== FILE MODE: piping audio directly ==="
    echo ""
    $PYTHON "$SCRIPT" --file "$AUDIO_FILE"
else
    # Speaker+mic mode — play through speakers, record with mic
    echo "=== SPEAKER+MIC MODE ==="
    echo "Playing audio through speakers in 3 seconds..."
    echo "Make sure your mic can hear your speakers!"
    echo ""
    sleep 3

    # Start the streaming transcriber in background
    $PYTHON "$SCRIPT" &
    STT_PID=$!

    # Give the model time to load
    echo "Waiting for model to load..." >&2
    sleep 2

    # Play the audio
    echo "Playing audio..." >&2
    ffplay -nodisp -autoexit -loglevel quiet "$AUDIO_FILE" 2>/dev/null || \
        aplay "$AUDIO_FILE" 2>/dev/null || \
        paplay "$AUDIO_FILE" 2>/dev/null

    # Give a moment for final processing
    sleep 2

    # Stop the transcriber
    kill $STT_PID 2>/dev/null || true
    wait $STT_PID 2>/dev/null || true

    echo ""
    echo "=== Done ==="
fi
