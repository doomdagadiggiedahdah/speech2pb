#!/bin/bash

SPOT="/home/mat/Documents/ProgramExperiments/stt_hk"
AUDIO_DIR="$SPOT/.output/audio"
TRANSCRIPT_DIR="$SPOT/.output/transcripts"

# Load your OpenAI API key from credentials file
source $SPOT/cred.txt

# Check if a specific file was provided as argument
if [[ -n "$1" ]]; then
    AUDIO_FILE="$AUDIO_DIR/original_$1.m4a"
else
    # Find the most recently created audio file
    AUDIO_FILE=$(ls -t "$AUDIO_DIR"/original_*.m4a 2>/dev/null | head -1)
    if [[ -z "$AUDIO_FILE" ]]; then
        echo "Error: No audio files found in $AUDIO_DIR"
        exit 1
    fi
fi

# Check if file exists
if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

echo "Transcribing: $AUDIO_FILE"

# Send the audio file to Whisper API and get transcription
stt_json=$(curl -s https://api.openai.com/v1/audio/transcriptions \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: multipart/form-data" \
    -F file="@$AUDIO_FILE" \
    -F model="whisper-1"
)

# Extract the transcribed text from the JSON response
stt_result=$(echo $stt_json | jq -r '.text')

# Display the transcription
echo "Transcription: $stt_result"

# Save the raw transcription to a text file
TIMESTAMP=$(basename "$AUDIO_FILE" | sed 's/original_//;s/.m4a//')
echo "$stt_result" > "$TRANSCRIPT_DIR/raw_transcript_$TIMESTAMP.txt"

# Optional: Copy the result to clipboard
echo "$stt_result" | xclip -selection clipboard
