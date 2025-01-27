#!/bin/bash

# Directory where your recording.wav is located
SPOT="/home/mat/Documents/ProgramExperiments/stt_hk"

# Load your OpenAI API key from credentials file
source $SPOT/cred.txt

# Send the audio file to Whisper API and get transcription
stt_json=$(curl https://api.openai.com/v1/audio/transcriptions \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: multipart/form-data" \
    -F file="@$SPOT/recording.wav" \
    -F model="whisper-1"
)

# Extract the transcribed text from the JSON response
stt_result=$(echo $stt_json | jq -r '.text')

# Display the transcription
echo "Transcription: $stt_result"

# Optional: Copy the result to clipboard
echo "$stt_result" | xclip -selection clipboard
