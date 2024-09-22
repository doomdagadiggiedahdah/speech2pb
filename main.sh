#!/bin/bash

### Trajectory ###
# - recorder starts / pop-up with button 
# - solo; "stop" is its focus
# - enter to end
# - audo; *sent* to whisper
# - text response / in paste buffer
# - program ends

# replace this directory with your own
SPOT="/home/mat/Documents/ProgramExperiments/stt_hk"
source $SPOT/cred.txt

transcribe_audio () {
    stt_json=$(curl https://api.openai.com/v1/audio/transcriptions \
          -H "Authorization: Bearer $OPENAI_API_KEY" \
          -H "Content-Type: multipart/form-data" \
          -F file="@$SPOT/recording.wav" \
          -F model="whisper-1"
    )
    stt_result=$(echo $stt_json | jq -r '.text')
}

format_text () {
    local stt_output="$1"  # Accept the STT output as an argument

    # Create the formatted message content
    local message_content="Take the following STT output and apply formatting to make it more 'text friendly'. Add no additional text. Add punctuation, capitalization, remove filler words 'uhh, um' and make ready for text usage: $stt_output"

    # Clean JSON structure with heredoc
    format_json=$(curl https://api.openai.com/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -d @- <<EOF
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "$message_content"
        }
      ]
    }
  ],
  "temperature": 1,
  "max_tokens": 2048,
  "top_p": 1,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "response_format": {
    "type": "text"
  }
}
EOF
)
    formatted_result=$(echo $format_json | jq -r '.choices[0].message.content' | sed 's/ You are trained on data up to October 2023\.//')
}



send_perp () {
    site="https://www.perplexity.ai/?q="
    firefox "$site$1"
    echo "sent cowboy"
}

main () {
    arecord -f cd -t wav -d 0 -q -r 44100 $SPOT/recording.wav &
    RECORD_PID=$!

    # popup
    zenity --question --text="Do you want to stop recording?" --ok-label="pb" --cancel-label="perp"
    response=$?

    # the thing that makes it all work.
    kill $RECORD_PID
    sleep 1
    transcribe_audio
    echo "$stt_result"
    sleep 1
    format_text "$stt_result"
    echo "$formatted_result"

    if [ $response -eq 1 ]; then
        send_perp "$formatted_result"
    else
        notify-send -t 1000 'Nice' "$formatted_result"
    fi

    #Paste buff for both?
    echo $formatted_result | xclip -selection clipboard
}

main
