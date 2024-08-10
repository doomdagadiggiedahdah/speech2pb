#!/bin/bash

### Trajectory ###
# - recorder starts / pop-up with button 
# - solo; "stop" is its focus
# - enter to end
# - audo; *sent* to whisper
# - text response / in paste buffer
# - program ends

# replace this directory with your own
SPOT="/home/mat/Documents/ProgramExperiments/speech2txt_hk"
source $SPOT/cred.txt

transcribe_audio () {
    text=$(curl https://api.openai.com/v1/audio/transcriptions \
          -H "Authorization: Bearer $OPENAI_API_KEY" \
          -H "Content-Type: multipart/form-data" \
          -F file="@$SPOT/recording.wav" \
          -F model="whisper-1"
    )
    text=$(echo $text | jq -r '.text')
}

send_perp () {
    #TODO: send to perp
    site="https://www.perplexity.ai/?q="
    firefox "$site$text&focus=internet"
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

    if [ $response -eq 1 ]; then
        send_perp
    else
        notify-send -t 1000 'Nice' "$text"
    fi

    #Paste buff for both?
    echo $text | xclip -selection clipboard
}

main
