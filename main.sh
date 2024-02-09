#!/bin/bash

### Description ###
# - audio recorder starts / window pops up with button that says "stop" and focus is on the button
# - press enter
# - audio is sent to whisper
# - text result is put into paste buffer
# - notification of completion pops up

# replace this directory with your own
SPOT="/home/mat/Documents/ProgramExperiments/speech2txt_hk"
arecord -f cd -t wav -d 0 -q -r 44100 $SPOT/recording.wav &
RECORD_PID=$!

# popup
zenity --question --text="Do you want to stop recording?"

# If "Yes" is clicked, kill the recording process
if [ $? = 0 ]; then
    kill $RECORD_PID
fi

# the thing that makes it all work.
# needs just a sec to create the new file I guess. lol
sleep 1

source $SPOT/cred.txt
text=$(curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@$SPOT/recording.wav" \
  -F model="whisper-1"
)

# parse and pasteBuff
text=$(echo $text | jq -r '.text')
echo $text
echo $text | xclip -selection clipboard
notify-send -t 1000 'Nice' "$text"
