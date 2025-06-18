# usage
- input your OPENAI_API_KEY and save file as `cred.txt`
- change SPOT dir to your favorite location
- setup hotkey to run `main.sh` and test it out
- use `main.sh --history` to see recent transcriptions (in case you copy over the paste buffer)
- or use `--debug` to see troubleshooting info

## bad things
- there's no handling for runaway audio files, so if you continue to record it'll eat up your whole system.
  - you can fix this by finding the process from this command and manually killing it
  - or restart computer. yo'ure welcome
