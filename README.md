# usage
- input your OPENAI_API_KEY and save file as `cred.txt`
- change SPOT dir to your favorite location
- setup hotkey to run `main.sh` and test it out

## neat things
- this speeds up audio (for reduced latency and OPENAI cost)
- you can press enter and it'll send to your paste buffer 
- or press esc and it'll send to perplexity (wow)

## bad things
- there's no handling for runaway audio files, so if you continue to record it'll eat up your whole system.
  - you can fix this by finding the process from this command and manually killing it
  - or restart computer. yo'ure welcome
