# stt_hk todo

- [ ] profile the full pipeline (record → atempo → whisper → GPT → clipboard) to find bottlenecks
- [x] test keeping the small model constantly loaded (e.g. as a daemon/socket server) to skip model load time on each invocation
- [x] set up stt_daemon as a systemd user service for auto-start
- [ ] clipboard manager integration — queued transcriptions overwrite each other in the paste buffer; use a clipboard stack (e.g. gpaste, clipman) so nothing is lost
- [ ] "fire and forget" destination targeting — after recording, tag a destination window (e.g. click on a Signal chat, a text field, etc.) and the daemon delivers the transcribed text there when ready. Lets you move on immediately without waiting for transcription. Could use xdotool to capture the target window ID + cursor position, then focus/paste when done.
