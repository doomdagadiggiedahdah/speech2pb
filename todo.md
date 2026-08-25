# stt_hk todo

- [ ] profile the full pipeline (record → atempo → whisper → GPT → clipboard) to find bottlenecks
- [x] test keeping the small model constantly loaded (e.g. as a daemon/socket server) to skip model load time on each invocation
- [x] set up stt_daemon as a systemd user service for auto-start
- [ ] clipboard manager integration — queued transcriptions overwrite each other in the paste buffer; use a clipboard stack (e.g. gpaste, clipman) so nothing is lost
- [ ] make the retry in `main.py` conditional on failure type — it currently retries any exception up to 3x, including timeouts, which a retry cannot fix. Before the result cache landed this meant re-running a full transcription just to time out again at the same wall. Now it's cheap but still wrong: a timeout should surface immediately, and only transient failures (daemon restart, model load race) should retry.
- [ ] dedup in-flight daemon requests — the result cache only populates on completion, so a client that times out *during* a transcription still queues a duplicate. Register each request's cache key in a dict of `threading.Event`s and have a matching later request wait on that event instead of re-entering `_model_lock`.
- [ ] "fire and forget" destination targeting — after recording, tag a destination window (e.g. click on a Signal chat, a text field, etc.) and the daemon delivers the transcribed text there when ready. Lets you move on immediately without waiting for transcription. Could use xdotool to capture the target window ID + cursor position, then focus/paste when done.

## Training data capture
- [ ] iterative editing — Ctrl+Alt+E on same audio should overwrite `human_edit` in existing manifest entry (match on audio_filepath) instead of appending a new one. Later: fuzzy text matching for edits after a new recording.
- [ ] nemotron_stream.py integration — buffer raw audio chunks so streaming flow can also save training samples
- [ ] bulk import script — convert ~2,468 existing raw_transcript + original_audio pairs into "accepted" training samples
