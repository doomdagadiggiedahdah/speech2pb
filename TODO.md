# TODO

## Branches
- [ ] Merge `local-whisper-gpu` into `main` (ctranslate2 4.6.3 fix validated, pyproject/lockfile added)

## Benchmarks
- [ ] Test `atempo=1.5` speed-up vs no speed-up with large-v3 on GPU — measure accuracy (WER) and latency tradeoff. Hypothesis: ~no accuracy loss at 1.5x but meaningful time saving since model processes less audio.
- [ ] Profile the full pipeline (record → atempo → whisper → clipboard) to find bottlenecks

## Features
- [ ] **Minimize hotkey-to-recording latency** — currently ~0.3s before audio capture starts;
      goal is to begin recording as close to instantaneous as possible so no leading words
      are lost. Things to investigate:
      - Python startup time is the main culprit — even with lazy imports, the interpreter
        itself takes ~100ms to start. Options:
        - **Daemon/socket server** (overlaps with item below) — keep a process already running
          that starts recording the instant it receives a signal (e.g. a unix socket or
          `kill -USR1`). Zero startup cost, sub-ms response.
        - **Pre-fork / preload** — have a long-lived process `fork()` on hotkey press so
          the child inherits an already-initialized interpreter and open audio stream.
        - **`pyc` warm cache** — minor, but ensure `.pyc` files are current so bytecode
          compilation doesn't add to cold-start time.
        - **Ring buffer always recording** — keep a small (e.g. 2s) rolling audio buffer
          running in the daemon so even if there's latency in receiving the signal, the
          first words are already captured.
- [ ] Parakeet TDT 0.6B v2 backend (`parakeet-backend` branch) — in progress
- [ ] **Eager processing** — chunk audio during recording, transcribe each chunk in background, dedup at boundaries. Gives near-zero post-recording latency without model-level streaming complexity. Preferred over cache-aware streaming for hotkey use case.
- [ ] **parakeet_realtime_eou_120m** — nvidia's purpose-built live dictation model with end-of-utterance detection built in. Very new (2026), worth revisiting once more battle-tested. Ref: https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1
- [ ] **Daemon/socket server** ⭐ HIGH PRIORITY — eliminates interpreter startup latency,
      keeps model loaded between calls, and serialises concurrent hotkey presses safely.
      Architecture:
      - `stt_daemon.py` — long-lived process, starts at login via systemd user service.
        Holds the model in VRAM. Listens on a unix socket at `/tmp/stt_hk.sock`.
        Handles "start" / "stop" / "status" messages. On "start": if idle, begins
        recording immediately; if busy (already recording or transcribing), queues the
        request and processes it next — **no double model load, no OOM crash**.
        Sends result back over the socket when done.
      - `stt_trigger.py` — tiny script the hotkey calls. Connects to socket, sends
        "start" or "stop" depending on daemon state, exits immediately. Falls back to
        launching `main.py` directly if daemon isn't running.
      - Toggle behaviour: same hotkey press starts and stops (state lives in daemon).
      - Pairs well with ring buffer (see above) — daemon always records last ~2s so
        no leading words are lost even with scheduling jitter.
      - **Solves the concurrent-invocation problem**: if hotkey is pressed while a
        transcription is already in progress, the second request queues rather than
        spawning a second model load (which would OOM the 4GB GPU).
