# TODO

## Branches
- [ ] Merge `local-whisper-gpu` into `main` (ctranslate2 4.6.3 fix validated, pyproject/lockfile added)

## Benchmarks
- [ ] Test `atempo=1.5` speed-up vs no speed-up with large-v3 on GPU — measure accuracy (WER) and latency tradeoff. Hypothesis: ~no accuracy loss at 1.5x but meaningful time saving since model processes less audio.
- [ ] Profile the full pipeline (record → atempo → whisper → clipboard) to find bottlenecks

## Features
- [ ] Parakeet TDT 0.6B v2 backend (`parakeet-backend` branch) — in progress
- [ ] Streaming / VAD-triggered transcription (detect end-of-utterance, transcribe in real time instead of record-then-process)
- [ ] Keep model constantly loaded as a daemon/socket server to skip model load time on each invocation
