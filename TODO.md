# TODO

## Benchmarks
- [ ] Test `atempo=1.5` speed-up vs no speed-up with large-v3 on GPU — measure accuracy (WER) and latency tradeoff. Hypothesis: ~no accuracy loss at 1.5x but meaningful time saving since model processes less audio.

## Features
- [ ] Streaming / VAD-triggered transcription (detect end-of-utterance, transcribe in real time instead of record-then-process)
