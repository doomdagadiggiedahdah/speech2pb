# TODO

## Benchmarks
- [ ] Test `atempo=1.5` speed-up vs no speed-up with large-v3 on GPU — measure accuracy (WER) and latency tradeoff. Hypothesis: ~no accuracy loss at 1.5x but meaningful time saving since model processes less audio.

## Features
- [ ] Streaming / VAD-triggered transcription (detect end-of-utterance, transcribe in real time instead of record-then-process)
- [ ] **Try `parakeet_realtime_eou_120m`** — nvidia's purpose-built live dictation model (120M params),
      end-of-utterance detection built in, much smaller than 0.6b. Ref chat: https://claude.ai/chat/188d1684-521c-4c01-becd-904f79def2e9
      HuggingFace: https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1
