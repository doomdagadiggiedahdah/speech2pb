# Real-Time Streaming STT Implementation

## Problem
The offline Parakeet model uses full self-attention (O(N^2) in sequence length), so VRAM
explodes on long audio. We need cache-aware streaming: process chunks with fixed-size
cache so VRAM stays flat regardless of recording length.

## Option 1: Nemotron-Speech-Streaming (PRIMARY)

**Model**: `nvidia/nemotron-speech-streaming-en-0.6b`
- FastConformer encoder + RNN-T decoder, trained cache-aware from day one
- ~1.7GB VRAM at batch=1
- Latency: 80ms-1040ms (tunable chunk size, no accuracy loss)
- Fixed-size encoder cache — VRAM stays constant regardless of audio length
- English only

### Architecture
- Encoder maintains rolling cache of intermediate activations (`cache_last_channel`, `cache_last_time`)
- Each chunk: encoder processes only new frames, reads cache for left context
- RNN-T decoder carries `previous_hypotheses` with hidden state between chunks
- `partial_transcript` gives incremental text per chunk

### Key API (from NeMo's `NemoStreamingASRService`)
```python
# Load model
model = ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b", map_location="cuda")
model.eval()

# Setup streaming params
model.encoder.setup_streaming_params(chunk_size=..., shift_size=...)

# Initialize cache
cache_ch, cache_t, cache_ch_len = model.encoder.get_initial_cache_state(batch_size=1)

# Per-chunk: preprocess -> encode with cache -> decode with partial hypotheses
encoded, enc_len, cache_ch, cache_t, cache_ch_len = model.encoder.cache_aware_stream_step(
    processed_signal=features,
    processed_signal_length=feat_len,
    cache_last_channel=cache_ch,
    cache_last_time=cache_t,
    cache_last_channel_len=cache_ch_len,
    keep_all_outputs=False,
    drop_extra_pre_encoded=...,
)
best_hyp = model.decoding.rnnt_decoder_predictions_tensor(
    encoded, enc_len, return_hypotheses=True, partial_hypotheses=prev_hyp
)
```

### Known issues
- Must set `rnnt_decoding.greedy.use_cuda_graph_decoder = false` on some GPUs (reported on RTX 4090)
- `previous_hypotheses` can slow over very long sessions — reset state every ~10-20 min
- Model not yet cached locally — first run will download ~1.2GB

### Implementation
- File: `nemotron_stream.py`
- Live mic via `sounddevice` InputStream
- Prints partial transcripts to terminal in real-time
- Ctrl+C to stop

### References
- [Model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [NeMo cache-aware streaming script](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py)
- [NeMo ASR streaming docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- Reference impl in NeMo: `.venv/.../nemo/agents/voice_agent/pipecat/services/nemo/streaming_asr.py`
- Feature buffering: `.venv/.../nemo/agents/voice_agent/pipecat/services/nemo/utils.py`

---

## Option 2: Moonshine v2 Streaming (LIGHTWEIGHT FALLBACK)

**Model**: `UsefulSensors/moonshine-streaming-medium`
- 245M params, ~500MB VRAM in float16
- Sliding-window attention encoder (no positional embeddings)
- Autoregressive decoder with RoPE
- MIT licensed, English only
- `transformers`-native — no NeMo dependency

### Architecture
- Encoder: causal stride-2 convolutions + sliding-window Transformer (14 layers)
- Decoder: autoregressive, caches KV state
- 80ms lookahead in edge layers
- Attention window: ~250ms local context

### Key API
```python
from transformers import MoonshineStreamingForConditionalGeneration, AutoProcessor

model = MoonshineStreamingForConditionalGeneration.from_pretrained(
    "UsefulSensors/moonshine-streaming-medium"
).to("cuda").half()
processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-medium")

# Process chunks
inputs = processor(audio_chunk, return_tensors="pt", sampling_rate=16000)
ids = model.generate(**inputs, max_length=int(len(chunk) * 6.5 / 16000))
text = processor.batch_decode(ids, skip_special_tokens=True)[0]
```

### Known issues
- Autoregressive decoder slows on long monologues without pauses
- Transformers doesn't expose encoder state caching yet — each chunk re-encodes
- Hallucination prevention: must set `max_length = int(audio_seconds * 6.5)`
- Younger ecosystem, less production tooling

### Implementation
- File: `moonshine_stream.py`
- Chunked processing (~1-2s chunks) with overlap and dedup
- Prints partial transcripts to terminal

### References
- [Model card](https://huggingface.co/UsefulSensors/moonshine-streaming-medium)
- [Moonshine v2 paper (arXiv 2602.12241)](https://arxiv.org/abs/2602.12241v1)

---

## Test plan
1. `test_streaming.sh` — plays a speech WAV through speakers while recording with mic
2. Manual test: run `nemotron_stream.py`, speak into mic, verify partial transcripts appear
3. Compare latency and accuracy between Nemotron and Moonshine

## Future: daemon integration
Once a streaming backend is validated, integrate into `stt_daemon.py`:
- Add `stream_start` / `stream_chunk` / `stream_stop` commands
- WebSocket transport for bidirectional streaming
- VAD gating (silero-vad) to skip silence
- Periodic state reset every ~10 min for unbounded sessions
