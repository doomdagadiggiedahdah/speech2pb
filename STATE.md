# stt_hk — where things stand (2026-11-16)

## Done
- **language="en"** added to `TRANSCRIBE_OPTIONS` in `stt_client.py` (override with `STT_LANGUAGE`).
  Whisper used to spend a whole extra pass just guessing the language. `PIPELINE` tag in
  `main.py` bumped to `unsliced-en-v1` so runs.jsonl separates before and after.
  Benchmarked through the real path (archived hotkey `fast_*.wav` files → `stt_client.send_transcribe`
  → gpu-queued, distil-large-v3.5, all on the local T1000, before/after interleaved, wall time):

  | clip (sent / spoken) | n | before med / min | after med / min | saved | words before / after |
  |---|---|---|---|---|---|
  | 22s / 32s   | 6 | 5.21 / 4.96   | 3.22 / 2.94   | 1.98s | 67 / 67 |
  | 73s / 110s  | 6 | 8.28 / 8.13   | 6.24 / 6.07   | 2.04s | 165 / 165 |
  | 242s / 363s | 4 | 25.63 / 24.63 | 23.11 / 22.98 | 2.52s | 657 / 657 |

  A flat ~2s per call regardless of length (detection is one 30s window). Queue time was 0 in every run.
  The earlier "only 1s" on `test_clip.wav` was single-run noise: rerun 3×3 it gives 14.98 → 12.91s
  median (2.1s). Not routing: hotkey jobs are `interactive`, which only the local GPU takes; the
  Jetsons only pull `batch` jobs.

## Decisions for Mat

### 1. Model mismatch (v3 vs v3.5) — DONE 2026-11-16 (Mat: "your call" → switched to v3.5)
The `FW_MODEL` default is now `distil-large-v3.5` in all three gpu-queued files (`gpu_queued.py`,
`gpuq_model.py`, `worker/gpuq_worker.py`; the worker was copied to both Jetsons' `~/gpuq-worker/`).
The model was rsynced from the laptop's HF cache to both Jetsons. All three services were restarted.
Verified: the laptop log says `model loaded (distil-large-v3.5/cuda/int8_float16)`. The orin-seahorse
`/health` shows v3.5, loaded. jetson-agent is configured for v3.5 but can't load any model right now,
because two llama-servers hold ~5.5GB of its 7.6GB. gpu-queued marks it down ("only ~1180 MB free")
and skips it. This was already the case before the switch. The runs.jsonl `model` field comes from
`main.py`'s `BENCH_MODELS`, not from the service, so it is now correct but still unchecked against
the service.

### (original note)
**Plain words:** The transcription service has quietly been using the older model
(distil-large-v3) all along, while the log records every run as the newer one (v3.5).
They're equally fast, but they produce slightly different text, about 7 words in 100.
**Recommendation:** Switch the service to v3.5, since that's what you thought you'd chosen and
what the newer benchmarks used. Set `FW_MODEL=distil-large-v3.5` in the gpu-queued systemd unit
and restart it, then spot-check a few dictations. If you'd rather keep v3, fix the log label instead.

### 2. Live transcription — BUILT & LIVE 2026-11-16 (on by default; `STT_ONLINE=0` disables)
`online.py` (OnlineTranscriber): one streaming ffmpeg atempo of the recorder's chunks, whole-stream
Silero VAD (+ energy detector for the close mic), 30s windows sent to gpu-queued during
recording (word timestamps, no prompt, v2p0 commit/dedupe/quality filters), tail at Stop.
Any error/timeout/empty result → automatic fallback to the old single call (`online_fallback` in runs.jsonl).
gpu-queued gained `word_timestamps` + per-segment quality fields (restarted 2026-11-16).
Replay (real gpu-queued, 3-4× speed), wait after Stop vs old single call:

| file len | mic | windows | online wait | single wait | word diff (single-vs-single noise) |
|---|---|---|---|---|---|
| 51s | default | 1 | 4.01 | 3.47 | .022 |
| 122s | default | 3 | 2.62 | 6.36 | .089 (.015) |
| 242s | default | 5 | 3.65 | 15.71 | .102 (.056) |
| 508s | default | 14 | 3.84 | 27.81 | .147 (.149) |
| 695s | default | 23 | 4.55 | 42.75 | .102 |
| 218s | close | 6 | 4.43 | 16.54 | .084 (.030) |
| 450s | close | 18 | 5.35 | 29.20 | .238 (.100) |
| 90s silent | both | 0 (2 skipped) | ~0.05 | 1.1–7.4 | 0 words |

Watch-outs: whispered close-mic audio is shaky (one 42-word run lost on 193845, unclear if online-specific);
short dictations can wait ~1-2s *longer* if a window is mid-decode at Stop; windows preempt local batch
jobs while recording; archived fast wav is now 16 kHz. Replay harness: `replay_online.py`.

**Original plain words:** Right now nothing happens until you press Stop, so a long dictation means a long
wait (about 25s after an 8-minute ramble). Instead, the hotkey could send each 30s chunk to the GPU while you're
still talking, so after Stop only the last few seconds remain, about 3s whatever the length.
**Recommendation:** Yes, build it. It's tested in replay (`bench_online2.py --prompt 0`): accuracy
matches a one-shot run within normal wobble, and it no longer invents text during silence.
It's a medium-sized job, so hand it to an agent and try it on real dictations before trusting it.

### 3. Cleanup — DONE 2026-11-16 (Mat: kill loops only, keep data)
Stuck loops are gone. The scratch folders were moved, not deleted, into the gitignored
`.output/investigations/2026-11-15_online-and-floor/` (seam/, floor/, diag*.py). They were checked with diff before the originals were removed.

### (original note)
**Plain words:** The test agents left about 9 tiny "wait until X finishes" shell loops that will never exit.
They're a bug in how they were written: each loop keeps finding itself. They also left two scratch
folders holding transcripts of your recordings: `scratchpad/seam/hyp/` and `scratchpad/seam/data/`,
under `/tmp/claude-1000/-home-mat-Documents-ProgramExperiments-stt-hk/3d1a891c-c16a-4d39-97d1-3df835466d2a/`.
**Recommendation:** Kill the loops and delete both folders. Nothing needs them, and the transcripts
are personal data sitting in /tmp. The folders go away on reboot anyway.

## Reference
- Per-call cost breakdown + speedup table: memory `whisper-per-call-cost.md`; agent scripts in the
  session scratchpad `floor/` (`analyze.py`).
- Online-transcription algorithm details: `bench_online2.py`, results in `.output/bench_online/results2.jsonl`.
