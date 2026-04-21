# Lessons Learned

## Local GPU Whisper — cuDNN / ctranslate2 version mismatch (2026-04-20)

### Problem
faster-whisper running on GPU (Quadro T1000, CUDA 12.0) produced consistent hallucinations:
- Japanese ("ご視聴ありがとうございました" — "Thank you for watching") on large-v3
- English repetition loops ("go ahead and go ahead and...") on medium/small
- Exclamation marks on float16

This happened across all compute types (int8, int8_float16, float16), all model sizes, with VAD on/off, with different audio files and formats. CPU inference worked correctly throughout.

### Root Cause
ctranslate2 **bundles its own cuDNN** inside the package (`ctranslate2.libs/`). ctranslate2 4.6.1 bundles cuDNN 9.1.0.

Whisper's audio encoder runs **1D convolutional layers** on the mel spectrogram before any matrix math. These conv layers use cuDNN. cuDNN 9 changed the API — the convolutions silently produced garbage activations. Everything downstream (language detection, token sampling, the decoder) hallucinated because it was working from corrupted encoder output.

This is why:
- Forcing `language='en'` didn't help — language detection is post-encoder
- Disabling VAD didn't help — VAD is pre-encoder
- The hallucinations were consistent per model, not random

float32 worked on the small model because small + float32 fits in VRAM and float32 convolutions don't use the same cuDNN code path.

### Fix
Downgrade ctranslate2 to 4.4.0, which bundles cuDNN 8.9.7:

```bash
uv pip install "ctranslate2==4.4.0"
```

After downgrade: large-v3 + int8 + GPU works correctly, uses ~1.5GB VRAM on the 4GB T1000.

### Key Diagnostic Step
```bash
ldd .venv/lib/python3.12/site-packages/ctranslate2/*.so | grep cudnn
```
This reveals which cuDNN is *actually* being used — not the system one, not the pip-installed one, but the one bundled inside ctranslate2.libs. LD_PRELOAD and LD_LIBRARY_PATH cannot override this.

### Environment
- GPU: Quadro T1000 Max-Q (TU117), 4GB VRAM, CUDA 12.0
- cuDNN 9.6.0 installed system-wide (installed to fix initial "cannot load libcudnn_ops" error)
- ctranslate2 4.6.1 → downgraded to 4.4.0
- faster-whisper 1.2.1
