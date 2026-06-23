# wwd_i — open wake-word detector

A from-scratch, always-on wake-word detector: tiny CPU-only footprint, and adding
a new word costs just a small per-word head, not a retrain. Built on mid-2026
tech with a modern architecture and a streamlined, high-quality-TTS training
process — an alternative to [`openWakeWord`](https://github.com/dscripka/openWakeWord)
rather than a fork of it.

CPU-only ONNX inference, torch-free at runtime. Targets Linux x86_64 and ARM64
(Raspberry Pi 5 class).

## Features

- **Tiny, always-on.** A fraction of one CPU core; ~1 MB of shared models plus
  ~100 KB per word. Real-time factor ~0.005 on desktop x86_64.
- **Cheap "add a new word."** One frozen backbone is shared by every word; a new
  word is a small head trained on top — no backbone retrain.
- **Multi-word on one pipeline.** N words run off a single mel + backbone pass per
  hop, so each extra word is a tiny head, not a second pipeline.
- **Torch-free inference.** Runtime depends only on `onnxruntime`, `numpy`,
  `soundfile`, `soxr`. PyTorch and the training stack never ship.
- **One-line streaming API.** Feed 16 kHz mono audio to `engine.push()`; get
  `Detection` events back. Streamed output matches batch exactly.
- **Calibrated operating point.** Each word ships a threshold + refractory tuned
  to a product gate (FA < 0.5/hr, FR < 5%).

## Quick start

`uv` project, Python 3.14.

```bash
# Inference only (torch-free — this is what you ship):
uv sync
```

CLI:

```bash
# Detect on a file:
uv run wwd-i speech.wav --head artifacts/Aliyah_head.onnx

# Detect on the mic (needs sounddevice: `uv add sounddevice`):
uv run wwd-i --mic --head artifacts/Aliyah_head.onnx

# Several words off the one shared backbone (repeat --head):
uv run wwd-i --mic --head artifacts/Samara_head.onnx --head artifacts/Aliyah_head.onnx
```

Library:

```python
from wwd_i.runtime import WakeWordEngine
from wwd_i.audio import load_wav

engine = WakeWordEngine("artifacts/Aliyah_head.onnx")  # threshold/refractory from sibling .json
for det in engine.push(load_wav("speech.wav")):
    print(f"{det.word} at {det.time_s:.2f}s (score {det.score:.3f})")
```

`push()` accepts chunks of any size and is stateful across calls — stream live
audio frame by frame, or push a whole file at once.

## How it works

Three-stage streaming pipeline; 16 kHz mono, 80 ms hop.

```
audio → mel front-end → frozen backbone (shared) → per-word head → threshold + refractory → Detection
        melspec.onnx     backbone.onnx              <word>_head.onnx
```

The mel front-end and backbone are word-independent and ship in-package; each word
adds one small head. See [docs/integration-guide.md](docs/integration-guide.md)
for the full runtime API, the audio contract, and deployment notes.

## Training

Words are trained on Google Colab GPU (high-quality TTS positives + mined
negatives), then exported to ONNX. The TTS backend is pluggable
(`src/wwd_i/data/tts.py`) — prefer a permissively-licensed local TTS so the
pipeline stays license-clean; see [data licensing](docs/data-licensing.md).
Install the training stack with `uv sync --group train`; see the notebooks under
`notebooks/` and the design docs.

## Docs

- [Integration guide](docs/integration-guide.md) — embed it: install, API, gotchas.
- [Architecture & design](docs/architecture.md)
- [Phased implementation plan](docs/implementation-plan.md)
- [Licensing & data provenance](docs/data-licensing.md) — code/model licenses, head-data terms.
