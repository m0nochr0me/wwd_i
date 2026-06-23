# wwd_i — Integration Guide

How to embed the `wwd_i` wake-word detector into your own application: install,
the runtime API, the artifacts you need, and the rules that keep detection
accurate in production.

This is the _how to use it_ doc. For _why it's built this way_ see
[architecture.md](architecture.md); for the _build phases_ see
[implementation-plan.md](implementation-plan.md). `src/wwd_i/config.py` and the
source are ground truth — where the older docs disagree, the code wins.

---

## 1. What you're integrating

A three-stage, always-on streaming pipeline. 16 kHz mono audio in, `Detection`
events out.

```
audio → mel front-end → frozen backbone (shared) → per-word head → threshold + refractory → Detection
        melspec.onnx     backbone.onnx              <word>_head.onnx
```

- **Inference is torch-free.** The runtime imports only `onnxruntime`, `numpy`,
  `soundfile`, and `soxr`. No PyTorch, no `datasets`, nothing from the training
  stack ships or loads at inference time.
- **The mel front-end and backbone are word-independent and shared.** Each 80 ms
  hop is embedded once; every head scores that same embedding. Running N words
  costs one backbone pass plus N tiny heads — not N pipelines.
- **Three ONNX artifacts.** `melspec.onnx` (~864 KB) and `backbone.onnx`
  (~204 KB) ship inside the package and are loaded automatically. The
  `<word>_head.onnx` (~100 KB) is per-word and you supply it by path.

| Stage           | Artifact           | Ships in package?     | I/O                                                     |
| --------------- | ------------------ | --------------------- | ------------------------------------------------------- |
| Mel front-end   | `melspec.onnx`     | yes (`wwd_i/models/`) | `audio[1,N] → logmel[1,frames,32]`                      |
| Backbone        | `backbone.onnx`    | yes (`wwd_i/models/`) | `mel[B,frames,32] → embedding[B,96]`                    |
| Head (per word) | `<word>_head.onnx` | no — you provide it   | `embedding[1,1,96], h0[1,1,48] → prob[1,1], hn[1,1,48]` |

You normally never touch the ONNX I/O directly — `WakeWordEngine` wires all three
together. The contract is listed for debugging and for swapping artifacts.

---

## 2. Install

`uv` project, Python 3.14. Distribution name `wwd-i`; import name `wwd_i`.

```bash
# Inference only (torch-free — this is what you ship):
uv sync

# or in a plain venv:
pip install wwd-i
```

That pulls only `numpy`, `onnx`, `onnxruntime`, `soundfile`, `soxr`. The
training/export/notebook stack (`torch`, `datasets`, a TTS backend, …) lives in
the `train` dependency group and is never needed to run detection.

**Microphone capture is optional.** `wwd_i.audio.mic_frames` needs
`sounddevice` (PortAudio), which is _not_ a runtime dependency. Add it only if
you use the built-in mic source: `uv add sounddevice`. If you already have your
own audio capture, you don't need it — just feed samples to `engine.push()`.

---

## 3. What you need per wake word

Each word is two files that live together, normally in `artifacts/`
(gitignored):

1. `<word>_head.onnx` — the trained head.
2. `<word>_head.json` — its calibration (operating point + metrics):

```json
{
  "word": "Aliyah",
  "threshold": 0.99,
  "refractory_seconds": 1.0,
  "fa_per_hr": 0.399,
  "fr": 0.108,
  "passed": false
}
```

The engine reads `word`, `threshold`, and `refractory_seconds` from the JSON
that sits beside the head (same stem, `.json` extension). `fa_per_hr` / `fr` /
`passed` are the calibration metrics — informational, not used at runtime.

- If the JSON is missing you **must** pass `threshold=` explicitly, or
  construction raises `ValueError`.
- `word` defaults to the head's filename stem; `refractory_seconds` defaults to
  `1.0`.

Heads are produced by the training pipeline (Phase 4,
`python -m wwd_i.train.train_head`). For integration you only consume the two
files.

---

## 4. Quick start (CLI)

The package installs a `wwd-i` command.

```bash
# Detect on a file:
uv run wwd-i speech.wav --head artifacts/Aliyah_head.onnx

# Detect on the microphone (needs sounddevice):
uv run wwd-i --mic --head artifacts/Aliyah_head.onnx

# Several words off the one shared backbone (repeat --head):
uv run wwd-i --mic --head artifacts/Samara_head.onnx --head artifacts/whispers_Samara_head.onnx
```

Each detection prints as `⚡ '<word>' @ <time>s score=<P(wake)>`. Useful flags:

| Flag                 | Effect                                                                     |
| -------------------- | -------------------------------------------------------------------------- |
| `--threshold X`      | override the calibrated threshold (single operating point)                 |
| `--refractory S`     | override the debounce window in seconds                                    |
| `--calibration PATH` | use a non-sibling calibration JSON (single `--head` only)                  |
| `--debug`            | print input level (dBFS) and max `P(wake)` ~once/sec, even below threshold |
| `--save out.wav`     | dump captured mic audio for offline replay                                 |
| `--no-normalize`     | disable input AGC — **debug only**, see §8                                 |

Running `wwd-i <file>` with no `--head` is the throughput harness, not
detection; it reports a real-time factor and nothing else.

---

## 5. Quick start (library)

The whole API is `WakeWordEngine.push()`. Feed it 16 kHz mono float32 audio in
chunks of any size; it returns the detections that fired during that call.

```python
import numpy as np
from wwd_i.runtime import WakeWordEngine
from wwd_i.audio import load_wav

engine = WakeWordEngine("artifacts/Aliyah_head.onnx")  # threshold/refractory from the sibling .json

# Offline: push a whole file at once.
audio = load_wav("speech.wav")        # mono float32 @ 16 kHz (resampled if needed)
for det in engine.push(audio):
    print(f"{det.word} at {det.time_s:.2f}s (score {det.score:.3f})")
```

Streaming a live source — push each chunk as it arrives:

```python
from wwd_i.runtime import WakeWordEngine
from wwd_i.audio import mic_frames   # needs sounddevice

engine = WakeWordEngine("artifacts/Aliyah_head.onnx")
for frame in mic_frames():            # 80 ms float32 frames, forever
    for det in engine.push(frame):
        on_wake(det.word)             # your handler
```

`push()` is incremental and stateful: the engine keeps rolling buffers across
calls, so streaming the same audio in small chunks yields the same detections as
one big push (the streamed==batch parity gate). Chunk size is your choice — it
need not be a frame multiple.

---

## 6. API reference

Import from the two public subpackages:

```python
from wwd_i.runtime import WakeWordEngine, Detection, detect_file
from wwd_i.audio import load_wav, to_frames, file_frames, mic_frames
```

### `WakeWordEngine(heads, *, ...)`

| Parameter       | Type                                 | Default                  | Meaning                                                 |
| --------------- | ------------------------------------ | ------------------------ | ------------------------------------------------------- |
| `heads`         | `str \| Path \| Sequence[str\|Path]` | —                        | one head path, or a list to run several words at once   |
| `calibration`   | `str \| Path \| None`                | sibling `<head>.json`    | explicit calibration JSON; **single head only**         |
| `backbone_path` | `str \| Path`                        | packaged `backbone.onnx` | override the shared backbone                            |
| `mel_path`      | `str \| Path`                        | packaged `melspec.onnx`  | override the mel front-end                              |
| `threshold`     | `float \| None`                      | from calibration         | override the detection threshold (applies to all heads) |
| `refractory_s`  | `float \| None`                      | from calibration         | override the debounce window (applies to all heads)     |
| `normalize`     | `bool`                               | `True`                   | input loudness normalization (AGC) — keep on, see §8    |

Construct **one engine per audio stream**. Explicit `threshold` / `refractory_s`
override the JSON and broadcast to every head; passing `calibration=` with more
than one head raises `ValueError` (each head must use its own sibling JSON).

**Methods**

- `push(samples: np.ndarray) -> list[Detection]` — feed a chunk (any length,
  16 kHz mono float32); return detections that fired during this call.
- `reset() -> None` — clear all streaming state (buffers, rolling embedding
  window, per-head refractory timers). Call between independent streams.

**Attributes**

- `heads` — the list of loaded heads; each has `.word`, `.threshold`,
  `.refractory_s`, `.last_score`.
- `word`, `threshold`, `refractory_s` — convenience accessors for the _first_
  head (single-word case).
- `last_score: float` — most recent hop's max `P(wake)` across all heads, even
  when below threshold. For diagnostics / level meters; not a detection.

### `Detection`

Frozen dataclass returned by `push()`:

| Field    | Type    | Meaning                                                          |
| -------- | ------- | ---------------------------------------------------------------- |
| `word`   | `str`   | which head fired (lets multiple words drive different responses) |
| `time_s` | `float` | audio time at the **end** of the window that triggered           |
| `score`  | `float` | `P(wake)` at that hop                                            |

### `detect_file(engine, path) -> list[Detection]`

Convenience for offline whole-file detection: resets the engine, then pushes the
entire file. This is the batch reference for the parity gate.

```python
from wwd_i.runtime import WakeWordEngine, detect_file
engine = WakeWordEngine("artifacts/Aliyah_head.onnx")
dets = detect_file(engine, "clip.wav")
```

### Audio helpers (`wwd_i.audio`)

- `load_wav(path) -> np.ndarray` — read any soundfile-supported file as mono
  float32 in [-1, 1], resampled to 16 kHz. Hand the result straight to `push()`.
- `to_frames(signal, *, pad=False) -> Iterator` — slice a signal into 80 ms
  (1280-sample) frames.
- `file_frames(path, *, pad=False) -> Iterator` — `load_wav` + `to_frames`.
- `mic_frames(device=None) -> Iterator` — endless 80 ms frames from an input
  device (needs `sounddevice`).

You do **not** have to frame audio for the engine — `push()` accepts arbitrary
chunk sizes. The frame helpers exist for the CLI and for sources that are
naturally framed.

---

## 7. Multiple wake words

Pass a list of heads. The mel front-end and backbone run once per hop and every
head scores the shared embedding, so adding a word costs one small head session,
not a second pipeline:

```python
engine = WakeWordEngine([
    "artifacts/Samara_head.onnx",
    "artifacts/whispers_Samara_head.onnx",
])
for det in engine.push(audio):
    if det.word == "Samara":
        ...
    elif det.word == "[whispers] Samara":
        ...
```

Each head keeps its own threshold, refractory window, and fire timer from its own
sibling JSON. Dispatch on `det.word`.

---

## 8. The audio contract (read this)

The engine assumes a specific input format and loudness. Violations don't error —
they silently degrade accuracy.

- **Format: 16 kHz, mono, float32, range [-1, 1].** `load_wav` produces exactly
  this (and resamples for you). If you capture audio yourself, match it. Integer
  PCM must be scaled to float; stereo must be downmixed; other sample rates must
  be resampled to 16 kHz.
- **Loudness: keep AGC on (`normalize=True`, the default).** The pipeline expects
  input near −20 dBFS, the level heads are calibrated at. The engine applies a
  causal moving-RMS automatic gain control to pull quiet input up before the mel
  front-end. Raw, quiet input lands off the calibrated manifold and detection
  becomes unreliable. `normalize=False` / `--no-normalize` is for A/B debugging
  only.
- **Cadence: designed for real time** (12.5 hops/s, one 80 ms hop at a time), but
  `push()` itself imposes no timing — you can stream faster than real time for
  batch processing or slower for a throttled source. Detection results are
  identical either way.

---

## 9. Operating point: threshold & refractory

Detection fires when `P(wake) ≥ threshold`, then a **refractory** window
suppresses repeat fires for `refractory_seconds` so one utterance yields one
event.

- **Threshold** trades false accepts against false rejects. The calibrated value
  in `<head>.json` is the chosen operating point (the product gate targets
  FA < 0.5/hr and FR < 5%). Raise it to fire less; lower it to fire more.
- **Refractory** is debounce, default 1.0 s. Raise it if a single long word
  double-fires; lower it if you need rapid back-to-back detections.

Override per engine without editing the JSON:

```python
engine = WakeWordEngine("artifacts/Aliyah_head.onnx", threshold=0.95, refractory_s=1.5)
```

To tune interactively, run the CLI with `--debug` and watch `max P(wake)` while
you speak — pick a threshold just below the values your real utterances reach and
above what background speech produces.

---

## 10. Always-on deployment

Built to run continuously on a small CPU (target: Linux x86_64 and ARM64, RPi5
class).

- **CPU: a fraction of one core.** The engine forces single-threaded ONNX
  sessions with thread spinning disabled. This is deliberate and important: ORT's
  default per-core busy-waiting worker pools peg several cores spinning through
  the idle gaps between hops for almost no real work. Don't override the session
  options back to defaults for an always-on detector.
- **Latency.** No detection can fire until the first full analysis window is
  buffered — about 0.76 s of audio — plus your chunk cadence. `time_s` marks the
  end of the triggering window.
- **Footprint per process:** ~864 KB (mel) + ~204 KB (backbone), shared across
  all words, plus ~100 KB per head. Memory is bounded — the rolling buffers are a
  few windows deep, not the whole stream.
- **Throughput:** real-time factor ~0.005 (≈200× faster than real time) on
  desktop x86_64, so the per-hop compute sits comfortably inside the 80 ms budget
  with large headroom for several words.

---

## 11. Worked example: react to a wake word

```python
from wwd_i.runtime import WakeWordEngine
from wwd_i.audio import mic_frames

def on_wake(word: str) -> None:
    print(f"heard {word!r} — waking up")
    # trigger your assistant / start ASR / ring a bell …

def main() -> None:
    engine = WakeWordEngine([
        "artifacts/Samara_head.onnx",
        "artifacts/whispers_Samara_head.onnx",
    ])
    print("listening — Ctrl-C to stop")
    try:
        for frame in mic_frames():
            for det in engine.push(frame):
                on_wake(det.word)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
```

Replace `mic_frames()` with your own audio source if you already capture audio —
yield 16 kHz mono float32 chunks and feed each to `engine.push()`.

---

## 12. Rules & gotchas

Things that won't raise an error but will quietly break detection if ignored:

1. **One engine per stream; `reset()` between streams.** The engine is stateful.
   Reusing it across two unrelated recordings without `reset()` leaks buffered
   audio and refractory timers from one into the other. `detect_file` resets for
   you.
2. **Keep AGC on.** See §8. The single most common cause of "it never fires on my
   mic" is quiet, un-normalized input.
3. **A head and its calibration JSON are a matched pair.** Ship both. The
   threshold in the JSON is meaningless for a different head, and a missing JSON
   forces you to supply `threshold=` by hand.
4. **Heads are bound to the backbone they were trained on.** The shipped
   `backbone.onnx` defines the embedding space. If you swap `backbone_path` for a
   different backbone, every existing head is invalidated and must be retrained —
   the file I/O is drop-in but the embedding space is not.
5. **Don't carry a single GRU state across the stream.** The engine deliberately
   re-scores each decision from a zero hidden state over the trailing window (the
   exact criterion the head was trained on). This is internal, but worth knowing
   before "optimizing" it: a persistent streaming state makes `P(wake)` collapse
   to ~0 after a few seconds.

---

## 13. Troubleshooting

| Symptom                                                | Likely cause                                   | Fix                                                                                   |
| ------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| Never fires on mic, even at a low threshold            | Input too quiet / AGC off / wrong input device | Keep `normalize=True`; check the device with `--debug` (watch dBFS and `max P(wake)`) |
| Fires constantly / on silence                          | Threshold too low, or AGC amplifying room tone | Raise `threshold`; verify input level with `--debug`                                  |
| One word double-fires                                  | Refractory too short                           | Raise `refractory_s`                                                                  |
| `ValueError: no threshold for <head>`                  | Calibration JSON missing                       | Place `<head>.json` beside the head, or pass `threshold=`                             |
| `RuntimeError: microphone capture needs 'sounddevice'` | Optional dep not installed                     | `uv add sounddevice`, or supply your own audio source                                 |
| High CPU when always-on                                | ORT session options overridden to defaults     | Let the engine set single-thread / no-spinning options                                |
| File-mode and stream-mode disagree                     | Different chunking _and_ a real bug            | They should match to ~1e-4; report it — this is a tested invariant                    |

---

## 14. See also

- [architecture.md](architecture.md) — design and rationale.
- [implementation-plan.md](implementation-plan.md) — build phases.
- [data-licensing.md](data-licensing.md) — code/model licenses and head-data terms.
- `src/wwd_i/config.py` — the audio/mel contract, single source of truth.
- `src/wwd_i/runtime/engine.py` — the engine, with the streaming invariants
  documented inline.
