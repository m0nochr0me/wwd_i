# wwd_i — Licensing & data provenance

What `wwd_i` is licensed under, what its dependencies and shipped models carry,
and — the part that actually constrains you — where the training data for a
distributed head may come from. This is engineering guidance, not legal advice;
verify the upstream terms yourself before redistributing.

## Code

`wwd_i`'s own source is **MIT**. Permissive, no copyleft. Nothing below forces a
different choice — the strongest pull would be the ML patent grant in Apache-2.0,
which MIT lacks; if patents aren't a concern, MIT is the right call.

## Runtime dependencies (what ships in the wheel)

All permissive; no strong copyleft (GPL/AGPL) anywhere in the runtime closure.

| Package       | License                                            |
| ------------- | -------------------------------------------------- |
| numpy         | BSD-3-Clause (and 0BSD / MIT / Zlib / CC0 vendored) |
| onnx          | Apache-2.0 (+ protobuf BSD-3, ml-dtypes Apache-2.0) |
| onnxruntime   | MIT (+ flatbuffers Apache-2.0)                      |
| soundfile     | BSD-3-Clause — **bundles libsndfile (LGPL-2.1+)**   |
| soxr          | LGPL-2.1-or-later — **bundles libsoxr (LGPL-2.1+)** |

**The two LGPL native libs are not a problem for an MIT project.** `libsndfile`
(via `soundfile`) and `libsoxr` (via `soxr`) ship as separate pip wheels with
their own license files and are dynamically loaded — the LGPL obligations stay
contained to those packages; MIT code depending on them is standard. The only
LGPL trigger ("provide a way to relink") fires if you ship a **frozen binary that
statically links** them (e.g. a PyInstaller single-file bundle); a normal
pip/wheel install needs nothing. If you do freeze a binary, include the bundled
LGPL license texts and a relink path.

## Shipped models

- **`melspec.onnx`** — a deterministic STFT/log-mel transform. No training data,
  no provenance to attribute.
- **`backbone.onnx`** — trained on **MSWC** (Multilingual Spoken Words Corpus),
  which is **CC-BY 4.0**. Attribution-only, no share-alike, so it's MIT-compatible;
  give MSWC credit when you redistribute the backbone (a line in your NOTICE / model
  card). Swapping the backbone for one trained on other data changes this.

## Per-word heads (you build these — they are not shipped from this repo)

A head's weights are trained on TTS positives + hard-negatives + background
negatives. **The provenance of that data is the head-builder's responsibility**,
and it's a stronger constraint than anything above, because some sources forbid
training on their output outright.

### TTS positives / hard-negatives — pick the backend by its terms

The positives train an ML model (the head), so the TTS backend's terms must
permit using its output that way. See `src/wwd_i/data/tts.py` for the backend
seam.

- **ElevenLabs (`data/elevenlabs.py`) is opt-in and restricted.** ElevenLabs'
  [Prohibited Use Policy](https://elevenlabs.io/use-policy) §9.k/§9.l forbid using
  its Output "as input for any machine learning or training of artificial
  intelligence models" or "as part of a dataset that may be used for training …
  any machine learning." That is exactly what head training does, and it is
  tier-independent (a paid commercial license does not waive the Prohibited Use
  Policy). Don't use this backend for distributed heads unless you have a custom
  agreement with ElevenLabs that covers it.
- **A permissively-licensed local TTS keeps the pipeline clean end-to-end.**
  Implement `LocalTtsBackend` (`data/local_tts.py`) on **Kokoro** (Apache-2.0),
  **Piper** (MIT), or **Parler-TTS** (Apache-2.0) — all permit training on output.
  Avoid models under non-commercial terms (e.g. Coqui XTTS / CPML).

### Background negatives

Used to train the head; check each before redistributing a head built on them:

- **AudioSet** — labels are CC-BY, but the audio is YouTube-sourced; you assemble
  it, it isn't redistributed by this project.
- **FMA** — per-track Creative Commons variants (some CC-BY, some CC-BY-NC/SA).
- **MSWC** — CC-BY 4.0 (same as the backbone corpus).
- **HF pools / vocal bursts** — per-dataset terms; check the dataset card.

## Bottom line

- Shipping the **framework** (code + `melspec.onnx` + `backbone.onnx`): MIT, plus
  an MSWC CC-BY attribution and the LGPL note for `libsndfile`/`libsoxr`.
- Distributing a **head**: clean only if its TTS + negatives permit ML-training
  use. Default to a permissive local TTS; treat the ElevenLabs path as opt-in.
