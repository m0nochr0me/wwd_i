# wwd_i — Phased Implementation Plan

Companion to [`architecture.md`](./architecture.md). Each phase has an explicit
**verify** gate; do not advance until it passes. Phases are ordered so each one
unblocks the next; early phases de-risk the parts most likely to be wrong.

Legend: 🎯 deliverable · ✅ verify gate · ⚠️ risk/decision.

---

## Phase 0 — Scaffolding & runtime harness
Lay down structure and prove we can stream audio through an (empty) ONNX pipeline.

- 🎯 Repo layout:
  ```
  wwd_i/
    src/wwd_i/
      audio/      # capture, resample, framing, ring buffers
      features/   # mel front-end (torch + onnx)
      models/     # backbone, head definitions
      train/      # backbone & head training loops
      data/       # sample gen (ElevenLabs), augmentation, datasets
      runtime/    # streaming inference engine, post-processing
      export/     # onnx export + quantization
    docs/
    tests/
    configs/      # yaml/py configs (audio, mel, model, train)
  ```
- 🎯 Central config (audio/mel constants in one place, imported everywhere).
- 🎯 Audio ingest: file + live mic → normalized 16 kHz mono float32, 80 ms framing,
  ring buffer.
- 🎯 ORT harness that loads an ONNX model and runs the stream loop (identity model ok).
- ✅ **Verify**: feed a wav file, confirm correct frame count/timing and a clean
  stream loop at real-time; `uv run` entrypoint works on x86_64.

## Phase 1 — Mel front-end (+ ONNX parity)
The contract every later stage depends on.

- 🎯 PyTorch/`torchaudio` log-mel with the §4 params.
- 🎯 Export to `melspec.onnx`; streaming wrapper over the ring buffer.
- ✅ **Verify**: torch vs ONNX outputs match within tolerance (e.g. max abs err
  < 1e-3) on a battery of clips; streaming output equals batch output on the same
  audio. Unit tests in `tests/`.

## Phase 2 — Backbone: architecture + training (the hard core)
Build and train the reusable frozen embedding model. **Highest-risk phase.**

- 🎯 Implement backbone (BC-ResNet default, swappable) producing a `D`-dim embedding.
- 🎯 Backbone training pipeline on a large keyword corpus (MSWC / Common Voice /
  LibriSpeech-derived) with **prototypical/triplet** metric loss (Path A).
- 🎯 Train on Colab GPU; export frozen `backbone.onnx`.
- ✅ **Verify (embedding quality, before touching wake words)**: few-shot probe —
  on **held-out, unseen** words, a kNN/prototype classifier over embeddings hits a
  pre-set accuracy bar (e.g. clearly above a mel-only baseline). This proves the
  embedding generalizes; if it fails, no head will work.
- ⚠️ **[DECISION]** objective Path A vs +B (SSL distillation); block size; `D`.
  Iterate architecture/training here, not later.

## Phase 3 — Data generation & augmentation (per word)
Streamlined, high-quality, few-sample positives + a strong negative pipeline.

- 🎯 ElevenLabs v3 sample generator: phrase → diverse clips (voices/accents/prosody),
  cached, deduped, loudness-normalized.
- 🎯 Hard-negative generation (near phrases / sub-phrases) + background negative
  loaders (speech/music/noise corpora).
- 🎯 On-the-fly augmentation: RIR, additive noise/music at varied SNR, gain,
  pitch/time/speed, codec/clipping.
- ✅ **Verify**: reproducible dataset for one wake word ("hey computer") with
  documented counts; listen-tests + spectrogram spot-checks confirm clips are
  on-phrase and augmentation is sane; class balance reported.

## Phase 4 — Wake-word head: train, evaluate, calibrate
First end-to-end detector for one word.

- 🎯 Head (streaming GRU default; windowed alt available) on **frozen** embeddings.
- 🎯 Training loop consuming Phase-3 data through Phase-1/2 features.
- 🎯 Export `<word>_head.onnx`; calibrate threshold + refractory.
- ✅ **Verify**: on a held-out augmented eval set, hit **FA < 0.5/hr** and
  **FR < 5%**; produce a DET/ROC curve and the chosen operating point. This is the
  project's first real success signal.
- ⚠️ **[DECISION]** head type confirmed by results here.

## Phase 5 — Streaming inference engine
Make it run continuously and correctly in real time.

- 🎯 Engine wiring mel → backbone → head with rolling buffers + (GRU) state,
  threshold, smoothing, refractory; clean detection events with timestamps.
- 🎯 Single-process always-on loop; live-mic demo.
- ✅ **Verify**: streaming detections match offline batch detections on the same
  audio (no edge/seam errors); live mic triggers on the word, ignores chatter over
  a multi-minute soak; latency measured (target sub-frame, ≤ a few hundred ms).

## Phase 6 — Cross-arch optimization & packaging
Hit the SBC budget and ship artifacts.

- 🎯 int8 quantization (dynamic → static/QAT if needed) with accuracy regression
  guard; XNNPACK EP eval on ARM64.
- 🎯 Benchmarks on **x86_64** and **RPi5 (ARM64)**: CPU %, RAM, latency per word
  and for N simultaneous words.
- 🎯 Package the three ONNX artifacts + wrapper; usage docs; "add a new wake word"
  guide (only Phases 3–4 rerun, backbone reused).
- ✅ **Verify**: meets §10 footprint/compute targets on RPi5 with no accuracy
  regression beyond an agreed delta; cold install + run reproduces a detection.

---

## Critical path & risk order
```
P0 ─ P1 ─ P2 ─ P3 ─ P4 ─ P5 ─ P6
            ▲          ▲
       biggest risk   first real
       (embedding)    success metric
```
- **Phase 2** is where the project lives or dies — budget the most iteration there;
  the few-shot probe gate catches a bad backbone before wasted downstream work.
- **Phases 3–4** are the per-word loop that users repeat; keep them cheap and
  reproducible.
- Quantization (P6) can regress accuracy — gate it; never ship an unguarded int8 model.

## Milestones
- **M1** — Phases 0–1: audio + mel, ONNX parity. (Foundations.)
- **M2** — Phase 2: a frozen backbone that passes the few-shot probe. (Core IP.)
- **M3** — Phases 3–4: first word hitting FA/FR targets offline. (It works.)
- **M4** — Phases 5–6: real-time on RPi5, packaged, "add a word" documented. (Shippable.)
```
