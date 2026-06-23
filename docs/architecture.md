# wwd_i — Architecture & Design

> Status: **implemented** — Phases 0–5 complete (audio + log-mel + ONNX parity, a
> frozen backbone past the few-shot probe gate, the per-word data/head pipeline,
> and the real-time streaming engine). Phase 6 (int8 quant, ARM64/XNNPACK, RPi5
> benchmarks, packaging) is not started.
>
> This document captures the design and its rationale; where it disagrees with the
> code, **the code wins** — `src/wwd_i/config.py` and the source are ground truth.
> The decisions once flagged **[DECISION]** are now resolved (BC-ResNet backbone,
> cosine-prototypical loss, GRU head, D=96, n_mels=32); see §11. For how to embed
> the runtime, see [integration-guide.md](integration-guide.md).

## 1. Goal & non-goals

**Goal.** A from-scratch, open wake-word detector that matches `openWakeWord`'s
tiny footprint and "train a new word cheaply" property, but is built on mid-2026
technology and a streamlined, high-quality training process.

**In scope**

- Always-on, streaming detection of short custom wake words/phrases.
- CPU-only inference on Linux **x86_64** and **ARM64**, tuned for a modern
  single-board server (RPi5 class: Cortex-A76, NEON, no usable NPU).
- A reusable, frozen **backbone** so adding a new wake word only trains a small head.
- Training that uses **few but high-quality** ElevenLabs v3 samples + augmentation.

**Non-goals (for now)**

- ASR / general speech-to-text.
- Far-field beamforming / multi-mic arrays.
- Microcontroller (sub-1 MB RAM) targets — we target SBC-class Linux, not MCUs.
- Speaker identification (a per-voice verifier is a possible later add-on).

## 2. Why not just reuse openWakeWord

openWakeWord is a 3-stage pipeline: ONNX mel → **frozen Google `speech_embedding`
backbone (TFHub, ~2020)** → small per-word classifier. The backbone is the legacy
bottleneck: TF heritage, fixed 96-dim, old training data, and a training recipe
that needs ~30k h of negatives and ~50k Piper-TTS positives. We keep the _shape_
of the pipeline (it's why it stays tiny and adds words cheaply) and replace the
backbone and the data pipeline.

## 3. System overview

Same streaming contract as openWakeWord so concepts transfer, but every stage is
ours.

```
 mic / PCM stream  (16 kHz, mono, 16-bit)
        │  80 ms frames (1280 samples)
        ▼
┌─────────────────────┐
│ 1. Mel front-end    │  log-mel, 32 bins, 10 ms hop  → 8 mel frames / 80 ms
└─────────────────────┘
        │  rolling mel buffer (76 frames ≈ 760 ms)
        ▼
┌─────────────────────┐
│ 2. Backbone (FROZEN)│  (76, 32) mel patch → D-dim embedding, stride 80 ms
└─────────────────────┘  ~12.5 embeddings / s   [trained once, reused for all words]
        │  rolling embedding buffer / streaming state
        ▼
┌─────────────────────┐
│ 3. Head (per word)  │  N embeddings (≈1.3 s) → P(wake)  ∈ [0,1]
└─────────────────────┘
        │
        ▼
   threshold + refractory (debounce) → detection event
```

### Audio contract (fixed)

| Param       | Value                             | Note                         |
| ----------- | --------------------------------- | ---------------------------- |
| Sample rate | 16 kHz                            | resample at ingest if needed |
| Format      | mono, 16-bit PCM / float32 [-1,1] |                              |
| Stream hop  | 80 ms = 1280 samples              | unit of streaming work       |

## 4. Stage 1 — Mel front-end

A deterministic log-mel spectrogram. Owned by us (not a black box) so backbone and
runtime agree exactly.

| Param         | Default        | Rationale                                                      |
| ------------- | -------------- | -------------------------------------------------------------- |
| `n_fft`       | 512            | covers 25 ms window at 16 kHz                                  |
| `win_length`  | 400 (25 ms)    | standard speech window                                         |
| `hop_length`  | 160 (10 ms)    | → 100 mel frames/s, 8 per 80 ms hop                            |
| `n_mels`      | 32             | footprint vs detail; matches OWW, room to try 40               |
| `fmin / fmax` | 0 / 8000       | full 16 kHz band                                               |
| compression   | `log(mel + ε)` | ε≈1e-6; per-window z-score (mean/var) norm baked into backbone |

- **Single definition, two implementations**: a reference PyTorch/`torchaudio`
  version used in training, and the **exported ONNX** version used at inference.
  A parity test (§ implementation plan, Phase 1) guarantees bit-close outputs.
- Streaming: maintain a rolling buffer; each 80 ms hop appends 8 new mel frames.

## 5. Stage 2 — Backbone (the from-scratch replacement)

Maps a `(76, 32)` mel patch (~760 ms of context) to a compact embedding, produced
once per 80 ms hop. Trained **once** on a large speech corpus, then **frozen** and
shared by every wake-word head.

### 5.1 Architecture — BC-ResNet

A small 2-D depthwise-separable CNN that collapses the frequency axis and projects
to the embedding. Chosen building block: **BC-ResNet (broadcasted residual)**
— combines a 2-D frequency conv with a 1-D temporal conv via a broadcasted residual,
giving strong KWS accuracy at very low MACs and mapping well to NEON/ONNX kernels.
Implemented in `models/backbone.py`.

```
mel (76×32×1)
  → stem conv
  → k× BC-ResNet blocks (channels grow, freq pooled to 1)   ← depthwise-separable
  → temporal aggregation (avg-pool / GAP over time)
  → linear projection → embedding (D)
```

| Param             | Default                            | Note                                        |
| ----------------- | ---------------------------------- | ------------------------------------------- |
| Embedding dim `D` | 96                                 | comparable to OWW; try 64 for smaller heads |
| Stride            | 80 ms (8 mel frames)               | one embedding per stream hop                |
| Target size       | ≤ ~1 MB fp32, ≤ ~300 KB int8       | always-on budget                            |
| Target compute    | < ~5% of one A76 core @ 12.5 inf/s | leaves room for several words               |

Alternatives considered: MatchboxNet / TC-ResNet (1-D depthwise TDNN — also good),
Tiny-CRNN (conv + GRU — streaming-friendly but RNN state complicates the _backbone_).
BC-ResNet is the chosen block; the I/O (`mel[B,T,32] → emb[B,96]`) is swappable.

### 5.2 Training objective — cosine-prototypical metric learning

For the head to learn a _new_ word from only synthetic positives + generic
negatives, the frozen embedding must generalize to **unseen** words. Two paths
were considered; **Path A shipped**:

- **Path A (shipped): supervised metric-learning over a large keyword vocab.**
  The backbone + a temporary head are trained to recognize a large set of spoken
  words from **MSWC / Multilingual Spoken Words** with a **cosine-prototypical**
  loss so embeddings of the _same_ word cluster and transfer to unseen words. This
  is the proven "few-shot KWS" recipe (and what Google's `speech_embedding`
  effectively did) — modernized architecture, newer data, current training stack.
  The temporary head is then dropped and the backbone frozen, gated by a held-out
  few-shot probe (`train/probe.py`) before any word head is trained.
- **Path B (later/optional): distillation** from a pretrained SSL encoder
  (wav2vec2 / HuBERT). Tiny student CNN regresses to teacher embeddings on
  unlabeled audio. Stronger representations, but adds teacher inference + a more
  complex pipeline. Can be layered on top of Path A — not pursued yet.

## 6. Stage 3 — Wake-word head (per word)

Consumes the embedding stream and emits `P(wake)`. **1-layer GRU, hidden 48, +
linear + sigmoid** (`models/head.py`), one head per word; the backbone is shared.
Size ~50–200 KB.

- **Written in primitive ops, not `nn.GRU`.** Two linears + explicit gates, so the
  exported ONNX matches torch to fp32. `nn.GRU`→ONNX uses a different reset-gate
  formulation that drifts ~4e-3 and breaks the torch-calibrated threshold.
- **Trained on ~1.5 s clips with a max-over-time logit.** The GRU runs across the
  clip's embeddings from a zero initial state and the supervision is the **max**
  over those per-hop logits — "the word appears somewhere in the window."
- **Re-scored from a ZERO state each hop at inference — _not_ a carried streaming
  state.** This is the one place the runtime departs from the early "stateful
  streaming GRU" sketch. Each hop the engine re-runs the head from zero over the
  trailing `HEAD_CONTEXT_HOPS=10` embeddings and takes the max `P(wake)` — exactly
  the criterion it was trained on. Carrying one hidden state across the whole
  stream instead lets `‖h‖` run away and collapses `P(wake)→0` after a few seconds.
  See `runtime/engine.py` and integration-guide §12.

### Post-processing

- **Threshold** calibrated per word (default ~0.5, tuned on a val set for the
  FA/FR operating point).
- **Refractory period** (debounce, ~1 s) after a trigger to avoid repeats.
- Optional **smoothing** (moving avg over a few hops) to cut spurious spikes.

## 7. Training data pipeline (streamlined)

| Class                    | Source                                                                                      | Notes                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Positives**            | **ElevenLabs v3** TTS of the wake phrase                                                    | few but high-diversity: voices, accents, prosody/emotion. Target order ~1–5k clips, not 50k |
| **Hard negatives**       | phonetically near phrases, sub-phrases, rhymes                                              | mine + TTS; critical for low FA                                                             |
| **Background negatives** | speech (Common Voice/LibriSpeech), music & noise (AudioSet/MUSAN)                           | large, generic                                                                              |
| **Augmentation**         | RIR convolution, additive noise/music at varied SNR, gain, pitch/time/speed, codec/clipping | applied on-the-fly during head training                                                     |

The **backbone** training set (§5.2) is separate and larger (general speech). The
**head** training set is per-word and small thanks to the frozen backbone + quality
TTS + augmentation.

## 8. Runtime & deployment

- **Inference engine**: ONNX Runtime, **CPU execution provider** (NEON on ARM64,
  AVX on x86_64). Optional **XNNPACK EP** for extra ARM throughput.
- **Quantization**: int8 (dynamic first; static/QAT if accuracy allows) for size +
  speed on ARM64 (ORT QGEMM).
- **Packaging**: three ONNX artifacts — `melspec.onnx`, `backbone.onnx` (shared),
  `<word>_head.onnx` (per word) — plus a thin streaming wrapper (Python first;
  C++/`librt` later if needed) that owns the ring buffers, state, threshold, and
  refractory logic.
- **Optional CPU gate**: a very cheap energy/spectral VAD before the backbone to
  skip silence and save power. Optional; adds complexity — defer unless idle CPU
  is a problem.

### Footprint budget (rough, per running word)

| Artifact          | fp32        | int8        |
| ----------------- | ----------- | ----------- |
| melspec           | ~tens of KB | — (kept fp) |
| backbone (shared) | ~1 MB       | ~300 KB     |
| head              | ~50–200 KB  | ~smaller    |

## 9. Toolchain

| Concern        | Choice                                                       |
| -------------- | ------------------------------------------------------------ |
| Language / pkg | Python 3.14, `uv`                                            |
| Training       | PyTorch + `torchaudio`, on **Google Colab** GPU; export ONNX |
| Inference      | ONNX Runtime (CPU EP)                                        |
| Sample gen     | ElevenLabs v3 API                                            |
| Target HW      | Linux x86_64 + ARM64 (RPi5 class)                            |

## 10. Evaluation targets

Match or beat openWakeWord's practical bar, measured on a held-out, augmented set:

- **False accepts < 0.5 / hour**
- **False rejects < 5%**
- Backbone inference **< ~5%** of one A76 core at 12.5 inf/s; full single-word
  pipeline comfortably real-time on RPi5 with headroom for several words.

## 11. Decisions (resolved)

The decisions originally open here are settled in code; `src/wwd_i/config.py` and
the source are ground truth.

1. **Backbone training objective** — Path A: **cosine-prototypical** metric
   learning on MSWC, frozen behind a few-shot probe gate. Path B (SSL distillation)
   not pursued. §5.2
2. **Backbone block** — **BC-ResNet**. §5.1
3. **Head type** — **GRU** (primitive ops), but re-scored from a zero state over a
   trailing window each hop (max-over-time), _not_ a carried streaming state. §6
4. **Embedding dim** — **D = 96**. §5.1
5. **n_mels** — **32**. §4
6. **CPU VAD gate** — **deferred** (not needed: the always-on engine already runs
   single-thread / no-spin at a fraction of one core). §8

```

```
