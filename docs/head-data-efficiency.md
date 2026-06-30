# wwd_i — Head Data-Efficiency Plan

How to train a head that meets the Phase-4 gate (FA < 0.5/hr, FR < 5%) with
**fewer and cheaper** generated positives/negatives. Companion to
[`implementation-plan.md`](./implementation-plan.md) Phases 3–4 and
[`architecture.md`](./architecture.md) §6–§7.

Legend: 🎯 deliverable · ✅ verify gate · ⚠️ risk/decision.

---

## Why "add more data" keeps being needed (diagnosis)

The recurring `Axana` gate failure is **FA on continuous speech / YT-TV media** —
a *negative-side* boundary problem. The fix history has bulk-bought broad TTS
positives + near-phrases that land **far** from where the head actually
false-fires, so each paid clip carries ≈0 boundary gradient. Two compounding
causes:

1. **The backbone is frozen and strongly invariant.** A positive that lands near
   an existing one in 96-d embedding space adds ≈0 information. Most paid clips
   are near-duplicates.
2. **The augmentation budget is mostly wasted.** `Augmenter` spends ~80% of its
   `n_aug` variants on a **gain axis the backbone provably erases**:
   `backbone._normalize_window` subtracts the `2·ln(a)` log-mel shift exactly, and
   the export gate `_assert_loudness_invariant` enforces it. A +6 dB variant is a
   literal embedding duplicate.

The only axis worth paid ElevenLabs is **positive timbre diversity** — the one
thing the backbone is *not* invariant to. Everything else you buy is either
redundant (dedup-able) or replaceable by free corpora already wired in
(`hf_audio` peoples_speech/common_voice, AudioSet/FMA backgrounds).

Each lever below was adversarially verified against the code (frozen backbone,
torch-free runtime, top-k-mean criterion + primitive-op GRU, the streaming
`calibrate_stream` gate, ONNX export). Verdicts are stated.

---

## Workstream 1 — Hard-negative mining from free media  ⭐ highest leverage
**Verdict: STRONG.** Genuinely cuts the expensive item; directly attacks the
failing distribution.

Stream a **free** continuous-media pool through the real engine and harvest the
exact embedding windows that score high — on-boundary negatives, $0, far higher
gradient-per-sample than any paid near-phrase. Replaces the ~4597 paid near-phrases
with a few hundred mined windows.

- 🎯 `engine.py`: add `record_embeddings: bool` to `WakeWordEngine`. In `push()`,
  right before the per-head scoring loop, set
  `self.last_emb_window = self._emb_buf.copy()`.
  **Do not** attach it to `Detection` — `_emb_buf` is shared across heads, the
  window is per-hop, not per-detection.
- 🎯 New `train/mine_neg.py`, mirroring `preprocess_bg`'s stream-embed-to-`.npy`
  loop: drive a `WakeWordEngine(record_embeddings=True, threshold=0, refractory_s=0)`
  over `--mine-bg` dirs; at every hop with `score >= --mine-thr` (start ~0.3),
  stash `last_emb_window` **only when** `shape[0] == HEAD_CONTEXT_HOPS` (skip the
  first 9 warm-up hops/file — short windows fail the `train_head.py:143` shape
  check); cap per-file so one talker can't dominate; write `[N, 10, 96]`
  `mined_neg.npy`.
- 🎯 `train_head.py`: make `--bg-neg-emb` `nargs='+'` and `np.concatenate` all
  caches into `neg_parts` (today single-path at `train_head.py:141-148`,
  shape-checked). Feed `mined_neg.npy` alongside `bg_neg.npy`.
- ⚠️ **Hard guard (raise, not convention):** assert `--mine-bg` paths are disjoint
  from `--calib-bg` and `--rhythm-impostors`, or the gate becomes test-on-train.
- ⚠️ **Whack-a-mole risk:** the tiny GRU (`hidden=48`) can memorize specific mined
  frames and shift FA elsewhere. Mitigate: mine from a **large, diverse** pool
  (hundreds of distinct files), keep `n_aug` augmentation on mined crops, cap the
  mined count to a few hundred, and re-mine **fresh** each round (never accumulate
  stale crops).
- ✅ **Verify:** re-gate each round on a **fresh** `--calib-bg` from media never
  seen in training/mining, **and** track field-FA on a frozen holdout
  round-over-round (confirms the win generalizes, not just that an easier gate
  flipped to PASS).

> Capturing `_emb_buf` (already AGC-correct — exactly what `_Head._score` consumed)
> sidesteps source re-crop + AGC-reproduction entirely: it *is* the firing point.

---

## Workstream 2 — Move paid generation to free/cheap TTS
**Verdict: VIABLE, low effort, ~zero gate risk for negatives.**

The hard-neg synthesis path is byte-identical across `gemini_tts.py:191`,
`groq_tts.py:136`, `elevenlabs.py:238` (same `TtsBackend` contract,
collision-safe `blake2b(phrase|tag)` caches, per-backend `rms_normalize` so
invariant #4 holds). Negatives only populate the rejection region — a free voice
is fully adequate. Cuts ~100% of paid hard-neg spend and ~80% of paid positive
volume.

- 🎯 **Negatives — no code change:** run
  `python -m wwd_i.data.gemini_tts --hard-negs-for <word> --llm-confusables N`
  instead of the ElevenLabs equivalent; `train_head --hard-neg` consumes the dir
  unchanged.
- 🎯 **Positives:** generate bulk on Gemini/Groq into `--positives`, then add a
  **small** ElevenLabs anchor tranche into the *same* dir (tag-keyed caches merge
  collision-safe).
- ⚠️ **Do not blind-swap positives.** Free-tier voice demographics are unverified;
  an under-covered positive manifold *raises* FR. Embed candidate free-voice
  positives through `backbone.onnx`, measure cosine coverage vs the existing
  ElevenLabs prototypes, and **size the anchor tranche by the measured gap** — not
  a fixed count.
- ⚠️ **Language:** verify Gemini per-voice handles the target language for
  non-English words; Groq Orpheus is English-only — exclude it for positives.
- ⚠️ **Licensing** ([`data-licensing.md`](./data-licensing.md)): ElevenLabs
  §9.k/§9.l and Gemini free-tier ToS restrict training-on-output. Only
  `LocalTtsBackend` (Kokoro/Apache-2.0, Piper/MIT) is unambiguously train-safe and
  free — but `local_tts.synthesize` is currently a **stub**. Verify ToS before
  shipping a distributable head.
- ✅ **Verify:** `calibrate_stream` PASS at equal or lower FR than the
  all-ElevenLabs baseline.

---

## Workstream 3 — Rebalance augmentation (free multiplier)
**Verdict: STRONG.** Turns each paid clip into genuinely-distinct training
*sequences* instead of duplicates.

- 🎯 `augment.py:164` — drop `p_gain` default `0.8 → 0.1` (proven no-op; see
  diagnosis).
- 🎯 Wire the **dead** `pitch_shift` (`augment.py:109` — torch-free numpy+soxr,
  test-pinned, currently never called) into `Augmenter.__call__` after
  `speed_perturb`: `p_pitch ~0.3`, `pitch_semitones=(-2, 2)`. Formant/timbre is an
  axis the backbone is *not* invariant to.
- 🎯 `train_head._augmented` (lines 103-110) — add small random front-pad temporal
  jitter (draw from `aug.rng`, the single shared RNG) before `fixed_length`, so
  word onset slides across the `WINDOW`. Measured: a 320 ms shift swings per-window
  cosine −0.31..+0.69 — genuinely distinct sequences for a top-k-mean-over-time GRU
  that keys on *which* windows are word-active.
- 🎯 `preprocess_bg` — add `--aug-frac` to run the `Augmenter` (RIR + noise + pitch,
  `p_gain=0`, RMS-normalize the crop to −20 dBFS first to match the AGC'd gate)
  over 25–50% of media crops before embedding. Keep clean crops too. This changes
  the `bg_neg.npy` contract → version/regen the shared cache.
- ⚠️ **Bound the jitter small.** Positives embed *without* runtime AGC; a
  near-silent padded clip labelled positive teaches "silence-ramp ⇒ fire" → FA.
- ⚠️ Pitch beyond ~±2 semitones risks vocoder artifacts off the MSWC manifold → FR.
  Over-densifying negatives near the boundary can pull the threshold up and blow
  FR < 5% (two-sided gate) — back off `--aug-frac` if FR creeps.
- ✅ **Verify:** validate **only** on `calibrate_stream` (it streams raw audio, so
  augmentation cannot inflate the gate). Expect ~1.2–1.5× effective coverage — do
  not bank 2–3×. Augmentation explores only the backbone's (near-)invariance
  directions; it adds **no** new phonetic/voice content and is **not** a substitute
  for Workstream 1 or 2.

---

## Workstream 4 — Pre-embed `GENERIC_CONFUSABLES` once
**Verdict: VIABLE, low effort. Cost amortization only — does not move the gate.**

`negatives.hard_negative_phrases()` (`negatives.py:44`) unconditionally unions the
same 10 word-independent phrases into *every* word, re-synthesized through the paid
grid each time (`elevenlabs.py:248`).

- 🎯 Synthesize `GENERIC_CONFUSABLES` once; a `preprocess_bg`-style wrapper applies
  `n_aug` `Augmenter` variants then `embed_clips` → `generic_neg.npy` `[N, 10, 96]`.
- 🎯 Inject via the same `nargs='+'` `--bg-neg-emb` seam as Workstream 1; keep
  `generic_neg.npy` **separate** from `bg_neg.npy` (preserves the word-independent
  abstraction; lets you regen one without the other on a backbone swap). Drop
  `GENERIC_CONFUSABLES` from `hard_negative_phrases()` (keep sub-words + `extra`).
- ⚠️ Savings = `10 × n_clips` renders per *future* word (state it honestly; not a
  fixed fraction). Ship alongside Workstream 1, not instead of it.
- ✅ **Verify:** pin a test asserting the cache shape `(*, 10, 96)` so a silent
  backbone swap is caught like `bg_neg.npy`.

---

## Workstream 5 — Embedding-space dedup as a budget *diagnostic*
**Verdict: WEAK as a saver — it saves $0 on the current set.** `build_embeddings`
runs *after* the ElevenLabs render, so dedup cannot recover spent API cost.

Use it only **forward-looking**, to learn the *next* word's saturation point.

- 🎯 Helper `coreset(emb, eps)` in `train_head.py` over **raw** (`n_aug=0`)
  per-clip embeddings; print kept/total and the marginal-coverage curve. Use the
  `[W, 96]` sequence (or min-over-window cosine), not the window-mean, so you keep
  the temporal structure the criterion trains on.
- ⚠️ Default **OFF** for the shipped head. Do not silently shrink the live training
  set; aggressive dedup prunes exactly the voice/accent diversity that matters.

---

## Quick wins (one-liners, do now)
- `augment.py:164` — `p_gain 0.8 → 0.1`. Stops wasting ~80% of `n_aug` slots.
- Switch hard-neg CLI from `elevenlabs` → `gemini_tts`/`groq_tts` (byte-identical
  path). Zero code, ~100% of paid hard-neg spend → $0.
- Raise `preprocess_bg --n-bg-neg` above 50k using a **continuous-speech** corpus
  (`hf_audio` peoples_speech) before any augmentation — sharper FA/hr at no new
  compute axis.
- Wire the dead `pitch_shift` into `Augmenter` (`p_pitch ~0.3`, ±2 semitones).

---

## Avoid (verified anti-patterns)
- **Class-balanced / focal-weighted BCE "to need fewer positives"** — illusory:
  FR/FA are set by the rank-based threshold sweep in `calibrate_stream`, not the
  loss scale. Upweighting positives starves negatives of gradient → *raises* media
  FA (the failing half) and pushes back toward the overfit prior fix (commit
  8344d05) corrected.
- **Coreset-*cutting* the 50k `bg_neg` cache** — wrong target (it's free) and
  dangerous: the head is *under*-fit on media negatives (it false-fires on them),
  so it needs **more** boundary-relevant negatives, not fewer. The
  boundary-proximity idea is salvageable only as *add* (Workstream 1), never as
  *replace*.
- **Sourcing confusables from isolated MSWC / Speech-Commands single-words** —
  corpus/gate mismatch: the gate fails on *continuous* media where a sliding 760 ms
  window straddles word boundaries; isolated words don't reproduce that
  distribution.
- **Generate-broad-then-coreset positives as a gate fix** — optimizes positive-
  manifold coverage (FR side) while the failure is FA (negative side).

---

## Stopping rule — when to stop generating
Stop when **both** hold on a fresh held-out media set:

1. **Gate** — the head passes FA < 0.5/hr **and** FR < 5% on `calibrate_stream`
   over a `--calib-bg` drawn from continuous media **never seen** in
   training/mining, **and** a frozen field-FA holdout does not regress
   round-over-round.
2. **Saturation** — on **raw** (`n_aug=0`) clips: (a) positives — a new paid batch
   no longer lowers nearest-neighbour cosine to the kept set (new voices land at
   cos > ~0.97 ⇒ $0 new info, stop buying positives); (b) negatives (the binding
   one) — a mining round over fresh free media yields fewer than ~20 hops scoring
   ≥ `mine_thr` ⇒ the boundary is covered, stop mining.

If the gate **still** fails, that is a negative/discrimination signal — **mine more
free media (Workstream 1)**; do **not** buy more positives or broad near-phrases.

---

## Suggested order
1. Quick wins (minutes; no retrain dependency beyond the next run).
2. Workstream 2 — backend swap (low effort, immediate spend cut).
3. Workstream 1 — mining (the lever that fixes media-FA). Needs the engine hook +
   `train/mine_neg.py`.
4. Workstream 3 — augmentation rebalance (regen `bg_neg.npy`).
5. Workstreams 4–5 — amortization + budget diagnostic, opportunistic.
