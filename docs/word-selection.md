# wwd_i — Choosing a Wake Word

How to pick a wake word that this detector can actually learn well, and how to
validate a candidate before you commit to it.

This is the _what to name it_ doc. For _how to embed the detector_ see
[integration-guide.md](integration-guide.md); for _why it's built this way_ see
[architecture.md](architecture.md). `src/wwd_i/config.py` is the audio contract,
ground truth.

---

## 0. The one thing to internalize

> **Phonetics sets the achievable ceiling; data and calibration decide where you
> land under it.**

A structurally good word _raises the best false-accept / false-reject tradeoff
you could ever reach_ — but the per-word positives, the negative mix, and the
calibrated threshold decide the operating point you actually ship, and in
practice they dominate the differences you'll see between words. So choosing a
good word is **necessary but not sufficient**: pair it with adequate data
([§7](#7-validate-a-candidate)) or even a phonetically ideal word will fail the
gate.

The product gate is strict — **FA < 0.5/hr and FR < 5%** (`train_head`
`--target-fa` / `--target-fr`). Word choice alone will not clear it; it widens
the door.

---

## 1. The two levers

Separate the two, because they have different fixes:

| Lever                         | What it controls                              | You change it by…                                          |
| ----------------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| **Phonetics** (word choice)   | the achievable DET-curve _envelope_           | picking the word (this doc)                                |
| **Data + calibration**        | the _operating point_ on that envelope        | positives volume/diversity, negative mix, threshold sweep  |

What this looked like in this project's own calibration runs (held-out,
augmented; small sample, all pre-gate — directional, not gospel):

- **Same word, only the negatives changed** → false-reject moved by ~⅓ at the
  same FA. Phonetics held constant; the data moved the operating point.
- A **phonetically textbook** word (4 syllables, hard stops, varied vowels)
  trained on **half** the positives posted the **worst** false-reject of the
  fleet. The "best" word lost because it was starved of data.
- The word that _felt_ hardest to trigger at the mic had the **best** held-out FA
  **and** FR. "Feels easy/hard" at the mic is mostly threshold + loudness (AGC),
  not phonetic destiny.

Takeaway: use the rules below to pick a word, then spend most of your effort on
**positives + negatives**, not on hunting for the "perfect" phonetic shape.

---

## 2. Structural rules

Ranked by how much we trust them (strong → moderate). All raise the envelope;
none guarantee a feasible operating point on their own.

1. **Uncommon in everyday speech — in _every_ deployment language.** _(strong)_
   False-accept resistance is dominated by distance from ordinary conversation. A
   word that overlaps common speech has no threshold meeting FA < 0.5/hr **and**
   FR < 5%. Avoid bare first names and command words. If you ship in English and
   Russian, it must be uncommon in both.
2. **≥ 3 syllables, ≥ 6 phonemes, varied vowels.** _(moderate)_ Short phrases are
   hard for humans and machines in noise; industry wake words cluster at 6+
   phonemes (Alexa 6, "OK Google" 8). Mix open/close vowels and add a diphthong
   so the head has more to key on. Necessary-not-sufficient — a 6-phoneme word
   with thin positives still fails.
3. **Carry at least one hard-to-fake high-band sound.** _(moderate)_ A sibilant
   or obstruent — `s`, `ш`/`sh`, `ц`/`ts`, a `ks`/`ст`/`sp` cluster — puts energy
   in the top mel bins where vowels and most ambient noise are nearly silent
   ([§5](#5-sounds--the-spectrum)). It is the cue a _voiced_ rhythm imposter
   cannot reproduce, and it also feeds the optional second-stage gate
   (`high_mel_energy`).
4. **Don't lean on a low-energy onset.** _(moderate, field robustness)_ A breathy
   `h` or other near-silent onset is easily elided in fast or far-field speech →
   recall loss in the field. (It may still calibrate fine on clean held-out
   audio; the cost shows up live.)
5. **Front-stress (trochaic) helps recall only.** _(moderate)_ A distinctive
   stress contour aids detection, but the head's top-k-mean-over-time scoring
   discards _where_ in the window the cue landed, so stress placement is a weak
   false-accept discriminator. Use it to help users get detected, not to reject
   noise.

---

## 3. Antipatterns

Avoid these — each is a real liability here:

- **Bare names / common words** ("Sam", "Joy", "computer"): overlap conversation;
  sub-words ("Sam", "are" in _Samara_) can fire on their own.
- **All-sonorant, open-vowel shapes** (`m`-`a`-`y`-`u`-`m`-`i`): identity rests
  entirely on the voiced envelope — the single most forgeable feature here, so
  rhythm imposters ([§6](#6-rhythm--false-triggers)) match it easily.
- **2-syllable or < 6-phoneme words**: underfill the ~0.8 s decision window and
  offer few constraints; flagged high-FA across the industry.
- **Words common in _one_ of your languages**: bilingual deployments widen the
  positive manifold (the head must accept several realizations), pulling it toward
  generic speech.
- **Relying on loudness/whisper to stand out**: the engine forces input to
  −20 dBFS (AGC) and the backbone z-scores each window, so a quiet TV wake-alike
  embeds like a shouted one. Loudness is not a discriminator by design.
- **Rolled `r` as a load-bearing cue** for child/accent use: not everyone
  produces it → recall failures and a wider positive manifold. (Yandex
  deliberately avoided `r` in _Alisa_ for exactly this.)

---

## 4. Worked candidates

Starting points, not guarantees — each still needs adequate per-word data and a
real validation pass ([§7](#7-validate-a-candidate)).

| Word                  | Lang        | Why it fits                                                                 | Watch out for                                              |
| --------------------- | ----------- | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| Maxine / Roxana       | en          | 3 syllables, `ks` cluster (Alexa's documented anchor), varied vowels       | names → conversational overlap; "Roxana" rolled-r variant |
| Castila / Festara     | en (coined) | uncommon, `s`/`st` anchor + 3 varied vowels, sits in a sparse speech region | coined words can feel unnatural; verify cross-language     |
| Оксана (Oksana)       | ru          | real name, `ks` anchor, renders cleanly in EN + RU                          | common RU name → mine near-phrase + rhythm negatives       |
| Сакура / Sakura       | ru / bi     | `s`+`k`, varied vowels a-u-a, uncommon in casual speech                     | rising brand/anime usage — check FA empirically            |
| Цецилия (Tsetsiliya)  | ru          | `ts` (ц) affricate fully below the 8 kHz Nyquist — a clean high-band anchor | long/formal (UX)                                           |
| Эспера (Espera)       | ru          | `sp` obstruent cluster: sibilant + plosive transient; uncommon             | near Spanish _espera_ if any Spanish exposure              |

---

## 5. Sounds & the spectrum

Why high-band consonants help, and what the model actually "hears":

- The front-end is **32 mel bins over 0–8 kHz**. Roughly the **top ~8–10 bins**
  (≳ 3.5 kHz) cover the sibilant/fricative band, where vowels and ambient noise
  have little energy — so frication stands out there. This is the band the
  second-stage gate's `high_mel_energy` summarizes.
- The backbone collapses the frequency axis progressively, so the 96-d embedding
  effectively records **"high-band energy is present,"** not a fine
  `s`-vs-`ʃ` shape. You get an anchor for _"is there frication,"_ not phoneme
  identification — which is exactly what's hard for a voiced imposter to fake.
- **Nyquist caveat:** `FMAX = 8000`, so the top of a plain `/s/` (peak ≈ 8 kHz+)
  is partly truncated. `/ʃ/` (ш), `/ts/` (ц) and `/ks/` sit **fully** in band, so
  they are _cleaner_ anchors than a bare `s` in this configuration. **Do not raise
  the sample rate** to capture more of `/s/` — it breaks the 16 kHz contract and
  invalidates every existing artifact.

The `s`-in-Alexa/Siri/Alisa intuition is **directionally right but
over-specified**: the documented vendor lever is "a hard, _uncommon_ consonant +
an uncommon word," not `/s/` specifically (Amazon named Alexa's `ks`; Yandex
avoided `r`; Apple chose Siri for naturalness). Treat a high-band consonant as
_one good ingredient_, not the cause.

---

## 6. Rhythm & false triggers

The detector keys partly on a word's **rhythm**, so a 3-beat _"мя-мя-мЯ"_ can
false-trigger a 3-syllable word like _Samara_ — it reproduces the voiced syllabic
energy bursts without any of the phonemes. Pure single-band rhythm (tapping)
should _not_ fire; it's the **multi-band voiced envelope** that matches.

This is mostly a **data/threshold** problem, not a word problem — so it's fixable
without re-choosing the word:

1. **Train against it.** `train_head --rhythm-neg <dir>` folds nonsense babble at
   the wake-word cadence into the training negatives so the head learns to reject
   rhythm-only matches. Generate the clips with
   `elevenlabs --rhythm-impostors-for "<word>"` (needs `GEMINI_API_KEY`).
2. **Measure it.** `train_head --rhythm-impostors <held-out dir>` adds an
   **imp-FA** column to the DET table — the fraction of rhythm impostors that
   false-fire at each threshold — surfacing the failure the continuous-background
   FA/hr can't see. Add `--target-impostor-far 0.05` to also gate on it. The
   Colab notebook wires both (cell A4 + Part C).
3. **Reject at runtime.** A client-side second-stage gate
   ([integration-guide §10](integration-guide.md#10-second-stage-gating-rhythm--babble-rejection))
   on `Detection.hops_above` (persistence) and `Detection.high_mel_energy`
   (the sibilant anchor a voiced imposter lacks) drops rhythm spikes without
   retraining.
4. **Choose for it.** A word with a hard-band consonant ([§5](#5-sounds--the-spectrum))
   gives the gate something to require — which is why rule 3 in
   [§2](#2-structural-rules) doubles as rhythm defense.

---

## 7. Validate a candidate

Don't trust intuition — measure. Cheapest-first:

1. **Generate near-phrase hard negatives** so the head learns the fine boundary:
   `elevenlabs --hard-negs-for "<word>" --llm-confusables 20` (Gemini writes
   acoustic confusables that are close but _not_ homophones).
2. **Train + calibrate** with a held-out streaming FA set and rhythm impostors:
   `train_head … --calib-bg <held-out audio> --rhythm-impostors <eval dir>`.
   Read the `[gate]` line: **PASS** = FA < 0.5/hr and FR < 5%; watch **imp-FA**.
3. **Probe rhythm directly.** Synthesize _"мя-мя-мЯ"_-style clips plus same-beat
   babble and run them through the engine:
   `uv run wwd-i <clip>.wav --head <word>_head.onnx` — a candidate that stays
   below its threshold on all of them is rhythm-robust.
4. **Conversation FA test.** Feed 30–60 min of real EN+RU TV/podcast and count
   detections → FA/hr. This is the real-world signal; compare candidates on it.
5. **A/B the data lever.** Retrain the _same_ word with vs without `--rhythm-neg`
   (and at different positive counts) and compare the chosen FA/FR — this tells
   you whether a failure is the **word** or the **data**, which have different
   fixes ([§1](#1-the-two-levers)).

---

## 8. Checklist

Before committing to a wake word:

- [ ] Uncommon in **every** deployment language; not a bare name or command word.
- [ ] ≥ 3 syllables, ≥ 6 phonemes, varied vowels.
- [ ] Carries a high-band consonant (`s`/`ш`/`ц`/`ks`/`sp`) — prefer `ш`/`ц`/`ks`
      over a bare `s` (Nyquist).
- [ ] No load-bearing low-energy onset (`h`) or rolled `r` if children/accents
      matter.
- [ ] Sub-words aren't themselves common words.
- [ ] Planned for adequate **positives** (don't ship a word under-provisioned) and
      a curated **negative** mix (near-phrases + rhythm impostors).
- [ ] Validated end-to-end ([§7](#7-validate-a-candidate)): passes the gate, low
      imp-FA, survives a real conversation FA test.

---

## 9. See also

- [integration-guide.md](integration-guide.md) — embed the detector; §10 is the
  second-stage gate.
- [architecture.md](architecture.md) — why the pipeline is shaped this way.
- `src/wwd_i/train/train_head.py` — the gate, `--rhythm-neg`, `--rhythm-impostors`.
- `src/wwd_i/data/confusables.py` — LLM confusables + rhythm-impostor generators.
