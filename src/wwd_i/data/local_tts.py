"""Local open-TTS backend (skeleton) — the drop-in slot for ``TtsBackend``.

This is where a permissively-licensed, self-hosted TTS plugs in so the whole
Phase-3 → Phase-4 pipeline is license-clean end-to-end (no provider ToS on the
head's training data; see ``docs/data-licensing.md``). Candidates whose licenses
permit using their output to train ML models:

  - **Kokoro**     (Apache-2.0) — small, fast, multi-voice
  - **Piper**      (MIT)        — CPU-friendly, many voice packs
  - **Parler-TTS** (Apache-2.0) — prompt-controllable prosody

``variants()`` below is already backend-shaped (a voice × speed grid, deterministic
in ``seed``); only ``synthesize()`` is a TODO — wire your chosen model and return
float32 mono @ ``SAMPLE_RATE`` (``generate_clips`` handles RMS-normalize + caching).
Keep the heavy TTS import lazy and in the ``train`` group so the runtime stays
torch-free. Then use it exactly like the ElevenLabs backend::

    from wwd_i.data.tts import generate_clips
    from wwd_i.data.local_tts import LocalTtsBackend

    generate_clips("hey computer", "data/pos", LocalTtsBackend(), n_clips=300)
"""

import numpy as np

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.tts import TtsBackend, Variant

# Replace with the speaker ids your model exposes.
DEFAULT_VOICES: tuple[str, ...] = ()
SPEEDS = (0.9, 1.0, 1.1)  # cheap prosody spread; extend with whatever knobs the model has


class LocalTtsBackend(TtsBackend):
    def __init__(self, *, voices: tuple[str, ...] = DEFAULT_VOICES, speeds: tuple[float, ...] = SPEEDS):
        self.voices = voices
        self.speeds = speeds

    def variants(self, n_clips: int, *, seed: int) -> list[Variant]:
        if not self.voices:
            raise RuntimeError("LocalTtsBackend: set voices= to your model's speaker ids")
        grid = [(v, sp) for v in self.voices for sp in self.speeds]
        np.random.default_rng(seed).shuffle(grid)
        chosen = [grid[i % len(grid)] for i in range(n_clips)]
        return [Variant(tag=f"{v}|{sp}", voice=v, params={"speed": sp}) for v, sp in chosen]

    def synthesize(self, phrase: str, variant: Variant) -> np.ndarray:
        # TODO: load your TTS (lazily) and render `phrase` with variant.voice / variant.params.
        # Return float32 mono @ SAMPLE_RATE in [-1, 1]; resample here if the model isn't 16 kHz.
        _ = SAMPLE_RATE  # the contract generate_clips expects
        raise NotImplementedError(
            "LocalTtsBackend.synthesize is a stub — wire a permissive TTS (Kokoro/Piper/Parler). "
            "See docs/data-licensing.md."
        )
