"""Central audio & feature constants — the single source of truth.

Every stage (front-end, backbone, head, runtime) imports these so that training
and inference agree exactly. See docs/architecture.md §3–§4.
"""

from typing import Final

# --- audio contract (fixed) -------------------------------------------------
SAMPLE_RATE: Final = 16_000  # Hz, mono
FRAME_MS: Final = 80  # streaming unit of work
FRAME_SAMPLES: Final = SAMPLE_RATE * FRAME_MS // 1000  # 1280

# --- log-mel front-end (consumed from Phase 1 onward) -----------------------
N_FFT: Final = 512
WIN_LENGTH: Final = 400  # 25 ms
HOP_LENGTH: Final = 160  # 10 ms -> 100 mel frames / s
N_MELS: Final = 32
FMIN: Final = 0.0
FMAX: Final = 8000.0

# mel frames produced per 80 ms stream frame
MEL_FRAMES_PER_FRAME: Final = FRAME_SAMPLES // HOP_LENGTH  # 8
