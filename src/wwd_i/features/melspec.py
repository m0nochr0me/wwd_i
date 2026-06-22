"""Log-mel front-end.

Implemented as a fixed 1-D convolution (a windowed DFT basis) followed by a mel
filterbank matmul and a log. Because it is just conv + matmul + elementwise ops,
the PyTorch module and its exported ONNX twin compute identical values — which is
what the Phase 1 parity gate checks. No normalization is applied here; the
backbone owns per-window normalization. See docs/architecture.md §4.

torch/torchaudio are training/export-only deps (the `train` group); the inference
runtime uses the exported ONNX model and never imports this module.
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from torch import Tensor, nn

from wwd_i.config import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WIN_LENGTH,
)

_EPS = 1e-6
_N_FREQS = N_FFT // 2 + 1


def _dft_conv_weight() -> Tensor:
    """Windowed DFT basis as conv1d weights of shape [2 * n_freqs, 1, win_length].

    Rows 0..n_freqs-1 are the real (cosine) filters, the rest are imaginary
    (negative sine) filters; the Hann window is folded into the weights.
    """
    window = torch.hann_window(WIN_LENGTH, periodic=True, dtype=torch.float64)
    n = torch.arange(WIN_LENGTH, dtype=torch.float64)
    k = torch.arange(_N_FREQS, dtype=torch.float64).unsqueeze(1)  # [F, 1]
    angle = 2.0 * torch.pi * k * n / N_FFT  # [F, win]
    real = torch.cos(angle) * window
    imag = -torch.sin(angle) * window
    weight = torch.cat([real, imag], dim=0).unsqueeze(1)  # [2F, 1, win]
    return weight.to(torch.float32)


def _mel_filterbank() -> Tensor:
    """Mel filterbank of shape [n_mels, n_freqs] (HTK scale, unnormalized)."""
    fb = AF.melscale_fbanks(
        n_freqs=_N_FREQS,
        f_min=FMIN,
        f_max=FMAX,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        norm=None,
        mel_scale="htk",
    )  # [F, n_mels]
    return fb.t().contiguous().to(torch.float32)


class MelSpectrogram(nn.Module):
    """Map audio ``[B, T]`` to log-mel ``[B, frames, n_mels]`` (center=False)."""

    dft: Tensor  # registered buffers, annotated so the type checker sees Tensor not Module
    mel_fb: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("dft", _dft_conv_weight())  # [2F, 1, win]
        self.register_buffer("mel_fb", _mel_filterbank())  # [n_mels, F]

    def forward(self, audio: Tensor) -> Tensor:
        spec = torch.conv1d(audio.unsqueeze(1), self.dft, stride=HOP_LENGTH)  # [B, 2F, frames]
        real, imag = spec[:, :_N_FREQS], spec[:, _N_FREQS:]
        power = real * real + imag * imag  # [B, F, frames]
        mel = torch.matmul(self.mel_fb, power)  # [B, n_mels, frames]
        log_mel = torch.log(mel + _EPS)
        return log_mel.transpose(1, 2).contiguous()  # [B, frames, n_mels]


_DEFAULT_MODEL: MelSpectrogram | None = None


def _default_model() -> MelSpectrogram:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = MelSpectrogram().eval()
    return _DEFAULT_MODEL


def compute_logmel(signal: np.ndarray) -> np.ndarray:
    """Batch log-mel for a 1-D float32 signal -> ``[frames, n_mels]`` float32."""
    x = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = _default_model()(x)
    return out.squeeze(0).numpy()


def export_onnx(path: str | Path) -> Path:
    """Export the front-end to ONNX (batch 1, dynamic sample length).

    Uses the torch.export-based exporter; the model is conv + matmul + log, so the
    ONNX graph matches the PyTorch module within float tolerance.

    Weights are written inline (``external_data=False``): the dynamo exporter
    otherwise spills them to a ``{path}.data`` sidecar, which silently breaks the
    model the moment the ``.onnx`` is moved or shipped without it (see
    ``models.backbone.export_onnx``).
    """
    from torch.export import Dim

    model = MelSpectrogram().eval()
    dummy = torch.zeros(1, SAMPLE_RATE)  # 1 s of silence
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        dynamo=True,
        input_names=["audio"],
        output_names=["logmel"],
        dynamic_shapes={"audio": {1: Dim("samples")}},
        external_data=False,
        verbose=False,
    )
    return path
