"""From-scratch BC-ResNet backbone (Phase 2).

A small broadcasted-residual CNN that maps a log-mel patch ``[B, T, n_mels]`` to a
compact L2-normalized embedding ``[B, D]``. Trained once with a metric loss (see
``train/proto.py``) over a large keyword vocabulary, then frozen and shared by
every per-word head. It is length-agnostic in time (global average pool over
time), so the same weights embed a whole training word (~1 s) and a 76-frame
streaming window at inference.

torch is a training/export-only dep (the `train` group); the inference runtime
uses the exported ``backbone.onnx`` and never imports this module.

See docs/architecture.md §5.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wwd_i.config import N_MELS


@dataclass(frozen=True)
class BackboneConfig:
    """Backbone hyper-parameters.

    Frequency is halved by the stem and by the first (transition) block of each
    stage, so with ``n_mels=32`` and 4 stages the frequency axis collapses to 1
    (32 → 16 → 8 → 4 → 2 → 1) before the final projection.
    """

    n_mels: int = N_MELS
    stem_channels: int = 16
    stage_channels: tuple[int, ...] = (16, 24, 32, 48)
    blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2)
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    embedding_dim: int = 96
    dropout: float = 0.1

    def __post_init__(self) -> None:
        n = len(self.stage_channels)
        if not (len(self.blocks_per_stage) == len(self.dilations) == n):
            raise ValueError("stage_channels, blocks_per_stage and dilations must have equal length")


class BCResBlock(nn.Module):
    """Broadcasted-residual block.

    Frequency path: a depthwise conv over frequency keeps a 2-D ``[C, F, T]``
    feature. Temporal path: average over frequency, run a dilated depthwise conv
    over time, then broadcast back over frequency and add. An identity skip is
    used only when the block preserves channels and frequency (a non-transition
    block); transition blocks change channels and/or stride frequency.
    """

    def __init__(
        self, in_ch: int, out_ch: int, *, freq_stride: int = 1, dilation: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.use_skip = in_ch == out_ch and freq_stride == 1
        # Channel transition (only when channels change); skip blocks pass through.
        if in_ch != out_ch:
            self.pw_in = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())
        else:
            self.pw_in = None
        # Frequency path: depthwise (3x1) over frequency, may stride frequency.
        self.freq_dw = nn.Conv2d(
            out_ch, out_ch, (3, 1), stride=(freq_stride, 1), padding=(1, 0), groups=out_ch, bias=False
        )
        self.freq_bn = nn.BatchNorm2d(out_ch)
        # Temporal path: dilated depthwise (1x3) over time + pointwise mix.
        self.temp_dw = nn.Conv2d(
            out_ch, out_ch, (1, 3), padding=(0, dilation), dilation=(1, dilation), groups=out_ch, bias=False
        )
        self.temp_bn = nn.BatchNorm2d(out_ch)
        self.temp_pw = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.pw_in is not None:
            x = self.pw_in(x)
        freq = self.freq_bn(self.freq_dw(x))  # [B, C, F', T]
        temp = freq.mean(dim=2, keepdim=True)  # [B, C, 1, T] — average over frequency
        temp = F.silu(self.temp_bn(self.temp_dw(temp)))
        temp = self.dropout(self.temp_pw(temp))
        out = F.relu(freq + temp)  # broadcast the temporal feature back over frequency
        if self.use_skip:
            out = out + identity
        return out


def _normalize_window(mel: Tensor) -> Tensor:
    """Per-window z-score over (time, mel) — makes the embedding loudness-invariant.

    A constant audio-gain factor ``a`` scales power by ``a²``, i.e. adds a uniform
    ``2·ln a`` to every log-mel bin; subtracting the per-window mean removes that
    offset exactly, and dividing by the std normalizes contrast. ``eps`` keeps a
    silent window (std≈0) finite — it maps to ~0 instead of blowing up.
    """
    centered = mel - mel.mean(dim=(1, 2), keepdim=True)
    std = centered.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
    return centered / (std + 1e-5)


class Backbone(nn.Module):
    """Log-mel ``[B, T, n_mels]`` -> L2-normalized embedding ``[B, D]``.

    The log-mel input is per-window z-score normalized (``_normalize_window``) so the
    embedding is invariant to input loudness; this is baked into ``forward`` so the
    exported ONNX carries it to head training and the runtime unchanged.
    """

    def __init__(self, config: BackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or BackboneConfig()
        c = self.config
        self.stem = nn.Sequential(
            nn.Conv2d(1, c.stem_channels, 5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c.stem_channels),
            nn.ReLU(),
        )
        blocks: list[nn.Module] = []
        in_ch = c.stem_channels
        for out_ch, n_blocks, dilation in zip(c.stage_channels, c.blocks_per_stage, c.dilations, strict=True):
            for b in range(n_blocks):
                freq_stride = 2 if b == 0 else 1  # first block of each stage halves frequency
                blocks.append(BCResBlock(in_ch, out_ch, freq_stride=freq_stride, dilation=dilation, dropout=c.dropout))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Linear(in_ch, c.embedding_dim)

    def forward(self, mel: Tensor) -> Tensor:
        mel = _normalize_window(mel)  # per-window z-score: loudness-invariant input
        x = mel.transpose(1, 2).unsqueeze(1)  # [B, T, n_mels] -> [B, 1, n_mels, T]
        x = self.stem(x)
        x = self.blocks(x)  # [B, C, 1, T]
        x = x.mean(dim=(2, 3))  # global average pool over frequency and time -> [B, C]
        x = self.proj(x)  # [B, D]
        return F.normalize(x, dim=1)


def export_onnx(model: Backbone, path: str | Path, *, n_frames: int = 76) -> Path:
    """Export a (frozen) backbone to ONNX with a dynamic time axis.

    The model is convs + batchnorm + pooling + linear, so the ONNX graph matches
    the eval-mode PyTorch module within float tolerance. ``n_frames`` only sizes
    the trace dummy; the exported graph accepts any time length.

    Weights are written inline (``external_data=False``): the dynamo exporter
    otherwise spills them to a ``{path}.data`` sidecar, which silently breaks the
    model the moment the ``.onnx`` is moved or downloaded without it.
    """
    import copy

    from torch.export import Dim

    # Export on CPU (the deployment target) from a copy: handles a GPU-trained model
    # and leaves the caller's model on its original device/mode.
    model = copy.deepcopy(model).eval().cpu()
    dummy = torch.zeros(2, n_frames, model.config.n_mels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        dynamo=True,
        input_names=["mel"],
        output_names=["embedding"],
        # Batch dynamic too (=1 at stream time, but lets head training embed many windows at once).
        dynamic_shapes={"mel": {0: Dim("batch", min=1), 1: Dim("frames", min=1)}},
        external_data=False,
        verbose=False,
    )
    return path
