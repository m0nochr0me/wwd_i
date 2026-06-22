"""GPU-batched waveform augmentation for backbone training (Phase 2).

The numpy ``data.augment.Augmenter`` runs per clip on the CPU. Wired into the
per-step episode that is a bottleneck: the backbone is tiny (~18 k params), so a
GPU step is a few milliseconds, while augmenting 300 clips on the CPU — the FFT
reverb above all — takes far longer and starves the GPU. This is a batched,
on-device twin used by ``train_backbone`` only. It takes the episode waveform
tensor ``[B, L]`` already on the GPU and applies the same family of perturbations
(speed-perturb, reverb via a per-clip RIR, additive background at a sampled SNR,
clipping) to the whole batch at once, so the augmentation runs on the otherwise
idle GPU instead of the CPU.

The numpy ``Augmenter`` is left untouched (head training still uses it); the knobs
and semantics here mirror it. See ``data/augment.py`` and docs/architecture.md §7.
"""

import numpy as np
import torch
import torchaudio.functional as AF
from torch import Tensor

from wwd_i.config import SAMPLE_RATE

_EPS = 1e-9


def _fix_len(x: Tensor, length: int) -> Tensor:
    """Crop or zero-pad the last dim of ``x`` to exactly ``length`` samples."""
    n = x.shape[-1]
    if n >= length:
        return x[..., :length]
    return torch.cat([x, x.new_zeros(*x.shape[:-1], length - n)], dim=-1)


def _make_noise_bank(pool: list[np.ndarray], length: int, device: torch.device) -> Tensor:
    """Stack variable-length pool clips into a fixed-length ``[P, length]`` tensor.

    Each clip is tiled (if short) and cropped to ``length`` so a random offset crop
    to the clip length is available at mix time. ``length`` is two clip-lengths, so
    the offset window is one full clip.
    """
    rows: list[Tensor] = []
    for clip in pool:
        t = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))
        if t.shape[0] < length:
            t = t.repeat((length + t.shape[0] - 1) // t.shape[0])
        rows.append(t[:length])
    return torch.stack(rows).to(device)


class GPUAugmenter:
    """Randomly perturb a batch of 16 kHz waveforms ``[B, L]`` on-device.

    Each component fires per clip with its probability (independent across the
    batch). Length-changing speed-perturb is re-fixed to ``L``. Pass a
    ``background_pool`` (raw waveforms, see ``backgrounds.py``) to enable additive
    noise; without one only speed/reverb/clip apply.
    """

    def __init__(
        self,
        background_pool: list[np.ndarray] | None = None,
        *,
        device: str | torch.device = "cpu",
        p_gain: float = 0.0,
        gain_db: tuple[float, float] = (-6.0, 6.0),
        p_speed: float = 0.5,
        speed: tuple[float, float] = (0.9, 1.1),
        p_rir: float = 0.5,
        rt60: tuple[float, float] = (0.1, 0.6),
        p_noise: float = 0.8,
        snr_db: tuple[float, float] = (0.0, 20.0),
        p_clip: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.p_gain, self.gain_db = p_gain, gain_db
        self.p_speed, self.speed = p_speed, speed
        self.p_rir, self.rt60 = p_rir, rt60
        self.p_noise, self.snr_db = p_noise, snr_db
        self.p_clip = p_clip
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        # 2 s per clip so a random ~1 s offset crop is available at mix time.
        self.bank = _make_noise_bank(background_pool, 2 * SAMPLE_RATE, self.device) if background_pool else None

    def _rand(self, n: int) -> Tensor:
        return torch.rand(n, generator=self.gen, device=self.device)

    def _uniform(self, n: int, lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * self._rand(n)

    def __call__(self, audio: Tensor) -> Tensor:
        b = audio.shape[0]
        out = audio
        if self.p_speed > 0:
            out = self._speed(out)
        if self.p_gain > 0:
            fire = self._rand(b) < self.p_gain
            gain = torch.where(
                fire, 10.0 ** (self._uniform(b, *self.gain_db) / 20.0), torch.ones(b, device=self.device)
            )
            out = out * gain[:, None]
        if self.p_rir > 0:
            out = self._reverb(out)
        if self.bank is not None and self.p_noise > 0:
            out = self._noise(out)
        if self.p_clip > 0:
            fire = self._rand(b) < self.p_clip
            out = torch.where(fire[:, None], out.clamp(-0.99, 0.99), out)
        peak = out.abs().amax(dim=1, keepdim=True)
        out = torch.where(peak > 1.0, out / (peak + _EPS), out)
        return out.contiguous()

    def _speed(self, audio: Tensor) -> Tensor:
        """Per-clip resample (tempo+pitch). Factors are quantized so the batch needs
        only a handful of grouped resample calls instead of one per clip."""
        b, length = audio.shape
        fire = self._rand(b) < self.p_speed
        if not bool(fire.any()):
            return audio
        factors = self._uniform(b, *self.speed)
        q = torch.round(factors / 0.025) * 0.025  # ~9-value grid over (0.9, 1.1)
        q = torch.where(fire, q, torch.ones_like(q))
        out = audio.clone()
        for f in torch.unique(q).tolist():
            if abs(f - 1.0) < 1e-6:
                continue
            idx = (q == f).nonzero(as_tuple=True)[0]
            resampled = AF.resample(audio.index_select(0, idx), int(round(SAMPLE_RATE * f)), SAMPLE_RATE)
            out[idx] = _fix_len(resampled, length)
        return out

    def _reverb(self, audio: Tensor) -> Tensor:
        """Convolve each clip with its own synthetic exponential-decay RIR.

        Batched FFT convolution (like the numpy ``apply_rir``): a long-kernel
        depthwise conv1d is inefficient, while a single batched rfft/irfft over the
        whole batch is cheap on the GPU. Taking the first ``L`` samples of the full
        convolution keeps the causal head and the original length.
        """
        b, length = audio.shape
        fire = self._rand(b) < self.p_rir
        rt = self._uniform(b, *self.rt60)  # [B]
        kmax = max(2, int(self.rt60[1] * SAMPLE_RATE))
        t = torch.arange(kmax, device=self.device, dtype=torch.float32)
        decay = torch.exp(-6.9 * t[None, :] / (rt[:, None] * SAMPLE_RATE))  # -60 dB at rt60, [B, K]
        rir = torch.randn(b, kmax, generator=self.gen, device=self.device) * decay
        rir[:, 0] += 1.0  # direct path
        rir = rir / rir.abs().amax(dim=1, keepdim=True).clamp_min(_EPS)
        nfft = 1 << (length + kmax - 2).bit_length()  # >= L+K-1, so circular conv == linear
        rev = torch.fft.irfft(torch.fft.rfft(audio, nfft) * torch.fft.rfft(rir, nfft), nfft)[:, :length]
        return torch.where(fire[:, None], rev, audio)

    def _noise(self, audio: Tensor) -> Tensor:
        """Add a random background crop at a sampled SNR (vectorized over the batch)."""
        bank = self.bank
        assert bank is not None  # forward() only calls _noise when the noise bank is set
        b, length = audio.shape
        fire = self._rand(b) < self.p_noise
        p, lpool = bank.shape
        nidx = torch.randint(0, p, (b,), generator=self.gen, device=self.device)
        max_off = lpool - length
        off = (
            torch.randint(0, max_off + 1, (b,), generator=self.gen, device=self.device)
            if max_off > 0
            else torch.zeros(b, dtype=torch.long, device=self.device)
        )
        cols = off[:, None] + torch.arange(length, device=self.device)
        noise = bank.index_select(0, nidx).gather(1, cols)  # [B, L]
        snr = self._uniform(b, *self.snr_db)
        gain = (audio.pow(2).mean(dim=1).sqrt() + _EPS) / (
            (noise.pow(2).mean(dim=1).sqrt() + _EPS) * 10.0 ** (snr / 20.0)
        )
        mixed = audio + noise * gain[:, None]
        return torch.where(fire[:, None], mixed, audio)
