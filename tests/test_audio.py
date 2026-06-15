import numpy as np
import soundfile as sf

from wwd_i.audio.io import load_wav, to_frames
from wwd_i.config import FRAME_SAMPLES, SAMPLE_RATE


def test_to_frames_drops_partial():
    sig = np.arange(FRAME_SAMPLES * 3 + 100, dtype=np.float32)
    frames = list(to_frames(sig))
    assert len(frames) == 3
    assert all(len(f) == FRAME_SAMPLES for f in frames)
    np.testing.assert_array_equal(frames[0], sig[:FRAME_SAMPLES])


def test_to_frames_pads_last():
    sig = np.arange(FRAME_SAMPLES + 100, dtype=np.float32)
    frames = list(to_frames(sig, pad=True))
    assert len(frames) == 2
    assert len(frames[1]) == FRAME_SAMPLES
    np.testing.assert_array_equal(frames[1][100:], np.zeros(FRAME_SAMPLES - 100, dtype=np.float32))


def test_load_wav_resamples_to_16k(tmp_path):
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, tone, sr)

    out = load_wav(path)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(len(out) - SAMPLE_RATE) <= 1
    assert float(np.abs(out).max()) <= 1.0


def test_load_wav_downmixes_stereo(tmp_path):
    sr = SAMPLE_RATE
    stereo = np.zeros((sr, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, sr)

    out = load_wav(path)
    assert out.ndim == 1
    assert len(out) == sr
    assert np.allclose(out, 0.0, atol=1e-6)
