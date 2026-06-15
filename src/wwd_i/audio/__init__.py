"""Audio ingest: load and resample to the canonical 16 kHz mono float32 format,
and split a stream into 80 ms frames from either a file or a live microphone."""

from wwd_i.audio.io import load_wav, to_frames
from wwd_i.audio.sources import file_frames, mic_frames

__all__ = ["file_frames", "load_wav", "mic_frames", "to_frames"]
