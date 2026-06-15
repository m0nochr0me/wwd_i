"""Mel front-end (Phase 1): log-mel spectrogram in PyTorch + an exported ONNX
twin, with a parity test, plus a streaming wrapper.

Submodules import torch/torchaudio (the `train` group); they are not imported at
package level so the inference runtime stays torch-free."""
