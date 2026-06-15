"""wwd_i — an alternative open wake-word detector (from-scratch, two-stage).

See docs/architecture.md for the design and docs/implementation-plan.md for the
phased plan. Phase 0 provides the audio ingest plus an ONNX-Runtime streaming
harness; later phases add the mel front-end, backbone, head and training.
"""

__version__ = "0.1.0"
