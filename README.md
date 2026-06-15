# Alternative open Wake Word Detector

While being a de facto standard, open-source Wake Word Detector,
`openWakeWord` rely on obsolete and old technologies (by mid 2026 standards).
Idea is to reinvent everything from scratch. While keeping the same small footprint.
Also streamline the training process (using fewer samples but of higher quality TTS engine).

## Stack

- Python 3.14
- Cuda 13
- ONNX
- Google Colab (training runtime)
- Elevenlabs v3 (sample generation)

## Design

- [Architecture & design](docs/architecture.md)
- [Phased implementation plan](docs/implementation-plan.md)

## References

https://github.com/dscripka/openWakeWord/tree/main
https://github.com/lgpearson1771/openwakeword-trainer
