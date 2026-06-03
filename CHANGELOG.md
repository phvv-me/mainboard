# Changelog

All notable changes to Mainboard are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project uses semantic versioning while it is published.

## [0.0.1] - 2026-06-01

### Added

- Concept-first `Machine`, `Unit`, `CPU`, `GPU`, and `NPU` API.
- Apple Silicon CPU, GPU, and Neural Engine detection.
- NVIDIA CUDA GPU detection with CUDA Runtime memory fallback.
- Rich terminal machine schematic through `mainboard` and `python -m mainboard`.
- Provider stubs for AMD, Intel, and Qualcomm.
- Typed Pydantic telemetry models for snapshots, memory, clocks, compilers, disks, and thermal state.
