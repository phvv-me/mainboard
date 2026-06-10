# Changelog

All notable changes to Mainboard are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project uses semantic versioning while it is published.

## [0.0.5] - 2026-06-10

### Added

- `mainboard.profile(fn, *, iters=...)` runs a callable under the profiler and returns a structured `ProfileReport` naming the dominant kernel, its share of GPU time, whether the work is memory- or compute-bound, the launch shape (occupancy proxy, registers per thread, static and dynamic shared memory), and the full per-kernel breakdown. It adapts the requested activity kinds to what the device supports, so an unavailable kind (for example `MEMORY` on GB10) is recorded in the report rather than crashing the run.
- `mainboard.gpu_busy(index=0)` and `mainboard.wait_for_idle(index=0, *, timeout, ...)` read the live NVML utilization and memory so a caller can take measurements in a clean, uncontended window.
- `KernelTrace` now splits shared memory into static and dynamic, and exposes `threads_per_block` and an `occupancy_pct` launch-shape proxy.

### Changed

- Typing is now mypy strict with `disallow_any_explicit`, added to CI. Provider, CUPTI, and NVML seams are typed with Protocols, removing all explicit `Any` and `object` from the source.
- The docs adopt the shared Open Props design language over mkdocs-material, with a legible app-icon as logo and favicon, and a working `llms.txt` from the english post-build hook.
- CI actions updated to setup-uv v7, upload-pages-artifact v5, deploy-pages v5, and gh-release v3.

## [0.0.4] - 2026-06-05

### Added

- `meter()` context manager for torch-free runtime metrics: times a region and tracks peak host and GPU memory from the live snapshot (`elapsed_s`, `peak_host_gb`, `peak_gpu_gb`, `host_delta_gb`).
- `MemoryHardware` model carrying physical DIMM slots and swap, split out of the host memory usage.

### Changed

- Collapsed `MemInfo`, `MemoryUsage`, and the host memory usage into one `Memory` model with `total_gb`/`used_gb`/`free_gb` and `percent_used`.
- Units and snapshots now expose memory the same way: every unit has `.memory` (a `Memory`) and `UnitSnapshot.memory` is a single `Memory`.
- `GPUSnapshot` drops the redundant `gpu_memory`/`gpu_clocks` fields in favor of the inherited `memory`/`clocks`.
- CUDA bindings moved into core dependencies (Linux only); the `mainboard[nvidia]` extra is gone, so NVIDIA GPUs are detected automatically on Linux.

## [0.0.1] - 2026-06-01

### Added

- Concept-first `Machine`, `Unit`, `CPU`, `GPU`, and `NPU` API.
- Apple Silicon CPU, GPU, and Neural Engine detection.
- NVIDIA CUDA GPU detection with CUDA Runtime memory fallback.
- Rich terminal machine schematic through `mainboard` and `python -m mainboard`.
- Provider stubs for AMD, Intel, and Qualcomm.
- Typed Pydantic telemetry models for snapshots, memory, clocks, compilers, disks, and thermal state.
