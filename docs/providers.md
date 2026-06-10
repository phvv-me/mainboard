# Providers

Providers detect vendor-specific hardware and telemetry while keeping the public API concept-first.

| provider | platform | status |
|---|---|---|
| `AppleGPU` | Apple Silicon macOS | GPU model, cores, Metal support, unified memory |
| `AppleNPU` | Apple Silicon macOS | Neural Engine identity and Core ML backend |
| `NvidiaGPU` | Linux + CUDA | CUDA architecture, SM count, memory, clocks where supported |

AMD, Intel, and Qualcomm providers are import-safe stubs today. They return unavailable so imports and CI do not require hardware or vendor SDKs.

Provider details should add telemetry, not rename the public concepts. A GPU is still a `GPU`. CUDA, Metal, ROCm, Level Zero, and Core ML are backend details.
