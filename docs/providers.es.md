# Proveedores

Los proveedores detectan hardware y telemetría específicos del fabricante manteniendo la API pública centrada en el concepto.

| proveedor | plataforma | estado |
|---|---|---|
| `AppleGPU` | macOS con Apple Silicon | modelo de GPU, núcleos, soporte de Metal, memoria unificada |
| `AppleNPU` | macOS con Apple Silicon | identidad del Neural Engine y backend de Core ML |
| `NvidiaGPU` | Linux + CUDA | arquitectura CUDA, conteo de SM, memoria, relojes donde se admite |

Los proveedores AMD, Intel y Qualcomm son hoy stubs seguros para importar. Devuelven no disponible para que las importaciones y la CI no requieran hardware ni SDK de fabricantes.

Los detalles del proveedor deben agregar telemetría, no renombrar los conceptos públicos. Una GPU sigue siendo una `GPU`; CUDA, Metal, ROCm, Level Zero y Core ML son detalles de backend.
