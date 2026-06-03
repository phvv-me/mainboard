# Providers

Os providers detectam hardware e telemetria específicos de cada vendor mantendo a API pública centrada no conceito.

| provider | plataforma | status |
|---|---|---|
| `AppleGPU` | macOS com Apple Silicon | modelo da GPU, núcleos, suporte a Metal, memória unificada |
| `AppleNPU` | macOS com Apple Silicon | identidade do Neural Engine e backend Core ML |
| `NvidiaGPU` | Linux + CUDA | arquitetura CUDA, contagem de SMs, memória e clocks onde houver suporte |

Os providers AMD, Intel e Qualcomm são stubs seguros para importação hoje. Eles retornam indisponível para que imports e CI não exijam hardware ou SDKs de vendor.

Os detalhes de provider devem adicionar telemetria, não renomear os conceitos públicos. Uma GPU continua sendo uma `GPU`; CUDA, Metal, ROCm, Level Zero e Core ML são detalhes de backend.
