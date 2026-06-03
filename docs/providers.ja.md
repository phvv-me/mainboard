# Providers

プロバイダは、公開 API をコンセプトファーストに保ちながら、ベンダー固有のハードウェアとテレメトリを検出します。

| プロバイダ | プラットフォーム | ステータス |
|---|---|---|
| `AppleGPU` | Apple Silicon macOS | GPU モデル、コア数、Metal サポート、統合メモリ |
| `AppleNPU` | Apple Silicon macOS | Neural Engine の識別情報と Core ML バックエンド |
| `NvidiaGPU` | Linux + CUDA | CUDA アーキテクチャ、SM 数、メモリ、サポートされる場合はクロック |

AMD、Intel、Qualcomm のプロバイダは、現在のところインポート安全なスタブです。これらは利用不可を返すため、インポートや CI でハードウェアやベンダー SDK を必要としません。

プロバイダの詳細はテレメトリを追加するべきであり、公開された概念の名前を変えるべきではありません。GPU は依然として `GPU` であり、CUDA、Metal、ROCm、Level Zero、Core ML はバックエンドの詳細です。
