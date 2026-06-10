from __future__ import annotations

from cyclopts import App

from . import NAME
from .machine import Machine
from .visual import MachineView

app = App(name=NAME, help="Inspect CPU, GPU, and NPU hardware topology.")


@app.default
def show(color: bool = True) -> None:
    """Render a Rich terminal view of the current machine."""
    MachineView(Machine()).print(color=color)


@app.command
def profile(
    target: str,
    *,
    auto: str = "",
    trace: bool = False,
    perfetto: str = "",
    color: bool = True,
) -> None:
    """Run a module or script under the profiler and show where time goes.

    target: a module name (``pkg.mod``) or a ``.py`` script path, run as ``__main__``.
    auto: comma-separated package prefixes to auto-annotate (every call becomes a region).
    trace: enable deep per-kernel tracing (CUDA). perfetto: also write a timeline JSON to
    this path (open at ui.perfetto.dev).
    """
    import runpy

    from .profiling import Profiler

    modules = tuple(part for part in auto.split(",") if part)
    with Profiler(trace=trace, auto=modules) as profiler:
        if target.endswith(".py"):
            runpy.run_path(target, run_name="__main__")
        else:
            runpy.run_module(target, run_name="__main__")
    result = profiler.result()
    result.show(color=color)
    if perfetto:
        result.perfetto(perfetto)


def main() -> None:
    """Run the Mainboard command-line interface."""
    app()


if __name__ == "__main__":
    main()
