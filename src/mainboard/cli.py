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


def main() -> None:
    """Run the Mainboard command-line interface."""
    app()


if __name__ == "__main__":
    main()
