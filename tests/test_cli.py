from __future__ import annotations

import pytest

from mainboard import cli


def test_show_renders_via_machine_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default command builds a `MachineView` and prints it with the color flag."""
    calls: list[bool] = []

    class FakeView:
        def __init__(self, machine: object) -> None:
            self.machine = machine

        def print(self, *, color: bool = True) -> None:
            calls.append(color)

    monkeypatch.setattr(cli, "MachineView", FakeView)
    cli.show(color=False)
    assert calls == [False]


def test_main_invokes_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main` runs the cyclopts application object."""
    ran: list[bool] = []
    monkeypatch.setattr(cli, "app", lambda: ran.append(True))
    cli.main()
    assert ran == [True]
