import sys
from typing import TYPE_CHECKING

import pytest

from mainboard.manifest import Tracking

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

# The variable the wandb sink reads, spelled here so a test can put one there or take it away.
_KEY = "WANDB_API_KEY"


class Bag:
    """A run's config or summary: a mapping something updates and a test then reads back."""

    def __init__(self) -> None:
        self.held: dict[str, JsonValue] = {}
        self.options: dict[str, bool] = {}

    def update(self, values: Mapping[str, JsonValue], **options: bool) -> None:
        self.held.update(values)
        self.options = dict(options)


class FakeRun:
    """One opened run, keeping what it was told so a test asserts on the mapping, not the SDK."""

    def __init__(self, step: int, options: dict[str, JsonValue]) -> None:
        self.options = options
        self.step = step
        self.history: list[tuple[int | None, dict[str, JsonValue]]] = []
        self.config = Bag()
        self.summary = Bag()
        self.exit_code: int | None = None
        self.finished = False
        self.url = "https://wandb.test/run"

    def finish(self, exit_code: int | None = None) -> None:
        self.finished = True
        self.exit_code = exit_code

    def log(self, data: Mapping[str, JsonValue], step: int | None = None) -> None:
        self.history.append((step, dict(data)))


class FakeWandb:
    """The `wandb` module as this sink uses it, with a resume position a test can pre-set."""

    def __init__(self, resume_at: int = 0) -> None:
        self.resume_at = resume_at
        self.runs: list[FakeRun] = []

    def init(self, **options: JsonValue) -> FakeRun:
        opened = FakeRun(self.resume_at, dict(options))
        self.runs.append(opened)
        return opened


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    """The tracking SDK, replaced by a stand-in for the length of one test."""
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def declared() -> Tracking:
    """The `[tracking]` table a test tunes, at the defaults a workspace that wrote none gets."""
    return Tracking()


def keyed(monkeypatch: pytest.MonkeyPatch, *, key: str = "") -> None:
    """Put a credential in this process's environment, or take the one there away.

    key: what the tracking variable should read as, absent when empty.
    """
    monkeypatch.delenv(_KEY, raising=False)
    if key:
        monkeypatch.setenv(_KEY, key)
