from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from patos import FrozenModel

from mainboard import Results
from mainboard.verbs.workspace.core import Caps

from .rendering import Rendering

_CAPS = Caps()


@pytest.fixture
def render(workspace: Path, capsys: pytest.CaptureFixture[str]) -> Callable[..., Rendering]:
    """`Query.render` over a stream of one statement in the workspace, as `mb query` calls it."""

    def run(
        sql: str,
        *,
        mode: str | None = None,
        max_rows: int = _CAPS.max_rows,
        max_bytes: int = _CAPS.max_bytes,
    ) -> Rendering:
        with Results(workspace).stream(sql) as query:
            stamp = query.render(mode=mode, max_rows=max_rows, max_bytes=max_bytes)
        printed = capsys.readouterr()
        return Rendering(out=printed.out, err=printed.err, stamp=stamp)

    return run


@pytest.fixture
def in_process_mb(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Callable[..., FrozenModel]:
    """The root `mb` fixture in its inspect lane: the CLI runs here, where a test may patch it."""
    monkeypatch.setenv("MB_GATE_INSPECT", "1")
    return request.getfixturevalue("mb")


@pytest.fixture
def read_only() -> Iterator[Callable[[Path], None]]:
    """Lock folders against writes for the test, and open them again at its end."""
    locked: list[Path] = []

    def lock(folder: Path) -> None:
        folder.chmod(0o555)
        locked.append(folder)

    yield lock
    for folder in locked:
        folder.chmod(0o755)
