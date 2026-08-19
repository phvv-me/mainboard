import os
import tomllib
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import Manifest
from mainboard.engines.compile.backend import Pixi

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture
def manifest_from() -> Callable[[str], Manifest]:
    """A factory: `manifest_from(text)` parses and validates inline TOML into a `Manifest`."""

    def make(text: str) -> Manifest:
        return Manifest.model_validate(tomllib.loads(text))

    return make


@pytest.fixture(autouse=True)
def tool_paths(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    """Stub a `pixi` executable on plumbum's PATH.

    Backend commands resolve without the real tool installed, and `pytest-subprocess`
    intercepts the actual invocation. Yields the tool's resolved absolute path, which is what
    plumbum runs and therefore what a fake registers.

    Autouse, because `PixiEngine.command` falls back to bootstrapping pixi when the name is
    absent from PATH. Without the stub these tests pass only on a machine that already has pixi
    installed, and on one that does not the bootstrap runs the installer as an extra subprocess
    that eats the registered fake before the call under test is ever made.
    """
    bindir = tmp_path_factory.mktemp("bin")
    executable = bindir / "pixi"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    with local.env(PATH=f"{bindir}{os.pathsep}{local.env['PATH']}"):
        yield {"pixi": str(executable)}


@pytest.fixture(autouse=True)
def isolated_pixi_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every test's global pixi home inside its own `tmp_path`, never the real `~/.pixi`."""
    home = tmp_path / "pixi-home"
    monkeypatch.setenv("PIXI_HOME", str(home))
    return home


@pytest.fixture
def pixi(tmp_path: Path) -> Pixi:
    """A Pixi backend pinned to a fresh workspace's generated env dir."""
    out = tmp_path / ".mainboard"
    out.mkdir()
    return Pixi(out)


@pytest.fixture
def stub_binary(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Callable[[str], str]]:
    """A factory placing a fake executable on plumbum's PATH, returning its resolved path.

    `pytest-subprocess` intercepts the run itself, so the file only has to exist and be
    executable for plumbum's lookup to resolve it, and that resolved path is what a fake
    process registers against.
    """
    bindir = tmp_path_factory.mktemp("stubs")

    def install(name: str) -> str:
        executable = bindir / name
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        return str(executable)

    with local.env(PATH=f"{bindir}{os.pathsep}{local.env['PATH']}"):
        yield install
