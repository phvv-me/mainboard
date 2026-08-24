import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pytest
from plumbum import local

from mainboard import Manifest
from mainboard.engines.compile import Ecosystem, SecondStage
from mainboard.engines.compile.backend import Pixi
from mainboard.engines.compile.compiler import Compiler
from mainboard.engines.compile.generated import GeneratedFiles, Writer
from mainboard.manifest import Toolchain

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mainboard.manifest.schema.spec import Json


class Bind(Protocol):
    """Binds one ecosystem implementation to a table body, in the fixture workspace."""

    def __call__[E: Ecosystem](self, kind: type[E], body: dict[str, Json]) -> E: ...


class Record(Protocol):
    """Writes one `dist-info` into a site-packages tree the way an installer leaves it."""

    def __call__(
        self,
        site_packages: Path,
        name: str,
        *,
        installer: str = ...,
        roots: str = ...,
        url: str = ...,
        editable: bool = ...,
        files: list[str] | None = ...,
    ) -> Path: ...


@pytest.fixture
def manifest_from() -> Callable[[str], Manifest]:
    """A factory where `manifest_from(text)` validates inline TOML into a `Manifest`."""

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
def files(pixi: Pixi) -> Iterator[Writer]:
    """The generated-file writer for the fixture workspace, its sync lock held for the test."""
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as writer:
        yield writer


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


@pytest.fixture
def bind(tmp_path: Path, pixi: Pixi) -> Bind:
    """A factory binding one ecosystem implementation to the table body a test declares."""

    def make[E: Ecosystem](kind: type[E], body: dict[str, Json]) -> E:
        return kind(
            Toolchain.model_validate(body),
            env="default",
            project="w",
            workspace=tmp_path,
            out=pixi.manifest.parent,
            pixi=pixi,
        )

    return make


@pytest.fixture
def stage_from(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> Callable[[str], SecondStage]:
    """A factory where `stage_from(text)` builds the second stage of an inline manifest."""

    def make(text: str) -> SecondStage:
        return SecondStage(tmp_path, manifest_from(text), pixi.manifest.parent, pixi)

    return make


@pytest.fixture
def compiler_from(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> Callable[[str], Compiler]:
    """A factory where `compiler_from(text)` builds the compiler of an inline manifest."""

    def make(text: str) -> Compiler:
        manifest = manifest_from(text)
        out = pixi.manifest.parent
        return Compiler(tmp_path, manifest, out, pixi, SecondStage(tmp_path, manifest, out, pixi))

    return make


@pytest.fixture
def record() -> Record:
    """A factory writing one `dist-info` the way an installer leaves it behind.

    Returns the import root the distribution declares, which is the path a damaged package is
    missing and a fake reinstall puts back.

    site_packages: the tree the record is written into.
    name: the distribution name, which is also what a repair reinstalls it by.
    installer: the manager claiming the record, `uv-pixi` for everything pixi installs.
    roots: the `top_level.txt` import roots, the file left out entirely when empty.
    url: where the install came from, its PEP 610 record left out entirely when empty.
    editable: whether that PEP 610 record marks the install as editable.
    files: the `RECORD` paths relative to site-packages, the file left out when `None`.
    """

    def write(
        site_packages: Path,
        name: str,
        *,
        installer: str = "uv-pixi",
        roots: str = "",
        url: str = "",
        editable: bool = False,
        files: list[str] | None = None,
    ) -> Path:
        metadata = site_packages / f"{name}-1.0.dist-info"
        metadata.mkdir()
        metadata.joinpath("METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
        metadata.joinpath("INSTALLER").write_text(installer)
        if roots:
            metadata.joinpath("top_level.txt").write_text(roots)
        if url:
            editability = str(editable).lower()
            metadata.joinpath("direct_url.json").write_text(
                f'{{"url": "{url}", "dir_info": {{"editable": {editability}}}}}'
            )
        if files is not None:
            metadata.joinpath("RECORD").write_text("".join(f"{path},,\n" for path in files))
        return site_packages / name.replace("-", "_")

    return write
