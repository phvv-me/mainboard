import os
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import Manifest
from mainboard.engines.compile import Ecosystem, SecondStage
from mainboard.engines.compile.backend import PIXI_VERSION, Pixi
from mainboard.engines.compile.compiler import Compiler
from mainboard.engines.compile.generated import GeneratedFiles, Writer
from mainboard.engines.compile.pixi_manifest import selected_manifest
from mainboard.engines.compile.vendor import Vendor
from mainboard.manifest import Toolchain

from .support import Bind, CompilerFrom, Record

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

    from mainboard.manifest.schema.spec import Json


@pytest.fixture
def manifest_from() -> Callable[[str], Manifest]:
    """A factory where `manifest_from(text)` validates inline TOML into a `Manifest`."""

    def make(text: str) -> Manifest:
        return Manifest.model_validate(tomllib.loads(text))

    return make


@pytest.fixture(autouse=True)
def tool_paths(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    """Stub `pixi` on plumbum's PATH, yielding the resolved path a fake registers against.

    Autouse, since without it `PixiEngine.command` bootstraps pixi on a machine lacking one and
    the installer eats the registered fake.
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
def solver_version(fp: FakeProcess, tool_paths: dict[str, str]) -> str:
    """Answer `pixi --version` with the pin, registered before a test's own `fp.any()` fakes."""
    fp.register([tool_paths["pixi"], "--version"], stdout=f"pixi {PIXI_VERSION}\n", occurrences=4)
    return PIXI_VERSION


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
    """A factory placing a fake executable on plumbum's PATH, returning its resolved path."""
    bindir = tmp_path_factory.mktemp("stubs")

    def install(name: str) -> str:
        # Windows runs a program only by a `PATHEXT` spelling, so a bare name stands as `.exe`.
        if os.name == "nt":
            executable = bindir / (name if Path(name).suffix else f"{name}.exe")
        else:
            executable = bindir / name.removesuffix(".exe")
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        # POSIX has no PATHEXT, so an `.exe` stub also stands bare there. `os.name` is the real
        # platform, since a test may be pretending to be Windows.
        if os.name != "nt" and executable.suffix == ".exe":
            bare = executable.with_suffix("")
            bare.write_text("#!/bin/sh\n")
            bare.chmod(0o755)
            return str(bare)
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
) -> CompilerFrom:
    """A factory where `compiler_from(text)` builds the compiler of an inline manifest."""

    def make(text: str, *, environment: str = "default") -> Compiler:
        manifest = manifest_from(text)
        projected = selected_manifest(manifest, environment)
        out = pixi.manifest.parent
        return Compiler(
            tmp_path,
            projected,
            out,
            pixi,
            SecondStage(tmp_path, projected, out, pixi),
            Vendor(tmp_path, manifest),
            environment=environment,
        )

    return make


@pytest.fixture
def record() -> Record:
    """A factory writing one `dist-info` as an installer leaves it, returning its import root.

    roots: `top_level.txt`, omitted when empty.
    url: the PEP 610 source, its record omitted when empty.
    files: `RECORD` paths relative to site-packages, omitted when `None`.
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
