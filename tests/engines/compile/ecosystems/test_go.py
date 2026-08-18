from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.ecosystems import Go
from mainboard.manifest.schema.spec import Spec
from mainboard.manifest.schema.toolchain import Toolchain

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
    from mainboard.manifest.schema.spec import Json


def _go(body: dict[str, Json], tmp_path: Path, pixi: Pixi) -> Go:
    return Go(
        Toolchain.model_validate(body),
        env="default",
        project="w",
        workspace=tmp_path,
        out=pixi.manifest.parent,
        pixi=pixi,
    )


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("github.com/owner/tool", "tool"),
        ("github.com/owner/tool/v2", "tool"),
        ("github.com/owner/tool/cmd/inner", "inner"),
        ("github.com/owner/tool/", "tool"),
        ("tool", "tool"),
    ],
)
def test_the_executable_name_follows_the_module_path(module: str, expected: str) -> None:
    """A major-version suffix names no executable, so `.../v2` still installs as the tool."""
    assert Go.executable(module) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("*", "example.com/tool@latest"),
        ("1.4.0", "example.com/tool@v1.4.0"),
        ("v1.4.0", "example.com/tool@v1.4.0"),
        ("main", "example.com/tool@main"),
        ("latest", "example.com/tool@latest"),
    ],
)
def test_a_requirement_go_can_resolve_becomes_a_module_reference(
    version: str, expected: str
) -> None:
    assert Go.reference("example.com/tool", Spec.model_validate(version)) == expected


def test_a_version_range_is_refused_where_it_is_declared() -> None:
    """Go resolves one version, so a range would reach the module proxy as a broken reference."""
    with pytest.raises(MissionError, match=r"example.com/tool.*never a range"):
        Go.reference("example.com/tool", Spec.model_validate(">=1.4"))


def test_modules_install_into_a_generated_directory_this_workspace_owns(
    tmp_path: Path, pixi: Pixi
) -> None:
    go = _go({"deps": {"example.com/tool": "*"}}, tmp_path, pixi)
    assert go.gobin == pixi.manifest.parent / "go" / "bin"
    assert go.binary_dirs() == (go.gobin,)


def test_sync_installs_every_declared_module_with_the_environments_go(
    tmp_path: Path, pixi: Pixi, fp: FakeProcess, tool_paths: dict[str, str]
) -> None:
    go = _go({"deps": {"example.com/tool": "v1.4.0"}}, tmp_path, pixi)
    fp.register([fp.any()], stdout="\n")

    go.sync()

    assert list(fp.calls[0]) == [
        tool_paths["pixi"],
        "run",
        "--manifest-path",
        str(pixi.manifest),
        "--environment",
        "default",
        "go",
        "install",
        "example.com/tool@v1.4.0",
    ]
    assert go.gobin.is_dir()


def test_sync_unlinks_an_executable_the_table_no_longer_declares(
    tmp_path: Path, pixi: Pixi, fp: FakeProcess, tool_paths: dict[str, str]
) -> None:
    go = _go({"deps": {"example.com/kept": "*"}}, tmp_path, pixi)
    go.gobin.mkdir(parents=True)
    (go.gobin / "kept").write_text("")
    (go.gobin / "dropped").write_text("")
    fp.register([fp.any()], stdout="\n")

    go.sync()

    assert sorted(path.name for path in go.gobin.iterdir()) == ["kept"]


def test_a_table_without_modules_installs_nothing_and_creates_no_directory(
    tmp_path: Path, pixi: Pixi, fp: FakeProcess
) -> None:
    go = _go({}, tmp_path, pixi)
    go.sync()
    assert not fp.calls
    assert not go.gobin.exists()
