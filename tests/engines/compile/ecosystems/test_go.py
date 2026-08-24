from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.ecosystems import Go
from mainboard.manifest import Spec

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi

    from ..conftest import Bind

_TOOL = "example.com/tool"


@pytest.mark.parametrize(
    ("module", "executable"),
    [
        pytest.param("github.com/owner/tool", "tool", id="the-last-element-of-the-module-path"),
        pytest.param("github.com/owner/tool/v2", "tool", id="a-major-version-suffix-names-none"),
        pytest.param("github.com/owner/tool/cmd/inner", "inner", id="a-nested-command"),
        pytest.param("github.com/owner/tool/", "tool", id="a-trailing-separator"),
        pytest.param("tool", "tool", id="a-module-path-of-one-element"),
    ],
)
def test_the_executable_name_follows_the_module_path(module: str, executable: str) -> None:
    """`example.com/tool/v2` still installs as `tool`, so the suffix names no executable."""
    assert Go.executable(module) == executable


@pytest.mark.parametrize(
    ("version", "reference"),
    [
        pytest.param("*", f"{_TOOL}@latest", id="an-unconstrained-requirement-is-latest"),
        pytest.param("1.4.0", f"{_TOOL}@v1.4.0", id="a-bare-semver-gains-the-v-prefix"),
        pytest.param("v1.4.0", f"{_TOOL}@v1.4.0", id="a-version-already-prefixed"),
        pytest.param("main", f"{_TOOL}@main", id="a-branch-rides-through-as-written"),
        pytest.param("latest", f"{_TOOL}@latest", id="the-latest-keyword-itself"),
    ],
)
def test_a_requirement_go_can_resolve_becomes_a_module_reference(
    version: str, reference: str
) -> None:
    assert Go.reference(_TOOL, Spec.model_validate(version)) == reference


def test_a_version_range_is_refused_where_it_is_declared() -> None:
    """Go resolves one version, so a range would reach the module proxy as a broken reference."""
    with pytest.raises(MissionError, match=r"example.com/tool.*never a range"):
        Go.reference(_TOOL, Spec.model_validate(">=1.4"))


def test_sync_installs_every_declared_module_and_unlinks_what_was_dropped(
    bind: Bind, pixi: Pixi, fp: FakeProcess, tool_paths: dict[str, str]
) -> None:
    """`GOBIN` is one directory inside the generated tree that this workspace owns outright,
    so an executable the table stopped declaring is pruned rather than left to shadow."""
    go = bind(Go, {"deps": {_TOOL: "v1.4.0"}})
    go.gobin.mkdir(parents=True)
    go.gobin.joinpath("tool").write_text("")
    go.gobin.joinpath("dropped").write_text("")
    fp.register([fp.any()], stdout="\n")

    go.sync()

    assert go.gobin == pixi.manifest.parent / "go" / "bin"
    assert go.binary_dirs() == (go.gobin,)
    assert list(fp.calls[0]) == [
        tool_paths["pixi"],
        "run",
        "--manifest-path",
        str(pixi.manifest),
        "--environment",
        "default",
        "go",
        "install",
        f"{_TOOL}@v1.4.0",
    ]
    assert sorted(path.name for path in go.gobin.iterdir()) == ["tool"]


def test_a_table_without_modules_installs_nothing_and_creates_no_directory(
    bind: Bind, fp: FakeProcess
) -> None:
    go = bind(Go, {})
    go.sync()
    assert not fp.calls
    assert not go.gobin.exists()
