from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.ecosystems import Go
from mainboard.manifest import Spec

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi

    from ..support import Bind

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
    assert Go.executable(module) == executable


@pytest.mark.parametrize(
    ("version", "reference"),
    [
        pytest.param("*", f"{_TOOL}@latest", id="an-unconstrained-requirement-is-latest"),
        pytest.param("1.4.0", f"{_TOOL}@v1.4.0", id="a-bare-semver-gains-the-v-prefix"),
        pytest.param("v1.4.0", f"{_TOOL}@v1.4.0", id="a-version-already-prefixed"),
        pytest.param("main", f"{_TOOL}@main", id="a-branch-rides-through-as-written"),
        pytest.param("latest", f"{_TOOL}@latest", id="the-latest-keyword-itself"),
        pytest.param("1" * 40, f"{_TOOL}@{'1' * 40}", id="a-commit-is-not-a-numbered-version"),
    ],
)
def test_a_requirement_go_can_resolve_becomes_a_module_reference(
    version: str, reference: str
) -> None:
    assert Go.reference(_TOOL, Spec.model_validate(version)) == reference


def test_a_version_range_is_refused_where_it_is_declared() -> None:
    with pytest.raises(MissionError, match=r"example.com/tool.*never a range"):
        Go.reference(_TOOL, Spec.model_validate(">=1.4"))


def test_sync_installs_every_declared_module_and_unlinks_what_was_dropped(
    bind: Bind, pixi: Pixi, fp: FakeProcess, tool_paths: Mapping[str, str]
) -> None:
    go = bind(Go, {"deps": {_TOOL: "v1.4.0"}})
    go.gobin.mkdir(parents=True)
    go.gobin.joinpath("tool").write_text("")
    go.gobin.joinpath("dropped").write_text("")
    fp.register([fp.any()], stdout="\n")

    go.sync()

    assert go.gobin == pixi.manifest.parent / "go" / "bin"
    assert go.binary_dirs() == (go.gobin,)
    assert " ".join(fp.calls[0]) == (
        f"{tool_paths['pixi']} run --manifest-path {pixi.manifest} --environment default "
        f"go install {_TOOL}@v1.4.0"
    )
    assert sorted(path.name for path in go.gobin.iterdir()) == ["tool"]


def test_a_table_without_modules_installs_nothing_and_creates_no_directory(
    bind: Bind, fp: FakeProcess
) -> None:
    go = bind(Go, {})
    go.sync()
    assert not fp.calls
    assert not go.gobin.exists()


@pytest.mark.parametrize("version", ["*", "latest", "main", "v1"])
def test_frozen_install_refuses_floating_go_references(version: str, bind: Bind) -> None:
    with pytest.raises(MissionError, match="frozen installation"):
        bind(Go, {"deps": {_TOOL: version}}).frozen_inputs()
