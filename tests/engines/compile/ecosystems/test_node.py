import json
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.ecosystems import Node

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
    from mainboard.engines.compile.generated import Writer

    from ..conftest import Bind


@pytest.mark.parametrize(
    ("app", "name"),
    [
        pytest.param(False, "w-npm", id="a-plain-toolchain-stays-inside-the-generated-directory"),
        pytest.param(True, "w", id="an-application-moves-to-the-workspace-root"),
    ],
)
def test_where_a_toolchain_installs_follows_whether_it_is_the_application(
    *, app: bool, name: str, bind: Bind, pixi: Pixi, tmp_path: Path, files: Writer
) -> None:
    """A bundler resolves `node_modules` from the application root, so `app` moves both there,
    and a toolchain that is not the application never claims the name the workspace publishes."""
    node = bind(Node, {"app": app, "deps": {"vite": ">=5"}})
    directory = tmp_path if app else pixi.manifest.parent

    node.generate(files)

    assert node.directory == directory
    assert node.manifest == directory / "package.json"
    assert node.binary_dirs() == (directory / "node_modules" / ".bin",)
    assert json.loads(node.manifest.read_text())["name"] == name


def test_runtime_dev_and_declared_fields_land_where_the_manager_reads_them(
    bind: Bind, files: Writer
) -> None:
    node = bind(
        Node,
        {
            "deps": {"prettier": ">=3"},
            "dev": {"eslint": "^10"},
            "package": {"type": "module", "engines": {"node": ">=22"}},
        },
    )

    node.generate(files)

    body = json.loads(node.manifest.read_text())
    assert body["dependencies"] == {"prettier": ">=3"}
    assert body["devDependencies"] == {"eslint": "^10"}
    assert body["private"] is True
    assert body["type"] == "module"
    assert body["engines"] == {"node": ">=22"}


def test_a_package_key_that_is_not_a_table_is_ignored_rather_than_merged(bind: Bind) -> None:
    """`package` names the fields table, so a scalar there cannot become manifest fields."""
    assert bind(Node, {"deps": {"vite": "*"}, "package": "module"}).fields == {}


def test_a_table_left_without_dependencies_drops_its_generated_manifest(
    bind: Bind, files: Writer
) -> None:
    """A surviving `package.json` would keep reinstalling what the manifest stopped declaring."""
    node = bind(Node, {})
    node.manifest.write_text('{"name": "w-npm"}\n')

    node.generate(files)

    assert not node.manifest.exists()


def test_a_source_requirement_npm_would_misread_is_refused_at_compile(bind: Bind) -> None:
    node = bind(Node, {"deps": {"private-lib": {"git": "https://example.com/lib.git"}}})
    with pytest.raises(MissionError, match=r"private-lib.*git"):
        node.compiled()


def test_sync_installs_through_the_declared_manager_once_there_is_a_manifest(
    bind: Bind, fp: FakeProcess, stub_binary: Callable[[str], str]
) -> None:
    """The manager needs no environment flag, since the directory it runs in is the
    environment it installs into."""
    pnpm = stub_binary("pnpm")
    node = bind(Node, {"manager": "pnpm", "deps": {"vite": "*"}})

    node.sync()
    assert not fp.calls

    node.manifest.write_text('{"name": "w-npm"}\n')
    fp.register([fp.any()], stdout="added 1 package\n")
    node.sync()

    assert list(fp.calls[0]) == [pnpm, "install"]
