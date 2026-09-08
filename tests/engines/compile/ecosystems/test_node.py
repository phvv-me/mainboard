import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.ecosystems import Node

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
    from mainboard.engines.compile.generated import Writer

    from ..support import Bind


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
    """`app` moves the whole node tree to the application root.

    A bundler resolves `node_modules` from there, and a toolchain that is not the
    application never claims the name the workspace publishes.
    """
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
    """The manager needs no environment flag.

    The directory it runs in is the environment it installs into.
    """
    pnpm = stub_binary("pnpm")
    node = bind(Node, {"manager": "pnpm", "deps": {"vite": "*"}})

    with pytest.raises(MissionError, match="missing despite declared Node"):
        node.sync(resolve=True)
    assert not fp.calls

    node.manifest.write_text('{"name": "w-npm"}\n')
    fp.register([fp.any()], stdout="added 1 package\n")
    node.sync(resolve=True)

    assert list(fp.calls[0]) == [pnpm, "install"]


@pytest.mark.parametrize(
    ("manager", "lock", "argv"),
    [
        ("npm", "package-lock.json", ["ci"]),
        ("pnpm", "pnpm-lock.yaml", ["install", "--frozen-lockfile"]),
        ("yarn", "yarn.lock", ["install", "--frozen-lockfile"]),
        ("bun", "bun.lock", ["install", "--frozen-lockfile"]),
    ],
)
def test_frozen_install_requires_and_preserves_the_native_lock(
    manager: str,
    lock: str,
    argv: list[str],
    bind: Bind,
    fp: FakeProcess,
    stub_binary: Callable[[str], str],
) -> None:
    executable = stub_binary(manager)
    node = bind(Node, {"manager": manager, "deps": {"vite": "*"}})
    node.manifest.write_text('{"name": "w-npm"}\n')
    with pytest.raises(MissionError, match="locally before shipping"):
        node.sync()
    assert not fp.calls
    locked = node.directory / lock
    locked.write_text("locked dependency versions\n")
    before = locked.read_bytes()
    fp.register([fp.any()], stdout="installed from lock\n")
    node.sync()
    assert list(fp.calls[0]) == [executable, *argv]
    assert node.frozen_inputs() == (node.manifest, locked)
    assert locked.read_bytes() == before


def test_remote_application_install_refuses_mutable_workspace_ownership(bind: Bind) -> None:
    node = bind(Node, {"app": True, "deps": {"vite": "*"}})
    with pytest.raises(MissionError, match="not an isolated prefix"):
        node.frozen_inputs()


def test_npm_shrinkwrap_takes_precedence_over_package_lock(bind: Bind) -> None:
    node = bind(Node, {"deps": {"vite": "*"}})
    for name in ("package-lock.json", "npm-shrinkwrap.json"):
        (node.directory / name).write_text("{}\n")
    assert node.lock().name == "npm-shrinkwrap.json"


@pytest.mark.parametrize("app", [False, True])
def test_package_fields_alone_still_require_a_manifest_and_frozen_lock(
    *, app: bool, bind: Bind, files: Writer, fp: FakeProcess
) -> None:
    node = bind(Node, {"app": app, "package": {"dependencies": {"vite": "^5"}}})
    with pytest.raises(MissionError, match="missing despite declared Node"):
        node.sync()
    assert not fp.calls
    node.generate(files)
    assert json.loads(node.manifest.read_text())["dependencies"] == {"vite": "^5"}
    message = "not an isolated prefix" if app else "no npm lock"
    with pytest.raises(MissionError, match=message):
        node.frozen_inputs()


def test_empty_node_table_does_not_install_a_stray_manifest(bind: Bind, fp: FakeProcess) -> None:
    node = bind(Node, {})
    node.manifest.write_text('{"dependencies":{"vite":"^5"}}\n')
    node.sync()
    assert node.frozen_inputs() == ()
    assert not fp.calls
