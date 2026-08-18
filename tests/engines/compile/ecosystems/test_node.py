import json
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.ecosystems import Node
from mainboard.engines.compile.generated import GeneratedFiles
from mainboard.manifest.schema.toolchain import Toolchain

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
    from mainboard.manifest.schema.spec import Json


def _node(body: dict[str, Json], tmp_path: Path, pixi: Pixi) -> Node:
    return Node(
        Toolchain.model_validate(body),
        env="default",
        project="w",
        workspace=tmp_path,
        out=pixi.manifest.parent,
        pixi=pixi,
    )


def _written(node: Node) -> dict[str, Json]:
    with GeneratedFiles(directory=node.out).locked() as files:
        node.generate(files)
    return json.loads(node.manifest.read_text())


def test_a_plain_toolchain_installs_inside_the_generated_directory(
    tmp_path: Path, pixi: Pixi
) -> None:
    node = _node({"deps": {"prettier": ">=3"}}, tmp_path, pixi)
    assert node.directory == pixi.manifest.parent
    assert node.manifest == pixi.manifest.parent / "package.json"
    assert node.binary_dirs() == (pixi.manifest.parent / "node_modules" / ".bin",)


def test_an_application_installs_at_the_workspace_root_under_its_own_name(
    tmp_path: Path, pixi: Pixi
) -> None:
    """A bundler resolves `node_modules` from the application root, so `app` moves both there."""
    node = _node({"app": True, "deps": {"vite": ">=5"}}, tmp_path, pixi)
    assert node.directory == tmp_path
    assert node.binary_dirs() == (tmp_path / "node_modules" / ".bin",)
    assert _written(node)["name"] == "w"


def test_a_toolchain_that_is_not_the_application_is_named_apart_from_the_workspace(
    tmp_path: Path, pixi: Pixi
) -> None:
    assert _written(_node({"deps": {"prettier": ">=3"}}, tmp_path, pixi))["name"] == "w-npm"


def test_runtime_and_dev_requirements_land_in_their_own_manifest_tables(
    tmp_path: Path, pixi: Pixi
) -> None:
    node = _node({"deps": {"prettier": ">=3"}, "dev": {"eslint": "^10"}}, tmp_path, pixi)
    body = _written(node)
    assert body["dependencies"] == {"prettier": ">=3"}
    assert body["devDependencies"] == {"eslint": "^10"}
    assert body["private"] is True


def test_declared_package_fields_ride_verbatim_into_the_generated_manifest(
    tmp_path: Path, pixi: Pixi
) -> None:
    node = _node(
        {"deps": {"vite": "*"}, "package": {"type": "module", "engines": {"node": ">=22"}}},
        tmp_path,
        pixi,
    )
    body = _written(node)
    assert body["type"] == "module"
    assert body["engines"] == {"node": ">=22"}


def test_a_package_key_that_is_not_a_table_is_ignored_rather_than_merged(
    tmp_path: Path, pixi: Pixi
) -> None:
    """`package` names the fields table, so a scalar there cannot become manifest fields."""
    node = _node({"deps": {"vite": "*"}, "package": "module"}, tmp_path, pixi)
    assert node.fields == {}


def test_a_table_left_without_dependencies_drops_its_generated_manifest(
    tmp_path: Path, pixi: Pixi
) -> None:
    """A surviving `package.json` would keep reinstalling what the manifest stopped declaring."""
    node = _node({}, tmp_path, pixi)
    node.manifest.write_text('{"name": "w-npm"}\n')
    with GeneratedFiles(directory=node.out).locked() as files:
        node.generate(files)
    assert not node.manifest.exists()


def test_a_source_requirement_npm_would_misread_is_refused_at_compile(
    tmp_path: Path, pixi: Pixi
) -> None:
    node = _node({"deps": {"private-lib": {"git": "https://example.com/lib.git"}}}, tmp_path, pixi)
    with pytest.raises(MissionError, match=r"private-lib.*git"):
        node.compiled()


def test_sync_installs_through_the_declared_manager_from_the_install_directory(
    tmp_path: Path, pixi: Pixi, fp: FakeProcess, stub_binary: Callable[[str], str]
) -> None:
    pnpm = stub_binary("pnpm")
    node = _node({"manager": "pnpm", "deps": {"vite": "*"}}, tmp_path, pixi)
    node.manifest.write_text('{"name": "w-npm"}\n')
    fp.register([fp.any()], stdout="added 1 package\n")

    node.sync()

    assert list(fp.calls[0]) == [pnpm, "install"]


def test_sync_is_a_no_op_until_a_manifest_has_been_generated_to_install_from(
    tmp_path: Path, pixi: Pixi, fp: FakeProcess, stub_binary: Callable[[str], str]
) -> None:
    stub_binary("npm")
    _node({"deps": {"vite": "*"}}, tmp_path, pixi).sync()
    assert not fp.calls
