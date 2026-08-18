from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.ecosystems import Ecosystem, Go, Node, Rust, SecondStage
from mainboard.engines.compile.generated import GeneratedFiles

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
    from mainboard.manifest import Manifest


def _stage(text: str, make: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi) -> SecondStage:
    return SecondStage(tmp_path, make(text), pixi.manifest.parent, pixi)


def test_every_registered_ecosystem_claims_its_own_manifest_table() -> None:
    assert {implementation.toolchain for implementation in Ecosystem.implementations()} == {
        "nodejs",
        "rust",
        "go",
    }


def test_the_base_table_reaches_the_ecosystem_that_owns_it(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    stage = _stage(
        '[workspace]\nname = "w"\n[nodejs.deps]\nprettier = ">=3"\n', manifest_from, tmp_path, pixi
    )
    assert set(stage.toolchains("default")) == {"nodejs"}
    node = next(eco for eco in stage.ecosystems("default") if isinstance(eco, Node))
    assert set(node.deps) == {"prettier"}


def test_a_table_pixi_compiles_itself_reaches_no_ecosystem(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """`[python]` becomes pypi dependencies in the generated pixi manifest, never a stage."""
    stage = _stage(
        '[workspace]\nname = "w"\n[python.deps]\ntorch = "*"\n', manifest_from, tmp_path, pixi
    )
    assert set(stage.toolchains("default")) == {"python"}
    assert all(not eco.deps for eco in stage.ecosystems("default"))


def test_every_ecosystem_is_bound_even_when_the_manifest_declares_none(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """A deleted table still needs its ecosystem, since cleaning up after it is its job."""
    stage = _stage('[workspace]\nname = "w"\n', manifest_from, tmp_path, pixi)
    assert [type(eco) for eco in stage.ecosystems("default")] == [Go, Node, Rust]


def test_the_dev_scope_joins_the_default_environment_only(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    stage = _stage(
        """
        [workspace]
        name = "w"
        [dev.rust.deps]
        bookokrat = ">=0.1"
        [envs.serving]
        """,
        manifest_from,
        tmp_path,
        pixi,
    )
    assert set(stage.toolchains("default")["rust"].deps) == {"bookokrat"}
    assert "rust" not in stage.toolchains("serving")


def test_a_platform_overlay_merges_over_the_base_table_for_this_machine(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    pixi: Pixi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    stage = _stage(
        """
        [workspace]
        name = "w"
        [rust.deps]
        ripgrep = ">=13"
        [on.linux-64.rust.deps]
        ripgrep = ">=14"
        bookokrat = ">=0.1"
        [on.osx.rust.deps]
        maclike = ">=1"
        """,
        manifest_from,
        tmp_path,
        pixi,
    )
    rust = stage.toolchains("default")["rust"]
    assert rust.deps["ripgrep"].version == ">=14"
    assert set(rust.deps) == {"ripgrep", "bookokrat"}


def test_a_named_environment_overrides_the_base_table_it_inherits(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    stage = _stage(
        """
        [workspace]
        name = "w"
        [nodejs]
        manager = "npm"
        [nodejs.deps]
        prettier = ">=3"
        [envs.web.nodejs]
        manager = "pnpm"
        [envs.web.nodejs.deps]
        vite = ">=5"
        """,
        manifest_from,
        tmp_path,
        pixi,
    )
    web = stage.toolchains("web")["nodejs"]
    assert set(web.all_deps()) == {"prettier", "vite"}
    assert (web.model_extra or {})["manager"] == "pnpm"


def test_an_isolated_environment_starts_from_nothing_but_itself(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    stage = _stage(
        """
        [workspace]
        name = "w"
        [nodejs.deps]
        prettier = ">=3"
        [envs.web]
        no-default = true
        [envs.web.nodejs.deps]
        vite = ">=5"
        """,
        manifest_from,
        tmp_path,
        pixi,
    )
    assert set(stage.toolchains("web")["nodejs"].all_deps()) == {"vite"}


def test_an_undeclared_environment_is_refused_with_the_declared_roster(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    stage = _stage('[workspace]\nname = "w"\n', manifest_from, tmp_path, pixi)
    with pytest.raises(MissionError, match="no environment 'web'"):
        stage.toolchains("web")


def test_generate_writes_what_the_declared_toolchains_install_from(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    stage = _stage(
        '[workspace]\nname = "w"\n[nodejs.deps]\nprettier = ">=3"\n', manifest_from, tmp_path, pixi
    )
    with GeneratedFiles(directory=stage.out).locked() as files:
        stage.generate(files, "default")
    assert "prettier" in (stage.out / "package.json").read_text()


def test_a_workspace_wide_toolchain_reads_every_environments_tables(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """One `package.json` serves every env, so provisioning one env may not narrow it."""
    stage = _stage(
        """
        [workspace]
        name = "w"
        [dev.nodejs.deps]
        prettier = ">=3"
        [envs.serving.nodejs.deps]
        vite = ">=5"
        [envs.serving.rust.deps]
        ripgrep = ">=14"
        """,
        manifest_from,
        tmp_path,
        pixi,
    )
    node = next(eco for eco in stage.ecosystems("serving") if isinstance(eco, Node))
    rust = next(eco for eco in stage.ecosystems("serving") if isinstance(eco, Rust))

    assert set(node.deps) == {"prettier", "vite"}
    assert set(rust.deps) == {"ripgrep"}
    assert set(next(eco for eco in stage.ecosystems("default") if isinstance(eco, Rust)).deps) == (
        set()
    )


def test_provisioning_an_environment_never_drops_another_ones_generated_manifest(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """The bug this closes deleted `package.json` and orphaned the node_modules beside it."""
    stage = _stage(
        '[workspace]\nname = "w"\n[dev.nodejs.deps]\nprettier = ">=3"\n[envs.serving]\n',
        manifest_from,
        tmp_path,
        pixi,
    )
    with GeneratedFiles(directory=stage.out).locked() as files:
        stage.generate(files, "default")
        stage.generate(files, "serving")

    assert "prettier" in (stage.out / "package.json").read_text()


def test_binary_dirs_gathers_every_directory_the_toolchains_link_into(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    stage = _stage('[workspace]\nname = "w"\n', manifest_from, tmp_path, pixi)
    assert stage.binary_dirs("default") == [
        stage.out / "go" / "bin",
        stage.out / "node_modules" / ".bin",
    ]


def test_install_runs_every_toolchains_installer_inside_the_provisioned_environment(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    pixi: Pixi,
    fp: FakeProcess,
    tool_paths: dict[str, str],
    stub_binary: Callable[[str], str],
) -> None:
    npm = stub_binary("npm")
    stage = _stage(
        """
        [workspace]
        name = "w"
        [nodejs.deps]
        prettier = ">=3"
        [rust.deps]
        ripgrep = ">=14"
        """,
        manifest_from,
        tmp_path,
        pixi,
    )
    with GeneratedFiles(directory=stage.out).locked() as files:
        stage.generate(files, "default")
    for _ in range(2):
        fp.register([fp.any()], stdout="done\n")

    stage.install("default")

    assert [next(iter(call)) for call in fp.calls] == [npm, tool_paths["pixi"]]
