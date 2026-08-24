from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile import Ecosystem, SecondStage
from mainboard.engines.compile.ecosystems import Go, Node, Rust

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.generated import Writer

_HEADER = '[workspace]\nname = "w"\n'
_PRETTIER = '[nodejs.deps]\nprettier = ">=3"\n'


def test_every_ecosystem_is_bound_even_when_the_manifest_declares_none(
    stage_from: Callable[[str], SecondStage],
) -> None:
    """A deleted table still needs its ecosystem, since cleaning up after it is its job."""
    assert {implementation.toolchain for implementation in Ecosystem.implementations()} == {
        "nodejs",
        "rust",
        "go",
    }
    assert [type(eco) for eco in stage_from(_HEADER).ecosystems("default")] == [Go, Node, Rust]


def test_a_table_reaches_the_ecosystem_that_owns_it_and_no_other(
    stage_from: Callable[[str], SecondStage],
) -> None:
    """`[python]` becomes pypi dependencies in the generated pixi manifest, never a stage."""
    stage = stage_from(f'{_HEADER}{_PRETTIER}[python.deps]\ntorch = "*"\n')
    assert set(stage.toolchains("default")) == {"nodejs", "python"}
    ecosystems = stage.ecosystems("default")
    assert {type(eco).__name__: set(eco.deps) for eco in ecosystems} == {
        "Go": set(),
        "Node": {"prettier"},
        "Rust": set(),
    }


@pytest.mark.parametrize(
    ("declared", "env", "resolved"),
    [
        pytest.param(
            '[dev.rust.deps]\nbookokrat = ">=0.1"\n[envs.serving]\n',
            "default",
            {"rust.bookokrat": ">=0.1"},
            id="the-dev-scope-joins-the-default-environment",
        ),
        pytest.param(
            '[dev.rust.deps]\nbookokrat = ">=0.1"\n[envs.serving]\n',
            "serving",
            {},
            id="and-no-named-environment-beside-it",
        ),
        pytest.param(
            """[rust.deps]
ripgrep = ">=13"
[on.linux-64.rust.deps]
ripgrep = ">=14"
bookokrat = ">=0.1"
[on.osx.rust.deps]
maclike = ">=1"
""",
            "default",
            {"rust.ripgrep": ">=14", "rust.bookokrat": ">=0.1"},
            id="a-platform-overlay-merges-over-the-base-table-for-this-machine",
        ),
        pytest.param(
            f'{_PRETTIER}[envs.web.nodejs.deps]\nprettier = ">=4"\nvite = ">=5"\n',
            "web",
            {"nodejs.prettier": ">=4", "nodejs.vite": ">=5"},
            id="a-named-environment-overrides-the-base-table-it-inherits",
        ),
        pytest.param(
            f'{_PRETTIER}[envs.web]\nno-default = true\n[envs.web.nodejs.deps]\nvite = ">=5"\n',
            "web",
            {"nodejs.vite": ">=5"},
            id="an-isolated-environment-starts-from-nothing-but-itself",
        ),
    ],
)
def test_a_toolchain_table_is_merged_over_every_scope_that_applies(
    declared: str,
    env: str,
    resolved: dict[str, str],
    stage_from: Callable[[str], SecondStage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Least specific scope first, so a later table overrides an earlier one key by key."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    stage = stage_from(f"{_HEADER}{declared}")
    assert {
        f"{name}.{dep}": spec.version
        for name, chain in stage.toolchains(env).items()
        for dep, spec in chain.all_deps().items()
    } == resolved


def test_an_undeclared_environment_is_refused_with_the_declared_roster(
    stage_from: Callable[[str], SecondStage],
) -> None:
    with pytest.raises(MissionError, match="no environment 'web'"):
        stage_from(_HEADER).toolchains("web")


def test_a_workspace_wide_toolchain_reads_every_environments_tables(
    stage_from: Callable[[str], SecondStage],
) -> None:
    """Each toolchain sees exactly the scope it installs into.

    One `package.json` serves every env, so provisioning one env may not narrow it, while a
    toolchain installing into the pixi prefix sees only the environment being provisioned.
    """
    stage = stage_from(
        f"{_HEADER}"
        '[dev.nodejs.deps]\nprettier = ">=3"\n'
        '[envs.serving.nodejs.deps]\nvite = ">=5"\n'
        '[envs.serving.rust.deps]\nripgrep = ">=14"\n'
    )
    serving = {type(eco).__name__: set(eco.deps) for eco in stage.ecosystems("serving")}
    default = {type(eco).__name__: set(eco.deps) for eco in stage.ecosystems("default")}

    assert serving["Node"] == {"prettier", "vite"}
    assert serving["Rust"] == {"ripgrep"}
    assert default["Rust"] == set()


def test_provisioning_an_environment_never_drops_another_ones_generated_manifest(
    stage_from: Callable[[str], SecondStage], files: Writer
) -> None:
    """The bug this closes deleted `package.json` and orphaned the node_modules beside it."""
    stage = stage_from(f'{_HEADER}[dev.nodejs.deps]\nprettier = ">=3"\n[envs.serving]\n')

    stage.generate(files, "default")
    stage.generate(files, "serving")

    assert "prettier" in (stage.out / "package.json").read_text()


def test_binary_dirs_gathers_every_directory_the_toolchains_link_into(
    stage_from: Callable[[str], SecondStage],
) -> None:
    stage = stage_from(_HEADER)
    assert stage.binary_dirs("default") == [
        stage.out / "go" / "bin",
        stage.out / "node_modules" / ".bin",
    ]


def test_install_runs_every_toolchains_installer_inside_the_provisioned_environment(
    stage_from: Callable[[str], SecondStage],
    files: Writer,
    fp: FakeProcess,
    tool_paths: Mapping[str, str],
    stub_binary: Callable[[str], str],
) -> None:
    """Every manager ships as a conda package pixi has just installed."""
    npm = stub_binary("npm")
    stage = stage_from(f'{_HEADER}{_PRETTIER}[rust.deps]\nripgrep = ">=14"\n')
    stage.generate(files, "default")
    for _ in range(2):
        fp.register([fp.any()], stdout="done\n")

    stage.install("default")

    assert [next(iter(call)) for call in fp.calls] == [npm, tool_paths["pixi"]]
