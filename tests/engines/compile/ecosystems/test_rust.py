from typing import TYPE_CHECKING

import pytest

from mainboard.engines.compile.ecosystems import Rust
from mainboard.manifest.schema.spec import Json, Spec

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi

    from ..conftest import Bind

_REGISTRY = "(registry+https://example.com)"


def _record(rust: Rust, *entries: str) -> None:
    """Write the `.crates.toml` cargo leaves under the install root."""
    rust.prefix.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'"{entry}" = ["binary"]' for entry in entries)
    (rust.prefix / ".crates.toml").write_text(f"[v1]\n{body}\n")


@pytest.mark.parametrize(
    ("declared", "args"),
    [
        pytest.param("*", [], id="an-unconstrained-requirement-pins-nothing"),
        pytest.param(">=14", ["--version", ">=14"], id="a-version-pin-becomes-a-version-flag"),
        pytest.param(
            {"version": ">=0.1", "git": "https://example.com/c.git", "rev": "abc", "locked": True},
            [
                "--version",
                ">=0.1",
                "--git",
                "https://example.com/c.git",
                "--rev",
                "abc",
                "--locked",
            ],
            id="every-source-extra-becomes-the-cargo-flag-of-the-same-name",
        ),
    ],
)
def test_a_declared_crate_becomes_the_cargo_flags_that_express_it(
    declared: Json, args: list[str]
) -> None:
    assert Rust.install_args(Spec.model_validate(declared)) == args


@pytest.mark.parametrize(
    ("constraint", "installed", "satisfied"),
    [
        pytest.param("*", "0.1.0", True, id="an-unconstrained-requirement-accepts-anything"),
        pytest.param(">=14", "14.2.0", True, id="a-version-inside-the-constraint"),
        pytest.param(">=14", "13.0.0", False, id="a-version-that-drifted-below-it"),
        pytest.param("^1.2", "1.3.0", True, id="a-caret-spelling-packaging-cannot-read"),
        pytest.param(">=14", "abc", True, id="a-recorded-version-answering-to-no-constraint"),
    ],
)
def test_only_a_readable_constraint_can_report_an_installed_crate_as_drifted(
    constraint: str, installed: str, *, satisfied: bool
) -> None:
    """Trusting cargo's own record beats reinstalling on every single sync."""
    assert Rust.satisfied(constraint, installed) is satisfied


@pytest.mark.parametrize(
    ("entries", "installed"),
    [
        pytest.param((), {}, id="no-install-record-reads-as-an-empty-environment"),
        pytest.param(
            (f"ripgrep 14.1.0 {_REGISTRY}", "broken"),
            {"ripgrep": "14.1.0"},
            id="an-entry-without-a-version-is-no-usable-record",
        ),
    ],
)
def test_what_cargo_recorded_under_the_prefix_is_read_back_by_name_and_version(
    entries: tuple[str, ...], installed: dict[str, str], bind: Bind
) -> None:
    """A `.crates.toml` key reads `name version (source)`, and a partial one records nothing."""
    rust = bind(Rust, {})
    if entries:
        _record(rust, *entries)
    assert rust.installed() == installed


def test_sync_installs_a_missing_crate_against_the_environment_prefix(
    bind: Bind, pixi: Pixi, fp: FakeProcess, tool_paths: dict[str, str]
) -> None:
    """Crates share the environment's own `bin/`, so activation exports them with everything
    else and no extra directory of this toolchain's own ever reaches PATH."""
    rust = bind(Rust, {"deps": {"ripgrep": ">=14"}})
    fp.register([fp.any()], stdout="installed\n")

    rust.sync()

    assert rust.prefix == pixi.env_prefix("default")
    assert rust.binary_dirs() == ()
    assert list(fp.calls[0]) == [
        tool_paths["pixi"],
        "run",
        "--manifest-path",
        str(pixi.manifest),
        "--environment",
        "default",
        "cargo",
        "install",
        "--root",
        str(rust.prefix),
        "--version",
        ">=14",
        "ripgrep",
    ]


def test_sync_leaves_a_crate_that_already_satisfies_its_constraint_alone(
    bind: Bind, fp: FakeProcess
) -> None:
    rust = bind(Rust, {"deps": {"ripgrep": ">=14"}})
    _record(rust, f"ripgrep 14.1.0 {_REGISTRY}")

    rust.sync()

    assert not fp.calls


def test_sync_uninstalls_what_was_dropped_and_forces_a_reinstall_over_what_drifted(
    bind: Bind, fp: FakeProcess
) -> None:
    rust = bind(Rust, {"deps": {"ripgrep": ">=14"}})
    _record(rust, f"ripgrep 13.0.0 {_REGISTRY}", f"orphan 1.0 {_REGISTRY}")
    for _ in range(2):
        fp.register([fp.any()], stdout="done\n")

    rust.sync()

    assert list(fp.calls[0])[-4:] == ["uninstall", "--root", str(rust.prefix), "orphan"]
    assert list(fp.calls[1])[-4:] == ["--version", ">=14", "--force", "ripgrep"]
