from typing import TYPE_CHECKING

import pytest

from mainboard.engines.compile.ecosystems import Rust
from mainboard.manifest.schema.spec import Spec
from mainboard.manifest.schema.toolchain import Toolchain

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
    from mainboard.manifest.schema.spec import Json


def _rust(body: dict[str, Json], tmp_path: Path, pixi: Pixi) -> Rust:
    return Rust(
        Toolchain.model_validate(body),
        env="default",
        project="w",
        workspace=tmp_path,
        out=pixi.manifest.parent,
        pixi=pixi,
    )


def _record(rust: Rust, *entries: str) -> None:
    """Write the `.crates.toml` cargo leaves under the install root."""
    rust.prefix.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'"{entry}" = ["binary"]' for entry in entries)
    (rust.prefix / ".crates.toml").write_text(f"[v1]\n{body}\n")


def test_crates_install_into_the_environment_prefix_activation_already_exports(
    tmp_path: Path, pixi: Pixi
) -> None:
    rust = _rust({"deps": {"ripgrep": "*"}}, tmp_path, pixi)
    assert rust.prefix == pixi.env_prefix("default")
    assert rust.binary_dirs() == ()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("*", []),
        (">=14", ["--version", ">=14"]),
    ],
)
def test_a_version_pin_becomes_a_cargo_version_flag(spec: str, expected: list[str]) -> None:
    assert Rust.install_args(Spec.model_validate(spec)) == expected


def test_source_and_locked_extras_become_the_cargo_flags_of_the_same_name() -> None:
    spec = Spec.model_validate(
        {"version": ">=0.1", "git": "https://example.com/c.git", "rev": "abc", "locked": True}
    )
    assert Rust.install_args(spec) == [
        "--version",
        ">=0.1",
        "--git",
        "https://example.com/c.git",
        "--rev",
        "abc",
        "--locked",
    ]


@pytest.mark.parametrize(
    ("constraint", "installed", "satisfied"),
    [
        ("*", "0.1.0", True),
        (">=14", "14.2.0", True),
        (">=14", "13.0.0", False),
        ("^1.2", "1.3.0", True),
        (">=14", "abc", True),
    ],
)
def test_only_a_readable_constraint_can_report_an_installed_crate_as_drifted(
    constraint: str, installed: str, *, satisfied: bool
) -> None:
    """A caret spelling or an unreadable version is trusted rather than reinstalled forever."""
    assert Rust.satisfied(constraint, installed) is satisfied


def test_no_install_record_reads_as_an_empty_environment(tmp_path: Path, pixi: Pixi) -> None:
    assert _rust({"deps": {"ripgrep": "*"}}, tmp_path, pixi).installed() == {}


def test_an_install_record_entry_without_a_version_is_skipped(tmp_path: Path, pixi: Pixi) -> None:
    """A `.crates.toml` key reads `name version (source)`, and a partial one records nothing."""
    rust = _rust({}, tmp_path, pixi)
    _record(rust, "ripgrep 14.1.0 (registry+https://example.com)", "broken")
    assert rust.installed() == {"ripgrep": "14.1.0"}


def test_sync_installs_a_missing_crate_against_the_environment_prefix(
    tmp_path: Path, pixi: Pixi, fp: FakeProcess, tool_paths: dict[str, str]
) -> None:
    rust = _rust({"deps": {"ripgrep": ">=14"}}, tmp_path, pixi)
    fp.register([fp.any()], stdout="installed\n")

    rust.sync()

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
    tmp_path: Path, pixi: Pixi, fp: FakeProcess, tool_paths: dict[str, str]
) -> None:
    rust = _rust({"deps": {"ripgrep": ">=14"}}, tmp_path, pixi)
    _record(rust, "ripgrep 14.1.0 (registry+https://example.com)")

    rust.sync()

    assert not fp.calls


def test_sync_forces_a_reinstall_only_over_a_crate_that_drifted(
    tmp_path: Path, pixi: Pixi, fp: FakeProcess, tool_paths: dict[str, str]
) -> None:
    rust = _rust({"deps": {"ripgrep": ">=14"}}, tmp_path, pixi)
    _record(rust, "ripgrep 13.0.0 (registry+https://example.com)")
    fp.register([fp.any()], stdout="installed\n")

    rust.sync()

    assert list(fp.calls[0])[-4:] == ["--version", ">=14", "--force", "ripgrep"]


def test_sync_uninstalls_a_crate_the_table_no_longer_declares(
    tmp_path: Path, pixi: Pixi, fp: FakeProcess, tool_paths: dict[str, str]
) -> None:
    rust = _rust({}, tmp_path, pixi)
    _record(rust, "ripgrep 14.1.0 (registry+https://example.com)")
    fp.register([fp.any()], stdout="removed\n")

    rust.sync()

    assert list(fp.calls[0])[-4:] == ["uninstall", "--root", str(rust.prefix), "ripgrep"]
