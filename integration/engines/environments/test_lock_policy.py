from collections.abc import Sequence
from pathlib import Path

import pytest

from mainboard import Board, MissionError
from mainboard.deps import Dependencies, Edit, Tables
from mainboard.engines.compile import LockPolicy, Provisioner
from mainboard.engines.compile.pixi.policy import Preparation

_MANIFEST = '[workspace]\nname = "it"\n\n[deps]\npython = ">=3.14"\n'


@pytest.fixture
def python_only(workspace: Path) -> Path:
    """A workspace declaring Python alone, the smallest environment that solves."""
    (workspace / "mb.toml").write_text(_MANIFEST, encoding="utf-8")
    return workspace


@pytest.fixture
def board(python_only: Path) -> Board:
    return Board(python_only)


def test_install_solves_a_stale_lock_and_locked_refuses_it(mb, python_only: Path) -> None:
    assert mb("install").code == 0
    assert (python_only / "mb.lock").is_file()
    assert mb("lock", "--check").code == 0

    assert mb("add", "--frozen", "--lang", "python", "six>=1.15").code == 0
    refused = mb("install", "--locked")
    assert refused.code == 1 and "was not solved from the manifest" in refused.said

    assert mb("lock", "--check").code == 1
    assert "six" in (python_only / "mb.lock").read_text(encoding="utf-8")
    assert mb("lock", "--check").code == 0
    assert mb("install", "--frozen").code == 0


def test_as_is_runs_what_is_installed_without_solving(mb, python_only: Path) -> None:
    assert mb("install").code == 0
    lock = (python_only / "mb.lock").read_bytes()
    (python_only / "mb.toml").write_text(_MANIFEST + '\n[python.deps]\nsix = ">=1.15"\n')

    ran = mb("run", "--as-is", "python", "-c", "print('ran as it is')")

    assert ran.code == 0 and "ran as it is" in ran.out
    assert (python_only / "mb.lock").read_bytes() == lock


def test_a_failed_solve_leaves_the_manifest_and_lock_as_they_were(mb, python_only: Path) -> None:
    assert mb("install").code == 0
    kept = {name: (python_only / name).read_bytes() for name in ("mb.toml", "mb.lock")}

    refused = mb("add", "--no-install", "--lang", "python", "six>=999")

    assert refused.code == 1
    assert {name: (python_only / name).read_bytes() for name in kept} == kept
    assert mb("lock", "--check").code == 0


def test_upgrade_leaves_a_member_named_by_version(mb, python_only: Path) -> None:
    (python_only / "member").mkdir()
    (python_only / "member/pyproject.toml").write_text(
        '[project]\nname = "member"\nversion = "0.1"\n', encoding="utf-8"
    )
    member = '[python.deps]\nmember = "*"\n'
    manifest = _MANIFEST.replace("\n\n", '\nmembers = ["member"]\n\n') + member
    (python_only / "mb.toml").write_text(manifest, encoding="utf-8")

    assert mb("upgrade", "member", "--no-install").code == 0
    assert (python_only / "mb.toml").read_text(encoding="utf-8") == manifest


@pytest.mark.parametrize(
    ("flags", "policy", "flag", "installs"),
    [
        pytest.param({}, LockPolicy.UPDATE, "", True, id="no-flag-solves-what-moved"),
        pytest.param({"locked": True}, LockPolicy.LOCKED, "--locked", True, id="locked"),
        pytest.param({"frozen": True}, LockPolicy.FROZEN, "--frozen", True, id="frozen"),
        pytest.param({"as_is": True}, LockPolicy.FROZEN, "--frozen", False, id="as-is"),
        pytest.param({"no_install": True}, LockPolicy.UPDATE, "", False, id="no-install"),
    ],
)
def test_the_flags_of_a_verb_name_the_lock_policy_pixi_spells(
    flags: dict[str, bool], policy: LockPolicy, flag: str, installs: bool
) -> None:
    preparation = Preparation(**flags)

    assert (preparation.policy, preparation.policy.flag, preparation.installs) == (
        policy,
        flag,
        installs,
    )


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param({"locked": True, "frozen": True}, id="locked-and-frozen"),
        pytest.param({"locked": True, "as_is": True}, id="locked-and-as-is"),
    ],
)
def test_locked_and_frozen_exclude_each_other(flags: dict[str, bool]) -> None:
    with pytest.raises(MissionError, match="--locked and --frozen exclude each other"):
        _ = Preparation(**flags).policy


def _break_provisioning(
    monkeypatch: pytest.MonkeyPatch, *, lock: Path, error: type[BaseException], message: str
) -> None:
    """Make the solve write a half lock and fail, as a full disk or a refused solve does."""

    def provision(
        self: Provisioner,
        env: str = "default",
        *,
        policy: LockPolicy = LockPolicy.UPDATE,
        update: Sequence[str] | None = None,
        install: bool = True,
    ) -> None:
        lock.write_text("version: 6\n", encoding="utf-8")
        raise error(message)

    monkeypatch.setattr(Provisioner, "provision", provision)


@pytest.mark.parametrize(
    ("error", "message", "lock_before"),
    [
        pytest.param(OSError, "No space left on device", None, id="a-full-disk-leaves-no-lock"),
        pytest.param(
            OSError, "No space left on device", "version: 5\n", id="a-full-disk-keeps-it"
        ),
        pytest.param(MissionError, "pixi lock failed", "version: 5\n", id="a-refused-solve"),
        pytest.param(KeyboardInterrupt, "interrupted", None, id="an-interrupt"),
    ],
)
def test_any_failed_edit_puts_back_the_manifest_and_the_lock_it_had_not(
    board: Board,
    monkeypatch: pytest.MonkeyPatch,
    error: type[BaseException],
    message: str,
    lock_before: str | None,
) -> None:
    lock = board.project.lock(board.root)
    if lock_before is not None:
        lock.write_text(lock_before, encoding="utf-8")
    _break_provisioning(monkeypatch, lock=lock, error=error, message=message)

    with pytest.raises(error, match=message):
        Dependencies(board).add("six>=1.15", edit=Edit(tables=Tables(ecosystem="python")))

    assert (board.root / "mb.toml").read_text(encoding="utf-8") == _MANIFEST
    assert (lock.read_text(encoding="utf-8") if lock.exists() else None) == lock_before
