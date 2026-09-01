import json
import os
from typing import TYPE_CHECKING

import pytest

from mainboard import staleness
from mainboard.staleness import Snapshot, check, digest, tool_root

if TYPE_CHECKING:
    from pathlib import Path

_RECEIPT = '[tool]\nrequirements = [{ name = "mainboard", extras = ["wandb"], directory = %s }]\n'


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """A uv tool layout beside a source checkout: the receipt, the package, and the tree."""
    source = tmp_path / "checkout"
    (source / "src" / "mainboard").mkdir(parents=True)
    (source / "src" / "mainboard" / "cli.py").write_text("code")
    (source / "src" / "mainboard" / "__pycache__").mkdir()
    (source / "src" / "mainboard" / "__pycache__" / "cli.pyc").write_text("bytecode")
    tool = tmp_path / "tool"
    package = tool / "lib" / "site-packages" / "mainboard"
    package.mkdir(parents=True)
    (tool / "uv-receipt.toml").write_text(_RECEIPT % json.dumps(str(source)), encoding="utf-8")
    return package


def touched(source: Path) -> None:
    """Move one source file's clock forward, the edit the whole check exists to catch."""
    edited = source / "src" / "mainboard" / "cli.py"
    stat = edited.stat()
    os.utime(edited, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


def test_the_check_records_on_first_run_then_names_the_reinstall_when_the_tree_moves(
    snapshot: Path,
) -> None:
    """The whole lifecycle: record, agree, drift, warn, reinstall, record again.

    The first run after an install is the baseline, an unchanged tree keeps agreeing with it,
    an edit flips the answer to stale with the receipt's own extras in the named command, and
    a reinstall (a rewritten receipt) invalidates the old baseline so the fresh snapshot
    records the edited tree as its own.
    """
    tool = snapshot.parents[2]
    first = check(snapshot)
    assert first == Snapshot(installed=True, detail="snapshot matches the source tree")
    assert first.warning == ""
    assert check(snapshot).stale is False
    source = tool.parent / "checkout"
    touched(source)
    stale = check(snapshot)
    assert stale.stale is True
    assert str(source) in stale.detail
    assert stale.fix == (
        "exec",
        "--spec",
        "uv=0.12.7",
        "uv",
        "tool",
        "install",
        "--from",
        f"{source}[wandb]",
        "mainboard",
        "--force",
    )
    assert stale.detail in stale.warning
    assert "mainboard self-update" in stale.warning
    receipt = tool / "uv-receipt.toml"
    stat = receipt.stat()
    os.utime(receipt, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert check(snapshot).stale is False


@pytest.mark.parametrize("code", [0, 1])
def test_refresh_runs_uv_inside_pixi_and_answers_its_exit_code(
    code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refresh delegates its exact argv to Pixi, never to a host-level uv binary."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "mainboard.staleness.PixiEngine.exit_code",
        lambda self, *args: calls.append(args) or code,
    )
    fix = ("exec", "--spec", "uv=0.12.7", "uv", "tool", "install", "mainboard")
    found = Snapshot(installed=True, stale=True, detail="drifted", fix=fix)

    assert staleness.refresh(found) == code
    assert calls == [fix]


def test_the_refresh_preserves_the_receipts_exact_python(snapshot: Path) -> None:
    """A tool refresh reuses its recorded interpreter instead of rediscovering one."""
    tool = snapshot.parents[2]
    source = tool.parent / "checkout"
    interpreter = source / "python.exe"
    (tool / "uv-receipt.toml").write_text(
        "[tool]\n"
        f"python = {json.dumps(str(interpreter))}\n"
        f'requirements = [{{ name = "mainboard", directory = {json.dumps(str(source))} }}]\n',
        encoding="utf-8",
    )
    check(snapshot)
    touched(source)

    assert check(snapshot).fix[4:7] == ("tool", "install", "--python")
    assert check(snapshot).fix[7] == str(interpreter)


def test_refresh_does_nothing_for_a_snapshot_with_no_fix_to_run() -> None:
    """A fresh or uninstalled snapshot has an empty `fix`, so there is nothing to run at all."""
    assert staleness.refresh(Snapshot(installed=True)) == 0


def test_a_checkout_running_its_own_source_has_nothing_to_be_stale_against(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "mainboard"
    package.mkdir(parents=True)
    assert check(package) == Snapshot(installed=False, detail="running from source")
    assert tool_root(package) is None


@pytest.mark.parametrize(
    ("receipt", "said"),
    [
        ("not toml at all [", "names no source directory"),
        ('[tool]\nrequirements = [{ name = "mainboard" }]\n', "names no source directory"),
        ('[tool]\nrequirements = [{ name = "other", directory = "x" }]\n', "names no source"),
    ],
    ids=["torn receipt", "no directory", "another tool's requirement"],
)
def test_a_receipt_that_cannot_vouch_for_a_source_is_installed_but_never_stale(
    snapshot: Path, receipt: str, said: str
) -> None:
    """A snapshot uv cannot explain warns nobody, since there is no tree to compare against."""
    (snapshot.parents[2] / "uv-receipt.toml").write_text(receipt)
    found = check(snapshot)
    assert found == Snapshot(installed=True, detail=found.detail)
    assert said in found.detail


def test_a_receipt_whose_directory_lost_its_source_tree_says_where_it_looked(
    snapshot: Path, tmp_path: Path
) -> None:
    (snapshot.parents[2] / "uv-receipt.toml").write_text(
        _RECEIPT % json.dumps(str(tmp_path / "gone")), encoding="utf-8"
    )
    found = check(snapshot)
    assert found.stale is False
    assert "no source tree at" in found.detail


def test_a_receipt_without_extras_still_names_the_wandb_extra(snapshot: Path) -> None:
    """The extra is load-bearing, so the named command never drops it."""
    source = snapshot.parents[2].parent / "checkout"
    (snapshot.parents[2] / "uv-receipt.toml").write_text(
        "[tool]\n"
        f'requirements = [{{ name = "mainboard", directory = {json.dumps(str(source))} }}]\n',
        encoding="utf-8",
    )
    check(snapshot)
    touched(source)
    assert any("[wandb]" in argument for argument in check(snapshot).fix)


def test_a_torn_state_file_and_an_unwritable_tool_directory_both_answer_fresh(
    snapshot: Path,
) -> None:
    """The check never fails the command that asked, whatever the state file's condition."""
    tool = snapshot.parents[2]
    (tool / "source-state.json").write_text("torn {")
    assert check(snapshot).stale is False
    held = json.loads((tool / "source-state.json").read_text())
    assert set(held) == {"marker", "digest"}
    (tool / "source-state.json").unlink()
    tool.chmod(0o555)
    try:
        assert check(snapshot).stale is False
    finally:
        tool.chmod(0o755)


def test_the_digest_reads_names_sizes_and_clocks_and_never_bytecode(snapshot: Path) -> None:
    """Content is unread on purpose, so the check stays in CLI-startup budget."""
    source = snapshot.parents[2].parent / "checkout" / "src"
    before = digest(source)
    assert digest(source) == before
    (source / "mainboard" / "__pycache__" / "extra.pyc").write_text("more")
    assert digest(source) == before
    touched(source.parent)
    assert digest(source) != before


def test_the_stale_state_survives_a_repeat_ask_without_rerecording(snapshot: Path) -> None:
    """A stale answer stays stale until a reinstall, never healed by asking twice."""
    check(snapshot)
    touched(snapshot.parents[2].parent / "checkout")
    assert staleness.check(snapshot).stale is True
    assert staleness.check(snapshot).stale is True
