import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from threading import Thread

import pytest
from filelock import FileLock
from plumbum import CommandNotFound
from plumbum.commands.processes import ProcessTimedOut

from mainboard import staleness
from mainboard.core.project import Project
from mainboard.engines.compile.backend.engine import PixiEngine
from mainboard.engines.compile.backend.result import CommandResult
from mainboard.staleness import Refresh, Snapshot, check, digest, tool_root

_RECEIPT = '[tool]\nrequirements = [{ name = "mainboard", extras = ["wandb"], directory = %s }]\n'


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """A uv tool layout beside a source checkout: the receipt, the package, and the tree."""
    source = tmp_path / "checkout"
    (source / "src" / "mainboard").mkdir(parents=True)
    (source / "src" / "mainboard" / "cli.py").write_text("code")
    (source / "pyproject.toml").write_text('[project]\nname = "mainboard"\n', encoding="utf-8")
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


def test_package_metadata_moves_the_snapshot_even_when_runtime_source_does_not(
    snapshot: Path,
) -> None:
    """A dependency edit must reinstall the tool rather than leaving its imports unchanged."""
    check(snapshot)
    metadata = snapshot.parents[2].parent / "checkout" / "pyproject.toml"
    metadata.write_text(
        '[project]\nname = "mainboard"\ndependencies = ["cuda-bindings"]\n',
        encoding="utf-8",
    )

    assert check(snapshot).stale is True


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
        "--reinstall-package",
        "mainboard",
        "--from",
        f"{source}[wandb]",
        "mainboard",
        "--force",
    )
    receipt = tool / "uv-receipt.toml"
    stat = receipt.stat()
    os.utime(receipt, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert check(snapshot).stale is False


def stale(snapshot: Path) -> Snapshot:
    """The fixture's snapshot recorded, then edited past: what every refresh starts from."""
    check(snapshot)
    touched(snapshot.parents[2].parent / "checkout")
    found = check(snapshot)
    assert found.stale
    return found


@pytest.mark.parametrize(
    ("installer", "updated"),
    [
        pytest.param(CommandResult(0, "Installed 1 executable\n", ""), True, id="installed"),
        pytest.param(
            CommandResult(2, "", "resolving\nerror: no wheel for cuda-bindings\n"),
            False,
            id="the-installer-refused",
        ),
        pytest.param(CommandResult(2, "", ""), False, id="the-installer-said-nothing"),
        pytest.param(ProcessTimedOut("uv hung", []), False, id="the-installer-hung"),
        pytest.param(CommandNotFound("pixi", []), False, id="no-pixi-anywhere"),
    ],
)
def test_a_stale_snapshot_reinstalls_itself_quietly_and_reexecutes_the_same_command(
    snapshot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    installer: CommandResult | Exception,
    updated: bool,
) -> None:
    """The nag is gone: the snapshot updates itself and answers on the new code.

    Nothing reaches stdout on any path, since stdout belongs to the verb about to run, and a
    reinstall that fails leaves the command answering from the snapshot it has, with the
    installer's own last word on stderr rather than a silent loop.
    """
    found = stale(snapshot)
    ran: list[tuple[str, ...]] = []
    replaced: list[list[str]] = []

    def install(
        self: PixiEngine, action: Callable[..., CommandResult], *argv: str
    ) -> CommandResult:
        ran.append(argv)
        if isinstance(installer, Exception):
            raise installer
        return installer

    monkeypatch.setattr("mainboard.staleness.platform.system", lambda: "Linux")
    monkeypatch.setattr(PixiEngine, "within_cwd", install)
    monkeypatch.setattr("mainboard.staleness.os.execv", lambda path, argv: replaced.append(argv))
    monkeypatch.setattr("mainboard.staleness.sys.orig_argv", ["python", "mainboard", "jobs"])
    monkeypatch.delenv(staleness.REFRESHED, raising=False)

    Refresh(found).run()

    printed = capfd.readouterr()
    assert printed.out == ""
    assert ran == [found.fix]
    assert (replaced == [[sys.executable, "mainboard", "jobs"]]) is updated
    assert (os.environ.get(staleness.REFRESHED) == "1") is updated
    monkeypatch.delenv(staleness.REFRESHED, raising=False)
    if updated:
        assert f"updated from {found.source}" in printed.err
        return
    assert "could not update itself" in printed.err
    assert printed.err.count("\n") == 1


def test_a_second_process_waits_its_turn_and_only_reexecutes_on_the_install_it_waited_for(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nine jobs starting at once on a synced host reinstall once, and one never waits forever."""
    found = stale(snapshot)
    receipt = snapshot.parents[2] / "uv-receipt.toml"
    stat = receipt.stat()
    os.utime(receipt, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    replaced: list[list[str]] = []
    monkeypatch.setattr("mainboard.staleness.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        PixiEngine, "within_cwd", lambda *args: pytest.fail("the install already happened")
    )
    monkeypatch.setattr("mainboard.staleness.os.execv", lambda path, argv: replaced.append(argv))

    Refresh(found).run()
    monkeypatch.delenv(staleness.REFRESHED, raising=False)
    assert len(replaced) == 1

    monkeypatch.setattr("mainboard.staleness._LOCK_SECONDS", 0.01)
    with FileLock(snapshot.parents[2] / "self-update.lock", thread_local=False):
        worker = Thread(target=Refresh(found).run)
        worker.start()
        worker.join()
    assert len(replaced) == 1
    assert "another update held its lock" in capsys.readouterr().err


def test_windows_hands_the_update_to_one_worker_that_outlives_the_launcher(
    snapshot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A running Windows interpreter cannot be replaced, so the update waits for it to exit.

    The command at hand answers from the snapshot it started on, and every command run before
    the worker has finished finds the pending marker and schedules nothing more, until the
    marker is old enough that its worker can no longer be on its way.
    """
    found = stale(snapshot)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr("mainboard.staleness.platform.system", lambda: "Windows")
    monkeypatch.setattr("mainboard.staleness.os.getpid", lambda: 314)
    monkeypatch.setattr("mainboard.staleness.PixiEngine.defer", lambda self, *a: calls.append(a))
    assert found.source is not None
    log = found.source / ".mainboard" / "self-update.log"

    Refresh(found).run()
    Refresh(found).run()

    assert calls == [
        (
            "exec",
            "--spec",
            "uv=0.12.7",
            "--spec",
            "python=3.14",
            "--spec",
            "psutil=7.2.2",
            "--spec",
            "cyclopts=4.23",
            "python",
            str(Path(staleness.__file__).with_name("_refresh.py")),
            "314",
            str(log),
            "--",
            *found.uv,
        )
    ]
    pending = log.with_suffix(".pending")
    assert pending.read_text(encoding="utf-8") == "314"
    assert capsys.readouterr().err.count("updates itself once this command exits") == 1
    os.utime(pending, (0, 0))
    Refresh(found).run()
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("moved", "again", "refreshed", "said"),
    [
        pytest.param(False, False, False, "", id="a-fresh-snapshot-does-nothing"),
        pytest.param(True, False, True, "", id="a-stale-snapshot-refreshes"),
        pytest.param(True, True, False, "the update did not take", id="a-reexecution-never-loops"),
    ],
)
def test_every_invocation_starts_on_its_sources_newest_code(
    snapshot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    moved: bool,
    again: bool,
    refreshed: bool,
    said: str,
) -> None:
    """The entry check, and the one environment flag that keeps a failed update from looping."""
    found = stale(snapshot) if moved else check(snapshot)
    monkeypatch.setattr("mainboard.staleness.check", lambda: found)
    ran: list[Snapshot] = []
    monkeypatch.setattr(Refresh, "run", lambda self: ran.append(self.found))
    if again:
        monkeypatch.setenv(staleness.REFRESHED, "1")

    staleness.current()

    assert bool(ran) is refreshed
    assert staleness.REFRESHED not in os.environ
    printed = capsys.readouterr()
    assert printed.out == ""
    assert said in printed.err


def test_the_refresh_preserves_an_existing_durable_interpreter(snapshot: Path) -> None:
    """A uv-managed interpreter remains the exact self-update contract."""
    tool = snapshot.parents[2]
    source = tool.parent / "checkout"
    interpreter = tool.parent / "uv" / "python" / "cpython-3.14.7" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("python")
    (tool / "uv-receipt.toml").write_text(
        "[tool]\n"
        f"python = {json.dumps(str(interpreter))}\n"
        f'requirements = [{{ name = "mainboard", directory = {json.dumps(str(source))} }}]\n',
        encoding="utf-8",
    )
    check(snapshot)
    touched(source)

    assert check(snapshot).fix[4:10] == (
        "tool",
        "install",
        "--reinstall-package",
        "mainboard",
        "--python",
        str(interpreter),
    )


def test_the_refresh_does_not_retain_a_project_environment_interpreter(snapshot: Path) -> None:
    """Replaceable generated state never becomes the public launcher's Python home."""
    tool = snapshot.parents[2]
    source = tool.parent / "checkout"
    interpreter = source / ".mainboard" / "envs" / "tool" / ".pixi" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("python")
    (tool / "uv-receipt.toml").write_text(
        "[tool]\n"
        f"python = {json.dumps(str(interpreter))}\n"
        f'requirements = [{{ name = "mainboard", directory = {json.dumps(str(source))} }}]\n',
        encoding="utf-8",
    )
    check(snapshot)
    touched(source)

    fix = check(snapshot).fix
    assert "--python" not in fix
    assert str(interpreter) not in fix


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


def test_refresh_advice_needs_no_git(snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    check(snapshot)
    touched(snapshot.parents[2].parent / "checkout")
    found = check(snapshot)
    assert found.stale
    assert "git" not in " ".join(found.fix)


def test_a_refresh_without_a_source_logs_beside_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A snapshot that names no source still gets a durable log, next to where it ran."""
    monkeypatch.chdir(tmp_path)
    assert staleness._refresh_log(None) == tmp_path / Project().out_dir / "self-update.log"


def test_a_reinstall_names_its_source_absolutely_and_the_worker_reads_it_off_the_snapshot(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One reading of the receipt answers both callers, and a bracket in a path survives it.

    The deferred worker used to recover its argv by slicing four tokens off the pixi command and
    its log directory by cutting the `--from` token at the first `[`, which is the extras
    separator and also an ordinary character in a directory name. Both now come off the snapshot
    that read the receipt.
    """
    tool = snapshot.parents[2]
    source = tool.parent / "check[out]"
    (source / "src" / "mainboard").mkdir(parents=True)
    (source / "src" / "mainboard" / "cli.py").write_text("code", encoding="utf-8")
    (tool / "uv-receipt.toml").write_text(_RECEIPT % json.dumps(str(source)), encoding="utf-8")
    check(snapshot)
    touched(source)

    found = check(snapshot)

    assert found.stale is True
    assert found.source == source
    assert found.uv[0] == "uv" and found.uv[-1] == "--force"
    assert f"{source}[wandb]" in found.uv
    assert found.fix == ("exec", "--spec", staleness._UV, *found.uv)
    assert staleness._refresh_log(found.source) == source / Project().out_dir / "self-update.log"

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr("mainboard.staleness.platform.system", lambda: "Windows")
    monkeypatch.setattr("mainboard.staleness.os.getpid", lambda: 7)
    monkeypatch.setattr(
        "mainboard.staleness.PixiEngine.defer", lambda self, *args: calls.append(args)
    )

    Refresh(found).run()

    handed = calls[0]
    assert handed[handed.index("--") + 1 :] == found.uv
    assert str(source / Project().out_dir / "self-update.log") in handed
