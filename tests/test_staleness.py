import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
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


def bump(path: Path) -> None:
    """Move `path`'s clock forward: an edit to a source file, a reinstall to a receipt."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


@dataclass
class Layout:
    """A uv tool layout beside a source checkout: the receipt, the package, and the tree."""

    package: Path
    tool: Path
    source: Path

    def declare(self, requirement: str = 'extras = ["wandb"], ', *, head: str = "") -> None:
        """Write the receipt naming `source`, with `requirement` fields and `head` lines."""
        entry = f'name = "mainboard", {requirement}directory = {json.dumps(str(self.source))}'
        (self.tool / "uv-receipt.toml").write_text(
            f"[tool]\n{head}requirements = [{{ {entry} }}]\n",
            encoding="utf-8",
        )

    def edit(self) -> None:
        bump(self.source / "src" / "mainboard" / "cli.py")

    def stale(self) -> Snapshot:
        """The snapshot recorded, then edited past: what every refresh starts from."""
        check(self.package)
        self.edit()
        found = check(self.package)
        assert found.stale
        return found


def layout_at(tmp_path: Path, checkout: str) -> Layout:
    source = tmp_path / checkout
    (source / "src" / "mainboard" / "__pycache__").mkdir(parents=True)
    (source / "src" / "mainboard" / "cli.py").write_text("code", encoding="utf-8")
    (source / "src" / "mainboard" / "__pycache__" / "cli.pyc").write_text("bytecode")
    (source / "pyproject.toml").write_text('[project]\nname = "mainboard"\n', encoding="utf-8")
    tool = tmp_path / "tool"
    package = tool / "lib" / "site-packages" / "mainboard"
    package.mkdir(parents=True, exist_ok=True)
    found = Layout(package=package, tool=tool, source=source)
    found.declare()
    return found


@pytest.fixture
def snapshot(tmp_path: Path) -> Layout:
    return layout_at(tmp_path, "checkout")


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """A Linux host whose re-executions are recorded rather than performed."""
    replaced: list[list[str]] = []
    monkeypatch.setattr("mainboard.staleness.platform.system", lambda: "Linux")
    monkeypatch.setattr("mainboard.staleness.os.execv", lambda path, argv: replaced.append(argv))
    monkeypatch.delenv(staleness.REFRESHED, raising=False)
    return replaced


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """A Windows host whose deferred workers are recorded rather than started."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr("mainboard.staleness.platform.system", lambda: "Windows")
    monkeypatch.setattr("mainboard.staleness.os.getpid", lambda: 314)
    monkeypatch.setattr("mainboard.staleness.PixiEngine.defer", lambda self, *a: calls.append(a))
    return calls


def test_the_check_records_on_first_run_then_names_the_reinstall_when_the_tree_moves(
    snapshot: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole lifecycle: record, agree, drift, warn, reinstall, record again.

    The first run after an install is the baseline, an unchanged tree keeps agreeing with it,
    an edit flips the answer to stale with the receipt's own extras in the named command (which
    needs no git), asking again never heals it, and a reinstall (a rewritten receipt)
    invalidates the old baseline so the fresh snapshot records the edited tree as its own.
    """
    monkeypatch.setenv("PATH", "")
    assert check(snapshot.package) == Snapshot(
        installed=True, detail="snapshot matches the source tree"
    )
    assert check(snapshot.package).stale is False
    snapshot.edit()
    stale = check(snapshot.package)
    assert stale.stale is True
    assert str(snapshot.source) in stale.detail
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
        f"{snapshot.source}[wandb]",
        "mainboard",
        "--force",
    )
    assert check(snapshot.package) == stale
    bump(snapshot.tool / "uv-receipt.toml")
    assert check(snapshot.package).stale is False


def test_package_metadata_moves_the_snapshot_even_when_runtime_source_does_not(
    snapshot: Layout,
) -> None:
    """A dependency edit must reinstall the tool rather than leaving its imports unchanged."""
    check(snapshot.package)
    (snapshot.source / "pyproject.toml").write_text(
        '[project]\nname = "mainboard"\ndependencies = ["cuda-bindings"]\n', encoding="utf-8"
    )
    assert check(snapshot.package).stale is True


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
    snapshot: Layout,
    linux: list[list[str]],
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
    found = snapshot.stale()
    ran: list[tuple[str, ...]] = []

    def install(
        self: PixiEngine, action: Callable[..., CommandResult], *argv: str
    ) -> CommandResult:
        ran.append(argv)
        if isinstance(installer, Exception):
            raise installer
        return installer

    monkeypatch.setattr(PixiEngine, "within_cwd", install)
    monkeypatch.setattr("mainboard.staleness.sys.orig_argv", ["python", "mainboard", "jobs"])

    Refresh(found).run()

    printed = capfd.readouterr()
    assert printed.out == ""
    assert ran == [found.fix]
    assert (linux == [[sys.executable, "mainboard", "jobs"]]) is updated
    assert (os.environ.get(staleness.REFRESHED) == "1") is updated
    monkeypatch.delenv(staleness.REFRESHED, raising=False)
    if updated:
        assert f"updated from {found.source}" in printed.err
        return
    assert "could not update itself" in printed.err
    assert printed.err.count("\n") == 1


def test_a_second_process_waits_its_turn_and_only_reexecutes_on_the_install_it_waited_for(
    snapshot: Layout,
    linux: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nine jobs starting at once on a synced host reinstall once, and one never waits forever."""
    found = snapshot.stale()
    bump(snapshot.tool / "uv-receipt.toml")
    monkeypatch.setattr(
        PixiEngine, "within_cwd", lambda *args: pytest.fail("the install already happened")
    )

    Refresh(found).run()
    monkeypatch.delenv(staleness.REFRESHED, raising=False)
    assert len(linux) == 1

    monkeypatch.setattr("mainboard.staleness._LOCK_SECONDS", 0.01)
    with FileLock(snapshot.tool / "self-update.lock", thread_local=False):
        worker = Thread(target=Refresh(found).run)
        worker.start()
        worker.join()
    assert len(linux) == 1
    assert "another update held its lock" in capsys.readouterr().err


def test_windows_hands_the_update_to_one_worker_that_outlives_the_launcher(
    snapshot: Layout, windows: list[tuple[str, ...]], capsys: pytest.CaptureFixture[str]
) -> None:
    """A running Windows interpreter cannot be replaced, so the update waits for it to exit.

    The command at hand answers from the snapshot it started on, and every command run before
    the worker has finished finds the pending marker and schedules nothing more, until the
    marker is old enough that its worker can no longer be on its way.
    """
    found = snapshot.stale()
    log = snapshot.source / ".mainboard" / "self-update.log"

    Refresh(found).run()
    Refresh(found).run()

    assert windows == [
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
    assert len(windows) == 2


@pytest.mark.parametrize(
    ("moved", "again", "refreshed", "said"),
    [
        pytest.param(False, False, False, "", id="a-fresh-snapshot-does-nothing"),
        pytest.param(True, False, True, "", id="a-stale-snapshot-refreshes"),
        pytest.param(True, True, False, "the update did not take", id="a-reexecution-never-loops"),
    ],
)
def test_every_invocation_starts_on_its_sources_newest_code(
    snapshot: Layout,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    moved: bool,
    again: bool,
    refreshed: bool,
    said: str,
) -> None:
    """The entry check, and the one environment flag that keeps a failed update from looping."""
    found = snapshot.stale() if moved else check(snapshot.package)
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


@pytest.mark.parametrize(
    ("interpreter", "kept"),
    [
        pytest.param(
            ("uv", "python", "cpython-3.14.7", "python.exe"), True, id="a-uv-managed-python"
        ),
        pytest.param(
            ("checkout", ".mainboard", "envs", "tool", ".pixi", "python.exe"),
            False,
            id="a-project-environment-python",
        ),
    ],
)
def test_the_refresh_keeps_only_a_durable_interpreter(
    snapshot: Layout, tmp_path: Path, interpreter: tuple[str, ...], kept: bool
) -> None:
    """A uv-managed interpreter remains the exact self-update contract, while replaceable
    generated state never becomes the public launcher's Python home.
    """
    python = tmp_path.joinpath(*interpreter)
    python.parent.mkdir(parents=True)
    python.write_text("python")
    snapshot.declare("", head=f"python = {json.dumps(str(python))}\n")

    fix = snapshot.stale().fix

    assert (
        fix[4:10]
        == ("tool", "install", "--reinstall-package", "mainboard", "--python", str(python))
    ) is kept
    assert (str(python) in fix) is kept
    assert "[wandb]" in fix[-3]


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
        pytest.param("not toml at all [", "names no source directory", id="torn receipt"),
        pytest.param(
            '[tool]\nrequirements = [{ name = "mainboard" }]\n',
            "names no source directory",
            id="no directory",
        ),
        pytest.param(
            '[tool]\nrequirements = [{ name = "other", directory = "x" }]\n',
            "names no source",
            id="another tool's requirement",
        ),
        pytest.param(
            '[tool]\nrequirements = [{ name = "mainboard", directory = "gone" }]\n',
            "no source tree at",
            id="a directory that lost its source tree",
        ),
    ],
)
def test_a_receipt_that_cannot_vouch_for_a_source_is_installed_but_never_stale(
    snapshot: Layout, receipt: str, said: str
) -> None:
    """A snapshot uv cannot explain warns nobody, since there is no tree to compare against."""
    (snapshot.tool / "uv-receipt.toml").write_text(receipt)
    found = check(snapshot.package)
    assert found == Snapshot(installed=True, detail=found.detail)
    assert said in found.detail


def test_a_torn_state_file_and_an_unwritable_tool_directory_both_answer_fresh(
    snapshot: Layout,
) -> None:
    """The check never fails the command that asked, whatever the state file's condition."""
    state = snapshot.tool / "source-state.json"
    state.write_text("torn {")
    assert check(snapshot.package).stale is False
    assert set(json.loads(state.read_text())) == {"marker", "digest"}
    state.unlink()
    snapshot.tool.chmod(0o555)
    try:
        assert check(snapshot.package).stale is False
    finally:
        snapshot.tool.chmod(0o755)


def test_the_digest_reads_names_sizes_and_clocks_and_never_bytecode(snapshot: Layout) -> None:
    """Content is unread on purpose, so the check stays in CLI-startup budget."""
    source = snapshot.source / "src"
    before = digest(source)
    assert digest(source) == before
    (source / "mainboard" / "__pycache__" / "extra.pyc").write_text("more")
    assert digest(source) == before
    snapshot.edit()
    assert digest(source) != before


def test_a_reinstall_names_its_source_absolutely_and_the_worker_reads_it_off_the_snapshot(
    tmp_path: Path, windows: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One reading of the receipt answers both callers, and a bracket in a path survives it.

    The deferred worker used to recover its argv by slicing four tokens off the pixi command and
    its log directory by cutting the `--from` token at the first `[`, which is the extras
    separator and also an ordinary character in a directory name. Both now come off the snapshot
    that read the receipt. A checkout without a pyproject still digests, and a snapshot naming
    no source logs beside the working directory.
    """
    snapshot = layout_at(tmp_path, "check[out]")
    (snapshot.source / "pyproject.toml").unlink()
    found = snapshot.stale()

    assert found.source == snapshot.source
    assert found.uv[0] == "uv" and found.uv[-1] == "--force"
    assert f"{snapshot.source}[wandb]" in found.uv
    assert found.fix == ("exec", "--spec", staleness._UV, *found.uv)
    log = snapshot.source / Project().out_dir / "self-update.log"
    assert staleness._refresh_log(found.source) == log

    Refresh(found).run()

    handed = windows[0]
    assert handed[handed.index("--") + 1 :] == found.uv
    assert str(log) in handed
    monkeypatch.chdir(tmp_path)
    assert staleness._refresh_log(None) == tmp_path / Project().out_dir / "self-update.log"
