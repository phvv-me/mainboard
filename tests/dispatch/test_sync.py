from contextlib import chdir
from typing import TYPE_CHECKING, NoReturn

import pytest
from plumbum import CommandNotFound
from plumbum.commands.processes import ProcessExecutionError

from mainboard.dispatch import STATE_DIR, GitignoreFilter, HostUnreachable, SyncLock
from mainboard.dispatch import sync as sync_module
from mainboard.dispatch.sync import ALWAYS_EXCLUDE, Rsync, binary, rsync

if TYPE_CHECKING:
    from pathlib import Path

_MIRROR = Rsync.ARCHIVE | Rsync.RELATIVE | Rsync.DELETE | Rsync.DELETE_AFTER


def seed(root: Path, *files: str) -> None:
    """Create each relative path under `root` (parents included) with token content."""
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)


# --- ALWAYS_EXCLUDE ---


def test_always_exclude_covers_git_env_state_and_mainboard_dirs() -> None:
    assert (
        ".git",
        ".env",
        f"{STATE_DIR}/",
        ".mainboard/",
        ".pixi/",
        "__pycache__/",
    ) == ALWAYS_EXCLUDE


# --- binary() ---


def test_binary_prefers_upstream_rsync_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class Upstream:
        def __call__(self, *_a) -> str:
            return "rsync  version 3.2.7\n"

    monkeypatch.setattr(sync_module, "local", {"rsync": Upstream()})
    assert binary(mirror=False) == "rsync"


def test_binary_skips_apple_openrsync_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class Openrsync:
        def __call__(self, *_a) -> str:
            return "openrsync: protocol version 29\n"

    class Upstream:
        def __call__(self, *_a) -> str:
            return "rsync  version 3.2.7\n"

    monkeypatch.setattr(
        sync_module,
        "local",
        {"rsync": Openrsync(), "/opt/homebrew/bin/rsync": Upstream()},
    )
    assert binary(mirror=True) == "/opt/homebrew/bin/rsync"


def test_binary_raises_when_only_openrsync_exists_and_mirroring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Openrsync:
        def __call__(self, *_a) -> str:
            return "openrsync: protocol version 29\n"

    def missing(name: str) -> Openrsync:
        if name == "rsync":
            return Openrsync()
        raise CommandNotFound(name, [])

    class FakeLocal:
        def __getitem__(self, name: str) -> Openrsync:
            return missing(name)

    monkeypatch.setattr(sync_module, "local", FakeLocal())
    with pytest.raises(RuntimeError, match="mirroring needs upstream rsync"):
        binary(mirror=True)


def test_binary_falls_back_to_plain_rsync_when_not_mirroring_and_only_openrsync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLocal:
        def __getitem__(self, name: str) -> NoReturn:
            raise CommandNotFound(name, [])

    monkeypatch.setattr(sync_module, "local", FakeLocal())
    assert binary(mirror=False) == "rsync"


# --- Rsync flag rendering ---


def test_rsync_flag_members_carry_their_literal_switches() -> None:
    assert Rsync.ARCHIVE.string == "-a"
    assert (Rsync.ARCHIVE | Rsync.RELATIVE).string == "-a -R"
    assert Rsync.DELETE.string == "--delete"


# --- rsync() against the real binary, local dir to local dir ---


def test_mirror_prunes_stale_files_but_protects_results(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    base = tmp_path_factory.mktemp("mirror")
    repo, host = base / "repo", base / "host"
    seed(repo, "src/new.py")
    seed(host, "src/stale.py", "src/results/e1.json")
    with chdir(repo):
        rsync(["src"], f"{host}/", _MIRROR, protect=["results/***"])
    assert (host / "src/new.py").is_file()
    assert not (host / "src/stale.py").exists()
    assert (host / "src/results/e1.json").is_file()


def test_rsync_exercises_every_optional_flag(tmp_path: Path) -> None:
    """rsh/bwlimit/timeout/protect/include/exclude/filters each add their own rsync argument."""
    repo, host = tmp_path / "repo", tmp_path / "host"
    seed(repo, "src/keep.py", "src/drop.py")
    seed(host, "src/results/e1.json")
    with chdir(repo):
        output = rsync(
            ["src"],
            f"{host}/",
            _MIRROR,
            include=["/src/keep.py"],
            exclude=["drop.py"],
            protect=["results/***"],
            filters=[],
            rsh="ssh",
            bwlimit=1000,
            timeout=30,
        )
    assert isinstance(output, str)
    assert (host / "src/keep.py").is_file()
    assert (host / "src/results/e1.json").is_file()


def test_gitignored_files_are_never_shipped_or_deleted(tmp_path: Path) -> None:
    repo, host = tmp_path / "repo", tmp_path / "host"
    seed(repo, ".gitignore", "src/run.py", "src/local.scratch")
    (repo / ".gitignore").write_text("*.scratch\n")
    seed(host, "src/host-only.scratch", "src/stale.py")
    with chdir(repo):
        rsync(["src"], f"{host}/", _MIRROR, filters=["merge,- .gitignore", ":- .gitignore"])
    assert (host / "src/run.py").is_file()
    assert not (host / "src/local.scratch").exists()
    assert (host / "src/host-only.scratch").is_file()
    assert not (host / "src/stale.py").exists()


class _FailingRsync:
    """A `local["rsync"]` double: the `("--version")` probe answers plain text, the bound
    `[args]()` invocation (the real transfer) raises `error`."""

    def __init__(self, error: ProcessExecutionError) -> None:
        self.error = error

    def __call__(self, *args: str) -> str:
        if args:
            return "rsync  version 3.2.7\n"
        raise self.error

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> _FailingRsync:
        return self


def test_rsync_accepts_vanished_source_as_a_partial_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """rsync code 24 (source files vanished) is swallowed for an ordinary changing mirror."""
    error = ProcessExecutionError(["rsync"], 24, "deleting old.py\n", "")
    monkeypatch.setattr(sync_module, "local", {"rsync": _FailingRsync(error)})
    assert rsync(["a"], "b/", allow_vanished=True) == "deleting old.py\n"


def test_rsync_raises_when_vanished_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ProcessExecutionError(["rsync"], 24, "partial\n", "")
    monkeypatch.setattr(sync_module, "local", {"rsync": _FailingRsync(error)})
    with pytest.raises(ProcessExecutionError):
        rsync(["a"], "b/", allow_vanished=False)


def test_rsync_raises_host_unreachable_on_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ProcessExecutionError(["rsync"], 255, "", "kex_exchange identification failed")
    monkeypatch.setattr(sync_module, "local", {"rsync": _FailingRsync(error)})
    with pytest.raises(HostUnreachable, match="kex_exchange"):
        rsync(["a"], "gold:b/", host="gold")


def test_rsync_reraises_a_non_vanished_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ProcessExecutionError(
        ["rsync"], 23, "partial\n", "some files could not be transferred"
    )
    monkeypatch.setattr(sync_module, "local", {"rsync": _FailingRsync(error)})
    with pytest.raises(ProcessExecutionError):
        rsync(["a"], "b/")


def test_log_deletions_warns_once_with_every_deleted_path(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[tuple[str, tuple[int, str]]] = []
    monkeypatch.setattr(sync_module.logger, "warning", lambda msg, *a: warnings.append((msg, a)))
    sync_module._log_deletions("deleting a.py\nkeeping b.py\ndeleting c.py\n")  # noqa: SLF001  reason=unit-tests the module-private helper since=2026-08-16
    assert len(warnings) == 1
    assert warnings[0][1] == (2, "a.py, c.py")


def test_log_deletions_is_silent_with_nothing_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[tuple[str | int, ...]] = []
    monkeypatch.setattr(sync_module.logger, "warning", lambda *a: warnings.append(a))
    sync_module._log_deletions("no deletions here\n")  # noqa: SLF001  reason=unit-tests the module-private helper since=2026-08-16
    assert warnings == []


# --- SyncLock ---


def test_sync_lock_acquires_and_releases_cleanly(tmp_path: Path) -> None:
    with SyncLock("gold", tmp_path) as lock:
        assert lock.file is not None
        assert lock.path.is_file()
    assert lock.file is None


def test_sync_lock_closes_its_file_when_acquisition_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = (tmp_path / "file").open("a+")

    def fail(fileno: int, operation: int) -> None:
        raise OSError("lock unavailable")

    monkeypatch.setattr(sync_module.Path, "open", lambda self, mode: file)
    monkeypatch.setattr(sync_module.fcntl, "flock", fail)
    with pytest.raises(OSError, match="lock unavailable"):
        SyncLock("gold", tmp_path).__enter__()
    assert file.closed


def test_sync_lock_defaults_root_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    lock = SyncLock("gold")
    assert lock.path == tmp_path / STATE_DIR / "locks" / lock.path.name


def test_sync_lock_exit_without_enter_is_a_no_op(tmp_path: Path) -> None:
    """`__exit__` on a lock that never acquired (`.file` still None) simply returns."""
    lock = SyncLock("gold", tmp_path)
    lock.__exit__(None, None, None)  # must not raise


# --- GitignoreFilter ---


def test_gitignore_filter_reads_root_gitignore_and_merges_filters(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.scratch\n")
    gitignore = GitignoreFilter(tmp_path)
    assert gitignore.filters == ["merge,- .gitignore", ":- .gitignore"]
    assert gitignore.ignored(tmp_path / "a.scratch")
    assert not gitignore.ignored(tmp_path / "a.py")


def test_gitignore_filter_without_a_root_gitignore_skips_the_merge_rule(tmp_path: Path) -> None:
    gitignore = GitignoreFilter(tmp_path)
    assert gitignore.filters == [":- .gitignore"]


def test_gitignore_filter_defaults_root_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert GitignoreFilter().root == tmp_path


def test_gitignore_filter_ignored_handles_relative_paths_too(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build/\n")
    gitignore = GitignoreFilter(tmp_path)
    assert gitignore.ignored("build/output.txt")
    assert not gitignore.ignored("src/output.txt")


def test_gitignore_filter_ignored_accepts_a_path_outside_root(tmp_path: Path) -> None:
    """An absolute path that is not under root is matched literally, not relativized."""
    gitignore = GitignoreFilter(tmp_path)
    assert not gitignore.ignored("/somewhere/else/file.txt")


def test_control_files_lists_ancestor_gitignores_root_first(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.scratch\n")
    (tmp_path / "research").mkdir()
    (tmp_path / "research/.gitignore").write_text("generated/\n")
    (tmp_path / "research/projects").mkdir()
    gitignore = GitignoreFilter(tmp_path)
    files = gitignore.control_files(["research/projects"])
    assert files == [".gitignore", "research/.gitignore"]


def test_control_files_deduplicates_shared_ancestors(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.scratch\n")
    (tmp_path / "research").mkdir()
    gitignore = GitignoreFilter(tmp_path)
    files = gitignore.control_files(["research/a", "research/b"])
    assert files == [".gitignore"]


def test_control_files_accepts_an_absolute_source_path(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.scratch\n")
    (tmp_path / "research").mkdir()
    gitignore = GitignoreFilter(tmp_path)
    files = gitignore.control_files([str(tmp_path / "research/projects")])
    assert files == [".gitignore"]
