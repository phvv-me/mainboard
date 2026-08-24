from collections.abc import Mapping
from contextlib import chdir
from pathlib import Path
from typing import NoReturn

import pytest
from plumbum import CommandNotFound
from plumbum.commands.processes import ProcessExecutionError

from mainboard.dispatch import GitignoreFilter, HostUnreachable, SyncLock
from mainboard.dispatch import sync as sync_module
from mainboard.dispatch.shared import STATE_DIR
from mainboard.dispatch.sync import ALWAYS_EXCLUDE, Rsync, binary, rsync

_MIRROR = Rsync.ARCHIVE | Rsync.RELATIVE | Rsync.DELETE | Rsync.DELETE_AFTER
_UPSTREAM = "rsync  version 3.2.7\n"
_APPLE = "openrsync: protocol version 29\n"


def seed(root: Path, *files: str) -> None:
    """Create each relative path under `root` (parents included) with token content."""
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)


class _Version:
    """A `local[name]` double answering only the `--version` probe `binary` runs."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __call__(self, *_a: str) -> str:
        return self.text


class _Missing:
    """A `local` double where every name but the ones given is off this machine."""

    def __init__(self, **present: _Version) -> None:
        self.present = present

    def __getitem__(self, name: str) -> _Version | NoReturn:
        if name in self.present:
            return self.present[name]
        raise CommandNotFound(name, [])


class _FailingRsync:
    """A `local["rsync"]` double whose `--version` probe answers but whose transfer raises."""

    def __init__(self, error: ProcessExecutionError) -> None:
        self.error = error
        self.argv: list[str] = []

    def __call__(self, *args: str) -> str:
        if args:
            return _UPSTREAM
        raise self.error

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> _FailingRsync:
        self.argv = list(args) if isinstance(args, list | tuple) else [args]
        return self


def test_the_denylist_covers_git_env_and_every_generated_directory() -> None:
    assert (
        ".git",
        ".env",
        f"{STATE_DIR}/",
        ".mainboard/",
        ".pixi/",
        "__pycache__/",
    ) == ALWAYS_EXCLUDE


@pytest.mark.parametrize(
    ("installed", "mirror", "chosen"),
    [
        ({"rsync": _Version(_UPSTREAM)}, False, "rsync"),
        (
            {"rsync": _Version(_APPLE), "/opt/homebrew/bin/rsync": _Version(_UPSTREAM)},
            True,
            "/opt/homebrew/bin/rsync",
        ),
        ({"rsync": _Version(_APPLE)}, True, ""),
        ({}, False, "rsync"),
    ],
)
def test_binary_prefers_upstream_rsync_and_refuses_to_mirror_with_apples(
    monkeypatch: pytest.MonkeyPatch, installed: Mapping[str, _Version], mirror: bool, chosen: str
) -> None:
    """openrsync cannot prune inside shared directories, so a remote mirror silently keeps them."""
    monkeypatch.setattr(sync_module, "local", _Missing(**{k: v for k, v in installed.items()}))
    if not chosen:
        with pytest.raises(RuntimeError, match="mirroring needs upstream rsync"):
            binary(mirror=mirror)
        return
    assert binary(mirror=mirror) == chosen


def test_every_option_becomes_its_own_argument_in_receiver_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """protect rules must reach the receiver before include and exclude or `--delete` wins."""
    assert Rsync.ARCHIVE.string == "-a"
    assert (Rsync.ARCHIVE | Rsync.RELATIVE).string == "-a -R"
    assert Rsync.DELETE.string == "--delete"
    command = _FailingRsync(ProcessExecutionError(["rsync"], 24, "", ""))
    monkeypatch.setattr(sync_module, "local", {"rsync": command})
    rsync(
        ["src", "docs"],
        "gold:/repo/",
        _MIRROR,
        include=["/src/keep.py"],
        exclude=["drop.py"],
        protect=["results/***"],
        filters=["merge,- .gitignore"],
        rsh="ssh -o BatchMode=yes",
        bwlimit=1000,
        timeout=30,
        extra=["--itemize-changes"],
    )
    assert command.argv == [
        "-aR",
        "--delete",
        "--delete-after",
        "-e",
        "ssh -o BatchMode=yes",
        "--bwlimit=1000",
        "--timeout=30",
        "--filter",
        "protect results/***",
        "--include",
        "/src/keep.py",
        "--filter",
        "merge,- .gitignore",
        "--exclude",
        "drop.py",
        "--itemize-changes",
        "src",
        "docs",
        "gold:/repo/",
    ]


def test_a_real_mirror_prunes_the_stale_and_the_ignored_while_protecting_the_remote_only(
    tmp_path: Path,
) -> None:
    """The one end-to-end proof that the filter order above actually behaves as intended."""
    repo, host = tmp_path / "repo", tmp_path / "host"
    seed(repo, ".gitignore", "src/run.py", "src/local.scratch")
    (repo / ".gitignore").write_text("*.scratch\n")
    seed(host, "src/stale.py", "src/host-only.scratch", "src/results/e1.json")
    with chdir(repo):
        rsync(
            ["src"],
            f"{host}/",
            _MIRROR,
            protect=["results/***"],
            filters=["merge,- .gitignore", ":- .gitignore"],
        )
    assert (host / "src/run.py").is_file()
    assert not (host / "src/local.scratch").exists()
    assert not (host / "src/stale.py").exists()
    assert (host / "src/host-only.scratch").is_file()
    assert (host / "src/results/e1.json").is_file()


def test_a_mirror_reads_its_sources_from_the_workspace_not_from_where_it_was_typed(
    tmp_path: Path,
) -> None:
    """Sync always runs from the workspace root.

    `--relative` rebuilds each source path under the destination, so a path read from a
    subdirectory would name a different file or none at all.
    """
    repo, host = tmp_path / "repo", tmp_path / "host"
    seed(repo, "src/run.py", "src/deep/nested.py")
    host.mkdir()
    with chdir(repo / "src" / "deep"):
        rsync(["src"], f"{host}/", _MIRROR, cwd=repo)
    assert (host / "src/run.py").is_file()
    assert (host / "src/deep/nested.py").is_file()


@pytest.mark.parametrize(
    ("retcode", "stdout", "stderr", "host", "allow_vanished", "raised", "detail"),
    [
        (24, "deleting old.py\n", "", "", True, None, "deleting old.py\n"),
        (24, "partial\n", "", "", False, ProcessExecutionError, "Unexpected exit code: 24"),
        (23, "partial\n", "no transfer", "", True, ProcessExecutionError, "exit code: 23"),
        (
            255,
            "",
            "kex_exchange identification failed",
            "gold",
            True,
            HostUnreachable,
            "kex_exchange",
        ),
        (30, "", "ssh: connection closed by remote", "gold", True, HostUnreachable, "connection"),
        (30, "", "", "gold", True, HostUnreachable, "transport timed out"),
    ],
)
def test_a_vanished_source_is_absorbed_only_when_allowed_and_a_dead_host_is_translated(
    monkeypatch: pytest.MonkeyPatch,
    retcode: int,
    stdout: str,
    stderr: str,
    host: str,
    allow_vanished: bool,
    raised: type[BaseException] | None,
    detail: str,
) -> None:
    """A required job-script transfer must never let a submission continue on a partial sync."""
    error = ProcessExecutionError(["rsync"], retcode, stdout, stderr)
    monkeypatch.setattr(sync_module, "local", {"rsync": _FailingRsync(error)})
    if raised is None:
        assert rsync(["a"], "b/", allow_vanished=allow_vanished, host=host) == detail
        return
    with pytest.raises(raised, match=detail):
        rsync(["a"], "b/", allow_vanished=allow_vanished, host=host)


def test_deleted_receiver_paths_are_noted_once_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[str, tuple[int | str, ...]]] = []
    monkeypatch.setattr(sync_module.logger, "warning", lambda msg, *a: warnings.append((msg, a)))
    sync_module._log_deletions("deleting a.py\nkeeping b.py\ndeleting c.py\n")  # ruff:ignore[private-member-access]  reason=unit-tests the module-private helper since=2026-08-16
    assert [args for _, args in warnings] == [(2, "a.py, c.py")]
    sync_module._log_deletions("no deletions here\n")  # ruff:ignore[private-member-access]  reason=unit-tests the module-private helper since=2026-08-16
    assert len(warnings) == 1


def test_the_sync_lock_releases_its_file_however_the_mirror_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SyncLock("gold", tmp_path) as lock:
        assert lock.file is not None
        assert lock.path.is_file()
    assert lock.file is None
    lock.__exit__(None, None, None)
    monkeypatch.chdir(tmp_path)
    assert SyncLock("gold").path == tmp_path / STATE_DIR / "locks" / lock.path.name
    handle = (tmp_path / "file").open("a+")

    def refuse(fileno: int, operation: int) -> None:
        raise OSError("lock unavailable")

    monkeypatch.setattr(sync_module.Path, "open", lambda self, mode: handle)
    monkeypatch.setattr(sync_module.fcntl, "flock", refuse)
    with pytest.raises(OSError, match="lock unavailable"):
        SyncLock("gold", tmp_path).__enter__()
    assert handle.closed


def test_the_gitignore_filter_reads_the_root_rules_and_merges_the_nested_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert GitignoreFilter(tmp_path).filters == [":- .gitignore"]
    (tmp_path / ".gitignore").write_text("*.scratch\nbuild/\n")
    ignores = GitignoreFilter(tmp_path)
    assert ignores.filters == ["merge,- .gitignore", ":- .gitignore"]
    assert ignores.ignored(tmp_path / "a.scratch")
    assert not ignores.ignored(tmp_path / "a.py")
    assert ignores.ignored("build/output.txt")
    assert not ignores.ignored("src/output.txt")
    assert not ignores.ignored("/somewhere/else/file.txt")
    monkeypatch.chdir(tmp_path)
    assert GitignoreFilter().root == tmp_path


def test_control_files_ship_every_ancestor_gitignore_from_the_root_down(tmp_path: Path) -> None:
    """A narrow source inherits a parent ignore file the transferred subtree never carries."""
    (tmp_path / ".gitignore").write_text("*.scratch\n")
    (tmp_path / "research").mkdir()
    (tmp_path / "research/.gitignore").write_text("generated/\n")
    (tmp_path / "research/projects").mkdir()
    ignores = GitignoreFilter(tmp_path)
    assert ignores.control_files(["research/projects"]) == [".gitignore", "research/.gitignore"]
    assert ignores.control_files(["research/a", "research/b"]) == [
        ".gitignore",
        "research/.gitignore",
    ]
    assert ignores.control_files([str(tmp_path / "docs/projects")]) == [".gitignore"]
