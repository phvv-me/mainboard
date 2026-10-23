"""What only a gate worker can prove: its guard refuses, its sandbox holds, its mb is this tree's.

A test of refusal never performs what it expects refused: it asks the guard for its decision, or
aims at a target that could not be written or resolved anyway, so a missing guard shows as
another error rather than as a forbidden write.
"""

import os
import socket
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

import mainboard

Guard = pytest.importorskip(
    "guard", reason="M00-GATE: only a gate worker carries the guard (integration/gates/run.sh)"
).Guard

_WRITE = os.O_WRONLY | os.O_CREAT
_PROOF = Path(os.environ.get("MB_GATE_ROOT", "/nonexistent")) / "os-isolation-proved"


@pytest.mark.parametrize("live", ["MB_GATE_PROJECTS", "MB_GATE_WORKTREE"])
def test_a_write_outside_the_roots_is_refused(tmp_path: Path, live: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(os.environ[live])
    guard = Guard([root])
    for target in (Path(os.environ[live]), root / "escape"):
        with pytest.raises(PermissionError, match="M00 denies write"):
            guard.audit("open", [str(target / "M00-forbidden"), "w", _WRITE])
    guard.audit("open", [str(root / "allowed"), "w", _WRITE])


def test_a_provider_name_is_refused() -> None:
    with pytest.raises(PermissionError, match="M00 denies DNS"):
        Guard([]).audit("socket.getaddrinfo", ["api.vast.ai", 443, 0, 0, 0])


def test_the_installed_hook_runs_first() -> None:
    """Neither a directory that does not exist nor a `.invalid` name could be reached anyway."""
    with pytest.raises(PermissionError, match="M00 denies write"):
        Path("/M00-no-such-directory/forbidden").write_text("forbidden", encoding="utf-8")
    with pytest.raises(PermissionError, match="M00 denies DNS"):
        socket.getaddrinfo("provider.invalid", 443)


@pytest.mark.skipif(not _PROOF.is_file(), reason="M00-OS: needs the OS-isolated suite/target lane")
def test_a_real_ssh_cannot_connect() -> None:
    """The kernel refuses before a packet leaves, to a reserved address no host answers at."""
    done = subprocess.run(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "192.0.2.1",
            "true",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert done.returncode == 255
    assert "Operation not permitted" in done.stderr or "Network is unreachable" in done.stderr


def test_a_subprocess_mb_runs_the_worktrees_source() -> None:
    """The console script imports `mainboard.cli` and nothing else, so its interpreter answers."""
    script = Path(sysconfig.get_path("scripts")) / "mb"
    interpreter = script.read_text(encoding="utf-8").splitlines()[0].removeprefix("#!")
    done = subprocess.run(
        [interpreter, "-c", "import mainboard.cli as cli; print(cli.__file__)"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    source = Path(mainboard.__file__).resolve().with_name("cli.py")
    assert Path(done.stdout.strip()).resolve() == source


def test_the_legacy_runtime_imports_only_its_own_archive(legacy_runtime) -> None:
    shown = legacy_runtime.run("import sys; print(sys.path)").stdout
    assert str(legacy_runtime.root / "src") in shown
    assert str(Path(mainboard.__file__).resolve().parents[1]) not in shown
    assert len(legacy_runtime.archive_sha256) == 64


def test_postgres_is_a_temporary_catalog(postgres: str) -> None:
    psql = Path(sysconfig.get_path("scripts")) / "psql"
    done = subprocess.run(
        [
            str(psql),
            postgres,
            "-XAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "CREATE SCHEMA gate; CREATE TABLE gate.proof (id integer PRIMARY KEY); "
            "INSERT INTO gate.proof VALUES (1); SELECT count(*) FROM gate.proof; "
            "SHOW data_directory; SHOW listen_addresses;",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    assert "\n1\n" in done.stdout
    assert os.environ["MB_GATE_ROOT"] in done.stdout
    assert done.stdout.endswith("\n\n")  # listen_addresses is empty: no TCP listener.


def test_seeded_extensions_copy_without_user_duckdb_or_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from . import runner

    home = tmp_path / "source-home"
    (home / ".duckdb").mkdir(parents=True)
    (home / ".duckdb/config").write_text("private user settings", encoding="utf-8")
    seed = tmp_path / "seed"
    seed.mkdir()
    parser = seed / "sitting_duck.duckdb_extension"
    parser.write_bytes(b"installed extension")
    monkeypatch.setattr(runner, "_SEED", seed)
    monkeypatch.setattr(Path, "home", lambda: home)
    worker = runner.Gate.__new__(runner.Gate)
    worker.root = tmp_path / "worker"
    worker._seed_extensions()
    cache = "home/Library/Caches/mb/duckdb" if sys.platform == "darwin" else "cache/mb/duckdb"
    copied = worker.root / cache / "sitting_duck.duckdb_extension"
    assert copied.read_bytes() == b"installed extension"
    assert not (worker.root / "home/.duckdb").exists()
    copied.write_bytes(b"worker modification")
    assert parser.read_bytes() == b"installed extension"
