import hashlib
import io
import stat
import tarfile
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from mainboard import MissionError, Project, load
from mainboard.core import Runner, ToolOutput, ToolRunner, Verdict
from mainboard.engines.compile.backend import Pixi
from mainboard.engines.compile.ecosystems import Binary
from mainboard.lint import Linter
from mainboard.manifest import Spec, Toolchain
from mainboard.verbs.constants import TREE

_SWIFTLINT = "swiftlint"


def archive_bytes(*, member: str = _SWIFTLINT, mode: int = stat.S_IFREG | 0o755) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w") as zipped:
        entry = ZipInfo(member)
        entry.external_attr = mode << 16
        zipped.writestr(entry, b"tool release one")
        zipped.writestr("../outside", b"never extracted")
    return output.getvalue()


@pytest.fixture
def binary(tmp_path: Path) -> Binary:
    content = archive_bytes()
    table = Toolchain(
        dev={
            _SWIFTLINT: Spec.model_validate(
                {
                    "version": "0.65.1",
                    "url": "https://releases.example/swiftlint.zip",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "member": _SWIFTLINT,
                }
            )
        }
    )
    return Binary(
        table,
        env="default",
        project="tests",
        workspace=tmp_path,
        out=tmp_path / ".mainboard",
        pixi=Pixi(tmp_path),
    )


def serve(monkeypatch, content: bytes) -> list[str]:
    calls: list[str] = []

    class Curl(Runner):
        def __init__(self, *, timeout: float, max_bytes: int = 32 << 20) -> None:
            assert timeout in (5, 125)

        def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
            assert tool == "curl"
            if args == ("--version",):
                return ToolOutput(stdout="curl 8.23.0 native")
            assert args[args.index("--max-time") + 1] == "120"
            assert args[args.index("--proto-redir") + 1] == "=https"
            calls.append(args[args.index("--url") + 1])
            Path(args[args.index("--output") + 1]).write_bytes(content)
            return ToolOutput()

    monkeypatch.setattr("mainboard.engines.compile.ecosystems.binary.ToolRunner", Curl)
    return calls


def run_cli(*words: str) -> int | str | None:
    with pytest.raises(SystemExit) as stopped:
        TREE.cli()(list(words))
    return stopped.value.code


def test_verified_archive_installs_only_named_member_and_cache_never_downloads(
    binary: Binary, monkeypatch
) -> None:
    calls = serve(monkeypatch, archive_bytes())
    assert binary.frozen_inputs() == ()
    binary.sync()
    executable = binary.binary_dirs()[0] / _SWIFTLINT
    assert (
        executable.read_bytes(),
        bool(executable.stat().st_mode & 0o111),
        (binary.directory / "outside").exists(),
    ) == (b"tool release one", True, False)
    binary.sync()
    assert len(calls) == 1
    executable.write_bytes(b"changed locally")
    binary.sync()
    assert (executable.read_bytes(), len(calls)) == (b"tool release one", 2)


def test_bad_download_never_replaces_installed_executable(binary: Binary, monkeypatch) -> None:
    serve(monkeypatch, archive_bytes())
    binary.sync()
    receipt = binary.directory / "swiftlint.json"
    receipt.write_text("broken", encoding="utf-8")
    serve(monkeypatch, b"altered download")
    with pytest.raises(MissionError, match="SHA-256"):
        binary.sync()
    assert (binary.binary_dirs()[0] / _SWIFTLINT).read_bytes() == b"tool release one"
    assert not list(binary.directory.glob(".install-*"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "http://releases.example/tool.zip"),
        ("url", "https://user:password@releases.example/tool.zip"),
        ("sha256", "not-a-hash"),
        ("member", "../outside"),
        ("member", "/absolute"),
        ("member", r"folder\outside"),
        ("version", "*"),
        ("version", ">=0.65.1"),
    ],
)
def test_invalid_coordinates_fail_before_network(
    binary: Binary, monkeypatch, field: str, value: str
) -> None:
    spec = binary.deps[_SWIFTLINT].model_dump()
    spec[field] = value
    binary.table.dev[_SWIFTLINT] = Spec.model_validate(spec)
    calls = serve(monkeypatch, archive_bytes())
    with pytest.raises(MissionError, match="invalid"):
        binary.sync()
    assert calls == []


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFDIR | 0o755])
def test_archive_links_and_directories_are_not_executed(
    binary: Binary, monkeypatch, mode: int
) -> None:
    content = archive_bytes(mode=mode)
    binary.table.dev[_SWIFTLINT] = Spec.model_validate(
        {**binary.deps[_SWIFTLINT].model_dump(), "sha256": hashlib.sha256(content).hexdigest()}
    )
    serve(monkeypatch, content)
    with pytest.raises(MissionError, match="regular executable"):
        binary.sync()
    assert not (binary.binary_dirs()[0] / _SWIFTLINT).exists()


def test_missing_member_and_invalid_zip_fail_closed(binary: Binary, monkeypatch) -> None:
    for content, message in [(archive_bytes(member="other"), "declared member"), (b"bad", "ZIP")]:
        binary.table.dev[_SWIFTLINT] = Spec.model_validate(
            {
                **binary.deps[_SWIFTLINT].model_dump(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        serve(monkeypatch, content)
        with pytest.raises(MissionError, match=message):
            binary.sync()
        assert not (binary.binary_dirs()[0] / _SWIFTLINT).exists()


def test_dropped_dependency_prunes_its_private_binary_and_receipt(
    binary: Binary, monkeypatch
) -> None:
    serve(monkeypatch, archive_bytes())
    binary.sync()
    binary.table.dev.clear()
    binary.sync()
    assert list(binary.binary_dirs()[0].iterdir()) == []
    assert list(binary.directory.glob("*.json")) == []


def test_download_timeout_is_a_failed_install(binary: Binary, monkeypatch) -> None:
    class Curl(Runner):
        def __init__(self, *, timeout: float, max_bytes: int = 32 << 20) -> None:
            pass

        def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
            return (
                ToolOutput(stdout="curl 8.23.0")
                if args == ("--version",)
                else ToolOutput(returncode=124, truncated=True)
            )

    monkeypatch.setattr("mainboard.engines.compile.ecosystems.binary.ToolRunner", Curl)
    with pytest.raises(MissionError, match="download failed"):
        binary.sync()
    assert not (binary.binary_dirs()[0] / _SWIFTLINT).exists()


def test_binary_cli_installs_platform_declared_tools_without_a_solver(
    workspace: Path, monkeypatch
) -> None:
    content = archive_bytes()
    digest = hashlib.sha256(content).hexdigest()
    (workspace / "mb.toml").write_text(
        '[workspace]\nname = "tests"\n'
        "[binary.dev]\n"
        f'swiftlint = {{ version = "0.65.1", url = "https://releases.example/swiftlint.zip", '
        f'sha256 = "{digest}", member = "swiftlint" }}\n',
        encoding="utf-8",
    )
    calls = serve(monkeypatch, content)
    monkeypatch.chdir(workspace)
    for _ in range(2):
        assert run_cli("dep", "binary") == 0
    assert len(calls) == 1
    assert list(Project().out(workspace).rglob("bin/swiftlint"))
    assert not list(workspace.rglob("pixi.lock"))


@pytest.mark.parametrize("name", ["binary", "bin"])
def test_install_refuses_linked_directories_before_pruning_other_files(
    binary: Binary, monkeypatch, name: str
) -> None:
    outside = binary.workspace / "outside"
    outside.mkdir()
    (outside / "unrelated").write_bytes(b"keep")
    link = binary.directory if name == "binary" else binary.binary_dirs()[0]
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    calls = serve(monkeypatch, archive_bytes())
    with pytest.raises(MissionError, match="symbolic links"):
        binary.sync()
    assert ((outside / "unrelated").read_bytes(), calls) == (b"keep", [])


def test_missing_tool_fails_lint_without_downloading_or_changing_swift(
    workspace: Path, monkeypatch
) -> None:
    source = workspace / "Example.swift"
    source.write_bytes(b"struct Example {}\n")
    manifest = workspace / "mb.toml"
    manifest.write_text(
        '[workspace]\nname = "tests"\n'
        "[lint.tools.swiftlint]\n"
        f'check = "{workspace}/missing/swiftlint lint --strict {{files}}"\n'
        'files = ["*.swift"]\n',
        encoding="utf-8",
    )
    calls = serve(monkeypatch, archive_bytes())
    report = Linter(workspace, load(manifest), check=True, only=(_SWIFTLINT,)).lint([source])
    assert [failure.code for failure in report.failures] == [127]
    assert (report.rewritten, source.read_bytes(), calls) == ((), b"struct Example {}\n", [])


def test_continuous_partial_progress_cannot_extend_outer_download_deadline(tmp_path: Path) -> None:
    script = tmp_path / "slow-curl"
    script.write_text("#!/bin/sh\nwhile :; do printf x; sleep 0.01; done\n", encoding="utf-8")
    script.chmod(0o755)
    result = ToolRunner(timeout=1).run("/bin/sh", (str(script),))
    assert result.returncode == 124 and result.truncated
    assert 0 < len(result.stdout) < 1024


def tarball_bytes(*, member: str = "./sink") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as packed:
        entry = tarfile.TarInfo(member)
        entry.size, entry.mode = len(b"sink release one"), 0o755
        packed.addfile(entry, io.BytesIO(b"sink release one"))
    return output.getvalue()


def doctor_verdict(binary: Binary) -> Verdict:
    row = binary.section()
    assert row is not None
    return row.verdict


def test_a_tar_gz_release_installs_its_member_and_the_doctor_row_follows_it(
    binary: Binary, monkeypatch
) -> None:
    content = tarball_bytes()
    binary.table.dev.clear()
    binary.table.dev["sink"] = Spec.model_validate(
        {
            "version": "0.7.3",
            "url": "https://releases.example/sink-v0.7.3-darwin-arm64.tar.gz",
            "sha256": hashlib.sha256(content).hexdigest(),
            "member": "sink",
        }
    )
    calls = serve(monkeypatch, content)
    before = doctor_verdict(binary)
    binary.sync()
    after = doctor_verdict(binary)
    assert (binary.binary_dirs()[0] / "sink").read_bytes() == b"sink release one"
    assert (before, after, len(calls)) == (Verdict.FAIL, Verdict.PASS, 1)


def test_a_tar_gz_that_fails_its_hash_is_never_opened(binary: Binary, monkeypatch) -> None:
    binary.table.dev["swiftlint"] = Spec.model_validate(
        {
            **binary.deps[_SWIFTLINT].model_dump(),
            "url": "https://releases.example/swiftlint.tar.gz",
        }
    )
    serve(monkeypatch, tarball_bytes(member="swiftlint"))
    with pytest.raises(MissionError, match="SHA-256"):
        binary.sync()
    assert not (binary.binary_dirs()[0] / _SWIFTLINT).exists()


def test_binary_cli_lists_and_locates_pinned_tools_and_fails_while_one_is_missing(
    workspace: Path, monkeypatch, capsys
) -> None:
    content = archive_bytes()
    (workspace / "mb.toml").write_text(
        '[workspace]\nname = "tests"\n'
        "[binary.dev]\n"
        f'swiftlint = {{ version = "0.65.1", url = "https://releases.example/swiftlint.zip", '
        f'sha256 = "{hashlib.sha256(content).hexdigest()}", member = "swiftlint" }}\n',
        encoding="utf-8",
    )
    calls = serve(monkeypatch, content)
    monkeypatch.chdir(workspace)
    missing = run_cli("dep", "binary", "list")
    assert run_cli("dep", "binary") == 0
    capsys.readouterr()
    listed, found = run_cli("dep", "binary", "list"), run_cli("dep", "binary", "path", _SWIFTLINT)
    assert (missing, listed, found, len(calls)) == (1, 0, 0, 1)
    assert capsys.readouterr().out.strip().endswith("/binary/bin/swiftlint")
