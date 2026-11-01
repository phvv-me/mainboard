"""The native SwiftSyntax helper is built once, refuses a host without its toolchain and reads a
project given by a relative root."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.analysis.providers.swift import SwiftSource
from mainboard.core import Runner, ToolOutput, ToolRunner

_VIEW_SWIFT = "View.swift"


def test_missing_native_toolchain_refuses_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mainboard.analysis.providers.swift.shutil.which", lambda name: None)

    class MissingToolchain(Runner):
        def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
            return ToolOutput(returncode=127, stderr="tool unavailable")

    with pytest.raises(MissionError, match="Swift analysis requires"):
        SwiftSource.native_helper(MissingToolchain())


@pytest.mark.swift_native
def test_native_provider_accepts_a_relative_cli_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / _VIEW_SWIFT).write_text("import SwiftUI\nstruct Screen: View {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rows = SwiftSource().run(Path("project"))
    assert rows["facts"][1][0][1] == _VIEW_SWIFT
    assert rows["facts"][1][0][5] == 0


@pytest.mark.swift_native
def test_native_helper_is_cached_and_reports_bad_requests_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls: list[tuple[str, ...]] = []

    class RecordingRunner(Runner):
        def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
            calls.append(tuple(args))
            return ToolRunner(timeout=120).run(tool, args, cwd=cwd)

    runner = RecordingRunner()
    first = SwiftSource.native_helper(runner)
    assert first == SwiftSource.native_helper(runner)
    assert sum("-o" in args for args in calls) == 1
    result = runner.run(str(first), (str(tmp_path / "missing-request.json"),))
    assert (result.returncode, result.stderr.startswith("SwiftSyntax: ")) == (1, True)
    assert "Fatal error" not in result.stderr


@pytest.mark.swift_native
def test_native_helper_compile_timeout_never_becomes_a_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class TimedOutCompiler(Runner):
        def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
            if "-o" in args:
                return ToolOutput(returncode=124, stderr="compile timed out", truncated=True)
            return ToolRunner(timeout=120).run(tool, args, cwd=cwd)

    with pytest.raises(MissionError, match="helper compilation failed: compile timed out"):
        SwiftSource.native_helper(TimedOutCompiler())
    assert not list(tmp_path.rglob("swift-syntax"))


def test_missing_native_host_libraries_fails_with_an_actionable_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = tmp_path / "usr/bin/swiftc"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"compiler location only")
    monkeypatch.setattr(
        "mainboard.analysis.providers.swift.shutil.which", lambda name: str(compiler)
    )

    class CompilerLocation(Runner):
        def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
            return ToolOutput(stdout=str(compiler))

    with pytest.raises(MissionError, match="SwiftParser host libraries at"):
        SwiftSource.native_helper(CompilerLocation())
