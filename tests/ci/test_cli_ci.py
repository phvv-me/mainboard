import json
import os
import subprocess
from pathlib import Path

import pytest

from mainboard.ci import LocalLeg, Matrix, Result, Verdict
from mainboard.cli import build

from .conftest import declare, say, step


def _ci(root: Path, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    """Run `ci` through the CLI rooted at `root`, answering its exit code, stdout and stderr."""
    with pytest.raises(SystemExit) as exited:
        build(root)(["ci", *argv])
    captured = capsys.readouterr()
    return int(exited.value.code or 0), captured.out, captured.err


def test_the_gate_runs_here_with_no_workspace_and_exits_on_its_first_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each step's own words reach stderr as it settles, and the table alone reaches stdout."""
    gate = step("lint", say("clean")) + step("test", say("red", 1)) + step("after", say("never"))
    package = declare(tmp_path / "pkg", gate)
    monkeypatch.chdir(package)

    code, out, err = _ci(tmp_path, capsys, "--json")

    assert code == 1
    rows = json.loads(out)
    assert [(row["step"], row["verdict"]) for row in rows] == [
        ("lint", "ok"),
        ("test", "failed"),
        ("after", "not run"),
    ]
    assert "after" not in err
    assert "lint: ok in" in err and "clean" in err
    assert "test: failed in" in err and "red" in err

    code, out, _ = _ci(tmp_path, capsys, str(package), "--agent", "--fields", "step,verdict")
    assert out.splitlines()[:2] == ["step\tverdict", "lint\tok"]


def test_the_matrix_prints_only_failures_and_names_every_platform_left_uncovered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mainboard.toml").write_text('[workspace]\nname = "w"\n', encoding="utf-8")
    package = declare(tmp_path / "pkg", step("lint", say("quiet")))
    here = LocalLeg(package).family
    settled = [
        Result(leg="local", os=here, step="lint", verdict=Verdict.OK, output="quiet"),
        Result(leg="box", os="win", step="test", verdict=Verdict.FAILED, output="E boom"),
    ]
    monkeypatch.setattr(Matrix, "run", lambda self: settled)

    code, out, err = _ci(tmp_path, capsys, str(package), "--matrix", "--json")

    assert code == 1
    assert [row["leg"] for row in json.loads(out)] == ["local", "box"]
    assert "quiet" not in err
    assert "box [win] test: failed" in err and "E boom" in err
    for family in {"linux", "osx", "win"} - {here}:
        assert f"no leg ran on {family}" in err


def test_install_hook_makes_every_push_run_the_matrix_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = declare(tmp_path / "pkg", step("lint", say("ok")))
    subprocess.run(["git", "init", "-q", os.fspath(package)], check=True)

    with pytest.raises(SystemExit, match="0"):
        build(tmp_path)(["ci", "install-hook", str(package / "src")])

    hook = Path(capsys.readouterr().out.strip())
    assert hook.name == "pre-push"
    assert hook.read_text(encoding="utf-8").splitlines()[-1] == "exec mainboard ci --matrix"
