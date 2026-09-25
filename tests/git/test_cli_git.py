import json

import pytest

from mainboard.cli import build

from .conftest import Workspace


def _run(workspace: Workspace, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    """Run one `git` verb through the CLI, answering its exit code and what it printed."""
    with pytest.raises(SystemExit) as exited:
        build(workspace.path)(["center", "git", *argv])
    return int(exited.value.code or 0), capsys.readouterr().out


def test_status_prints_one_row_per_owned_repository(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(workspace, capsys, "status", "--json")
    assert code == 0
    assert [row["repo"] for row in json.loads(out)] == [".", "packages/lib"]
    code, out = _run(workspace, capsys, "status", "--agent", "--fields", "repo,branch")
    assert out.splitlines()[0] == "repo\tbranch"


def test_commit_and_push_read_the_workspace_manifest_and_exit_on_what_settled(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The manifest's 0.001 MB ceiling withholds the heavy file, and a held repo exits 1."""
    (workspace.lib / "lib.txt").write_text("edited\n", encoding="utf-8")
    (workspace.lib / "heavy.dat").write_bytes(b"x" * 2048)

    code, out = _run(workspace, capsys, "commit", "-m", "Edit", "--json")
    assert code == 0
    lib, root = json.loads(out)
    assert (lib["outcome"], root["outcome"]) == ("done", "done")
    assert "withheld heavy.dat" in lib["detail"]

    code, out = _run(workspace, capsys, "push", "--json")
    assert code == 0
    assert [step["outcome"] for step in json.loads(out)] == ["done", "done"]

    code, out = _run(workspace, capsys, "pull", "--json")
    assert code == 0
    workspace.forge.commit(workspace.lib, "local", {})
    workspace.git(workspace.lib, "switch", "-q", "--detach", "HEAD")
    workspace.forge.commit(workspace.lib, "detached", {"x.txt": "x\n"})
    code, _ = _run(workspace, capsys, "push")
    assert code == 1


def test_check_exits_one_only_on_a_failure(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(workspace, capsys, "check", "--json")
    assert code == 0
    assert [row["verdict"] for row in json.loads(out)] == ["warn"]
    workspace.forge.commit(workspace.path, "heavy", {"heavy.txt": "x" * 2048})
    code, _ = _run(workspace, capsys, "check")
    assert code == 1
