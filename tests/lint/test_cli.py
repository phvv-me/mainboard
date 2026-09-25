import json

import pytest

from mainboard.cli import build

from .conftest import Repository, tool

_FLAG = f"""
[lint.tools.flag]
check = "{tool("flag bad {files}")}"
files = ["*.py"]
"""


@pytest.fixture
def workspace(repository: Repository, monkeypatch: pytest.MonkeyPatch) -> Repository:
    """The repository declaring one check, entered as the working directory."""
    repository.manifest(_FLAG)
    repository.commit()
    monkeypatch.chdir(repository.root)
    return repository


def test_bare_lint_reads_the_changed_files_and_fails_on_a_rewrite_until_it_is_clean(
    workspace: Repository, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.write("notes.md", "trailing   \n")

    with pytest.raises(SystemExit, match="1"):
        build(workspace.root)(["lint"])
    assert capsys.readouterr().out == "lint: 1 files, rewrote 1 (notes.md), failed: none\n"

    with pytest.raises(SystemExit, match="0"):
        build(workspace.root)(["lint"])


def test_a_path_widens_the_pass_and_prints_each_finding_before_the_summary(
    workspace: Repository, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.write("pkg/one.py", "x = 'bad'\n")
    workspace.write("pkg/two.py", "y = 1\n")
    workspace.commit()

    with pytest.raises(SystemExit, match="1"):
        build(workspace.root)(["lint", "pkg"])

    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("flag [.] exited 1")
    assert printed[1:] == ["pkg/one.py", "lint: 2 files, rewrote 0, failed: flag"]


def test_a_json_check_leaves_the_files_alone_and_prints_the_whole_report(
    workspace: Repository, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.write("notes.md", "trailing   \n")
    workspace.write("one.py", "x = 'bad'\n")

    with pytest.raises(SystemExit, match="1"):
        build(workspace.root)(["lint", "--check", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert (workspace.root / "notes.md").read_text(encoding="utf-8") == "trailing   \n"
    assert report["files"] == 2
    assert report["rewritten"] == []
    assert [(failure["step"], failure["output"].strip()) for failure in report["failures"]] == [
        ("text", "notes.md: needs repair: trailing whitespace"),
        ("flag", "one.py"),
    ]


def test_only_runs_the_named_steps(
    workspace: Repository, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.write("one.py", "x = 'bad'   \n")

    with pytest.raises(SystemExit, match="1"):
        build(workspace.root)(["lint", "--only", "text"])

    assert capsys.readouterr().out == "lint: 1 files, rewrote 1 (one.py), failed: none\n"
