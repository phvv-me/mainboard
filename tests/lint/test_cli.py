import io
import json

import pytest

from mainboard.cli import build

from .conftest import Repository, tool

_FLAG = f"""
[lint.tools.flag]
run = "{tool("flag bad {files}")}"
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


def test_the_commit_entry_reads_only_what_is_staged_and_says_to_restage_a_rewrite(
    workspace: Repository, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.write("staged.txt", "a \n")
    workspace.write("loose.py", "x = 'bad'\n")
    workspace.git("add", "staged.txt")

    with pytest.raises(SystemExit, match="1"):
        build(workspace.root)(["lint", "commit"])

    assert capsys.readouterr().out.splitlines() == [
        "lint: 1 files, rewrote 1 (staged.txt), failed: none",
        "stage the rewritten files and commit again",
    ]


@pytest.mark.parametrize(
    ("name", "content", "answered"),
    [
        ("dirty.py", "x = 'bad'   \n", True),
        ("clean.py", "x = 1   \n", False),
        ("../outside.py", "x = 'bad'\n", False),
        ("", "", False),
    ],
    ids=[
        "a finding rides back as context",
        "a repair alone is silent",
        "a file under no workspace is left alone",
        "a tool that wrote no file",
    ],
)
def test_the_edit_entry_repairs_silently_and_answers_only_with_findings(
    workspace: Repository,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    content: str,
    answered: bool,
) -> None:
    edited = workspace.root / name
    if name:
        edited.write_text(content, encoding="utf-8")
    payload = {"cwd": str(workspace.root), "tool_input": {"file_path": name}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    with pytest.raises(SystemExit, match="0"):
        build(workspace.root)(["lint", "edit"])

    out = capsys.readouterr().out
    if name and name != "../outside.py":
        assert edited.read_text(encoding="utf-8") == content.replace("   \n", "\n")
    if answered:
        context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        assert context.startswith("flag [.] exited 1") and name in context
    else:
        assert out == ""


def test_install_hook_prints_where_the_hook_landed(
    workspace: Repository, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(workspace.root)(["lint", "install-hook"])

    assert capsys.readouterr().out.strip().endswith("pre-commit")
