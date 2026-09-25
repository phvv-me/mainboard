import json
import os
from pathlib import Path

import pytest
from pydantic import JsonValue

from mainboard import MissionError
from mainboard.lint import EditHook, GitHook

from .conftest import Repository


@pytest.mark.parametrize("hooks_path", ["", "tools/hooks"], ids=["git's own", "core.hooksPath"])
def test_the_hook_lands_where_git_reads_it_and_calls_lint_commit(
    repository: Repository, hooks_path: str
) -> None:
    if hooks_path:
        repository.git("config", "core.hooksPath", hooks_path)

    path = GitHook(repository.root, hook="pre-commit", command="lint commit").install()

    expected = repository.root / (hooks_path or ".git/hooks") / "pre-commit"
    assert path.resolve() == expected.resolve()
    assert path.read_bytes().splitlines()[-1] == b"exec mainboard lint commit"
    assert path.read_bytes().startswith(b"#!/bin/sh\n")
    assert os.name == "nt" or os.access(path, os.X_OK)


def test_a_hook_needs_a_git_work_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

    with pytest.raises(MissionError, match="not in a git work tree"):
        GitHook(tmp_path, hook="pre-commit", command="lint commit").install()


@pytest.mark.parametrize(
    ("payload", "found"),
    [
        ({"cwd": "{root}", "tool_input": {"file_path": "a.py"}}, "a.py"),
        ({"tool_input": {"file_path": "{root}/a.py"}, "extra": 1}, "a.py"),
        ({"cwd": "{root}", "tool_input": {"notebook_path": "n.ipynb"}}, "n.ipynb"),
        ({"cwd": "{root}", "tool_input": {"file_path": "gone.py"}}, None),
        ({"cwd": "{root}", "tool_input": {"command": "ls"}}, None),
        ({}, None),
    ],
    ids=[
        "a relative path read against the session",
        "an absolute path, unknown keys ignored",
        "a notebook",
        "a file already gone",
        "a tool that names no file",
        "an empty payload",
    ],
)
def test_the_edit_payload_names_the_one_file_just_written(
    tmp_path: Path, payload: dict[str, JsonValue], found: str | None
) -> None:
    for name in ("a.py", "n.ipynb"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    spelled = json.dumps(payload).replace("{root}", tmp_path.as_posix())

    edited = EditHook.model_validate_json(spelled).edited

    assert edited == (None if found is None else (tmp_path / found).resolve())


def test_the_edit_answer_hands_bounded_findings_to_the_agent() -> None:
    answer = json.loads(EditHook.context("E" * 5000))

    assert answer["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert answer["hookSpecificOutput"]["additionalContext"] == "E" * 4000
