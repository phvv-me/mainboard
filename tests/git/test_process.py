import shutil
import subprocess
from pathlib import Path

import pytest

from mainboard.core.errors import MissionError
from mainboard.engines.compile.backend.result import CommandResult
from mainboard.git import process
from mainboard.git.process import Git, said

_WHICH = shutil.which


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (CommandResult(1, "", "one\ntwo\n"), "two"),
        (
            CommandResult(1, "", "remote: GH006: Protected\nremote: \nerror: failed to push\n"),
            "GH006: Protected",
        ),
        (CommandResult(7, "", ""), "exit 7"),
    ],
)
def test_git_is_quoted_by_the_remotes_last_word_else_its_own_last_line(
    result: CommandResult, expected: str
) -> None:
    assert said(result) == expected


def test_a_query_that_has_to_succeed_is_refused_with_gits_own_words(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="not a git repository"):
        Git(tmp_path).out("rev-parse", "HEAD")


def test_no_git_on_path_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process._executable.cache_clear()
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(MissionError, match="git is not on PATH"):
        Git(tmp_path).run("status")
    process._executable.cache_clear()


@pytest.mark.parametrize("gh", [None, "/opt/gh tools/bin/gh"])
def test_a_network_call_offers_the_gh_login_after_the_machines_own_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gh: str | None
) -> None:
    """The helper is appended, so git still asks whatever this machine configures first."""
    monkeypatch.setattr(shutil, "which", lambda name: gh if name == "gh" else _WHICH(name))
    Git(tmp_path).out("init", "-q")
    helpers = Git(tmp_path).run(
        "config", "--get-all", "credential.https://github.com.helper", network=True
    )
    expected = "" if gh is None else '!"/opt/gh tools/bin/gh" auth git-credential\n'
    assert helpers.stdout == expected


def test_a_network_call_that_never_answers_is_a_timed_out_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def hang(argv: list[str], **options: float) -> None:
        raise subprocess.TimeoutExpired(argv, options["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    result = Git(tmp_path).run("fetch", network=True)
    assert (result.returncode, result.stderr) == (124, "git fetch gave no answer in 600 s")
