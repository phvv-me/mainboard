"""`check` ends on one line: its findings and the seconds each stage of the run took. It lists the
project once, whoever asks, from the rules it configures to the files it records."""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from mainboard.analysis.engine.lint import Selection, check
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger

_ONE_LISTING = ["ls-files", "ls-files", "check-attr"]
_SECONDS = r"\d+\.\d+"
_SUMMARY = re.compile(
    rf"1 findings; facts {_SECONDS} s \(catalog {_SECONDS}, parse {_SECONDS}, "
    rf"derive {_SECONDS}, install {_SECONDS}\), rules {_SECONDS} s"
)


def test_a_check_ends_with_the_seconds_of_its_catalog_parse_derive_install_and_rules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.mcmr]\n", encoding="utf-8")
    (tmp_path / "printed.py").write_text('print("one")\n', encoding="utf-8")

    check(tmp_path, selection=Selection(rule=(TextAroundTheLogger.id,)))

    assert _SUMMARY.fullmatch(capsys.readouterr().out.splitlines()[-1])


def test_a_check_asks_git_for_the_files_of_the_project_once_from_start_to_end(
    repository: Path, git_verbs: Callable[[], list[str]]
) -> None:
    check(repository, selection=Selection(rule=(TextAroundTheLogger.id,)))

    assert git_verbs() == _ONE_LISTING
