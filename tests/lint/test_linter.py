import os

import pytest

from mainboard import MissionError, Project, load
from mainboard.lint import Inventory, Linter, Report
from mainboard.lint import linter as linter_module
from mainboard.lint.process import Outcome

from .conftest import Repository, tool

# Two formatters that both rewrite Python, and two checks that read it: one per file, one over
# its whole owner. The formatters' order is the declaration's, which the appended words prove,
# and each one's read-only form fails on a file still missing its word.
_TOOLS = f"""
[lint]
exclude = ["vendor/"]
owners = ["pkgs/*"]
max-kb = 1

[lint.tools.first]
check = "{tool("lacks first {files}")}"
fix = "{tool("append first {files}")}"
files = ["*.py"]

[lint.tools.second]
check = "{tool("lacks second {files}")}"
fix = "{tool("append second {files}")}"
files = ["*.py"]
exclude = ["pkgs/b/"]

[lint.tools.flag]
check = "{tool("flag bad {files}")}"
files = ["*.py"]

[lint.tools.whole]
check = "{tool("where {root}")}"
files = ["*.toml"]
"""


@pytest.fixture
def workspace(repository: Repository) -> Repository:
    """The repository declaring the tools above, with one committed file in each of two owners."""
    repository.manifest(_TOOLS)
    repository.write("pkgs/a/pyproject.toml", "[project]\nname = 'a'\n")
    repository.write("pkgs/a/mod.py", "x = 1\n")
    repository.write("pkgs/b/mod.py", "y = 1\n")
    repository.commit()
    return repository


def test_writers_run_in_declared_order_then_checks_read_their_result_per_owner(
    workspace: Repository,
) -> None:
    workspace.write("pkgs/a/mod.py", "x = 'bad'   \r\n")
    workspace.write("pkgs/b/mod.py", "y = 2\n")
    workspace.write("vendor/lib.py", "z = 'bad'  \n")

    report = workspace.linter().lint(Inventory(workspace.root).changed())

    assert (workspace.root / "pkgs/a/mod.py").read_text() == "x = 'bad'\nfirst\nsecond\n"
    assert (workspace.root / "pkgs/b/mod.py").read_text() == "y = 2\nfirst\n"
    assert (workspace.root / "vendor/lib.py").read_text() == "z = 'bad'  \n"
    assert report.rewritten == ("pkgs/a/mod.py", "pkgs/b/mod.py")
    assert [
        (failure.step, failure.owner, failure.output.strip()) for failure in report.failures
    ] == [("flag", "pkgs/a", "mod.py")]
    assert report.files == 2
    assert not report.clean


def test_a_check_writes_nothing_and_fails_with_every_writers_and_the_hygienes_own_words(
    workspace: Repository,
) -> None:
    workspace.write("pkgs/a/mod.py", "x = 'bad'   \r\n")
    workspace.write("pkgs/b/mod.py", "y = 2\nfirst\n")
    workspace.write("notes.md", "fine\n")
    before = {name: (workspace.root / name).read_bytes() for name in ("pkgs/a/mod.py", "notes.md")}

    report = _linter(workspace, check=True).lint(Inventory(workspace.root).changed())

    assert {name: (workspace.root / name).read_bytes() for name in before} == before
    assert report.rewritten == ()
    assert sorted(
        (failure.step, failure.owner, failure.output.strip()) for failure in report.failures
    ) == [
        ("first", "pkgs/a", "mod.py"),
        ("flag", "pkgs/a", "mod.py"),
        ("second", "pkgs/a", "mod.py"),
        ("text", ".", "pkgs/a/mod.py: needs repair: line endings, trailing whitespace"),
    ]


@pytest.mark.parametrize("check", [False, True], ids=["writing", "check"])
def test_only_the_named_steps_run(workspace: Repository, check: bool) -> None:
    workspace.write("pkgs/a/mod.py", "x = 'bad'   \n")

    report = _linter(workspace, check=check, only=["flag"]).lint(
        [workspace.root / "pkgs/a/mod.py"]
    )

    assert [failure.step for failure in report.failures] == ["flag"]
    assert (workspace.root / "pkgs/a/mod.py").read_text() == "x = 'bad'   \n"


def test_a_step_nobody_declared_is_refused_with_the_steps_there_are(workspace: Repository) -> None:
    with pytest.raises(MissionError, match="no lint step 'ruff'; the steps are text, first"):
        _linter(workspace, only=["text", "ruff"])


def test_a_whole_owner_check_wakes_for_a_deleted_file_and_runs_inside_that_owner(
    workspace: Repository,
) -> None:
    (workspace.root / "pkgs/a/pyproject.toml").unlink()
    workspace.write("pkgs/b/extra.toml", "k = 1\n")

    report = workspace.linter().lint(Inventory(workspace.root).changed())

    assert [failure.step for failure in report.failures] == ["whole", "whole"]
    for failure, owner in zip(report.failures, ("a", "b"), strict=True):
        where, root = failure.output.split()
        assert os.path.samefile(where, workspace.root / "pkgs" / owner)
        assert root == str(workspace.root)
    assert report.rewritten == ()


def test_the_text_step_names_what_it_could_not_repair_and_a_new_file_past_the_size_limit(
    workspace: Repository,
) -> None:
    workspace.write("broken.toml", "k = [\n")
    workspace.write("big.txt", "x" * 2048 + "\n")
    workspace.write("old.txt", "y" * 2048 + "\n")
    workspace.git("add", "old.txt")
    workspace.git("commit", "-q", "-m", "old")
    workspace.write("old.txt", "z" * 2048 + "\n")
    workspace.write("image.png", b"\x89PNG\0\0" + b"\0" * 2048)

    report = workspace.linter().lint(Inventory(workspace.root).changed())

    text = next(failure for failure in report.failures if failure.step == "text")
    assert text.output.splitlines() == [
        "big.txt: is 2 KB, above the 1 KB limit",
        "broken.toml: does not parse: Invalid value (at end of document)",
        "image.png: is 2 KB, above the 1 KB limit",
    ]


def test_per_file_commands_split_so_no_command_line_outgrows_the_budget(
    workspace: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(linter_module, "_BATCH", 6)
    names = [f"pkgs/b/m{index}.py" for index in range(3)]
    for name in names:
        workspace.write(name, "ok = 'bad'\n")

    report = workspace.linter().lint([workspace.root / name for name in names])

    flagged = [failure.output.split() for failure in report.failures if failure.step == "flag"]
    assert sorted(flagged) == [["m0.py"], ["m1.py"], ["m2.py"]]


def test_nothing_to_read_runs_nothing_and_is_clean(workspace: Repository) -> None:
    report = workspace.linter().lint([])

    assert report == Report(files=0)
    assert report.clean
    assert report.summary() == "lint: 0 files, rewrote 0, failed: none"


def test_a_per_file_tool_whose_files_are_all_gone_does_not_run(workspace: Repository) -> None:
    (workspace.root / "pkgs/a/mod.py").unlink()

    report = workspace.linter().lint([workspace.root / "pkgs/a/mod.py"])

    assert report.clean


def test_the_report_heads_each_finding_with_its_step_owner_and_exit() -> None:
    report = Report(
        files=3,
        rewritten=("a.py",),
        failures=(Outcome(step="ruff", owner="pkgs/a", code=1, seconds=0.25, output="E1\n"),),
    )

    assert report.findings() == "ruff [pkgs/a] exited 1 after 0.2s\nE1"
    assert report.summary() == "lint: 3 files, rewrote 1 (a.py), failed: ruff"


def _linter(
    workspace: Repository, *, check: bool = False, only: list[str] | None = None
) -> Linter:
    manifest = load(workspace.root / Project().manifest)
    return Linter(workspace.root, manifest, check=check, only=only or ())
