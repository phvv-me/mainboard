"""What a lint pass reads and what it answers for.

A pass reads the repositories the workspace owns and nothing pinned from elsewhere, and on
2026-10-07 `mb lint .` recursed forever into a submodule never initialized, which git lists as
the directory itself. A check that reads its whole owner (ty, pyrefly, mcmr) answers for the
files the pass reads and no others, so linting a few files is red only for those files.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mainboard.analysis.apply import Plan
from mainboard.analysis.edits import Delete
from mainboard.analysis.findings import Span
from mainboard.core.project import Project
from mainboard.lint import Inventory, Linter
from mainboard.lint.process import TIMED_OUT, Outcome
from mainboard.manifest.loading import load
from mainboard.manifest.schema.git import GitPolicy
from mainboard.manifest.schema.lint import LintTool

from .conftest import committed, git


def test_a_pass_reads_owned_repositories_and_skips_an_uninitialized_one(
    tracked: Path, tmp_path_factory
) -> None:
    # Beside the root's own remote is owned; anywhere else is someone else's.
    remote = Path(git(tracked, "remote", "get-url", "origin").stdout.strip())
    owned = committed(remote.parent / "owned")
    absent = committed(remote.parent / "absent")
    foreign = committed(tmp_path_factory.mktemp("elsewhere") / "foreign")
    for repository, place in ((owned, "owned"), (absent, "absent"), (foreign, "foreign")):
        git(tracked, "submodule", "add", "--quiet", repository.as_posix(), place)
    git(tracked, "commit", "--quiet", "-m", "three submodules")
    git(tracked, "submodule", "deinit", "--quiet", "--force", "absent")

    listed = Inventory(tracked, GitPolicy()).under([tracked])

    names = {path.relative_to(tracked).as_posix() for path in listed}
    assert "owned/README" in names
    assert not any(name.startswith(("absent", "foreign/")) for name in names)


def _outcome(output: str, *, owner: str = ".", code: int = 1) -> Outcome:
    return Outcome(step="ty", owner=owner, code=code, seconds=0.1, output=output)


def test_a_whole_owner_check_answers_for_the_files_the_pass_reads(workspace: Path) -> None:
    linter = Linter(workspace, load(Project().manifest(workspace)))
    printed = "a.py:3:1: error[one] first\nsrc/b.py:9:2: error[two] second\nFound 2 diagnostics"

    narrowed = linter._narrowed(_outcome(printed), frozenset({"a.py"}))
    elsewhere = linter._narrowed(_outcome(printed), frozenset({"c.py"}))
    owned = linter._narrowed(_outcome(printed, owner="pkg"), frozenset({"pkg/src/b.py"}))

    assert narrowed.failed and narrowed.output == "a.py:3:1: error[one] first"
    assert not elsewhere.failed
    assert owned.output == "src/b.py:9:2: error[two] second"


def test_a_check_that_broke_or_timed_out_keeps_its_failure(workspace: Path) -> None:
    linter = Linter(workspace, load(Project().manifest(workspace)))
    broken = _outcome("ty: failed to read the configuration")
    late = _outcome("b.py:1:1: error[x] y\nty exceeded its 300s deadline", code=TIMED_OUT)

    assert linter._narrowed(broken, frozenset({"a.py"})).failed
    assert linter._narrowed(late, frozenset({"a.py"})).failed


def test_a_checker_names_its_path_only_where_its_own_configuration_is_absent(
    workspace: Path,
) -> None:
    (workspace / "mb.toml").write_text(
        '[workspace]\nname = "it"\n\n[lint.tools.checker]\ncheck = "checker run"\n'
        'files = ["*.py"]\npath-when-unconfigured = "."\n',
        encoding="utf-8",
    )
    for owner, pyproject in {"configured": "[tool.checker]\n", "bare": "[tool.other]\n"}.items():
        (workspace / owner).mkdir()
        (workspace / owner / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        (workspace / owner / "module.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "loose.py").write_text("x = 1\n", encoding="utf-8")
    linter = Linter(workspace, load(Project().manifest(workspace)))

    files = [
        workspace / "configured/module.py",
        workspace / "bare/module.py",
        workspace / "loose.py",
    ]
    argv = {job.owner: job.argv for job in linter._invocations("checker", files)}

    assert argv == {
        "configured": ("checker", "run"),
        "bare": ("checker", "run", "."),
        ".": ("checker", "run", "."),
    }


def test_no_step_rewrites_a_file_a_registration_sealed(tracked: Path) -> None:
    sealed = tracked / "experiments" / "registered"
    drafted = tracked / "experiments" / "drafted"
    sealed.mkdir(parents=True)
    drafted.mkdir()
    node = "---\nregistration_sha256: {}\n---\n# Node\n"
    (sealed / "node.md").write_text(node.format("ab" * 32), encoding="utf-8")
    (drafted / "node.md").write_text(node.format("0" * 64), encoding="utf-8")
    pins = sealed / "pins.json"
    pins.write_bytes(b'{"shapes":{}}')
    (sealed / "nested").mkdir()
    deeper = sealed / "nested" / "notes.txt"
    deeper.write_bytes(b"trailing   \n")
    open_draft = drafted / "pins.json"
    open_draft.write_bytes(b'{"shapes":{}}')
    linter = Linter(tracked, load(Project().manifest(tracked)))

    report = linter.lint([pins, deeper, open_draft])

    assert pins.read_bytes() == b'{"shapes":{}}'
    assert deeper.read_bytes() == b"trailing   \n"
    assert open_draft.read_bytes() == b'{"shapes":{}}\n'
    assert report.rewritten == ("experiments/drafted/pins.json",)


def test_an_analysis_fix_waits_when_a_sealed_file_is_among_its_edits(tmp_path: Path) -> None:
    sealed = tmp_path / "node"
    sealed.mkdir()
    (sealed / "node.md").write_text(
        f"---\nregistration_sha256: {'cd' * 32}\n---\n# Node\n", encoding="utf-8"
    )
    (sealed / "protocol.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "free.py").write_text("x = 1\n", encoding="utf-8")
    plan = Plan(tmp_path)

    assert not plan.add([_deletion("node/protocol.py")])
    assert plan.add([_deletion("free.py")])
    assert plan.deferred == 1


def _deletion(path: str) -> Delete:
    return Delete(span=Span(path=path, start_line=1, end_line=1))


def test_a_fix_that_names_no_files_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="must take"):
        LintTool(check="fmt --check {files}", fix="fmt .", files=("*.py",))
