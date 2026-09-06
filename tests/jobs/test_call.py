import sys
from pathlib import Path

import pytest

from mainboard.dispatch.provenance import listing
from mainboard.dispatch.shared import CLOSURE_VAR, FIRST_PARTY_VAR
from mainboard.jobs import call
from mainboard.jobs.closure import Closure
from mainboard.jobs.target import Target

from ..support import Lab


def sealed(lab: Lab, monkeypatch: pytest.MonkeyPatch, *, without: str = "") -> Path:
    """Stand in the lab with its closure listed and the guard's roster exported, as a job does.

    without: a shipped file left out of the listing, to see the guard refuse it.
    """
    from mainboard.dispatch.provenance import Repositories

    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    closure = Closure.of(target, root=lab.root, distributions=Lab.DISTRIBUTIONS)
    _, rows = Repositories(lab.root).seal(closure.owner, closure.files)
    written = lab.root / ".mainboard/closure.tsv"
    written.parent.mkdir(exist_ok=True)
    written.write_text(listing(row for row in rows if row.path != without), encoding="utf-8")
    monkeypatch.chdir(lab.root)
    monkeypatch.setenv(CLOSURE_VAR, str(written))
    monkeypatch.setenv(FIRST_PARTY_VAR, ":".join(closure.first_party))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    # What `PYTHONPATH` carries for a job: every import root of the closure, the node's first.
    for place in reversed(closure.roots):
        monkeypatch.syspath_prepend(str(lab.root / place))
    return written


def test_an_application_gets_the_arguments_and_exits_with_what_it_answered(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tool() * x + THING` is `1 * 3 + 1`, and the application's exit is the job's."""
    sealed(lab, monkeypatch)
    with pytest.raises(SystemExit) as exited:
        call.main([f"{Lab.JOB}::app", "--", "--x", "3"])
    assert exited.value.code == 4
    assert "experiments.node.run" in sys.modules


def test_a_function_gets_no_arguments_and_answers_its_return(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed(lab, monkeypatch)
    assert call.main([f"{Lab.JOB}::plain"]) == 7
    assert call.main([f"{Lab.JOB}::plain", "--"]) == 7
    with pytest.raises(SystemExit, match="takes no arguments"):
        call.main([f"{Lab.JOB}::plain", "--", "--x"])
    with pytest.raises(SystemExit, match="neither an application nor a function"):
        call.main([f"{Lab.JOB}::THING_LESS"]) if False else call.called(3, "three", [])
    assert call.called(lambda: None, "none", []) == 0


def test_the_runner_refuses_an_empty_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["call"])
    with pytest.raises(SystemExit, match="usage"):
        call.main()
    with pytest.raises(SystemExit, match="usage"):
        call.main(["--", "x"])


def test_a_bare_script_is_imported_from_its_own_directory(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = lab.write("research/camp/lone_script.py", "def main() -> int:\n    return 11\n")
    monkeypatch.chdir(lab.root)
    monkeypatch.delenv(CLOSURE_VAR, raising=False)
    try:
        assert call.main([f"{script.relative_to(lab.root)}::main"]) == 11
    finally:
        sys.modules.pop("lone_script", None)


def test_a_first_party_import_the_closure_left_out_is_refused_never_read_from_elsewhere(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module is right there on disk, one `sys.path` entry away, and that is the fault."""
    sealed(lab, monkeypatch, without="packages/core/src/core/util.py")
    with pytest.raises(ModuleNotFoundError, match="core.util is first-party code outside"):
        call.main([f"{Lab.JOB}::plain"])
    # Third-party names are never the guard's business, and a name it never heard of passes.
    guard = next(finder for finder in sys.meta_path if isinstance(finder, call.Guard))
    assert guard.find_spec("cyclopts") is None
    assert guard.find_spec("json") is None


def test_the_guard_answers_portions_and_paths_outside_the_tree_by_the_listing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib.machinery import ModuleSpec

    sealed(lab, monkeypatch)
    guard = call.Guard(["core"], ["packages/core/src/core/util.py"], lab.root)
    assert guard.holds(ModuleSpec("core", None, origin=None))
    assert guard.holds(
        ModuleSpec("core.util", None, origin=str(lab.root / "packages/core/src/core/util.py"))
    )
    assert not guard.holds(ModuleSpec("core.util", None, origin="/elsewhere/core/util.py"))
    assert call.Guard.armed(lab.root) is not None
    monkeypatch.delenv(CLOSURE_VAR)
    assert call.Guard.armed(lab.root) is None
