import runpy
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from cyclopts import App

from mainboard.dispatch.evidence import RECEIPTS_VAR
from mainboard.dispatch.provenance import listing
from mainboard.dispatch.shared import CLOSURE_VAR, DEFERRED_VAR, FIRST_PARTY_VAR
from mainboard.jobs import call
from mainboard.jobs.beacon import Progress
from mainboard.jobs.closure import Closure
from mainboard.jobs.target import Target

from ..support import Lab

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess
    from pytest_subprocess.fake_popen import FakePopen


def sealed(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    *,
    without: str = "",
    distributions: tuple[str, ...] = Lab.DISTRIBUTIONS,
    environment: Path | None = None,
) -> Path:
    """Stand in the lab with its closure listed and the guard's roster exported, as a job does.

    without: a shipped file left out of the listing, to see the guard refuse it.
    distributions: the import roots the manifest installs editable, `Lab.DISTRIBUTIONS` by
        default.
    environment: the compiled target environment the closure reads, an empty one when None.
    """
    from mainboard.dispatch.provenance import SourceTree

    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=distributions,
        environment=environment or lab.root / Lab.ENVIRONMENT,
    )
    _, rows = SourceTree(lab.root).seal(closure.files, built=closure.built)
    written = lab.root / ".mainboard/closure.tsv"
    written.parent.mkdir(exist_ok=True)
    written.write_text(listing(row for row in rows if row.path != without), encoding="utf-8")
    monkeypatch.chdir(lab.root)
    monkeypatch.setenv(CLOSURE_VAR, str(written))
    monkeypatch.setenv(FIRST_PARTY_VAR, ":".join(closure.first_party))
    monkeypatch.setenv(DEFERRED_VAR, ":".join(closure.deferred))
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
        call.called(3, "three", [])
    assert call.called(lambda: None, "none", []) == 0
    # An application built to hand its answer back rather than exit with it is answered for:
    # nothing is a clean exit, an int is the exit.
    quiet = App(result_action="return_value")
    quiet.default(lambda: None)
    assert call.called(quiet, "quiet", []) == 0
    loud = App(result_action="return_value")
    loud.default(lambda: 5)
    assert call.called(loud, "loud", []) == 5


def test_a_namespace_application_ships_and_loads_its_relative_imports(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    (lab.root / "research/camp/experiments/node/__init__.py").unlink()
    written = sealed(lab, monkeypatch)
    assert "research/camp/experiments/helper/tools.py" in written.read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as exited:
        call.main([f"{Lab.JOB}::app", "--", "--x", "3"])
    assert exited.value.code == 4
    assert "experiments.node.run" in sys.modules


def test_the_runner_refuses_an_empty_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["call"])
    with pytest.raises(SystemExit, match="usage"):
        call.main()
    with pytest.raises(SystemExit, match="usage"):
        call.main(["--", "x"])
    # The module is what `python -m` runs, and it exits with what `main` answers.
    with pytest.raises(SystemExit, match="usage"):
        runpy.run_module("mainboard.jobs.call", run_name="__main__", alter_sys=True)


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


def test_a_deferred_distribution_is_admitted_regardless_of_what_the_closure_shipped(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cutoken's `_native` resolves from the environment, never from a closure ships none of it.

    A pure-Python stand-in masquerades as the extension so the import can actually run: the
    closure defers `ext` on the metadata's word alone and ships nothing of it, and the guard
    steps aside for the name rather than refusing an import the listing was never going to
    carry, the same way it already does for anything that is not first-party at all.
    """
    lab.write("packages/ext/src/ext/__init__.py", "")
    lab.write("packages/ext/src/ext/_native.py", "VALUE = 42\n")
    lab.write(
        "research/camp/experiments/node/run.py",
        # `..helper.tools` stays imported so the node's ancestor packages keep shipping exactly
        # as they do for the plain job; only `ext._native` is new here.
        """from ..helper.tools import tool

import ext._native


def main() -> int:
    tool()
    return ext._native.VALUE
""",
    )
    environment = lab.compiled(
        "camp-ext",
        lab.root / f"{Lab.ENVIRONMENT}/lib/python3.14/site-packages/ext/"
        "_native.cpython-314-x86_64-linux-gnu.so",
    )
    written = sealed(
        lab,
        monkeypatch,
        distributions=(*Lab.DISTRIBUTIONS, "packages/ext/src"),
        environment=environment,
    )
    listed = written.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("packages/ext/src/") for line in listed)
    # The environment already has `ext`, independent of what the closure shipped or put on
    # `PYTHONPATH`; the guard's job is to let that resolution happen, not to answer it.
    monkeypatch.syspath_prepend(str(lab.root / "packages/ext/src"))
    assert call.main([f"{Lab.JOB}::main"]) == 42


def _overrun(process: FakePopen) -> None:
    """A child still running when its deadline passed, as `subprocess.run` reports one."""
    raise subprocess.TimeoutExpired(process.args, 30)


@pytest.mark.parametrize(
    ("outcomes", "code", "ran"),
    [
        pytest.param((0, 0, 0), 0, ["a", "b", "c"], id="every-cell-passes"),
        pytest.param((0, 3, 0), 3, ["a", "b"], id="the-first-failure-stops-the-group"),
        pytest.param((None, 0, 0), 124, ["a"], id="a-cell-past-its-timeout-is-killed"),
    ],
)
@pytest.mark.parametrize("dispatched", [False, True], ids=["at-a-terminal", "dispatched"])
def test_a_fresh_group_runs_every_cell_as_its_own_process_until_one_fails(
    outcomes: tuple[int | None, ...],
    code: int,
    ran: list[str],
    dispatched: bool,
    fp: FakeProcess,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed cell inherits nothing from the one before it, and a broken card stops the lane.

    A cell that never answers is the one outcome with no exit code of its own, so the child
    stands in for it by overrunning the deadline the runner handed `subprocess.run`. In a
    dispatched job the lane is one session to its waiter: every cell declared up front, a cell
    killed at its timeout reported failed, and the session ended on the lane's own exit code.
    """
    monkeypatch.delenv(RECEIPTS_VAR, raising=False)
    if dispatched:
        monkeypatch.setenv(RECEIPTS_VAR, "/tmp/receipts")
    for identity, returncode in zip("abc", outcomes, strict=True):
        cell = [sys.executable, "-m", "mainboard.jobs.call", f"lane.py::test[{identity}]"]
        if returncode is None:
            fp.register([*cell, "--", "-q"], callback=_overrun)
        else:
            fp.register([*cell, "--", "-q"], returncode=returncode)

    fresh = ["--fresh", "--timeout", "30", "a", "b", "c", "--", "-q"]

    assert call.main(["lane.py::test", "--", *fresh]) == code
    assert [list(command)[3] for command in fp.calls] == [f"lane.py::test[{cell}]" for cell in ran]
    read = Progress.read(capfd.readouterr().out)
    if not dispatched:
        assert read == Progress()
        return
    assert (read.total, read.session) == (3, code)
    killed = code == 124
    assert read.cells == ((("lane.py::test[a]", "failed"),) if killed else ())
