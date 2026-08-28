from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from mainboard.trials import Dataset, Declaration, pytest_plugin
from mainboard.trials import session as session_module

from .support import PROBED, declaration


class Hooked:
    """The one hook call `pytest_configure` makes, answering whatever the test declared."""

    def __init__(self, declared: Declaration | None) -> None:
        self.declared = declared

    def pytest_trials_declaration(self) -> Declaration | None:
        return self.declared


class Plugins:
    """A plugin manager that only has to say which plugins are present."""

    def __init__(self, present: Sequence[str]) -> None:
        self.present = present

    def hasplugin(self, name: str) -> bool:
        return name in self.present


class Configured:
    """A session config as `pytest_configure` reads one, with nothing else attached."""

    def __init__(self, declared: Declaration | None, plugins: Sequence[str] = ()) -> None:
        self.hook = Hooked(declared)
        self.pluginmanager = Plugins(plugins)
        self.option = SimpleNamespace()
        self.stash = pytest.Stash()
        self.registered: list[str] = []

    def addinivalue_line(self, name: str, line: str) -> None:
        self.registered.append(line)


# A consumer's whole conftest: one hook returning one declaration. Everything the reference
# scaffolding spelled out by hand, the markers, the coverage rule, the provenance probe and the
# arithmetic pin, is derived from these four statements.
CONFTEST = """
from pathlib import Path

from mainboard.trials import Declaration, Flag, Universe, Vocabulary, Word
from mainboard.trials.vocabulary import Stance

pytest_plugins = ["mainboard.trials.pytest_plugin"]

HERE = Path(__file__).resolve().parent
STATE = {"policy": "pinned"}


def pytest_trials_declaration() -> Declaration:
    return Declaration(
        universe=Universe(root=HERE, axes=("card", "model"), probed=("polars",)),
        words=Vocabulary(
            words=(
                Word(name="validated", letter="V", stance=Stance.CONFIRMS),
                Word(name="refuted", stance=Stance.REFUTES),
                Word(name="undecided"),
            )
        ),
        flags=(
            Flag(
                name="policy",
                read=lambda: STATE["policy"],
                write=lambda value: STATE.__setitem__("policy", value),
            ),
        ),
    )
"""

LANES = """
import pytest

from conftest import STATE


@pytest.mark.parametrize("model", ["qwen", "llama"])
def test_law_holds(trial, model):
    trial.validated("the law held", ratio=1.5)


def test_law_moves_the_knob(trial):
    STATE["policy"] = "default"
    trial.refuted("it did not survive", seen=STATE["policy"])


def test_law_reads_the_knob(trial):
    trial.undecided("the separation is below the noise floor", seen=STATE["policy"])


@pytest.mark.gpu
def test_needs_a_card(trial):
    trial.validated("measured")


@pytest.mark.paid
def test_costs_money(trial):
    trial.validated("bought")
"""


LEAKS = """
import pytest

from conftest import STATE


@pytest.fixture(scope="session", autouse=True)
def moves_it():
    STATE["policy"] = "default"


def test_reads(trial):
    trial.validated("measured")
"""

QUIET = """
def test_settles_nothing(trial):
    assert True


def test_breaks(trial):
    raise ZeroDivisionError
"""

STAGED = """
LOADS = []


def test_first(trial, stage, run):
    stage.kept("weights", lambda: LOADS.append("x") or "bundle")
    trial.validated("loaded", loads=len(LOADS), run=run)


def test_second(trial, stage, run):
    stage.kept("weights", lambda: LOADS.append("x") or "bundle")
    trial.validated("reused", loads=len(LOADS), run=run)
"""


HUNT = """
import pytest


@pytest.mark.adversarial
def test_a_law_is_hunted(trial):
    trial.validated("nothing broke it")
"""


@pytest.fixture
def universe(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    """A consumer workspace with one claim, its provenance fixed so no test touches silicon."""
    monkeypatch.setattr(session_module, "provenance", lambda *args, **kwargs: dict(PROBED))
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(**{"alpha/test_law": LANES})
    return pytester


def ran(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    """One inner session, with the one warning running pytest inside pytest always raises.

    The outer module imports the plugin to drive its hooks directly, so the inner run finds it
    already imported and says it can no longer rewrite its assertions, which is true and is about
    this suite rather than about anything under test.
    """
    return pytester.runpytest_inprocess("-W", "ignore::pytest.PytestAssertRewriteWarning", *args)


def test_a_collected_adaptive_lane_has_its_driver_imported_before_any_trial_runs(
    universe: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package first imported inside a running test leaves that test's frame reachable.

    The frame holds the fixture values pytest passed it, which for a claim is a loaded
    checkpoint, so the residency check reports a card that never came back and refuses a run that
    released everything it owned. Collection is where that import belongs.
    """
    warmed: list[str] = []
    monkeypatch.setattr(pytest_plugin, "driver", warmed.append)
    universe.makepyfile(**{"alpha/test_hunt": HUNT})

    assert ran(universe, "alpha/test_hunt.py").ret == 0
    assert warmed == ["adversarial"]


def store(pytester: pytest.Pytester, node: str = "alpha") -> Dataset:
    """The claim's receipt store as a reader outside the run sees it."""
    return Dataset(Path(pytester.path) / node / "evidence" / "receipts", axes=("card", "model"))


def test_a_session_with_no_declaration_stays_completely_inert(
    pytester: pytest.Pytester,
) -> None:
    """The plugin loads with pytest itself, so most sessions must never notice it at all.

    The one thing it does say is why, since a lane asking for an evidence line in a workspace
    that declared none has made a mistake nobody else can diagnose for it.
    """
    pytester.makeconftest('pytest_plugins = ["mainboard.trials.pytest_plugin"]')
    pytester.makepyfile(
        plain="def test_ordinary():\n    assert True\n",
        asking="def test_wants_evidence(trial):\n    trial.validated('x')\n",
    )
    quiet = ran(pytester, "plain.py")
    quiet.assert_outcomes(passed=1)
    quiet.stdout.no_fnmatch_line("*evidence on*")

    asked = ran(pytester, "asking.py")
    asked.stdout.fnmatch_lines(["*declared no trials*pytest_trials_declaration*"])


def test_a_declared_run_settles_its_own_words_and_leaves_one_compacted_store(
    universe: pytest.Pytester,
) -> None:
    """A dead hypothesis exits zero, so the colour is the whole difference between the words.

    Nobody learns to ignore a red line that only ever meant a prediction died, which is why the
    exit code is about the instrument and the word is about the claim.
    """
    run = ran(universe, "--paid")
    assert run.ret == 0
    run.stdout.fnmatch_lines(
        [
            "evidence on Test Card (GPU-1111):",
            "*missing*test_law_holds on GPU-1111, llama*",
        ]
    )
    assert run.parseoutcomes() == {"validated": 4, "refuted": 1, "undecided": 1}

    taken = store(universe)
    assert len(taken.parts) == 1
    rows = taken.rows()
    assert sorted(str(row["verdict"]) for row in rows) == [
        "refuted",
        "undecided",
        "validated",
        "validated",
        "validated",
        "validated",
    ]
    assert {str(row["node"]) for row in rows} == {"alpha"}
    assert {str(row["session_policy"]) for row in rows} == {"pinned"}


def test_a_lane_that_moves_a_tracked_knob_never_reaches_the_lane_collected_after_it(
    universe: pytest.Pytester,
) -> None:
    """The pin leak closed at the acquisition end rather than only at the audit end.

    A knob one lane moved and left moved is measured by every lane after it, and both readings
    look perfectly reasonable, which is why detection alone was never the fix.
    """
    run = ran(universe, "--paid")
    assert run.ret == 0
    rows = {str(row["run_id"]): row for row in store(universe).rows()}
    assert rows["test_law_moves_the_knob"]["policy"] == "default"
    assert rows["test_law_moves_the_knob"]["measured"] == {"seen": "default"}
    assert rows["test_law_reads_the_knob"]["policy"] == "pinned"
    assert rows["test_law_reads_the_knob"]["measured"] == {"seen": "pinned"}


def test_a_run_that_ends_with_a_knob_off_baseline_is_refused_and_names_the_leaker(
    universe: pytest.Pytester,
) -> None:
    """Refusal over warning, because a warning in a scroll-back is not a gate.

    The move here is made by a session fixture, which is exactly the shape `held` cannot wrap:
    it is not inside any one trial, so the audit at close is what has to catch it.
    """
    universe.makepyfile(**{"alpha/test_leaks": LEAKS})
    run = ran(universe, "alpha/test_leaks.py")
    assert run.ret == pytest.ExitCode.TESTS_FAILED
    run.stderr.fnmatch_lines(
        [
            "trials REFUSE this session*",
            "*policy: opened at 'pinned', ended at 'default'*alpha/test_leaks.py::test_reads*",
            "*mainboard.trials.held*",
        ]
    )


def test_a_complete_lane_skips_with_the_run_that_satisfied_it_named(
    universe: pytest.Pytester,
) -> None:
    """A lane declares its keys by being collected, so nothing about the grid is retyped.

    Coverage is asked per card and per subject, so a key alone, which names neither, could never
    have answered this question at all.
    """
    assert ran(universe, "--paid").ret == 0
    again = ran(universe, "--paid")
    assert again.parseoutcomes() == {"skipped": 6}
    again.stdout.fnmatch_lines(["*complete*test_law_holds on GPU-1111, qwen*"])

    forced = ran(universe, "--paid", "--rerun")
    assert forced.parseoutcomes() == {"validated": 4, "refuted": 1, "undecided": 1}
    assert len(store(universe).runs) == 2


def test_a_host_with_no_card_and_a_wallet_nobody_opened_both_skip(
    universe: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine cannot measure a device it does not have, and nothing spends money unasked."""
    monkeypatch.setattr(
        session_module,
        "provenance",
        lambda *args, **kwargs: {**PROBED, "card": "", "card_name": "", "card_probed": "absent"},
    )
    run = ran(universe)
    assert run.parseoutcomes() == {
        "skipped": 2,
        "validated": 2,
        "refuted": 1,
        "undecided": 1,
    }
    run.stdout.fnmatch_lines(["evidence on no card:"])
    assert not any(str(row["card"]) for row in store(universe).rows())


def test_a_trial_that_measured_nothing_fails_and_still_leaves_a_row(
    universe: pytest.Pytester,
) -> None:
    """A broken instrument is the one outcome that must never be silent.

    A trial that raised has already reported itself, so it writes its row and is not failed
    twice; a trial that passed while settling nothing is the case only this check can see.
    """
    universe.makepyfile(**{"alpha/test_quiet": QUIET})
    run = ran(universe, "alpha/test_quiet.py")
    assert run.ret == pytest.ExitCode.TESTS_FAILED
    assert run.parseoutcomes() == {"passed": 1, "failed": 1, "errors": 1}

    rows = {str(row["run_id"]): row for row in store(universe).rows()}
    assert set(rows) == {"test_settles_nothing", "test_breaks"}
    assert all(row["outcome"] == "failed" for row in rows.values())
    assert all(not row["verdict"] for row in rows.values())


def test_a_claim_holds_its_measure_once_work_and_the_run_names_itself(
    universe: pytest.Pytester,
) -> None:
    """The stage is what a session-scoped fixture should have been, scoped to one claim.

    The run id is the same string for every trial of a session, because it is the directory the
    fragments land in and a second one would split a run in two.
    """
    universe.makepyfile(**{"alpha/test_stage": STAGED})
    assert ran(universe, "alpha/test_stage.py").ret == 0
    rows = {str(row["run_id"]): row for row in store(universe).rows()}
    assert rows["test_first"]["measured"]["loads"] == 1
    assert rows["test_second"]["measured"]["loads"] == 1
    assert rows["test_first"]["measured"]["run"] == rows["test_second"]["measured"]["run"]


def test_a_shuffling_plugin_is_held_still_and_the_declared_markers_are_registered(
    tmp_path: Path, probed: None
) -> None:
    """A trial set is order sensitive: a lane leaves the card warm and the allocator fragmented.

    A shuffled suite therefore measures a different machine every run, so the guard is set the
    moment such a plugin is present rather than the day somebody debugs the result. Driven
    against a stand-in config because the guard has to fire whatever order the real plugin
    happened to register in, which is a fact about this hook and not about that plugin.
    """
    config = Configured(declaration(tmp_path), plugins=("randomly",))
    pytest_plugin.pytest_configure(cast("pytest.Config", config))
    assert config.option.randomly_reorganize is False
    assert config.registered == [
        "gpu: needs a real card, skipped where there is none",
        "slow: runs for minutes rather than seconds",
        "paid: could bill money, skipped unless --paid is passed",
        "adversarial: hunts a counterexample by shrinking, so what it finds is a candidate",
        "search: proposes its own points adaptively, so what it finds is a candidate",
    ]
    assert config.stash[pytest_plugin.SESSION].run

    # A suite nothing shuffles still opens its run, and the guard is not invented on a config
    # that carries no such option to hold.
    unshuffled = Configured(declaration(tmp_path))
    pytest_plugin.pytest_configure(cast("pytest.Config", unshuffled))
    assert not hasattr(unshuffled.option, "randomly_reorganize")
    assert unshuffled.stash[pytest_plugin.SESSION].run

    bare = cast("pytest.Config", Configured(None))
    pytest_plugin.pytest_configure(bare)
    assert pytest_plugin.SESSION not in bare.stash
    assert pytest_plugin.pytest_report_collectionfinish(bare) == []
