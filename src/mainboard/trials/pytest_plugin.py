# THE PLUGIN, WHICH IS HOOKS AND TWO FIXTURES AND DELIBERATELY NOTHING ELSE.
#
# Registered the ordinary way, `pytest_plugins = ["mainboard.trials.pytest_plugin"]` in a rootdir
# conftest, and inert until that conftest also implements `pytest_trials_declaration`. Everything
# below reads the declaration and nothing below knows a single one of the consumer's words.
#
# NOT A `pytest11` ENTRY POINT, on purpose. An always-loaded plugin would put `--paid` and
# `--rerun` into every pytest session on a machine this tool is installed beside, and it would be
# imported before pytest-cov starts its engine, which makes the whole subsystem read as uncovered
# however thoroughly it is tested. One line in the conftest that already implements the hook buys
# both back.
#
# THERE IS NO SECOND EVENT SYSTEM HERE. pytest already defines hook ordering, hook exception
# policy and where each hook sits relative to setup, call and teardown, so a lifecycle of our own
# invention on top of it would be a worse copy of all three. Cross-cutting concerns are hookimpls
# and fixtures, which is what they already are.
#
# THE COMPLETENESS CHECK IS DATA-LEVEL AND NOT A WORKFLOW ENGINE. A lane declares the keys it
# would run BY BEING COLLECTED, so the parametrize decorator is the declaration and nothing is
# retyped. A complete lane skips with the run that satisfied it named, unless `--rerun` says
# otherwise. There is no DAG, no scheduler and no edge between lanes, because a lane needing
# another lane's output is a promotion done by a person.
#
# COVERAGE IS ASKED AT THE DECLARED COORDINATE, which is what a key alone can never answer. A key
# is a parametrize id and names neither machine nor subject, so a lane satisfied on one card read
# complete on the next and skipped without measuring anything, which is how a cross-architecture
# campaign would have published one card's rows four times. A two-model campaign has the same hole
# one axis over. Both are the same fix, and it is configuration rather than code.
#
# THE EXIT CODE SAYS WHETHER THE INSTRUMENT WORKED AND NOTHING ELSE. Whatever word a trial settles
# on, it exits zero, so a dead hypothesis is a result and nobody learns to ignore a red line. A
# trial that settled no receipt at all fails, because that is the instrument breaking. A session
# that ends with a tracked flag off its baseline fails too, for the same reason one layer up.
#
# EVERY NAME A HOOK OR A FIXTURE ANNOTATES IS IMPORTED AT RUNTIME, deliberately, because pluggy
# and pytest both read these signatures when they register them and a deferred annotation is a
# `NameError` at plugin load. Only `Declaration` stays deferred, since it appears in a local
# annotation that is never evaluated, and importing it would pull a dataframe engine into every
# pytest session on a machine this tool is installed beside.

import sys
from collections.abc import Generator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import JsonValue

from . import hookspecs
from .flags import held
from .session import WORD, Session, Trial, lane_of, params_of
from .stage import Stage
from .vocabulary import Outcome

if TYPE_CHECKING:
    from .coverage import Cell, LaneStatus
    from .declaration import Declaration


# The run this session is working under, minted once and read by every hook and fixture below.
SESSION = pytest.StashKey[Session]()

# Whether one item's call phase itself passed, which the settled-nothing check reads at teardown
# so a trial that already failed on its own is never reported failing twice.
PASSED = pytest.StashKey[bool]()


def pytest_addhooks(pluginmanager: pytest.PytestPluginManager) -> None:
    """Teach pytest the one hook a consumer implements to declare its trials."""
    pluginmanager.add_hookspecs(hookspecs)


def pytest_addoption(parser: pytest.Parser) -> None:
    """`--paid` is the only opt-in that spends money, `--rerun` the only one that ignores data."""
    group = parser.getgroup("trials", "receipt-backed experiment trials")
    group.addoption("--paid", action="store_true", default=False, help="run the paid lanes")
    group.addoption(
        "--rerun", action="store_true", default=False, help="run lanes whose data is complete"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Open the run, register the declared markers and hold the collection order still.

    A trial set is ORDER SENSITIVE: a lane leaves the card warm, a cache built and an allocator
    fragmented, and a shuffled suite measures a different machine every run.
    """
    found: Declaration | None = config.hook.pytest_trials_declaration()
    if found is None:
        return
    for name, why in found.markers.items():
        config.addinivalue_line("markers", f"{name}: {why}")
    if config.pluginmanager.hasplugin("randomly"):
        config.option.randomly_reorganize = False
    config.stash[SESSION] = Session(found)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip what this host cannot run, what nobody asked to pay for, and what is already taken."""
    session = config.stash.get(SESSION, None)
    if session is None:
        return
    root = session.declared.universe.root.resolve()
    mine = [item for item in items if root in Path(str(item.path)).resolve().parents]
    _unrunnable(session, mine, paid=bool(config.getoption("--paid")))
    session.lanes = _surveyed(session, mine)
    if not config.getoption("--rerun"):
        _satisfied(session, mine)


def _unrunnable(session: Session, items: Sequence[pytest.Item], *, paid: bool) -> None:
    """Skip the marked trials this machine cannot take and the ones nobody opened a wallet for."""
    for item in items:
        if "gpu" in item.keywords and not session.card:
            item.add_marker(pytest.mark.skip(reason="no device on this host"))
        if "paid" in item.keywords and not paid:
            item.add_marker(pytest.mark.skip(reason="costs money, pass --paid to opt in"))


def _surveyed(session: Session, items: Sequence[pytest.Item]) -> tuple[LaneStatus, ...]:
    """Every collected lane's completeness at its own cell, in claim then lane order.

    A lane declares the keys it would run BY BEING COLLECTED, so the parametrize decorator is the
    declaration and nothing about the grid is retyped anywhere.
    """
    grids: dict[tuple[str, str, tuple[tuple[str, str], ...]], set[str]] = {}
    cells: dict[tuple[tuple[str, str], ...], Cell] = {}
    for item in items:
        lane, key = lane_of(item)
        node = session.declared.universe.node_of(Path(str(item.path)))
        cell = session.cell(params_of(item))
        cells[cell.key] = cell
        grids.setdefault((node, lane, cell.key), set()).add(key)
    return tuple(
        session.declared.universe.dataset(node).status(lane, keys, cells[where])
        for (node, lane, where), keys in sorted(grids.items())
    )


def _satisfied(session: Session, items: Sequence[pytest.Item]) -> None:
    """Skip every trial whose data a previous run already took, naming the run that took it."""
    complete = {
        (status.lane, status.cell.key): status.run
        for status in session.lanes
        if status.state == "complete"
    }
    for item in items:
        where = (lane_of(item)[0], session.cell(params_of(item)).key)
        if where in complete:
            taken = complete[where]
            item.add_marker(
                pytest.mark.skip(reason=f"complete, run {taken} took it; --rerun to force")
            )


def pytest_report_collectionfinish(config: pytest.Config) -> list[str]:
    """One line per lane before anything runs, so a session opens by saying what it already has.

    The heading NAMES THE MACHINE, because coverage is scoped to it and a `complete` that did not
    say which one satisfied it is the line that would let a campaign skip three architectures.
    """
    session = config.stash.get(SESSION, None)
    if session is None or not session.lanes:
        return []
    return [session.heading, *(status.line() for status in session.lanes)]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Record whether the call phase itself passed, which a settled-nothing check then reads."""
    report = yield
    if report.when == "call":
        item.stash[PASSED] = report.passed
    return report


def pytest_report_teststatus(
    report: pytest.TestReport, config: pytest.Config
) -> tuple[str, str, tuple[str, dict[str, bool]]] | None:
    """Print the consumer's own word for a settled trial rather than a green PASSED saying less.

    The exit code is untouched, so a dead hypothesis still exits zero and only a trial that could
    not be taken at all exits nonzero.

    THE LAST WORD IS THE ONE PRINTED, because a trial may settle several rows. A search lane
    narrates one row per ask-tell iteration and then settles the study, so reading the first word
    would print how its opening point scored and hide the outcome the whole budget was spent to
    reach. A lane that settles once is unaffected, since its first word is also its last.
    """
    session = config.stash.get(SESSION, None)
    if session is None or report.when != "call" or not report.passed:
        return None
    settled = next(
        (str(value) for name, value in reversed(report.user_properties) if name == WORD), ""
    )
    if settled not in session.declared.words:
        return None
    word = session.declared.words[settled]
    return settled, word.mark, (settled.upper(), word.markup)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Compact every store this run wrote, then refuse a run that left a tracked flag moved."""
    run = session.config.stash.get(SESSION, None)
    if run is None:
        return
    refusal = run.close()
    if refusal:
        sys.stderr.write(refusal + "\n")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _declared(config: pytest.Config) -> Session:
    """This session's run, refusing clearly where the workspace declared no trials at all."""
    found = config.stash.get(SESSION, None)
    if found is None:
        raise pytest.UsageError(
            "this session declared no trials, so there is nothing to write a receipt into. "
            "Implement `pytest_trials_declaration` in the rootdir conftest, returning one "
            "`mainboard.trials.Declaration`"
        )
    return found


@pytest.fixture(scope="session")
def run(request: pytest.FixtureRequest) -> str:
    """This session's identity, which NAMES ITS OWN DIRECTORY of receipt fragments."""
    return _declared(request.config).run


@pytest.fixture(autouse=True)
def held_flags(request: pytest.FixtureRequest) -> Iterator[dict[str, JsonValue]]:
    """Hold every tracked knob around this trial, writing each one back on the way out.

    Autouse because a lane that had to remember to ask for isolation would one day forget, and
    forgetting is the whole defect: a knob one lane moved and left moved is measured by every
    lane collected after it, and both readings look perfectly reasonable. Yields the baseline,
    so a lane deliberately sweeping both sides of a knob can say what it started from.

    A session that declared no trials gets an empty hold, since this fixture runs in every
    pytest session the tool is installed beside and has nothing to say to most of them.
    """
    session = request.config.stash.get(SESSION, None)
    if session is None:
        yield {}
        return
    with held(*session.declared.flags) as baseline:
        yield baseline


@pytest.fixture
def trial(request: pytest.FixtureRequest, held_flags: Mapping[str, JsonValue]) -> Iterator[Trial]:
    """This trial's evidence line, derived from the claim folder, the node id, the host and git.

    Depending on the hold is the whole of the ordering guarantee: the flags are taken before this
    fixture is built and released after it is torn down, so the receipt is COMMITTED WHILE THE
    STATE IS STILL HELD and a restore that itself fails cannot destroy the evidence of the trial
    that just ran.
    """
    written = _declared(request.config).trial(request.node)
    yield written
    if written.settled:
        return
    written.record(
        "",
        reason="the trial settled no receipt, so the instrument is what failed",
        measured={},
        outcome=Outcome.FAILED,
    )
    if request.node.stash.get(PASSED, False):
        pytest.fail("this trial passed without settling a receipt, so it measured nothing")


@pytest.fixture
def stage(request: pytest.FixtureRequest, trial: Trial) -> Stage:
    """What this claim loads once, dropped the moment collection leaves the claim.

    Depends on the evidence line because that is what opens the claim, so a holding made here can
    never land in the previous claim's stage.
    """
    return _declared(request.config).staged
