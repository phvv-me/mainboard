# The plugin: hooks and fixtures, and deliberately nothing else.
#
# Registered with `pytest_plugins = ["mainboard.trials.pytest_plugin"]` in a rootdir conftest and
# inert until that conftest implements `pytest_trials_declaration`. Not a `pytest11` entry point:
# that would put `--paid` and `--rerun` into every pytest session on the machine and import before
# pytest-cov starts, reading the subsystem as uncovered. There is no second event system; pytest's
# hook ordering and setup, call and teardown phases are the lifecycle.
#
# Completeness is data-level, not a workflow engine: a lane declares its keys by being collected,
# and a complete lane skips naming the run that satisfied it unless `--rerun`. There is no DAG,
# since a lane needing another's output is a promotion a person makes. Coverage is asked at the
# declared coordinate, because a key names neither machine nor subject: a lane satisfied on one
# card read complete on the next, which would have published one card's rows four times.
#
# The exit code says whether the instrument worked (see `vocabulary`): a trial that settled nothing
# fails, and so does a session ending with a tracked flag off its baseline.
#
# Every name a hook or fixture annotates is imported at runtime, since pluggy and pytest read these
# signatures at registration. Only `Declaration` stays deferred, in a never-evaluated local
# annotation, because importing it pulls a dataframe engine into every pytest session.

import sys
from collections.abc import Generator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import JsonValue

from . import hookspecs
from .adaptive import DRIVERS, driver
from .flags import held
from .lease import Busy
from .lints import findings
from .log import Log
from .session import WORD, Session, Trial, lane_of, params_of
from .stage import Stage
from .vocabulary import Outcome

if TYPE_CHECKING:
    from .coverage import Cell, LaneStatus
    from .declaration import Declaration


# The run this session is working under, minted once and read by every hook and fixture below.
SESSION = pytest.StashKey[Session]()

# Whether one item's call phase passed, so a trial that already failed is never failed twice.
PASSED = pytest.StashKey[bool]()


def pytest_addhooks(pluginmanager: pytest.PytestPluginManager) -> None:
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

    A trial set is order sensitive: a lane leaves the card warm, a cache built and an allocator
    fragmented, so a shuffled suite measures a different machine every run.
    """
    found: Declaration | None = config.hook.pytest_trials_declaration()
    if found is None:
        return
    for name, why in found.markers.items():
        config.addinivalue_line("markers", f"{name}: {why}")
    if config.pluginmanager.hasplugin("randomly"):
        config.option.randomly_reorganize = False
    # Collection identifies cases; it must not archive data, probe hardware, or open receipts.
    if config.option.collectonly:
        return
    config.stash[SESSION] = Session(found)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip what this host cannot run, what nobody asked to pay for, and what is already taken."""
    session = config.stash.get(SESSION, None)
    if session is None:
        return
    root = session.declared.universe.root.resolve()
    mine = [item for item in items if root in Path(str(item.path)).resolve().parents]
    _warmed(mine)
    _unrunnable(session, mine, paid=bool(config.getoption("--paid")))
    if not config.option.collectonly and any("gpu" in item.keywords for item in mine):
        try:
            session.claim()
        except Busy as busy:
            raise pytest.UsageError(str(busy)) from None
    session.lanes = _surveyed(session, mine)
    if not config.getoption("--rerun"):
        _satisfied(session, mine)


def _warmed(items: Sequence[pytest.Item]) -> None:
    """Import the driver of every adaptive lane kind collected, before any trial runs.

    NOT AN OPTIMISATION. A package first imported inside a running test leaves that test's frame,
    and so its fixture values (a claim's loaded checkpoint), reachable for the rest of the process,
    and `Stage` then refuses a run that released everything it owned; that cost 1.33 GB and an
    afternoon to find. It also turns a missing driver into a refusal at collection.
    """
    for kind in DRIVERS:
        if any(kind in item.keywords for item in items):
            driver(kind)


def _unrunnable(session: Session, items: Sequence[pytest.Item], *, paid: bool) -> None:
    """Skip the marked trials this machine cannot take and the ones nobody opened a wallet for."""
    for item in items:
        if "gpu" in item.keywords and not session.card:
            item.add_marker(pytest.mark.skip(reason="no device on this host"))
        if "paid" in item.keywords and not paid:
            item.add_marker(pytest.mark.skip(reason="costs money, pass --paid to opt in"))


def _surveyed(session: Session, items: Sequence[pytest.Item]) -> tuple[LaneStatus, ...]:
    """Every collected lane's completeness at its own cell, in claim then lane order."""
    grids: dict[tuple[str, str, tuple[tuple[str, str], ...]], set[str]] = {}
    cells: dict[tuple[tuple[str, str], ...], Cell] = {}
    for item in items:
        lane, key = lane_of(item)
        node = session.declared.universe.node_of(Path(str(item.path)))
        cell = session.cell(params_of(item, session.declared.universe.axes))
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
    axes = session.declared.universe.axes
    for item in items:
        where = (lane_of(item)[0], session.cell(params_of(item, axes)).key)
        if where in complete:
            reason = f"complete, run {complete[where]} took it; --rerun to force"
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_report_collectionfinish(config: pytest.Config) -> list[str]:
    """One line per lane before anything runs, under a heading naming the machine they cover."""
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
    """Print the consumer's own word for a settled trial, leaving the exit code untouched.

    The LAST word settled is printed: a search lane narrates a row per iteration before settling
    the study, and its first word would hide the outcome the budget was spent to reach.
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


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    """Print what the lints found over every store this run wrote.

    After the session, because each lint asks about a whole store across its runs; never fatal,
    because a gate that could not have failed is a finding, not a broken instrument.
    """
    run = config.stash.get(SESSION, None)
    if run is None or not run.writers:
        return
    universe, vocabulary = run.declared.universe, run.declared.words
    found = [
        finding
        for node in sorted(run.writers)
        for finding in findings(universe.dataset(node), vocabulary)
    ]
    if not found:
        return
    terminalreporter.write_sep("-", f"trials lints, {len(found)} finding(s)")
    for finding in found:
        terminalreporter.write_line(finding.line())


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Close the run, failing the session on whatever refusal the close returns."""
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
    """This session's identity, which names its own directory of receipt fragments."""
    return _declared(request.config).run


@pytest.fixture(autouse=True)
def held_flags(request: pytest.FixtureRequest) -> Iterator[dict[str, JsonValue]]:
    """Hold every tracked knob around this trial and yield the baseline.

    Autouse because a lane that had to remember isolation would one day forget. A session that
    declared no trials gets an empty hold.
    """
    session = request.config.stash.get(SESSION, None)
    if session is None:
        yield {}
        return
    with held(*session.declared.flags) as baseline:
        yield baseline


@pytest.fixture
def trial(request: pytest.FixtureRequest, held_flags: Mapping[str, JsonValue]) -> Iterator[Trial]:
    """This trial's evidence line, derived from the claim folder, the node id and the host.

    Depending on the hold orders it: the receipt is committed while the flags are still held, so
    a restore that fails cannot destroy the evidence of the trial that just ran.
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
def log(trial: Trial, request: pytest.FixtureRequest) -> Iterator[Log]:
    """One inferred experiment context backed by the existing trial lifecycle."""
    bound = Log(trial)
    try:
        yield bound
    finally:
        bound.close(passed=request.node.stash.get(PASSED, False))


@pytest.fixture
def stage(request: pytest.FixtureRequest, trial: Trial) -> Stage:
    """What this claim loads once, dropped the moment collection leaves the claim.

    Depends on `trial`, which opens the claim, so a holding never lands in the previous claim's.
    """
    return _declared(request.config).staged
