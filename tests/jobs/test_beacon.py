import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.dispatch.evidence import RECEIPTS_VAR, printed
from mainboard.jobs.beacon import CELL, CELLS, NESTED, SESSION, Progress, unbeaconed
from mainboard.jobs.pytest import Beacon, Runner

from ..strategies import WORDS

# A cell id as pytest spells one, parameters and all, spaces included since an id may hold them.
_CELLS = st.tuples(WORDS, WORDS, st.sampled_from(["", " ", "-"])).map(
    lambda parts: f"test_{parts[0]}.py::test[{parts[1]}{parts[2]}x]"
)

# One cell's report: which cell, and the outcome one of its phases decided.
_REPORTS = st.tuples(_CELLS, st.sampled_from(["passed", "failed", "skipped"]))

# What pytest's own progress puts on a line before a marker lands on it.
_DOTS = st.text(alphabet=".FsxE", max_size=4)


@given(
    reports=st.lists(st.tuples(_REPORTS, _DOTS)),
    totals=st.lists(st.integers(0, 50), max_size=2),
    session=st.one_of(st.none(), st.integers(0, 5)),
    noise=st.lists(st.sampled_from(["collected 3 items", "E   assert 2 == 3", "1 passed"])),
)
def test_the_markers_add_up_and_come_back_out_leaving_pytest_s_output_byte_for_byte(
    reports: list[tuple[tuple[str, str], str]],
    totals: list[int],
    session: int | None,
    noise: list[str],
) -> None:
    """The first declared total stands, a cell settles on its worst phase, the last session wins.

    Taking the markers out gives pytest's own output back exactly, dots included, since a marker
    lands in the middle of a line of them and runs only to that line's end.
    """
    said = "".join(f"{line}\n" for line in noise)
    log = "".join(
        [
            *(f"{CELLS} {total}\n" for total in totals),
            *(f"{dots}{CELL} {outcome} {cell}\n" for (cell, outcome), dots in reports),
            said,
            f"{SESSION} {session}\n" if session is not None else "",
        ]
    )
    own = "".join(dots for _, dots in reports) + said

    read = Progress.read(log)

    worst: dict[str, str] = {}
    for (cell, outcome), _ in reports:
        ranks = ("failed", "skipped", "passed")
        held = worst.get(cell, outcome)
        worst[cell] = min(held, outcome, key=ranks.index)
    assert read.total == (totals[0] if totals else None)
    assert read.cells == tuple(worst.items())
    assert read.session == session
    assert read.done == len(worst)
    assert read.failed == sum(outcome == "failed" for outcome in worst.values())
    assert unbeaconed(log) == own
    assert printed(log) == own


@pytest.mark.parametrize(
    ("log", "counted", "total", "session"),
    [
        pytest.param("", "", None, None, id="nothing-reported"),
        pytest.param(f"{CELL} passed a.py::t\n", "1/?", None, None, id="cells-before-a-total"),
        pytest.param(f"{CELLS} 3\n", "0/3", 3, None, id="a-total-before-any-cell"),
        pytest.param(f"{CELLS} x\n{SESSION} torn\n", "", None, None, id="torn-values"),
        pytest.param(f"{CELL} weird a.py::t\n", "1/?", None, None, id="a-word-pytest-never-uses"),
    ],
)
def test_a_log_reports_only_what_its_markers_actually_say(
    log: str, counted: str, total: int | None, session: int | None
) -> None:
    read = Progress.read(log)
    assert (read.counted, read.total, read.session, read.failed) == (counted, total, session, 0)


_LANE = """import pytest


@pytest.fixture
def broken():
    yield
    raise RuntimeError("teardown")


@pytest.mark.parametrize("model", ["gpt2", "qwen3"])
def test_cell(model):
    assert model == "gpt2"


@pytest.mark.skip(reason="not today")
def test_skipped():
    pass


def test_torn_down(broken):
    pass
"""


@pytest.mark.parametrize("nested", [False, True], ids=["a-whole-session", "one-fresh-cell"])
def test_a_real_session_reports_every_cell_through_the_job_s_own_descriptor(
    pytester: pytest.Pytester, capfd: pytest.CaptureFixture[str], nested: bool
) -> None:
    """Whatever pytest is capturing, the markers reach the descriptor the job's log is.

    A call decides a cell that ran, a skipped setup decides one that never did, and a teardown
    that failed turns a passed cell failed. A fresh cell reports only itself, since the lane
    around it declares the total and ends the session.
    """
    pytester.makepyfile(test_lane=_LANE)
    pytester.runpytest_inprocess("-q", "-p", "no:randomly", plugins=[Beacon(nested=nested)])

    read = Progress.read(capfd.readouterr().out)

    assert dict(read.cells) == {
        "test_lane.py::test_cell[gpt2]": "passed",
        "test_lane.py::test_cell[qwen3]": "failed",
        "test_lane.py::test_skipped": "skipped",
        "test_lane.py::test_torn_down": "failed",
    }
    assert (read.total, read.session) == ((None, None) if nested else (4, 1))


@pytest.mark.parametrize(
    ("environment", "beacons"),
    [
        pytest.param({}, [], id="a-run-at-this-terminal"),
        pytest.param({RECEIPTS_VAR: "/tmp/r"}, [False], id="a-dispatched-job"),
        pytest.param({RECEIPTS_VAR: "/tmp/r", NESTED: "1"}, [True], id="a-dispatched-fresh-cell"),
    ],
)
def test_only_a_dispatched_job_writes_the_markers_its_waiter_reads(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], beacons: list[bool]
) -> None:
    """A terminal keeps pytest's output as it always was; only a log a machine reads gets them."""
    monkeypatch.delenv(RECEIPTS_VAR, raising=False)
    monkeypatch.delenv(NESTED, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert [plugin.nested for plugin in Runner.plugins()] == beacons
