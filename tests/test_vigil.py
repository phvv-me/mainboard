from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.dispatch.state import RunRecord
from mainboard.jobs.beacon import Progress
from mainboard.pulse import Pulse
from mainboard.vigil import LINGER_SECONDS, LOOK_SECONDS, Linger, Look, Vigil, heartbeat

from .support import Clock, run


class Looked:
    """The looks at the hosts, answering whatever pulses the test last put in front of them."""

    def __init__(self) -> None:
        self.pulses: dict[RunRecord, Pulse] = {}
        self.asked = 0

    def taken(self, records: Sequence[RunRecord]) -> dict[RunRecord, Pulse]:
        self.asked += 1
        return self.pulses


def pulse(
    handle: str = "1",
    *,
    cells: tuple[tuple[str, str], ...] = (),
    total: int | None = 3,
    session: int | None = None,
    quiet: int | None = 0,
    gpu: int | None = None,
) -> Pulse:
    """One running job's pulse as a look found it."""
    progress = Progress(total=total, cells=cells, session=session)
    return Pulse(handle=handle, target="gold", progress=progress, quiet_s=quiet, gpu_pct=gpu)


def test_a_wait_says_each_cell_once_as_it_lands_and_beats_once_per_look() -> None:
    """Looks are spaced, since each reads whole logs over the network, and a cell is said once."""
    looked, clock, said = Looked(), Clock(), []
    vigil = Vigil(looked, say=said.append, clock=clock)
    records = [run("1")]
    looked.pulses = {records[0]: pulse(cells=(("a.py::t[x]", "passed"),), gpu=97)}

    assert vigil.look(records) == Look()
    assert said == [
        "1 passed a.py::t[x]",
        "heartbeat: 1 running, 1/3 cells, 0 failed, output 0s ago, gpu 97%",
    ]

    clock.now += LOOK_SECONDS / 2
    assert vigil.look(records) == Look()
    assert looked.asked == 1

    looked.pulses = {
        records[0]: pulse(cells=(("a.py::t[x]", "passed"), ("a.py::t[y]", "failed")), quiet=None)
    }
    clock.now += LOOK_SECONDS
    vigil.look(records)
    assert said[2:] == ["1 failed a.py::t[y]", "heartbeat: 1 running, 2/3 cells, 1 failed"]

    looked.pulses = {}
    clock.now += LOOK_SECONDS
    assert vigil.look(records) == Look()
    assert len(said) == 4


@pytest.mark.parametrize(
    ("seen", "stall", "stalled"),
    [
        pytest.param(
            pulse(quiet=1300, gpu=0),
            1200.0,
            "printed nothing for 1300s and its busiest card at 0%",
            id="silent-on-an-idle-card",
        ),
        pytest.param(
            pulse(quiet=1300),
            1200.0,
            "printed nothing for 1300s",
            id="silent-where-no-card-is-readable",
        ),
        pytest.param(pulse(quiet=1300, gpu=97), 1200.0, "", id="silent-but-the-card-is-working"),
        pytest.param(pulse(quiet=600, gpu=0), 1200.0, "", id="not-silent-long-enough"),
        pytest.param(pulse(quiet=None, gpu=0), 1200.0, "", id="seen-only-once"),
        pytest.param(pulse(quiet=1300, gpu=0), 0.0, "", id="a-wait-that-never-calls-a-stall"),
        pytest.param(
            pulse(quiet=1300, gpu=0, session=0),
            1200.0,
            "",
            id="a-finished-session-lingers-instead",
        ),
    ],
)
def test_a_job_is_stalled_only_when_silent_past_the_threshold_with_no_busy_card_to_excuse_it(
    seen: Pulse, stall: float, stalled: str
) -> None:
    looked = Looked()
    looked.pulses = {run("1"): seen}
    look = Vigil(looked, stall=stall, say=lambda line: None, clock=Clock()).look([run("1")])
    assert (stalled in look.stalled) and (bool(look.stalled) is bool(stalled))


@pytest.mark.parametrize(
    ("seen", "lingering"),
    [
        pytest.param(
            pulse(session=0, quiet=int(LINGER_SECONDS)),
            (Linger(handle="1", target="gold", session=0),),
            id="past-its-grace",
        ),
        pytest.param(
            pulse(session=1, quiet=int(LINGER_SECONDS) + 5),
            (Linger(handle="1", target="gold", session=1),),
            id="a-failed-session",
        ),
        pytest.param(pulse(session=0, quiet=5), (), id="still-inside-its-grace"),
        pytest.param(pulse(session=0, quiet=None), (), id="seen-only-once"),
        pytest.param(pulse(quiet=600), (), id="a-session-still-running"),
    ],
)
def test_a_session_that_ended_while_its_process_lives_on_is_handed_back_to_settle(
    seen: Pulse, lingering: tuple[Linger, ...]
) -> None:
    looked = Looked()
    looked.pulses = {run("1"): seen}
    look = Vigil(looked, say=lambda line: None, clock=Clock()).look([run("1")])
    assert look.lingering == lingering


@given(
    seen=st.lists(
        st.tuples(
            st.one_of(st.none(), st.integers(0, 9)),
            st.integers(0, 9),
            st.one_of(st.none(), st.integers(0, 999)),
            st.one_of(st.none(), st.integers(0, 100)),
        ),
        min_size=1,
    )
)
def test_one_heartbeat_sums_the_cells_and_names_the_freshest_output_and_the_busiest_card(
    seen: list[tuple[int | None, int, int | None, int | None]],
) -> None:
    pulses = [
        pulse(
            str(index),
            cells=tuple((f"c{cell}", "failed" if cell % 2 else "passed") for cell in range(done)),
            total=total,
            quiet=quiet,
            gpu=gpu,
        )
        for index, (total, done, quiet, gpu) in enumerate(seen)
    ]
    line = heartbeat(pulses)
    totals = [total for total, *_ in seen]
    counted = "?" if None in totals else str(sum(total or 0 for total in totals))
    assert line.startswith(
        f"heartbeat: {len(seen)} running, {sum(done for _, done, *_ in seen)}/{counted} cells"
    )
    quiet = [quiet for *_, quiet, _ in seen if quiet is not None]
    freshest = [f", output {min(quiet)}s ago"] if quiet else []
    cards = [gpu for *_, gpu in seen if gpu is not None]
    busiest = [f", gpu {max(cards)}%"] if cards else []
    assert line.endswith(" failed" + "".join([*freshest, *busiest]))
