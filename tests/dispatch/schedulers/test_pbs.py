import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.dispatch.schedulers import Pbs, build_qsub_flags
from mainboard.dispatch.schedulers.pbs import (
    JobInfo,
    PbsState,
    bare,
    parse_job_state,
    parse_qstat_full,
    pbs_verdict,
)
from mainboard.dispatch.vocabulary import Resources

from ...strategies import WORDS
from ..support import machine_with


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("R", PbsState.RUNNING),
        ("RUNNING", PbsState.RUNNING),
        ("BEGUN", PbsState.ARRAY_BEGUN),
        ("mystery", "mystery"),
    ],
)
def test_a_pbs_state_token_parses_from_its_letter_or_its_word(
    token: str, expected: PbsState | str
) -> None:
    assert parse_job_state(token) == expected


@given(handle=WORDS)
@example(handle="2435326.opbs")
@example(handle="2435326")
def test_a_handle_joins_on_its_bare_job_number_whichever_way_qsub_spelled_it(handle: str) -> None:
    """`qsub` prints a bare id on some wrappers and `<id>.<server>` elsewhere, `qstat -f` never."""
    assert bare(handle) == bare(f"{handle}.opbs")


def test_qstat_full_reads_every_field_it_recognizes_across_records() -> None:
    record = """Job Id: 2435326.opbs
    Job_Name = train.sh
    job_state = F
    Exit_status = 0
    resources_used.walltime = 00:01:00
    queue = short-g
junk line with no equals
Job Id: 2.opbs
    job_state = R
"""
    first, second = parse_qstat_full(record)
    assert first == JobInfo(
        job_id="2435326.opbs",
        name="train.sh",
        state=PbsState.FINISHED,
        queue="short-g",
        exit_status=0,
    )
    assert (second.job_id, second.state) == ("2.opbs", PbsState.RUNNING)
    assert parse_qstat_full("") == []


@given(
    resources=st.builds(
        Resources,
        queue=st.one_of(st.none(), WORDS),
        walltime=st.one_of(st.none(), st.sampled_from(["00:10:00", "06:00:00"])),
        account=WORDS | st.just(""),
        mem_gb=st.one_of(st.none(), st.integers(min_value=1, max_value=512)),
        nodes=st.integers(min_value=1, max_value=4),
    )
)
@example(resources=Resources())
@example(
    resources=Resources(
        queue="short-g", walltime="06:00:00", account="xg25g007", mem_gb=100, nodes=2
    )
)
def test_only_a_set_resource_becomes_a_qsub_flag(resources: Resources) -> None:
    flags = build_qsub_flags(resources)
    pairs = list(zip(flags[::2], flags[1::2], strict=True))
    expected: list[tuple[str, str]] = []
    if resources.queue is not None:
        expected.append(("-q", resources.queue))
    if resources.walltime is not None:
        expected.append(("-l", f"walltime={resources.walltime}"))
    if resources.account:
        expected.append(("-W", f"group_list={resources.account}"))
    if resources.mem_gb is not None:
        expected.append(("-l", f"select={resources.nodes}:mem={resources.mem_gb}gb"))
    assert pairs == expected


@pytest.mark.parametrize(
    ("state", "exit_code", "verdict"),
    [
        (None, None, "vanished"),
        ("R", None, "running"),
        ("Q", None, "running"),
        ("F", 0, "ok"),
        ("F", 1, "failed"),
        ("E", None, "unknown"),
        ("F", None, "unknown"),
    ],
)
def test_pbs_verdict_never_reads_a_finished_job_with_no_exit_status_as_ok(
    state: str | None, exit_code: int | None, verdict: str
) -> None:
    """A job `qdel`'d while still queued produced nothing, so a wait must not report success."""
    assert pbs_verdict(state, exit_code) == verdict


@pytest.mark.parametrize(
    ("printed", "handle"),
    [
        ("2435326.opbs\n", "2435326"),
        ("2435326\n", "2435326"),
        ("qsub: Resource invalid\n", ""),
        ("", ""),
    ],
)
def test_submitting_returns_the_job_id_qsub_printed_or_refuses_with_its_output(
    printed: str, handle: str
) -> None:
    remote = machine_with(printed)
    if not handle:
        with pytest.raises(SystemExit, match="qsub failed"):
            Pbs().submit(remote, "/repo", script="job.sh", args=(), resources=Resources())
        return
    submitted = Pbs().submit(
        remote, "/repo", script="job.sh", args=(), resources=Resources(queue="short-g")
    )
    assert submitted == handle
    # qsub is invoked from the workspace root so the workspace-relative script path resolves and
    # PBS_O_WORKDIR (which the generated script cds back to) is the root, not the login home.
    assert remote.calls[-1] == ["bash", "-lc", "cd /repo && qsub -q short-g job.sh"]
    assert Pbs._extract_job_id("no leading digits here") == "no leading digits here"


def test_a_handle_resolves_live_then_from_history_then_from_its_exit_artifact() -> None:
    """A job the server purged still reconciles from the exit the job script trapped on disk."""
    backend = Pbs()
    live = "Job Id: 1.opbs\n    job_state = R\n    queue = short-g\n"
    history = "Job Id: 2.opbs\n    job_state = F\n    Exit_status = 0\n"
    both = backend.states(machine_with(live, history), "/repo", ["1.opbs", "2.opbs"])
    assert (both["1.opbs"].verdict, both["2.opbs"].verdict) == ("running", "ok")
    assert backend.states(machine_with(), "/repo", []) == {}
    assert backend.state(machine_with(live), "/repo", handle="1.opbs").verdict == "running"
    settled = backend.state(machine_with("", "", "exit=0\n"), "/repo", handle="9999.opbs")
    assert (settled.verdict, settled.exit_code) == ("ok", 0)
    autopsied = backend.autopsy(machine_with("exit=1\n"), "/repo", handle="42.opbs")
    assert (autopsied.verdict, autopsied.exit_code) == ("failed", 1)
    assert backend.autopsy(machine_with(""), "/repo", handle="42.opbs").verdict == "vanished"


def test_every_backend_operation_rides_one_login_shell_command() -> None:
    backend = Pbs()
    assert backend.logs(machine_with("job output\n"), "/repo", handle="2435326.opbs") == (
        "job output\n"
    )
    cancelled = machine_with("")
    backend.cancel(cancelled, "/repo", handle="1.opbs")
    assert cancelled.calls[-1] == ["bash", "-lc", "qdel 1.opbs"]


def test_an_interactive_allocation_reuses_the_batch_flags_and_takes_no_command() -> None:
    """PBS hands over a terminal on the node it allocates and runs no command of its own."""
    resources = Resources(queue="interact-g", walltime="01:00:00")
    assert Pbs().interactive(env="default", command=(), resources=resources) == (
        "qsub -I -q interact-g -l walltime=01:00:00"
    )
    with pytest.raises(MissionError, match="runs no command of its own"):
        Pbs().interactive(env="default", command=("true",), resources=resources)
