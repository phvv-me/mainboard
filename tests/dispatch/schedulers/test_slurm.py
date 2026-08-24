from collections.abc import Sequence

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.dispatch.schedulers import Slurm, build_sbatch_flags, slurm_verdict
from mainboard.dispatch.schedulers import slurm as slurm_mod
from mainboard.dispatch.schedulers.slurm import (
    SlurmJob,
    SlurmState,
    build_sacct_command,
    build_squeue_command,
    parse_exit_code,
    parse_sacct_output,
    parse_slurm_state,
    parse_squeue_output,
)
from mainboard.dispatch.vocabulary import Resources

from ...strategies import WORDS
from ..support import machine_with

_WALLTIMES = st.one_of(st.none(), st.sampled_from(["00:10:00", "06:00:00"]))
_ROWS = st.lists(
    st.tuples(st.integers(min_value=1, max_value=9999).map(str), WORDS, WORDS),
    max_size=4,
)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("CANCELLED by 1000", SlurmState.CANCELLED),
        ("running", SlurmState.RUNNING),
        ("  COMPLETED ", SlurmState.COMPLETED),
        ("mystery-state", "mystery-state"),
    ],
)
def test_a_slurm_state_token_parses_to_its_member_or_stays_verbatim(
    token: str, expected: SlurmState | str
) -> None:
    """`sacct` appends a node reason to `CANCELLED`, which must not hide the state itself."""
    assert parse_slurm_state(token) == expected
    assert {
        SlurmState.PENDING,
        SlurmState.RUNNING,
        SlurmState.SUSPENDED,
        SlurmState.COMPLETING,
    } == slurm_mod._SLURM_LIVE


@pytest.mark.parametrize(
    ("field", "expected"), [("0:0", 0), ("1:0", 1), ("0:9", 9), ("", None), ("garbage", None)]
)
def test_an_sacct_exit_field_reports_the_return_code_or_the_signal_that_killed_it(
    field: str, expected: int | None
) -> None:
    assert parse_exit_code(field) == expected


@given(
    resources=st.builds(
        Resources,
        gpus=st.integers(min_value=0, max_value=8),
        walltime=_WALLTIMES,
        queue=st.one_of(st.none(), WORDS),
        account=WORDS | st.just(""),
        mem_gb=st.one_of(st.none(), st.integers(min_value=1, max_value=512)),
    )
)
@example(resources=Resources())
@example(resources=Resources(gpus=2, walltime="01:00:00", queue="gpu", account="proj", mem_gb=32))
def test_only_a_set_resource_becomes_an_sbatch_flag(resources: Resources) -> None:
    """A CPU-only job must carry no `--gpus`, since a cluster without GPU GRES rejects one."""
    flags = build_sbatch_flags(resources, "job.sh")
    assert flags[0] == "sbatch"
    assert flags[1] == "--output=.mainboard/dispatch/logs/%j.log"
    assert flags[-1] == "job.sh"
    optional = {flag.split("=", 1)[0]: flag.split("=", 1)[1] for flag in flags[2:-1]}
    assert optional.get("--gpus") == (str(resources.gpus) if resources.gpus else None)
    assert optional.get("--time") == resources.walltime
    assert optional.get("--partition") == resources.queue
    assert optional.get("--account") == (resources.account or None)
    assert optional.get("--mem") == (f"{resources.mem_gb}G" if resources.mem_gb else None)


@given(rows=_ROWS)
@example(rows=[("123", "train", "gpu")])
@example(rows=[])
def test_a_squeue_row_round_trips_through_the_format_the_command_asks_for(
    rows: Sequence[tuple[str, str, str]],
) -> None:
    """The parser reads back exactly the `%i|%j|%T|%P|%M` layout the built command requests."""
    assert build_squeue_command(me=True)[-1] == "--me"
    assert build_squeue_command(me=False) == [
        "squeue",
        "--noheader",
        "--format=%i|%j|%T|%P|%M",
    ]
    rendered = "\n".join(f"{i}|{name}|RUNNING|{part}|00:05:00" for i, name, part in rows)
    parsed = parse_squeue_output(f"{rendered}\n\nbad|row\n")
    assert parsed == [
        SlurmJob(
            job_id=i,
            name=name,
            state=SlurmState.RUNNING,
            partition=part,
            elapsed="00:05:00",
        )
        for i, name, part in rows
    ]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("123.batch|COMPLETED|0:0\n123|COMPLETED|0:0\n", 0),
        ("", None),
        ("\n123|only-two-fields\n", None),
        ("999|COMPLETED|0:0\n", None),
    ],
)
def test_an_sacct_listing_yields_only_the_top_level_row_of_the_job_asked_for(
    output: str, expected: int | None
) -> None:
    assert build_sacct_command("123")[:3] == ["sacct", "--jobs", "123"]
    job = parse_sacct_output(output, job_id="123")
    assert (job.exit_code if job else None) == expected


@pytest.mark.parametrize(
    ("state", "exit_code", "verdict"),
    [
        (None, None, "vanished"),
        (SlurmState.RUNNING, None, "running"),
        (SlurmState.PENDING, None, "running"),
        (SlurmState.COMPLETED, 0, "ok"),
        (SlurmState.COMPLETED, 1, "failed"),
        (SlurmState.FAILED, 1, "failed"),
        (SlurmState.CANCELLED, None, "failed"),
    ],
)
def test_slurm_verdict_reads_one_word_out_of_a_state_and_an_exit_code(
    state: SlurmState | None, exit_code: int | None, verdict: str
) -> None:
    assert slurm_verdict(state, exit_code) == verdict


@pytest.mark.parametrize(
    ("printed", "handle", "refusal"),
    [
        ("Submitted batch job 456\n", "456", ""),
        ("456\n", "456", ""),
        ("sbatch: error: invalid partition\n", "", "sbatch failed"),
        ("", "", "no output"),
    ],
)
def test_submitting_returns_the_job_id_sbatch_printed_or_refuses_with_its_output(
    printed: str, handle: str, refusal: str
) -> None:
    remote = machine_with(printed)
    if not refusal:
        assert (
            Slurm().submit(remote, "/repo", script="job.sh", args=(), resources=Resources(gpus=1))
            == handle
        )
        return
    with pytest.raises(SystemExit, match=refusal):
        Slurm().submit(remote, "/repo", script="job.sh", args=(), resources=Resources())


def test_every_backend_operation_rides_one_login_shell_command() -> None:
    backend = Slurm()
    listing = machine_with("42|train|RUNNING|gpu|00:01:00\n")
    batched = backend.states(listing, "/repo", ["42"])
    assert (batched["42"].handle, batched["42"].verdict) == ("42", "running")
    assert backend.logs(machine_with("output\n"), "/repo", handle="42") == "output\n"
    assert backend.state(machine_with("42|COMPLETED|0:0\n"), "/repo", handle="42").verdict == "ok"
    gone = backend.state(machine_with(""), "/repo", handle="42")
    assert (gone.verdict, gone.state) == ("vanished", None)
    cancelled = machine_with("")
    backend.cancel(cancelled, "/repo", handle="42")
    assert cancelled.calls[-1] == ["bash", "-lc", "scancel 42"]


def test_an_interactive_allocation_reuses_the_batch_flags_and_can_carry_a_command() -> None:
    """`srun` takes the command to run on the allocated node, so a probe rides one allocation."""
    resources = Resources(queue="gpu", walltime="01:00:00", gpus=1)
    assert Slurm().interactive(env="default", command=(), resources=resources) == (
        "srun --pty --gpus=1 --time=01:00:00 --partition=gpu bash -l"
    )
    assert Slurm().interactive(env="default", command=("pwd",), resources=Resources()) == (
        "srun --pty pwd"
    )
