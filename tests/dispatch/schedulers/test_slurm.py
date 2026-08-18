import pytest

from mainboard.dispatch.schedulers import Resources, Slurm, build_sbatch_flags, slurm_verdict
from mainboard.dispatch.schedulers.slurm import (
    SLURM_LIVE,
    SlurmJob,
    SlurmState,
    build_sacct_command,
    build_sinfo_command,
    build_squeue_command,
    parse_exit_code,
    parse_sacct_output,
    parse_sinfo_output,
    parse_slurm_state,
    parse_squeue_output,
)

from ..conftest import machine_with

# --- parse_slurm_state / parse_exit_code ---


def test_parse_slurm_state_strips_a_cancelled_by_suffix() -> None:
    assert parse_slurm_state("CANCELLED by 1000") is SlurmState.CANCELLED
    assert parse_slurm_state("running") is SlurmState.RUNNING
    assert parse_slurm_state("mystery-state") == "mystery-state"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0:0", 0), ("1:0", 1), ("0:9", 9), ("", None), ("garbage", None)],
)
def test_parse_exit_code(value: str, expected: int | None) -> None:
    assert parse_exit_code(value) == expected


# --- squeue ---


def test_build_squeue_command_with_and_without_me() -> None:
    assert build_squeue_command(me=True)[-1] == "--me"
    assert build_squeue_command(me=False) == [
        "squeue",
        "--noheader",
        f"--format={build_squeue_command()[2].split('=', 1)[1]}",
    ]


def test_parse_squeue_output_reads_pipe_delimited_rows() -> None:
    output = "123|train|RUNNING|gpu|00:05:00\n\nbad|row\n"
    [job] = parse_squeue_output(output)
    assert job == SlurmJob(
        job_id="123", name="train", state=SlurmState.RUNNING, partition="gpu", elapsed="00:05:00"
    )


# --- sacct ---


def test_build_sacct_command_shape() -> None:
    command = build_sacct_command("123")
    assert command[:3] == ["sacct", "--jobs", "123"]


def test_parse_sacct_output_keeps_only_the_top_level_row() -> None:
    output = "123.batch|COMPLETED|0:0\n123|COMPLETED|0:0\n"
    job = parse_sacct_output(output, job_id="123")
    assert job is not None
    assert job.state is SlurmState.COMPLETED
    assert job.exit_code == 0


def test_parse_sacct_output_returns_none_when_the_job_is_gone() -> None:
    assert parse_sacct_output("", job_id="123") is None


def test_parse_sacct_output_skips_blank_and_short_lines() -> None:
    assert parse_sacct_output("\n123|only-two-fields\n", job_id="123") is None


# --- sinfo ---


def test_build_sinfo_command_shape() -> None:
    assert build_sinfo_command() == ["sinfo", "--noheader", "--format=%P"]


def test_parse_sinfo_output_strips_default_marker_and_dedupes() -> None:
    output = "gpu*\ncpu\ngpu*\n"
    assert parse_sinfo_output(output) == ["gpu", "cpu"]


# --- build_sbatch_flags ---


def test_build_sbatch_flags_only_set_fields() -> None:
    flags = build_sbatch_flags(Resources(), "job.sh")
    assert flags[0] == "sbatch"
    assert flags[-1] == "job.sh"
    assert not any(f.startswith("--gpus") for f in flags)


def test_build_sbatch_flags_includes_every_set_field() -> None:
    flags = build_sbatch_flags(
        Resources(gpus=2, walltime="01:00:00", queue="gpu", account="proj", mem_gb=32), "job.sh"
    )
    assert "--gpus=2" in flags
    assert "--time=01:00:00" in flags
    assert "--partition=gpu" in flags
    assert "--account=proj" in flags
    assert "--mem=32G" in flags


# --- slurm_verdict ---


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
def test_slurm_verdict(state: SlurmState | None, exit_code: int | None, verdict: str) -> None:
    assert slurm_verdict(state, exit_code) == verdict


def test_slurm_live_covers_every_non_terminal_state() -> None:
    assert {
        SlurmState.PENDING,
        SlurmState.RUNNING,
        SlurmState.SUSPENDED,
        SlurmState.COMPLETING,
    } == SLURM_LIVE


# --- Slurm backend ---


def test_submit_returns_the_job_id() -> None:
    remote = machine_with("Submitted batch job 456\n")
    handle = Slurm().submit(remote, "/repo", script="job.sh", args=(), resources=Resources(gpus=1))
    assert handle == "456"


def test_submit_raises_system_exit_when_sbatch_rejects() -> None:
    remote = machine_with("sbatch: error: invalid partition\n")
    with pytest.raises(SystemExit, match="sbatch failed"):
        Slurm().submit(remote, "/repo", script="job.sh", args=(), resources=Resources())


def test_submit_raises_system_exit_with_no_output_at_all() -> None:
    remote = machine_with("")
    with pytest.raises(SystemExit, match="no output"):
        Slurm().submit(remote, "/repo", script="job.sh", args=(), resources=Resources())


def test_jobs_lists_the_current_users_squeue_rows() -> None:
    remote = machine_with("42|train|RUNNING|gpu|00:01:00\n")
    [state] = Slurm().jobs(remote, "/repo")
    assert state.handle == "42"
    assert state.verdict == "running"


def test_logs_reads_the_state_dir_log_file() -> None:
    remote = machine_with("output\n")
    assert Slurm().logs(remote, "/repo", handle="42") == "output\n"


def test_state_reads_sacct() -> None:
    remote = machine_with("42|COMPLETED|0:0\n")
    state = Slurm().state(remote, "/repo", handle="42")
    assert state.verdict == "ok"


def test_state_when_sacct_has_nothing_reads_as_vanished() -> None:
    remote = machine_with("")
    state = Slurm().state(remote, "/repo", handle="42")
    assert state.verdict == "vanished"
    assert state.state is None


def test_states_delegates_to_jobs() -> None:
    remote = machine_with("42|train|RUNNING|gpu|00:01:00\n")
    states = Slurm().states(remote, "/repo", ["42"])
    assert "42" in states


def test_wait_polls_state_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = machine_with("42|COMPLETED|0:0\n")
    monkeypatch.setattr(
        "mainboard.dispatch.schedulers.slurm.poll_until_done", lambda probe: probe()
    )
    assert Slurm().wait(remote, "/repo", handle="42").verdict == "ok"


def test_stream_drains_the_log_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = machine_with("42|COMPLETED|0:0\n")
    monkeypatch.setattr(
        "mainboard.dispatch.schedulers.slurm.stream_until_done", lambda probe, drain: probe()
    )
    assert Slurm().stream(remote, "/repo", handle="42").verdict == "ok"


def test_cancel_calls_scancel() -> None:
    remote = machine_with("")
    Slurm().cancel(remote, "/repo", handle="42")
    assert remote.calls[-1] == ["bash", "-lc", "scancel 42"]


def test_revive_is_unsupported_for_a_site_managed_scheduler() -> None:
    remote = machine_with()
    with pytest.raises(SystemExit, match="site-managed"):
        Slurm().revive(remote, "/repo")


def test_queues_lists_partitions() -> None:
    remote = machine_with("gpu*\ncpu\n")
    assert Slurm().queues(remote, "/repo") == ["gpu", "cpu"]
