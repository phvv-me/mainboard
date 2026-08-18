import pytest
from patos import IllegalTransition

from mainboard.dispatch import HostUnreachable
from mainboard.dispatch.schedulers import (
    JobState,
    Resources,
    exit_reason,
    failure_reason,
    log_excerpt,
    login_run,
    poll_until_done,
    read_log,
    short_reason,
    stream_until_done,
    verdict_line,
)
from mainboard.dispatch.schedulers.base import (
    _settle,
    drain_log,
    log_path,
    meaningful_lines,
    resilient,
)
from mainboard.dispatch.verdicts import OK, RUNNING, tracker

from ..conftest import machine_with

# --- login_run ---


def test_login_run_returns_stdout_on_a_clean_command() -> None:
    remote = machine_with("hello\n")
    assert login_run(remote, "echo hello") == "hello\n"


def test_login_run_raises_host_unreachable_on_a_transport_failure() -> None:
    class FailingCommand:
        def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> FailingCommand:
            return self

        def run(self, *_, **__) -> tuple[int, str, str]:
            return (255, "", "kex_exchange identification failed")

    remote = {"bash": FailingCommand()}
    with pytest.raises(HostUnreachable):
        login_run(remote, "qstat")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


# --- Resources defaults ---


def test_resources_widened_fields_default_sensibly() -> None:
    resources = Resources()
    assert resources.nodes == 1
    assert resources.account == ""
    assert resources.container == ""
    assert resources.gpus == 0


# --- resilient / poll_until_done / stream_until_done ---


def test_resilient_retries_host_unreachable_then_returns_the_real_state() -> None:
    calls = {"n": 0}

    def probe() -> JobState:
        calls["n"] += 1
        if calls["n"] == 1:
            raise HostUnreachable("blip")
        return JobState(handle="1", verdict="ok")

    wrapped = resilient(probe, interval=0.0, sleeper=lambda s: None, retries=3)
    assert wrapped().verdict == "ok"
    assert calls["n"] == 2


def test_resilient_reraises_after_the_retry_budget_is_spent() -> None:
    def always_down() -> JobState:
        raise HostUnreachable("still down")

    wrapped = resilient(always_down, interval=0.0, sleeper=lambda s: None, retries=2)
    with pytest.raises(HostUnreachable):
        wrapped()


def test_poll_until_done_waits_while_running_then_returns_the_terminal_state() -> None:
    states = iter(
        [
            JobState(handle="1", verdict="running"),
            JobState(handle="1", verdict="running"),
            JobState(handle="1", verdict="ok", exit_code=0),
        ]
    )
    slept: list[float] = []
    result = poll_until_done(lambda: next(states), interval=0.0, sleeper=slept.append)
    assert result.verdict == "ok"
    assert len(slept) == 2


def test_poll_until_done_returns_immediately_when_first_probe_is_terminal() -> None:
    result = poll_until_done(
        lambda: JobState(handle="1", verdict="failed"), sleeper=lambda s: None
    )
    assert result.verdict == "failed"


def test_poll_until_done_raises_on_a_genuine_regression_from_a_terminal() -> None:
    """A scheduler that reports `ok` and then, on a later re-read, `running` is a real bug.

    `poll_until_done` stops the instant it observes `ok` (a terminal), so this exercises the
    tracker directly: a fresh wait that inherited an already-settled state must still refuse a
    reported regression rather than resume polling.
    """
    machine = tracker(OK)
    with pytest.raises(IllegalTransition):
        _settle(machine, RUNNING)


def test_poll_until_done_accepts_vanished_as_the_first_observation() -> None:
    """`queued` legally reaches `vanished` directly (the table's declared edge)."""
    result = poll_until_done(
        lambda: JobState(handle="1", verdict="vanished"), sleeper=lambda s: None
    )
    assert result.verdict == "vanished"


@pytest.mark.parametrize("terminal", ["ok", "failed", "timeout"])
def test_poll_until_done_forgives_a_terminal_that_skips_running(terminal: str) -> None:
    """A fast job (or a scheduler that never reports an intermediate running tick) settles fine."""
    result = poll_until_done(
        lambda: JobState(handle="1", verdict=terminal), sleeper=lambda s: None
    )
    assert result.verdict == terminal


def test_stream_until_done_drains_between_polls_and_once_more_at_the_end() -> None:
    states = iter([JobState(handle="1", verdict="running"), JobState(handle="1", verdict="ok")])
    drained: list[int] = []

    def drain(offset: int) -> int:
        drained.append(offset)
        return 5

    result = stream_until_done(lambda: next(states), drain, interval=0.0, sleeper=lambda s: None)
    assert result.verdict == "ok"
    assert drained == [0, 5]  # one mid-loop drain, one final drain from the grown offset


# --- log_path / read_log / drain_log ---


def test_log_path_strips_a_pbs_server_suffix() -> None:
    assert log_path("/repo", handle="2435326.opbs") == "/repo/.mainboard/dispatch/logs/2435326.log"
    assert log_path("/repo", handle="2435326") == "/repo/.mainboard/dispatch/logs/2435326.log"


def test_read_log_tails_from_the_given_offset() -> None:
    remote = machine_with("chunk\n")
    assert read_log(remote, "/repo", handle="42", offset=10) == "chunk\n"
    assert remote.calls[-1] == [
        "bash",
        "-lc",
        "tail -c +11 /repo/.mainboard/dispatch/logs/42.log 2>/dev/null",
    ]


def test_drain_log_prints_and_reports_the_byte_count(capsys: pytest.CaptureFixture[str]) -> None:
    remote = machine_with("abc")
    consumed = drain_log(remote, "/repo", handle="42", offset=0)
    assert consumed == 3
    assert capsys.readouterr().out == "abc"


# --- exit_reason / failure_reason / meaningful_lines / log_excerpt / short_reason / verdict ---


def test_python_traceback_returns_the_raised_exception() -> None:
    log = """loading model...
  File "rotation.py", line 65, in forward_rows
ModuleNotFoundError: No module named 'fast_hadamard_transform'
"""
    assert failure_reason(log) == "ModuleNotFoundError: No module named 'fast_hadamard_transform'"


def test_scheduler_rejection_when_there_is_no_python_error() -> None:
    log = 'running setup...\nqsub: Resource invalid in "select" specification: ngpus\n'
    assert "ngpus" in failure_reason(log)


def test_falls_back_to_last_nonempty_line() -> None:
    log = "step 1 ok\nstep 2 ok\njob killed by walltime\n\n"
    assert failure_reason(log) == "job killed by walltime"


def test_empty_log_is_handled() -> None:
    assert failure_reason("   \n\n") == "(no log output)"


def test_sigkill_exit_137_reads_as_oom_or_walltime_when_log_is_silent() -> None:
    log = "loading shards...\nstep 200 ok\nstep 400 ok\n"
    reason = failure_reason(log, exit_code=137)
    assert "memory" in reason and "walltime" in reason


def test_timeout_exit_124_reads_as_walltime_exceeded() -> None:
    assert "walltime" in failure_reason("warming up...\n", exit_code=124).lower()


def test_a_real_traceback_wins_over_the_exit_code() -> None:
    log = "training...\ntorch.cuda.OutOfMemoryError: CUDA out of memory.\n"
    assert failure_reason(log, exit_code=137).startswith("torch.cuda.OutOfMemoryError")


def test_a_plain_nonzero_exit_does_not_invent_a_signal_reason() -> None:
    assert failure_reason("step 1 ok\nboom\n", exit_code=1) == "boom"


@pytest.mark.parametrize(
    ("code", "needle"), [(124, "walltime"), (137, "memory"), (139, "segfault"), (143, "SIGTERM")]
)
def test_exit_reason_maps_externally_imposed_codes(code: int, needle: str) -> None:
    reason = exit_reason(code)
    assert reason is not None and needle in reason


@pytest.mark.parametrize("code", [None, 0, 1, 2, 42])
def test_exit_reason_is_none_for_ordinary_codes(code: int | None) -> None:
    assert exit_reason(code) is None


def test_the_wrappers_walltime_kill_marker_is_the_authoritative_reason() -> None:
    marker = "mainboard: killed at walltime 02:00:00 (exit 124)"
    log = f"step 1 ok\nValueError: from an earlier retry\n{marker}\n"
    assert failure_reason(log) == marker


def test_last_line_fallback_skips_rich_panel_borders() -> None:
    log = "shutting down\n╭──────────╮\n│ all done │\n╰──────────╯\n\x1b[32m\x1b[0m\n"
    assert failure_reason(log) == "all done"


def test_meaningful_lines_strip_ansi_and_borders_but_keep_content() -> None:
    log = "\x1b[1mheader\x1b[0m\n╭───╮\n│ body │\n╰───╯\n\n  plain  \n"
    assert meaningful_lines(log) == ["header", "body", "plain"]


def test_log_excerpt_returns_the_last_meaningful_lines() -> None:
    log = "\n".join(f"line {i}" for i in range(20)) + "\n╭───╮\n"
    assert log_excerpt(log, limit=3) == ["line 17", "line 18", "line 19"]


def test_short_reason_for_vanished() -> None:
    assert "vanished" in short_reason("vanished", None)


def test_short_reason_decodes_a_known_signal_exit() -> None:
    assert "memory" in short_reason("failed", 137)


def test_short_reason_reports_a_plain_exit_code() -> None:
    assert short_reason("failed", 3) == "exited 3"


def test_short_reason_falls_back_to_failed_with_no_exit_code() -> None:
    assert short_reason("failed", None) == "failed"


def test_verdict_line_leads_with_handle_verdict_then_decoded_exit_and_age() -> None:
    state = JobState(handle="2435326", state="F", exit_code=137, verdict="failed")
    line = verdict_line(state, submitted_age="11 days ago")
    assert line.startswith("2435326 failed (exit 137, killed by SIGKILL")
    assert "submitted 11 days ago" in line


def test_verdict_line_is_bare_for_a_running_job_without_details() -> None:
    assert verdict_line(JobState(handle="7", state="R", verdict="running")) == "7 running"


def test_verdict_line_with_a_plain_exit_code_skips_the_decoded_reason() -> None:
    line = verdict_line(JobState(handle="7", state="F", exit_code=1, verdict="failed"))
    assert line == "7 failed (exit 1)"


def test_verdict_line_keeps_the_rendered_age_verbatim() -> None:
    line = verdict_line(JobState(handle="7", verdict="vanished"), submitted_age="t0")
    assert line == "7 vanished (submitted t0)"
