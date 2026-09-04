from datetime import UTC, datetime, timedelta

import pytest

from mainboard.dispatch import HostUnreachable
from mainboard.dispatch.schedulers import (
    exit_reason,
    failure_reason,
    is_quota_refusal,
    log_excerpt,
    login_run,
    read_log,
    short_reason,
    standing,
    verdict_line,
)
from mainboard.dispatch.schedulers.base import (
    log_path,
    meaningful_lines,
    workspace_session,
)
from mainboard.dispatch.shared import since
from mainboard.dispatch.vocabulary import JobState, Resources

from ..support import machine_with

_TRACEBACK = """loading model...
  File "rotation.py", line 65, in forward_rows
ModuleNotFoundError: No module named 'fast_hadamard_transform'
"""
_PANEL = "shutting down\n╭──────────╮\n│ all done │\n╰──────────╯\n\x1b[32m\x1b[0m\n"
_WALLTIME_KILL = "mainboard: killed at walltime 02:00:00 (exit 124)"


def test_login_run_returns_stdout_but_raises_when_the_transport_itself_failed() -> None:
    """An empty answer from a refused ssh session used to end a wait as a false `vanished`."""
    assert login_run(machine_with("hello\n"), "echo hello") == "hello\n"
    broken = machine_with(rules=[("qstat", 255, "kex_exchange identification failed")])
    with pytest.raises(HostUnreachable, match="kex_exchange"):
        login_run(broken, "qstat")


def test_a_bare_resource_request_asks_for_one_cpu_node_and_nothing_else() -> None:
    resources = Resources()
    assert (resources.nodes, resources.gpus, resources.account, resources.container) == (
        1,
        0,
        "",
        "",
    )
    assert (resources.walltime, resources.queue, resources.mem_gb) == (None, None, None)


def test_a_workspace_session_hands_the_terminal_to_the_hosts_own_tool() -> None:
    """An ssh host is already the machine the work runs on, so its own tool owns activation."""
    resources = Resources()
    assert workspace_session(env="serving", command=(), resources=resources) == (
        "mainboard shell serving"
    )
    assert workspace_session(env="default", command=("nvidia-smi", "-L"), resources=resources) == (
        "mainboard run --env default -- nvidia-smi -L"
    )


def test_a_log_is_read_from_the_state_dir_path_the_job_template_writes() -> None:
    assert log_path("/repo", handle="2435326.opbs") == "/repo/.mainboard/dispatch/logs/2435326.log"
    assert log_path("/repo", handle="2435326") == "/repo/.mainboard/dispatch/logs/2435326.log"
    remote = machine_with("abc")
    assert read_log(remote, "/repo", handle="42", offset=10) == "abc"
    assert remote.calls[-1] == [
        "bash",
        "-lc",
        "tail -c +11 /repo/.mainboard/dispatch/logs/42.log 2>/dev/null",
    ]


@pytest.mark.parametrize(
    ("log", "exit_code", "expected"),
    [
        (_TRACEBACK, None, "ModuleNotFoundError: No module named 'fast_hadamard_transform'"),
        (_TRACEBACK, 137, "ModuleNotFoundError: No module named 'fast_hadamard_transform'"),
        (
            'setup...\nqsub: Resource invalid in "select" specification: ngpus\n',
            None,
            'qsub: Resource invalid in "select" specification: ngpus',
        ),
        (
            "cmake...\nfatal error: cuda.h: No such file\n",
            None,
            "fatal error: cuda.h: No such file",
        ),
        (f"step 1 ok\nValueError: earlier retry\n{_WALLTIME_KILL}\n", None, _WALLTIME_KILL),
        ("step 1 ok\nstep 2 ok\njob killed by walltime\n\n", None, "job killed by walltime"),
        ("   \n\n", None, "(no log output)"),
        (
            "loading shards...\nstep 200 ok\n",
            137,
            "killed by SIGKILL (out of memory or walltime, exit 137)",
        ),
        ("warming up...\n", 124, "timed out (walltime exceeded)"),
        ("step 1 ok\nboom\n", 1, "boom"),
        (_PANEL, None, "all done"),
    ],
)
def test_failure_reason_reports_the_strongest_marker_the_log_carries(
    log: str, exit_code: int | None, expected: str
) -> None:
    """A real traceback outranks the exit code, and the walltime kill outranks the traceback."""
    assert failure_reason(log, exit_code) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (124, "timed out (walltime exceeded)"),
        (125, "timeout failed to start the job"),
        (137, "killed by SIGKILL (out of memory or walltime, exit 137)"),
        (139, "crashed with SIGSEGV (segfault, exit 139)"),
        (143, "terminated by SIGTERM (walltime or cancel, exit 143)"),
        (None, None),
        (0, None),
        (1, None),
        (42, None),
    ],
)
def test_exit_reason_decodes_an_externally_imposed_code_and_invents_nothing_else(
    code: int | None, expected: str | None
) -> None:
    assert exit_reason(code) == expected


def test_terminal_noise_is_stripped_before_a_log_is_quoted_as_a_cause() -> None:
    log = "\x1b[1mheader\x1b[0m\n╭───╮\n│ body │\n╰───╯\n\n  plain  \n"
    assert meaningful_lines(log) == ["header", "body", "plain"]
    numbered = "\n".join(f"line {index}" for index in range(20)) + "\n╭───╮\n"
    assert log_excerpt(numbered, limit=3) == ["line 17", "line 18", "line 19"]


@pytest.mark.parametrize(
    ("verdict", "exit_code", "expected"),
    [
        ("vanished", None, "vanished (the scheduler no longer remembers the job)"),
        ("failed", 137, "killed by SIGKILL (out of memory or walltime, exit 137)"),
        ("failed", 3, "exited 3"),
        ("failed", None, "failed"),
    ],
)
def test_short_reason_explains_a_cached_state_without_touching_the_network(
    verdict: str, exit_code: int | None, expected: str
) -> None:
    assert short_reason(verdict, exit_code) == expected


@pytest.mark.parametrize(
    ("state", "age", "expected"),
    [
        (
            JobState(handle="2435326", state="F", exit_code=137, verdict="failed"),
            "11 days ago",
            "2435326 failed (exit 137, killed by SIGKILL (out of memory or walltime, exit 137), "
            "submitted 11 days ago)",
        ),
        (JobState(handle="7", state="R", verdict="running"), "", "7 running"),
        (JobState(handle="7", state="F", exit_code=1, verdict="failed"), "", "7 failed (exit 1)"),
        (JobState(handle="7", verdict="vanished"), "t0", "7 vanished (submitted t0)"),
    ],
)
def test_verdict_line_leads_with_the_handle_then_whatever_details_exist(
    state: JobState, age: str, expected: str
) -> None:
    assert verdict_line(state, submitted_age=age) == expected


@pytest.mark.parametrize(
    ("state", "submitted_at", "host", "expected"),
    [
        (
            JobState(
                handle="3289319",
                state="Q",
                verdict="queued",
                note="estimated start Thu Sep  4 14:00:00 2026",
            ),
            "2026-09-04T09:12:04+00:00",
            "miyabi-g",
            "3289319 is queued on miyabi-g; scheduler state Q; "
            "submitted 2026-09-04T09:12:04+00:00 ({age} ago); "
            "estimated start Thu Sep  4 14:00:00 2026",
        ),
        (
            JobState(handle="7", state="R", verdict="running"),
            "",
            "gold",
            "7 is running on gold; scheduler state R",
        ),
        (JobState(handle="7", verdict="running"), "", "", "7 is running"),
    ],
)
def test_a_job_that_printed_nothing_still_says_where_it_stands(
    state: JobState, submitted_at: str, host: str, expected: str
) -> None:
    """An empty log is either a job that has not started or one that started and said nothing."""
    assert standing(state, submitted_at=submitted_at, host=host) == expected.format(
        age=since(submitted_at)
    )


def test_a_waiting_time_reads_compactly_and_never_from_a_stamp_that_is_not_one() -> None:
    """The stamp comes off a durable record, which older workspaces wrote in other shapes."""
    now = datetime.now(UTC)
    assert since((now - timedelta(hours=3, minutes=12)).isoformat()) == "3h12m"
    assert since((now - timedelta(days=2, hours=5)).isoformat()) == "2d5h"
    assert since((now - timedelta(seconds=9)).isoformat()) == "9s"
    # A naive stamp is read as UTC, since reading it locally would invent a timezone of waiting.
    assert since(now.replace(tzinfo=None).isoformat()).endswith("s")
    assert since("t0") == ""


@pytest.mark.parametrize(
    ("reason", "quota"),
    [
        pytest.param(
            "qsub failed (rc=39): qsub: would exceed group xg25g007's limit on resource njobs-g",
            True,
            id="pbs-refuses-on-the-groups-job-count",
        ),
        pytest.param(
            "qsub: would exceed complex's per-user limit on resource njobs",
            True,
            id="pbs-refuses-on-the-users-job-count",
        ),
        pytest.param(
            "sbatch: error: Batch job submission failed: Job violates QOSMaxJobsPerUserLimit",
            True,
            id="slurm-refuses-on-the-qos-job-count",
        ),
        pytest.param(
            "sbatch: error: AssocMaxSubmitJobLimit",
            True,
            id="slurm-refuses-on-the-association-submit-count",
        ),
        pytest.param("qsub: Maximum number of jobs already in queue", True, id="queue-is-full"),
        pytest.param("qsub: Unknown queue: short-q", False, id="a-queue-that-does-not-exist"),
        pytest.param(
            "qsub: Job violates queue and/or server resource limits",
            False,
            id="a-request-the-queue-will-never-take",
        ),
        pytest.param("sbatch: error: invalid partition specified", False, id="a-bad-partition"),
        pytest.param("Permission denied", False, id="an-account-without-permission"),
    ],
)
def test_only_a_refusal_about_how_many_jobs_are_queued_is_worth_asking_again(
    reason: str, *, quota: bool
) -> None:
    """A count quota is "not now" and everything else is "no".

    Holding a rejection would re-ask it every twenty minutes forever, and dropping a quota
    refusal loses the job, which is what cost a wave four of its thirteen (miyabi-g njobs-g,
    2026-09-04). The line the scheduler printed is the only thing that tells them apart.
    """
    assert is_quota_refusal(reason) is quota
    assert is_quota_refusal(reason.upper()) is quota
