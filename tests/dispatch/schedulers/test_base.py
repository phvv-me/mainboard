import pytest

from mainboard.dispatch import HostUnreachable
from mainboard.dispatch.schedulers import (
    exit_reason,
    failure_reason,
    log_excerpt,
    login_run,
    read_log,
    short_reason,
    verdict_line,
)
from mainboard.dispatch.schedulers.base import (
    log_path,
    meaningful_lines,
    workspace_session,
)
from mainboard.dispatch.vocabulary import JobState, Resources

from ..conftest import machine_with

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
