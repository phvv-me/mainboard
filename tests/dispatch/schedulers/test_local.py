import pytest

from mainboard.dispatch.schedulers import Local, Resources

from ..conftest import machine_with


def test_submit_runs_the_script_through_bash_and_returns_it_as_the_handle() -> None:
    remote = machine_with("")
    handle = Local().submit(
        remote, "/repo", script="job.sh", args=("--x", "1"), resources=Resources()
    )
    assert handle == "job.sh"
    assert remote.calls[-1] == ["bash", "job.sh", "--x", "1"]


def test_jobs_is_always_empty() -> None:
    assert Local().jobs(machine_with(), "/repo") == []


def test_logs_reads_the_state_dir_log_file() -> None:
    remote = machine_with("output\n")
    assert Local().logs(remote, "/repo", handle="job.sh") == "output\n"


def test_state_is_always_vanished() -> None:
    state = Local().state(machine_with(), "/repo", handle="job.sh")
    assert state.verdict == "vanished"


def test_states_is_always_empty() -> None:
    assert Local().states(machine_with(), "/repo", ["job.sh"]) == {}


def test_wait_reports_the_already_finished_foreground_run() -> None:
    state = Local().wait(machine_with(), "/repo", handle="job.sh")
    assert state.verdict == "ok"
    assert state.exit_code == 0


def test_stream_delegates_to_wait() -> None:
    state = Local().stream(machine_with(), "/repo", handle="job.sh")
    assert state.verdict == "ok"


def test_cancel_logs_and_does_nothing() -> None:
    Local().cancel(machine_with(), "/repo", handle="job.sh")  # must not raise


def test_revive_is_unsupported_with_no_daemon() -> None:
    with pytest.raises(SystemExit, match="no daemon to revive"):
        Local().revive(machine_with(), "/repo")


def test_queues_is_always_empty() -> None:
    assert Local().queues(machine_with(), "/repo") == []
