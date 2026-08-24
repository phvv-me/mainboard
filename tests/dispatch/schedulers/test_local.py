from mainboard.dispatch.schedulers import Local
from mainboard.dispatch.vocabulary import Resources

from ..support import machine_with


def test_the_local_backend_runs_the_script_in_the_foreground_with_nothing_to_queue() -> None:
    """`submit` already ran the job and raised on a non-zero exit, so a handle here finished."""
    backend = Local()
    remote = machine_with("output\n")
    handle = backend.submit(
        remote, "/repo", script="job.sh", args=("--x", "1"), resources=Resources()
    )
    assert handle == "job.sh"
    assert remote.calls[-1] == ["bash", "job.sh", "--x", "1"]
    assert backend.logs(remote, "/repo", handle=handle) == "output\n"
    # A queue that keeps nothing answers for every handle asked about rather than leaving it
    # absent, so a caller batching a whole host never re-asks one job at a time.
    assert backend.states(remote, "/repo", [handle]) == {
        handle: backend.state(remote, "/repo", handle=handle)
    }
    assert backend.state(remote, "/repo", handle=handle).verdict == "vanished"
    backend.cancel(remote, "/repo", handle=handle)


def test_a_bare_bash_host_hands_its_terminal_to_its_own_tool() -> None:
    """No daemon stands between the caller and the machine, so nothing is allocated for it."""
    assert Local().interactive(env="default", command=(), resources=Resources()) == (
        "mainboard shell default"
    )
    assert Local().interactive(env="default", command=("pwd",), resources=Resources()) == (
        "mainboard run --env default -- pwd"
    )
