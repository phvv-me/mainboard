from pathlib import PurePosixPath

import pytest

from mainboard.dispatch import HostUnreachable, SshTransport
from mainboard.dispatch import wrapping as wrapping_module
from mainboard.dispatch.wrapping import activation, activation_stage, argv, connection, wrap
from mainboard.manifest import Container, HostProfile

from .support import plan

# Mirrors wrapping.py's own private per-user install dirs, kept as a literal here rather than
# imported so a new bin dir demands a deliberate test update, not silent inherited coverage.
_USER_BINS = ("$HOME/.local/bin", "$HOME/.pixi/bin", "$HOME/.cargo/bin")
_CONTAINER = Container(image="nvcr.io/nvidia/pytorch:25.06-py3")


class _FakeMachine:
    """The only parts of an `SshMachine` that `_open` reaches for, its PATH and its cwd."""

    def __init__(self) -> None:
        self.env = type("Env", (), {"path": []})()
        self.cwd = PurePosixPath("/home/user")


def test_wrap_stages_cd_then_path_then_modules_before_the_environment() -> None:
    steps = wrap(plan(), "/repo", command="python -m foo").split(" && ")
    assert steps[0] == "cd /repo"
    assert steps[1] == f"export PATH={':'.join(_USER_BINS)}:$PATH"
    assert "if [ -f /repo/.mainboard/activate.sh ]" in steps[2]
    assert "export PATH=/repo/.mainboard/envs/default/.pixi/envs/default/bin:$PATH" in steps[2]
    assert steps[-1] == "python -m foo"
    assert "module" not in " && ".join(steps)
    moduled = wrap(
        plan(profile=HostProfile(modules={"cuda": "13.0", "gcc": ""})), "/repo", command="run"
    )
    assert "module purge" in moduled
    assert "module load cuda/13.0" in moduled
    assert "module load gcc" in moduled
    assert "module load gcc/" not in moduled


def test_wrap_without_activation_stops_at_the_footing_an_unprovisioned_host_offers() -> None:
    """An onboarding stands on `cd`, PATH and modules while the host has no environment yet."""
    host = HostProfile(modules={"cuda": "13.0"}, container="ngc")
    line = wrap(
        plan(profile=host, container=_CONTAINER),
        "/repo",
        command="uv tool install .",
        activate=False,
    )
    assert line.split(" && ")[0] == "cd /repo"
    assert "module load cuda/13.0" in line
    assert "activate.sh" not in line
    assert line.endswith("uv tool install .")


def test_wrap_containerized_delegates_to_the_injected_builder_or_refuses_without_one() -> None:
    containerized = plan(profile=HostProfile(container="ngc"), container=_CONTAINER)
    with pytest.raises(LookupError, match="no container argv builder"):
        wrap(containerized, "/repo", command="python -m foo")
    line = wrap(
        containerized,
        "/repo",
        command="python -m foo",
        containerize=lambda inner: ["apptainer", "exec", "image.sif", *inner],
    )
    assert "apptainer exec image.sif bash -c 'python -m foo'" in line
    assert "activate.sh" not in line


def test_the_activation_stage_tries_the_named_script_then_the_prefix_then_refuses() -> None:
    """A silent wrong interpreter costs far more to find than a command that refuses to start."""
    assert activation("/repo") == "/repo/.mainboard/activate.sh"
    assert activation("/repo", env="serving") == "/repo/.mainboard/activate-serving.sh"
    default = activation_stage(plan(), "/repo")
    assert "/repo/.chefe/activate.sh" in default
    assert "elif [ -d /repo/.mainboard/envs/default/.pixi/envs/default/bin ]" in default
    assert default.endswith("exit 1; fi")
    serving = activation_stage(plan(env="serving"), "/repo")
    assert "if [ -f /repo/.mainboard/activate-serving.sh ]" in serving
    assert ".chefe" not in serving
    assert "envs/default" not in serving


def test_the_refusal_names_the_command_that_provisions_the_environment_where_it_ran() -> None:
    remote = activation_stage(plan(env="vserve"), "/repo")
    assert (
        "found no vserve environment at /repo/.mainboard/envs/vserve/.pixi/envs/vserve on gold"
        in remote
    )
    assert "mainboard install vserve --on gold" in remote
    here = activation_stage(plan(host="local", env="vserve"), "/repo")
    assert "mainboard install vserve`" in here
    assert "--on" not in here


def test_a_named_environment_is_never_optional_however_the_caller_asks() -> None:
    """Naming an environment is the user stating which interpreter they want."""
    assert "exit 1" in activation_stage(plan(env="vserve"), "/repo", optional=True)
    assert "found no vserve environment" in wrap(
        plan(env="vserve"), "/repo", command="python -c 1"
    )


@pytest.mark.parametrize(("login", "flag"), [(True, "-lc"), (False, "-c")])
def test_argv_wraps_the_staged_line_in_a_login_or_a_plain_bash(login: bool, flag: str) -> None:
    built = argv(plan(), "/repo", command="run", login=login)
    assert built[:2] == ["bash", flag]
    assert built[2] == wrap(plan(), "/repo", command="run")


def test_open_warms_the_host_before_plumbum_and_prepends_the_user_install_dirs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warm-up rides a robust one-shot channel so an expired ControlMaster relogs there."""
    warmed: list[str] = []
    machine = _FakeMachine()
    monkeypatch.setattr(SshTransport, "warm", lambda self, host: warmed.append(host))
    monkeypatch.setattr(SshTransport, "machine", lambda self, host: machine)
    assert wrapping_module._open("gold", SshTransport()) is machine  # ruff:ignore[private-member-access]  reason=unit-tests the module-private connection opener since=2026-08-16
    assert warmed == ["gold"]
    assert [str(path) for path in machine.env.path] == [
        f"/home/user/{bindir.removeprefix('$HOME/')}" for bindir in _USER_BINS
    ]


@pytest.mark.parametrize(
    ("failures", "error", "attempts", "raised"),
    [
        (0, HostUnreachable("blip"), 1, None),
        (1, HostUnreachable("blip"), 2, None),
        (9, HostUnreachable("still down"), 3, HostUnreachable),
        (9, ConnectionError("host key verification failed"), 1, ConnectionError),
    ],
)
def test_connection_rides_out_a_transient_blip_but_never_retries_a_host_key_failure(
    monkeypatch: pytest.MonkeyPatch,
    failures: int,
    error: BaseException,
    attempts: int,
    raised: type[BaseException] | None,
) -> None:
    """A rotated host key is not transient, so retrying it only delays the real diagnosis."""
    monkeypatch.setattr(wrapping_module, "_CONNECT_ATTEMPTS", 3)
    monkeypatch.setattr(wrapping_module, "_CONNECT_BACKOFF", 0.0)
    tried: list[str] = []

    def opening(host: str, ssh: SshTransport) -> str:
        tried.append(host)
        if len(tried) <= failures:
            raise error
        return "SESSION"

    monkeypatch.setattr(wrapping_module, "_open", opening)
    if raised is None:
        assert connection("gold") == "SESSION"
    else:
        with pytest.raises(raised):
            connection("gold")
    assert len(tried) == attempts
