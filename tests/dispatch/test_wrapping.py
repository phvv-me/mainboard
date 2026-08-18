from pathlib import PurePosixPath

import pytest

from mainboard import ExecutionPlan
from mainboard.dispatch import HostUnreachable, SshTransport
from mainboard.dispatch import wrapping as wrapping_module
from mainboard.dispatch.wrapping import activation, activation_stage, argv, connection, wrap
from mainboard.manifest import Container, HostProfile

type PlanField = str | HostProfile | Container | dict[str, str] | None

# Mirrors wrapping.py's own private per-user install dirs; kept as a literal here rather than
# imported so a new bin dir demands a deliberate test update, not silent inherited coverage.
_USER_BINS = ("$HOME/.local/bin", "$HOME/.pixi/bin", "$HOME/.cargo/bin")


def plan(**overrides: PlanField) -> ExecutionPlan:
    fields: dict[str, PlanField] = {"host": "gold", "profile": HostProfile(), "env": "default"}
    fields.update(overrides)
    return ExecutionPlan.model_validate(fields)


# --- wrap() ---


def test_wrap_stages_cd_path_and_bare_activation() -> None:
    line = wrap(plan(), "/repo", command="python -m foo")
    steps = line.split(" && ")
    assert steps[0] == "cd /repo"
    assert steps[1] == f"export PATH={':'.join(_USER_BINS)}:$PATH"
    assert "if [ -f /repo/.mainboard/activate.sh ]" in steps[2]
    assert steps[-1] == "python -m foo"


def test_wrap_falls_back_to_a_path_prepend_when_activate_sh_is_absent() -> None:
    line = wrap(plan(), "/repo", command="python -m foo")
    assert "export PATH=/repo/.mainboard/.pixi/envs/default/bin:$PATH" in line


def test_wrap_loads_declared_modules_with_and_without_a_pinned_version() -> None:
    host = HostProfile(modules={"cuda": "13.0", "gcc": ""})
    line = wrap(plan(profile=host), "/repo", command="run")
    assert "module purge" in line
    assert "module load cuda/13.0" in line
    assert "module load gcc" in line
    assert "module load gcc/" not in line


def test_wrap_without_activation_stops_after_the_staging() -> None:
    line = wrap(plan(), "/repo", command="uv tool install .", activate=False)
    steps = line.split(" && ")
    assert steps[0] == "cd /repo"
    assert steps[-1] == "uv tool install ."
    assert "activate.sh" not in line


def test_wrap_without_activation_still_loads_the_declared_modules() -> None:
    host = HostProfile(modules={"cuda": "13.0"})
    line = wrap(plan(profile=host), "/repo", command="uv --version", activate=False)
    assert "module load cuda/13.0" in line


def test_wrap_without_activation_never_enters_the_container() -> None:
    host = HostProfile(container="ngc")
    container = Container(image="nvcr.io/nvidia/pytorch:25.06-py3")
    line = wrap(
        plan(profile=host, container=container), "/repo", command="uv --version", activate=False
    )
    assert line.endswith("uv --version")


def test_activation_names_the_workspace_script() -> None:
    assert activation("/repo") == "/repo/.mainboard/activate.sh"
    assert activation("/repo", "serving") == "/repo/.mainboard/activate-serving.sh"


def test_wrap_sources_the_requested_environments_own_activation() -> None:
    """`--env serving` must never reach the default environment's script or prefix."""
    line = wrap(plan(env="serving"), "/repo", command="python -c 'import sys'")
    assert "if [ -f /repo/.mainboard/activate-serving.sh ]" in line
    assert "/repo/.mainboard/activate.sh" not in line
    assert "export PATH=/repo/.mainboard/.pixi/envs/serving/bin:$PATH" in line
    assert "envs/default" not in line


def test_wrap_skips_module_stage_when_none_declared() -> None:
    line = wrap(plan(), "/repo", command="run")
    assert "module" not in line


def test_wrap_containerized_without_a_builder_raises() -> None:
    host = HostProfile(container="ngc")
    container = Container(image="nvcr.io/nvidia/pytorch:25.06-py3")
    with pytest.raises(LookupError, match="no container argv builder"):
        wrap(plan(profile=host, container=container), "/repo", command="python -m foo")


def test_wrap_containerized_delegates_to_the_injected_builder() -> None:
    host = HostProfile(container="ngc")
    container = Container(image="nvcr.io/nvidia/pytorch:25.06-py3")

    def containerize(inner: list[str]) -> list[str]:
        return ["apptainer", "exec", "image.sif", *inner]

    line = wrap(
        plan(profile=host, container=container),
        "/repo",
        command="python -m foo",
        containerize=containerize,
    )
    assert "apptainer exec image.sif bash -c 'python -m foo'" in line
    assert "activate.sh" not in line


# --- argv() ---


def test_argv_uses_login_shell_by_default() -> None:
    assert argv(plan(), "/repo", command="run")[:2] == ["bash", "-lc"]


def test_argv_uses_plain_shell_when_login_is_false() -> None:
    assert argv(plan(), "/repo", command="run", login=False)[:2] == ["bash", "-c"]


# --- connection() / _open() ---


class _FakeMachine:
    def __init__(self) -> None:

        self.env = type("Env", (), {"path": []})()
        self.cwd = PurePosixPath("/home/user")


def test_open_warms_the_host_and_prepends_user_bins(monkeypatch: pytest.MonkeyPatch) -> None:
    warmed: list[str] = []
    machine = _FakeMachine()

    monkeypatch.setattr(SshTransport, "warm", lambda self, host: warmed.append(host))
    monkeypatch.setattr(SshTransport, "machine", lambda self, host: machine)
    result = wrapping_module._open("gold", SshTransport())  # ruff:ignore[private-member-access]  reason=unit-tests the module-private connection opener since=2026-08-16
    assert warmed == ["gold"]
    assert result is machine
    expected = [f"/home/user/{b.removeprefix('$HOME/')}" for b in _USER_BINS]
    assert [str(path) for path in machine.env.path] == expected


def test_connection_returns_on_the_first_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_open(host: str, ssh: SshTransport) -> str:
        calls.append(host)
        return "SESSION"

    monkeypatch.setattr(wrapping_module, "_open", fake_open)
    assert connection("gold") == "SESSION"
    assert calls == ["gold"]


def test_connection_retries_a_transient_blip_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wrapping_module, "_CONNECT_ATTEMPTS", 3)
    monkeypatch.setattr(wrapping_module, "_CONNECT_BACKOFF", 0.0)
    attempts = {"n": 0}

    def flaky(host: str, ssh: SshTransport) -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise HostUnreachable("blip")
        return "SESSION"

    monkeypatch.setattr(wrapping_module, "_open", flaky)
    assert connection("gold") == "SESSION"
    assert attempts["n"] == 2


def test_connection_gives_up_after_the_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrapping_module, "_CONNECT_ATTEMPTS", 2)
    monkeypatch.setattr(wrapping_module, "_CONNECT_BACKOFF", 0.0)

    def always_down(host: str, ssh: SshTransport) -> str:
        raise HostUnreachable("still down")

    monkeypatch.setattr(wrapping_module, "_open", always_down)
    with pytest.raises(HostUnreachable, match="still down"):
        connection("gold")


def test_connection_does_not_retry_a_host_key_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrapping_module, "_CONNECT_ATTEMPTS", 5)
    monkeypatch.setattr(wrapping_module, "_CONNECT_BACKOFF", 0.0)
    attempts = {"n": 0}

    def bad_key(host: str, ssh: SshTransport) -> str:
        attempts["n"] += 1
        raise ConnectionError("host key verification failed")

    monkeypatch.setattr(wrapping_module, "_open", bad_key)
    with pytest.raises(ConnectionError):
        connection("gold")
    assert attempts["n"] == 1


def test_activation_stage_falls_back_to_the_prefix_before_refusing() -> None:
    stage = activation_stage(plan(), "/repo")
    assert "elif [ -d /repo/.mainboard/.pixi/envs/default/bin ]" in stage
    assert stage.endswith("exit 1; fi")


def test_activation_stage_offers_the_chefe_script_only_to_the_default_environment() -> None:
    """chefe only ever wrote one activation, so a named env sourcing it would get the wrong env."""
    assert "/repo/.chefe/activate.sh" in activation_stage(plan(), "/repo")
    assert ".chefe" not in activation_stage(plan(env="serving"), "/repo")


def test_activation_stage_refusal_names_the_command_that_provisions_the_environment() -> None:
    """A silent wrong interpreter costs far more to find than a command that refuses to start."""
    stage = activation_stage(plan(env="vserve"), "/repo")
    assert "found no vserve environment at /repo/.mainboard/.pixi/envs/vserve on gold" in stage
    assert "mainboard install vserve --on gold" in stage


def test_activation_stage_refusal_on_this_machine_names_a_local_install() -> None:
    stage = activation_stage(plan(host="local", env="vserve"), "/repo")
    assert "mainboard install vserve`" in stage
    assert "--on" not in stage


def test_a_named_environment_is_never_optional_however_the_caller_asks() -> None:
    """Naming an environment is the user stating which interpreter they want."""
    assert "exit 1" in activation_stage(plan(env="vserve"), "/repo", optional=True)


def test_wrap_refuses_a_named_environment_the_machine_never_provisioned() -> None:
    line = wrap(plan(env="vserve"), "/repo", command="python -c 1")
    assert "found no vserve environment" in line
    assert "exit 1" in line
