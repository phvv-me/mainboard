"""The declared children and the process-compose YAML they become: which run, which are off and
why, and a file that is strict, unexpanded and secret-free."""

from collections.abc import Sequence, Set
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from mainboard import MissionError
from mainboard.serve.compose import Compose
from mainboard.serve.constants import SERVICES, Service
from mainboard.serve.enums import Kind
from mainboard.serve.keeper import Layout

from .fakes import programs

_DECLARED = {service.name: service for service in SERVICES}

# One command with the characters a YAML, a shell or process-compose's expansion could bend.
_COMMAND = ("/bin/sh", "-c", "exit $1 # 50% {{.X}}")
_NAMES = st.text(alphabet="abcdefgh", min_size=1, max_size=6)


def _services(*, every: st.SearchStrategy[str], off: st.SearchStrategy[str]) -> st.SearchStrategy:
    return st.lists(
        st.builds(
            Service,
            name=_NAMES,
            command=st.just(_COMMAND),
            every=every,
            timeout=st.integers(1, 900),
            off=off,
            stop=st.integers(1, 30),
        ),
        unique_by=lambda service: service.name,
        max_size=6,
    )


def _written(root: Path, services: Sequence[Service], dotenv: Set[str] = frozenset()):
    """The YAML `Compose` writes for `services` under `root`, read back as plain data."""
    layout = Layout(root / ".mb" / "serve")
    layout.state.mkdir(parents=True, exist_ok=True)
    Compose(services, root=root, layout=layout, programs=programs(root), dotenv=dotenv).write()
    return yaml.safe_load(layout.compose.read_text(encoding="utf-8"))


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(services=_services(every=st.sampled_from(["", "60s"]), off=st.sampled_from(["", "later"])))
def test_the_yaml_holds_exactly_the_children_that_are_not_off_and_is_strict(
    tmp_path: Path, services: list[Service]
) -> None:
    written = _written(tmp_path, services)
    assert sorted(written["processes"] or {}) == sorted(s.name for s in services if not s.off)
    assert (written["is_strict"], written["disable_env_expansion"]) == (True, True)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(services=_services(every=st.just(""), off=st.just("")))
def test_a_resident_child_runs_exactly_as_written_and_is_always_restarted(
    tmp_path: Path, services: list[Service]
) -> None:
    children = _written(tmp_path, services)["processes"] or {}
    assert {name: child["command"] for name, child in children.items()} == dict.fromkeys(
        children, "exec /bin/sh -c 'exit $1 # 50% {{.X}}'"
    )
    assert {child["availability"]["restart"] for child in children.values()} <= {"always"}
    assert all(child["is_template_disabled"] for child in children.values())


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(services=_services(every=st.sampled_from(["60s", "20m"]), off=st.just("")))
def test_a_pass_runs_under_proc_timeout_once_at_a_time_on_its_interval(
    tmp_path: Path, services: list[Service]
) -> None:
    children = _written(tmp_path, services)["processes"] or {}
    tool = tmp_path / "env" / "bin" / "mainboard"
    assert {name: child["command"].split()[:6] for name, child in children.items()} == {
        s.name: ["exec", str(tool), "proc", "timeout", str(s.timeout), "/bin/sh"] for s in services
    }
    assert {name: child["schedule"] for name, child in children.items()} == {
        s.name: {"interval": s.every, "run_on_start": True, "max_concurrent": 1} for s in services
    }


def test_the_written_yaml_is_owner_only_and_the_stop_allows_every_childs_timeout(
    tmp_path: Path, stand_ins: list[Service]
) -> None:
    layout = Layout(tmp_path / ".mb" / "serve")
    layout.state.mkdir(parents=True)
    compose = Compose(
        stand_ins, root=tmp_path, layout=layout, programs=programs(tmp_path), dotenv=set()
    )
    compose.write()
    assert layout.compose.stat().st_mode & 0o777 == 0o600
    assert compose.stop_seconds() == 3 + 2 + 10


@pytest.mark.parametrize(
    ("environment", "refusal"),
    [
        pytest.param(
            (("OPENAI_API_KEY", "x"),), "which the workspace .env defines", id="a-dotenv-name"
        ),
        pytest.param(
            (("PATH", "/opt/bin"),), "redefines PATH of the allowlist", id="the-allowlisted-path"
        ),
    ],
)
def test_a_child_asking_for_a_secret_or_another_path_is_refused_and_nothing_is_written(
    tmp_path: Path, environment: tuple[tuple[str, str], ...], refusal: str
) -> None:
    asking = Service(name="sink", command=("/bin/true",), environment=environment)
    with pytest.raises(MissionError, match=refusal):
        _written(tmp_path, [asking], dotenv={"OPENAI_API_KEY"})
    assert not (tmp_path / ".mb" / "serve" / "compose.yaml").exists()


def test_a_childs_own_variables_reach_the_yaml_as_written(tmp_path: Path) -> None:
    token = Service(name="sink", command=("/bin/true",), environment=(("TOKEN_FILE", "/s/t"),))
    assert _written(tmp_path, (token,))["processes"]["sink"]["environment"] == ["TOKEN_FILE=/s/t"]


def test_a_program_named_but_not_found_here_is_refused_by_name(tmp_path: Path) -> None:
    missing = Service(name="sink", command=("duckdb-otlp", "serve"))
    with pytest.raises(MissionError, match="duckdb-otlp is not installed"):
        _written(tmp_path, (missing,))


def test_every_child_starts_from_the_same_six_variables_and_nothing_else() -> None:
    allowed = Compose.allowed()
    assert sorted(allowed) == ["HOME", "LANG", "LOGNAME", "PATH", "TMPDIR", "USER"]
    assert allowed["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"


def test_the_registry_declares_the_roadmaps_children_once_each() -> None:
    assert list(_DECLARED) == [
        *("sink", "claude", "codex"),
        *("save", "collect", "settle", "drain", "reap", "tidy"),
    ]
    assert {name for name, service in _DECLARED.items() if service.kind is Kind.RESIDENT} == {
        "sink",
        "claude",
        "codex",
    }


@pytest.mark.parametrize("name", ["claude", "codex"])
def test_a_remote_control_child_is_declared_but_stays_off_until_n1(name: str) -> None:
    assert "N1 is not applied" in _DECLARED[name].off
    assert _DECLARED[name].command[1] == "remote-control"


def test_codex_remote_control_would_run_asking_and_inside_the_workspace() -> None:
    assert _DECLARED["codex"].command[2:] == (
        *("-c", "approval_policy=on-request"),
        *("-c", "sandbox_mode=workspace-write"),
    )


def test_the_settle_pass_waits_off_while_todays_settler_unit_runs_until_cvpr() -> None:
    settle = _DECLARED["settle"]
    assert (settle.command, settle.every) == (("mainboard", "job", "list", "--json"), "20m")
    assert "Nov 16" in settle.off


@pytest.mark.parametrize(
    ("declared", "refusal"),
    [
        pytest.param({"name": "sink"}, "runs but has no command", id="a-running-child-with-none"),
        pytest.param(
            {"name": "save", "command": ("/bin/true",), "every": "60s"},
            "runs every 60s with no timeout",
            id="an-unbounded-pass",
        ),
    ],
)
def test_a_child_that_runs_needs_a_command_and_a_pass_needs_a_bound(
    declared: dict[str, str | tuple[str, ...]], refusal: str
) -> None:
    with pytest.raises(ValidationError, match=refusal):
        Service.model_validate(declared)


def test_a_child_that_is_off_needs_neither_a_command_nor_a_bound() -> None:
    assert Service(name="drain", every="5m", off="v2-drain builds it").kind is Kind.PERIODIC
