from typing import TYPE_CHECKING

import pytest

from mainboard import ExecutionPlan, MissionError
from mainboard.dispatch import onboard as onboard_module
from mainboard.dispatch.onboard import (
    HostSetup,
    Onboarding,
    RemoteShell,
    facts_command,
    installers,
    read_facts,
)
from mainboard.dispatch.state import Cache
from mainboard.manifest import HostProfile

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from types import TracebackType

_FACTS_JSON = '{"schema_version": 1, "hostname": "gold-1", "cpu_logical_cores": 72}'

# What the stock capability probe prints back, the shape `probe_capabilities` parses.
_CAPABILITIES = (
    "root=/home/me/projects\nkind=ssh\ngpu=NVIDIA GH200, 97871\nmem=536870912\n"
    "account=me\nqueue=\npixi=/home/me/.pixi/bin/pixi\nuv=/home/me/.local/bin/uv\n"
    "platform=Linux aarch64\n"
)


class Shell:
    """A host-shell double: every `bash -lc <line>` is answered from an ordered rule table.

    rules: `(marker, retcode, stdout)` triples; the first marker found in the line wins, and a
    line matching nothing succeeds silently, the way a `test`/`command -v` probe usually does.
    """

    def __init__(self, rules: Sequence[tuple[str, int, str]] = ()) -> None:
        self.rules = list(rules)
        self.lines: list[str] = []

    def __getitem__(self, name: str) -> Command:
        return Command(self, [name])

    def __enter__(self) -> Shell:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    def answer(self, line: str) -> tuple[int, str, str]:
        """The scripted `(retcode, stdout, stderr)` for `line`, recording it as run."""
        self.lines.append(line)
        for marker, retcode, out in self.rules:
            if marker in line:
                return retcode, out, f"{marker}: refused" if retcode else ""
        return 0, "", ""

    def ran(self, marker: str) -> bool:
        """Whether any line run so far carried `marker`."""
        return any(marker in line for line in self.lines)


class Command:
    """A plumbum-command double binding argv until it is called or `run`."""

    def __init__(self, shell: Shell, bound: list[str]) -> None:
        self.shell = shell
        self.bound = bound

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> Command:
        extra = list(args) if isinstance(args, list | tuple) else [args]
        return Command(self.shell, [*self.bound, *extra])

    def __call__(self, *_: object, **__: object) -> str:
        return self.shell.answer(self.bound[-1])[1]

    def run(self, *_: object, **__: object) -> tuple[int, str, str]:
        return self.shell.answer(self.bound[-1])


class FakeDispatcher:
    """A dispatch-core double: it records the mirror and keeps a real state cache."""

    def __init__(self, cache: Cache) -> None:
        self.cache = cache
        self.mirrored: list[tuple[str, str]] = []

    def rsync_up(self, plan: ExecutionPlan, root: str) -> None:
        self.mirrored.append((plan.host, root))


def plan(**overrides: str | HostProfile) -> ExecutionPlan:
    fields: dict[str, str | HostProfile] = {
        "host": "gold",
        "profile": HostProfile(root="/repo"),
        "env": "default",
    }
    fields.update(overrides)
    return ExecutionPlan.model_validate(fields)


def onboarding(shell: Shell, cache: Cache, **overrides: str) -> tuple[Onboarding, FakeDispatcher]:
    """An `Onboarding` over `shell`, with its connection and dispatcher stubbed out."""
    dispatcher = FakeDispatcher(cache)
    fields: dict[str, str] = {"root": "/repo"}
    fields.update(overrides)
    return Onboarding(dispatcher, plan(), **fields), dispatcher


def onboarded_shell() -> Shell:
    """A host that answers every onboarding step the way a healthy machine would."""
    return Shell(
        [
            ("MemTotal", 0, _CAPABILITIES),
            ("facts --json", 0, f"module chatter\n{_FACTS_JSON}\n"),
            ("--version", 0, "0.1.0\n"),
        ]
    )


# --- RemoteShell ---


def test_remote_shell_runs_a_bare_command_without_any_activation() -> None:
    shell = Shell()
    assert not RemoteShell(shell, plan(), "/repo").run("uv --version")
    [line] = shell.lines
    assert line.startswith("cd /repo && export PATH=")
    assert "activate.sh" not in line
    assert line.endswith("uv --version")


def test_remote_shell_activated_sources_the_workspace_activation() -> None:
    shell = Shell()
    RemoteShell(shell, plan(), "/repo").run("mainboard facts", activate=True)
    assert "/repo/.mainboard/activate.sh" in shell.lines[0]


def test_remote_shell_raises_a_mission_error_naming_the_failing_command() -> None:
    shell = Shell([("broken", 1, "")])
    with pytest.raises(MissionError, match="`broken` failed on 'gold'"):
        RemoteShell(shell, plan(), "/repo").run("broken")


def test_remote_shell_ok_reports_a_probe_without_raising() -> None:
    shell = Shell([("missing", 1, "")])
    remote = RemoteShell(shell, plan(), "/repo")
    assert remote.ok("command -v uv")
    assert not remote.ok("missing")


# --- installers ---


def test_installers_offer_uv_then_a_bootstrapped_uv_then_pip() -> None:
    routes = installers(RemoteShell(Shell(), plan(), "/repo"), "packages/tool")
    assert routes.names == ["uv", "uv-bootstrap", "pip"]
    assert all("packages/tool" in routes.select(name).command for name in routes.names)
    assert "astral.sh/uv" in routes.select("uv-bootstrap").command


def test_bootstrap_falls_through_to_pip_and_keeps_every_rejection(tmp_path: Path) -> None:
    shell = Shell([("command -v uv", 1, ""), ("command -v curl", 1, "")])
    setup, _ = onboarding(shell, Cache(tmp_path / "db.sqlite"))
    resolution = setup.bootstrap(RemoteShell(shell, plan(), "/repo"))
    assert resolution.winner == "pip"
    assert [name for name, _ in resolution.rejected] == ["uv", "uv-bootstrap"]
    assert shell.ran("pip install --user")


def test_bootstrap_refuses_a_host_with_no_install_route(tmp_path: Path) -> None:
    shell = Shell([("command -v", 1, ""), ("pip --version", 1, "")])
    setup, _ = onboarding(shell, Cache(tmp_path / "db.sqlite"))
    with pytest.raises(MissionError, match="cannot install mainboard on 'gold'"):
        setup.bootstrap(RemoteShell(shell, plan(), "/repo"))


# --- read_facts ---


def test_read_facts_ignores_the_shell_chatter_above_the_json() -> None:
    facts = read_facts(f"module: loading cuda\n{_FACTS_JSON}\n")
    assert facts.hostname == "gold-1"


def test_read_facts_refuses_output_carrying_no_snapshot() -> None:
    with pytest.raises(MissionError, match="no host facts"):
        read_facts("command not found: mainboard\n")


def test_facts_command_asks_the_installed_tool_for_json() -> None:
    assert facts_command() == "mainboard facts --json"


# --- Onboarding ---


def test_onboarding_mirrors_installs_provisions_and_records_the_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = onboarded_shell()
    monkeypatch.setattr(onboard_module, "connection", lambda host: shell)
    cache = Cache(tmp_path / "db.sqlite")
    stages: list[str] = []
    setup, dispatcher = onboarding(shell, cache, watch=stages.append)
    report = setup.run()
    assert dispatcher.mirrored == [("gold", "/repo")]
    assert shell.ran("uv tool install")
    assert shell.ran("mainboard install default --resolve --profile gold")
    assert shell.ran("test -f /repo/.mainboard/activate.sh")
    assert report.installer == "uv"
    assert report.activate == "/repo/.mainboard/activate.sh"
    assert report.tool == "0.1.0"
    assert report.capabilities is not None
    assert report.capabilities.pixi.endswith("/pixi")
    assert report.hardware is not None
    assert report.hardware.hostname == "gold-1"
    assert report.onboarded_at
    assert cache.host("gold").root == "/repo"
    assert [record.host for record in cache.hosts()] == ["gold"]
    assert [stage.split()[0] for stage in stages] == [
        "probing",
        "mirroring",
        "installing",
        "provisioning",
        "reading",
    ]


def test_onboarding_a_named_environment_verifies_that_environments_own_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host provisioned for `serving` must be checked and recorded against its own script."""
    shell = onboarded_shell()
    monkeypatch.setattr(onboard_module, "connection", lambda host: shell)
    dispatcher = FakeDispatcher(Cache(tmp_path / "db.sqlite"))
    report = Onboarding(dispatcher, plan(env="serving"), root="/repo", env="serving").run()
    assert shell.ran("mainboard install serving --resolve --profile gold")
    assert shell.ran("test -f /repo/.mainboard/activate-serving.sh")
    assert report.activate == "/repo/.mainboard/activate-serving.sh"


def test_onboarding_discovers_a_root_the_profile_never_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = onboarded_shell()
    shell.rules.append(("ls -d /work", 0, "/work/grp/me/projects\n"))
    monkeypatch.setattr(onboard_module, "connection", lambda host: shell)
    setup, dispatcher = onboarding(shell, Cache(tmp_path / "db.sqlite"), root="")
    report = setup.run()
    assert report.root == "/work/grp/me/projects"
    assert dispatcher.mirrored == [("gold", "/work/grp/me/projects")]


def test_onboarding_announces_its_stages_through_the_logger_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    shell = onboarded_shell()
    monkeypatch.setattr(onboard_module, "connection", lambda host: shell)
    setup, _ = onboarding(shell, Cache(tmp_path / "db.sqlite"))
    with caplog.at_level("INFO", logger="mainboard.dispatch"):
        setup.run()
    assert any("probing gold" in message for message in caplog.messages)
    assert any("onboarded" in message for message in caplog.messages)


def test_onboarding_refuses_a_provisioning_that_left_no_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = onboarded_shell()
    shell.rules.append(("test -f", 1, ""))
    monkeypatch.setattr(onboard_module, "connection", lambda host: shell)
    setup, _ = onboarding(shell, Cache(tmp_path / "db.sqlite"))
    with pytest.raises(MissionError, match=r"has no /repo/\.mainboard/activate\.sh"):
        setup.run()


# --- HostSetup ---


def test_host_setup_defaults_describe_a_machine_nothing_probed() -> None:
    setup = HostSetup(host="gold", root="/repo")
    assert setup.env == "default"
    assert setup.rejected == ()
    assert setup.capabilities is None
    assert setup.hardware is None
