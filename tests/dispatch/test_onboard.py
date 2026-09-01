from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.dispatch import HostSetup
from mainboard.dispatch import onboard as onboard_module
from mainboard.dispatch.onboard import (
    Onboarding,
    RemoteShell,
    facts_command,
    installers,
    read_facts,
)
from mainboard.dispatch.state import Cache

from .support import RecordingMachine, Rule, cache, machine_with, plan

if TYPE_CHECKING:
    from mainboard import ExecutionPlan

# One keyword `Onboarding` accepts past its dispatcher and plan, so the helper below forwards a
# test's overrides without widening them to anything the constructor would refuse.
type Setting = str | Sequence[str] | bool | Callable[[str], None]

_FACTS_JSON = '{"schema_version": 1, "hostname": "gold-1", "cpu_logical_cores": 72}'

# What the stock capability probe prints back, the shape `probe_capabilities` parses.
_CAPABILITIES = """root=/home/me/projects
kind=ssh
gpu=NVIDIA GH200, 97871
mem=536870912
account=me
queue=
pixi=/home/me/.pixi/bin/pixi
uv=/home/me/.local/bin/uv
platform=Linux aarch64
"""

# A machine that answers every onboarding step the way a healthy one would.
_HEALTHY: tuple[Rule, ...] = (
    ("MemTotal", 0, _CAPABILITIES),
    ("facts --json", 0, f"module chatter\n{_FACTS_JSON}\n"),
    ("--version", 0, "0.1.0\n"),
)


class FakeDispatcher:
    """A dispatch-core double that records the mirror and keeps a real state cache."""

    def __init__(self, store: Cache) -> None:
        self.cache = store
        self.mirrored: list[tuple[str, str]] = []
        self.required: list[Sequence[str]] = []

    def rsync_up(
        self, execution: ExecutionPlan, root: str, *, required: Sequence[Sequence[str]] = ()
    ) -> None:
        self.mirrored.append((execution.host, root))
        self.required = list(required)


def onboarding(
    host: RecordingMachine, monkeypatch: pytest.MonkeyPatch, **overrides: Setting
) -> tuple[Onboarding, FakeDispatcher]:
    """An `Onboarding` over `host`, with its connection and dispatcher stubbed out."""
    monkeypatch.setattr(onboard_module, "connection", lambda alias: host)
    dispatcher = FakeDispatcher(cache())
    fields: dict[str, Setting] = {"root": "/repo"}
    fields.update(overrides)
    return Onboarding(dispatcher, plan(), **fields), dispatcher


def test_the_remote_shell_stages_a_bare_command_and_activates_only_when_asked() -> None:
    """An unprovisioned machine has nothing to source, so onboarding stands on `cd` and PATH."""
    host = machine_with(rules=[("broken", 1, "")])
    shell = RemoteShell(host, plan(), "/repo")
    assert not shell.run("uv --version")
    assert host.lines[0].startswith("cd /repo && export PATH=")
    assert "activate.sh" not in host.lines[0]
    assert host.lines[0].endswith("uv --version")
    shell.run("mainboard facts", activate=True)
    assert "/repo/.mainboard/activate.sh" in host.lines[1]
    assert shell.ok("command -v uv")
    assert not shell.ok("missing broken thing")
    with pytest.raises(MissionError, match="`broken` failed on 'gold'"):
        shell.run("broken")


def test_the_install_routes_are_offered_best_first_and_all_read_the_synced_source() -> None:
    """uv leads because it needs no interpreter on the host new enough to run the tool."""
    routes = installers(RemoteShell(machine_with(), plan(), "/repo"), "packages/tool")
    assert routes.names == ["uv", "uv-bootstrap", "pip"]
    assert all("packages/tool" in routes.select(name).command for name in routes.names)
    assert "astral.sh/uv" in routes.select("uv-bootstrap").command


def test_bootstrap_falls_through_to_pip_keeping_every_rejection_it_passed_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with(rules=[("command -v uv", 1, ""), ("command -v curl", 1, "")])
    setup, _ = onboarding(host, monkeypatch)
    resolution = setup.bootstrap(RemoteShell(host, plan(), "/repo"))
    assert resolution.winner == "pip"
    assert [name for name, _ in resolution.rejected] == ["uv", "uv-bootstrap"]
    assert host.ran("pip install --user")


def test_bootstrap_refuses_a_host_no_route_can_reach_before_anything_assumes_the_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with(rules=[("command -v", 1, ""), ("pip --version", 1, "")])
    setup, _ = onboarding(host, monkeypatch)
    with pytest.raises(MissionError, match="cannot install mainboard on 'gold'"):
        setup.bootstrap(RemoteShell(host, plan(), "/repo"))


def test_read_facts_starts_at_the_first_brace_and_refuses_output_carrying_no_snapshot() -> None:
    assert facts_command() == "mainboard facts --json"
    assert read_facts(f"module: loading cuda\n{_FACTS_JSON}\n").hostname == "gold-1"
    with pytest.raises(MissionError, match="no host facts"):
        read_facts("command not found: mainboard\n")


def test_onboarding_probes_mirrors_installs_provisions_then_reads_the_host_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = machine_with(rules=_HEALTHY)
    setup, dispatcher = onboarding(host, monkeypatch)
    with caplog.at_level("INFO", logger="mainboard.dispatch"):
        report = setup.run()
    assert dispatcher.mirrored == [("gold", "/repo")]
    assert host.ran("uv tool install")
    assert host.ran("mainboard install default --profile gold")
    assert host.ran("test -f /repo/.mainboard/activate.sh")
    assert (report.installer, report.tool, report.env) == ("uv", "0.1.0", "default")
    assert report.activate == "/repo/.mainboard/activate.sh"
    assert report.capabilities is not None and report.capabilities.pixi.endswith("/pixi")
    assert report.hardware is not None and report.hardware.hostname == "gold-1"
    assert report.onboarded_at
    assert dispatcher.cache.host("gold").root == "/repo"
    assert [record.host for record in dispatcher.cache.hosts()] == ["gold"]
    stages = [message.split()[0] for message in caplog.messages]
    assert stages == ["probing", "mirroring", "installing", "provisioning", "reading", "onboarded"]
    bare = HostSetup(host="gold", root="/repo")
    assert (bare.env, bare.rejected, bare.capabilities, bare.hardware) == (
        "default",
        (),
        None,
        None,
    )


def test_onboarding_a_named_environment_verifies_that_environments_own_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host provisioned for `serving` must be checked and recorded against its own script."""
    host = machine_with(rules=_HEALTHY)
    monkeypatch.setattr(onboard_module, "connection", lambda alias: host)
    dispatcher = FakeDispatcher(cache())
    report = Onboarding(dispatcher, plan(env="serving"), root="/repo").run()
    assert host.ran("mainboard install serving --profile gold")
    assert host.ran("test -f /repo/.mainboard/activate-serving.sh")
    assert report.activate == "/repo/.mainboard/activate-serving.sh"


@pytest.mark.parametrize(
    ("artifact", "resolve", "installed"),
    [
        (
            (
                ".mainboard/envs/default/pixi.toml",
                ".mainboard/envs/default/pixi.lock",
                ".mainboard/envs/default/state.toml",
            ),
            False,
            "mainboard install default --profile gold",
        ),
        ((), True, "mainboard install default --resolve --profile gold"),
    ],
)
def test_onboarding_ships_the_compiled_artifact_unless_told_to_solve_on_the_host(
    monkeypatch: pytest.MonkeyPatch,
    artifact: tuple[str, ...],
    resolve: bool,
    installed: str,
) -> None:
    """A host's own compiler must never sit in the lock's dependency path."""
    host = machine_with(rules=_HEALTHY)
    watched: list[str] = []
    setup, dispatcher = onboarding(
        host, monkeypatch, artifact=artifact, resolve=resolve, watch=watched.append
    )
    setup.run()
    assert dispatcher.required == ([artifact] if artifact else [])
    assert host.ran(installed)
    assert [stage.split()[0] for stage in watched] == [
        "probing",
        "mirroring",
        "installing",
        "provisioning",
        "reading",
    ]


def test_onboarding_discovers_a_root_the_profile_never_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with(rules=(*_HEALTHY, ("ls -d /work", 0, "/work/grp/me/projects\n")))
    setup, dispatcher = onboarding(host, monkeypatch, root="")
    report = setup.run()
    assert report.root == "/work/grp/me/projects"
    assert dispatcher.mirrored == [("gold", "/work/grp/me/projects")]


def test_onboarding_refuses_a_provisioning_that_left_no_activation_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with(rules=(*_HEALTHY, ("test -f", 1, "")))
    setup, _ = onboarding(host, monkeypatch)
    with pytest.raises(MissionError, match=r"has no /repo/\.mainboard/activate\.sh"):
        setup.run()


def test_onboarding_stamps_the_manifest_digest_it_was_given_onto_the_recorded_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`doctor` tells a diverged host apart from a fresh one by comparing this field."""
    host = machine_with(rules=_HEALTHY)
    setup, dispatcher = onboarding(host, monkeypatch, digest="deadbeef")
    report = setup.run()
    assert report.digest == "deadbeef"
    assert dispatcher.cache.host("gold").digest == "deadbeef"


def test_sync_only_stamps_the_digest_it_was_given_and_keeps_the_old_one_when_given_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with()
    setup, dispatcher = onboarding(host, monkeypatch, digest="cafe")
    dispatcher.cache.save_host(HostSetup(host="gold", root="/repo", digest="stale"))
    assert setup.run(sync_only=True).digest == "cafe"

    bare, dispatcher = onboarding(host, monkeypatch)
    dispatcher.cache.save_host(HostSetup(host="gold", root="/repo", digest="stale"))
    assert bare.run(sync_only=True).digest == "stale"


def test_sync_only_re_mirrors_and_re_provisions_without_bootstrap_or_hardware_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast path back to a host whose environment drifted from a manifest that moved.

    Neither the tool nor the hardware changed, only the workspace and what compiles from it, so
    this must never reach the bootstrap cascade or the facts probe the way a full onboarding does.
    """
    host = machine_with()
    setup, dispatcher = onboarding(host, monkeypatch, root="")
    dispatcher.cache.save_host(HostSetup(host="gold", root="/repo", installer="uv", tool="0.1.0"))

    report = setup.run(sync_only=True)

    assert dispatcher.mirrored == [("gold", "/repo")]
    assert host.ran("mainboard install default --profile gold")
    assert not host.ran("uv tool install")
    assert not host.ran("facts --json")
    assert not host.ran("--version")
    assert (report.root, report.installer, report.tool) == ("/repo", "uv", "0.1.0")


def test_sync_only_prefers_a_given_root_over_the_recorded_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with()
    setup, dispatcher = onboarding(host, monkeypatch)
    dispatcher.cache.save_host(HostSetup(host="gold", root="/other", installer="uv"))
    report = setup.run(sync_only=True)
    assert report.root == "/repo"
    assert dispatcher.mirrored == [("gold", "/repo")]


def test_sync_only_refuses_a_host_that_was_never_onboarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with()
    setup, _ = onboarding(host, monkeypatch)
    with pytest.raises(LookupError, match="'gold' has never been set up"):
        setup.run(sync_only=True)
