import shlex
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.dispatch import Facts, HostSetup
from mainboard.dispatch import onboard as onboard_module
from mainboard.dispatch.onboard import (
    Bootstrap,
    Onboarding,
    facts_command,
    gpus_command,
    installers,
    read_facts,
    satisfied_by,
)
from mainboard.dispatch.shells import PosixShell
from mainboard.dispatch.state import Cache
from mainboard.engines.compile.backend import PIXI_VERSION

from .support import (
    RecordingMachine,
    Rule,
    cache,
    machine_with,
    plan,
    run_record,
)

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
    ("facts --json", 0, f"module chatter\n{_FACTS_JSON}\n"),
    ("pixi --version", 0, f"pixi {PIXI_VERSION}\n"),
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
    """An `Onboarding` over `host`, with its probe, connection and dispatcher stubbed out."""
    monkeypatch.setattr(
        onboard_module,
        "probe_capabilities",
        lambda alias, ssh=None: Facts.parsed(alias, _CAPABILITIES),
    )
    monkeypatch.setattr(
        onboard_module,
        "open_shell",
        lambda execution, root, ssh=None: PosixShell(host, execution, root),
    )
    dispatcher = FakeDispatcher(cache())
    fields: dict[str, Setting] = {"root": "/repo"}
    fields.update(overrides)
    return Onboarding(dispatcher, plan(), **fields), dispatcher


def test_the_remote_shell_stages_a_bare_command_and_activates_only_when_asked() -> None:
    """An unprovisioned machine has nothing to source, so onboarding stands on `cd` and PATH."""
    host = machine_with(rules=[("broken", 1, "")])
    shell = PosixShell(host, plan(), "/repo")
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
    routes = installers(PosixShell(machine_with(), plan(), "/repo"), "packages/tool")
    assert routes.names == ["uv", "uv-bootstrap", "pip"]
    assert all("packages/tool" in routes.select(name).command for name in routes.names)
    assert "astral.sh/uv" in routes.select("uv-bootstrap").command
    assert "--python '>=3.14'" in routes.select("uv").command


def test_a_workspace_that_vendors_no_source_installs_the_version_it_declares() -> None:
    """A standalone workspace consumes the tool from an index and ships no source at all.

    Every route used to test for that directory, so all three refused on a host whose uv, pip
    and mainboard were all present, and the refusal blamed the host's tooling for something the
    workspace had never sent (miyabi-g, 2026-09-05).
    """
    shell = PosixShell(machine_with(), plan(), "/repo")
    routes = installers(shell, "packages/tool", vendored=False, floor=">=0.4.8")

    assert routes.names == ["present", "uv-index", "uv-bootstrap-index", "pip-index"]
    assert all("packages/tool" not in routes.select(name).command for name in routes.names)
    assert routes.select("uv-index").command == (
        "uv tool install --force --python '>=3.14' 'mainboard>=0.4.8'"
    )
    assert routes.select("pip-index").command.endswith("--upgrade 'mainboard>=0.4.8'")
    assert "astral.sh/uv" in routes.select("uv-bootstrap-index").command
    # A machine that already runs it installs nothing at all.
    assert routes.select("present").command == "true"


@pytest.mark.parametrize(
    ("floor", "wanted"),
    [
        pytest.param(">=0.4.8", "mainboard>=0.4.8", id="a-declared-floor"),
        pytest.param("0.4.8", "mainboard==0.4.8", id="a-bare-version-means-that-one"),
        pytest.param("*", "mainboard", id="any-version-at-all"),
        pytest.param("", "mainboard", id="a-workspace-that-declares-none"),
    ],
)
def test_the_declared_version_reaches_the_index_command_the_way_a_requirement_spells_it(
    floor: str, wanted: str
) -> None:
    """A manifest writes a version the way its own resolver spells one, operator or not."""
    routes = installers(
        PosixShell(machine_with(), plan(), "/repo"), "packages/tool", vendored=False, floor=floor
    )
    assert routes.select("uv-index").command == (
        f"uv tool install --force --python '>=3.14' {shlex.quote(wanted)}"
    )


@pytest.mark.parametrize(
    ("found", "floor", "satisfied"),
    [
        pytest.param("0.4.9", ">=0.4.8", True, id="newer-than-the-floor"),
        pytest.param("0.4.8", ">=0.4.8", True, id="exactly-the-floor"),
        pytest.param("0.4.7", ">=0.4.8", False, id="older-than-the-floor"),
        pytest.param("0.4.8", "0.4.8", True, id="the-bare-version-it-names"),
        pytest.param("0.4.9", "0.4.8", False, id="not-the-bare-version-it-names"),
        pytest.param("0.4.9", "", True, id="anything-when-none-is-declared"),
        pytest.param("", ">=0.4.8", False, id="a-machine-running-no-tool"),
        pytest.param("nonsense", ">=0.4.8", False, id="a-version-nobody-can-parse"),
    ],
)
def test_a_machine_keeps_its_own_tool_only_when_it_already_satisfies_the_workspace(
    found: str, floor: str, *, satisfied: bool
) -> None:
    """Skipping the install is only right when the machine already runs what was asked for."""
    assert satisfied_by(found, floor) is satisfied


def test_a_host_already_running_the_declared_tool_is_onboarded_without_installing_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the index family: a machine that needs nothing is left alone.

    And it is still onboarded, since a setup that installs nothing still mirrors the workspace,
    provisions the environment and records what the host became, which is the record `hosts`
    and the snapshot pruner both read.
    """
    host = machine_with(
        rules=[
            # Ahead of the healthy set, whose bare `--version` rule answers an older tool.
            ("mainboard --version", 0, "mainboard 0.4.9\n"),
            ("[ -d packages/mainboard ]", 1, ""),
            *_HEALTHY,
        ]
    )
    setup, dispatcher = onboarding(host, monkeypatch, floor=">=0.4.8")

    recorded = setup.run()

    assert recorded.installer == "present"
    assert not host.ran("uv tool install")
    assert not host.ran("pip install")
    assert dispatcher.cache.host("gold").installer == "present"
    assert host.ran("mainboard install default")


def test_a_host_that_can_reach_no_route_says_which_family_was_being_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`uv: reported unavailable` reads as a host with no tooling, which it was not.

    The refusal names the condition that actually decided the routes: whether this workspace
    ships the tool's source, and which version it asks for when it does not.
    """
    bare = machine_with(
        rules=[("command -v", 1, ""), ("pip --version", 1, ""), ("-d packages", 1, "")]
    )
    setup, _ = onboarding(bare, monkeypatch, floor=">=0.4.8")
    with pytest.raises(MissionError, match=r"installing mainboard>=0.4.8 from an index"):
        Bootstrap(PosixShell(bare, plan(), "/repo"), floor=">=0.4.8").tool()

    vendoring = machine_with(rules=[("command -v", 1, ""), ("pip --version", 1, "")])
    with pytest.raises(MissionError, match="source this workspace vendors at packages/mainboard"):
        Bootstrap(PosixShell(vendoring, plan(), "/repo")).tool()
    del setup


def test_bootstrap_falls_through_to_pip_keeping_every_rejection_it_passed_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with(rules=[("command -v uv", 1, ""), ("command -v curl", 1, "")])
    onboarding(host, monkeypatch)
    resolution = Bootstrap(PosixShell(host, plan(), "/repo")).tool()
    assert resolution.winner == "pip"
    assert [name for name, _ in resolution.rejected] == ["uv", "uv-bootstrap"]
    assert host.ran("pip install --user")


def test_bootstrap_refuses_a_host_no_route_can_reach_before_anything_assumes_the_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with(rules=[("command -v", 1, ""), ("pip --version", 1, "")])
    onboarding(host, monkeypatch)
    with pytest.raises(MissionError, match="cannot install mainboard on 'gold'"):
        Bootstrap(PosixShell(host, plan(), "/repo")).tool()


@pytest.mark.parametrize(
    ("recorded", "verdict", "refused"),
    [
        pytest.param("older", "F1", True, id="a-changed-environment-under-a-queued-wave"),
        pytest.param("current", "F1", False, id="the-same-environment-a-sync-reships-all-day"),
        pytest.param("older", "ok", False, id="a-host-with-nothing-left-to-owe"),
    ],
)
def test_a_host_owing_runs_is_never_given_a_different_environment_under_them(
    monkeypatch: pytest.MonkeyPatch, recorded: str, verdict: str, *, refused: bool
) -> None:
    """Every pinned tree on a host symlinks its environment back to the one prefix in the mirror.

    So shipping a compiled manifest that differs from the one a queued wave was pinned against
    replaces what those jobs will activate, and the two waves then fight over the prefix: job
    3296353 of five died exactly that way (2026-09-05). A sync that changes nothing is what a
    campaign runs between waves all day and is left alone.
    """
    host = machine_with(rules=list(_HEALTHY))
    setup, dispatcher = onboarding(host, monkeypatch, digest="current")
    dispatcher.cache.save_host(HostSetup(host="gold", root="/repo", digest=recorded))
    dispatcher.cache.record(
        run_record("F1", target="gold").model_copy(
            update={"verdict": None if verdict == "F1" else verdict}
        )
    )

    if not refused:
        assert setup.run().host == "gold"
        return
    with pytest.raises(MissionError, match=r"still owes 1 run\(s\) an outcome \(F1\)"):
        setup.run()
    assert not host.ran("mainboard install")


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
    assert stages == [
        "probing",
        "mirroring",
        "installing",
        "checking",
        "provisioning",
        "checking",
        "reading",
        "onboarded",
    ]
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
    monkeypatch.setattr(
        onboard_module,
        "probe_capabilities",
        lambda alias, ssh=None: Facts.parsed(alias, _CAPABILITIES),
    )
    monkeypatch.setattr(
        onboard_module,
        "open_shell",
        lambda execution, root, ssh=None: PosixShell(host, execution, root),
    )
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
        "checking",
        "provisioning",
        "checking",
        "reading",
    ]


def test_onboarding_discovers_a_root_the_profile_never_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe already answered where the workspace goes, so nothing asks the host twice."""
    host = machine_with(rules=_HEALTHY)
    setup, dispatcher = onboarding(host, monkeypatch, root="")
    report = setup.run()
    assert report.root == "/home/me/projects"
    assert dispatcher.mirrored == [("gold", "/home/me/projects")]
    assert not host.ran("ls -d /work")


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
    host = machine_with(rules=list(_HEALTHY))
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
    host = machine_with(rules=list(_HEALTHY))
    setup, dispatcher = onboarding(host, monkeypatch, root="")
    dispatcher.cache.save_host(HostSetup(host="gold", root="/repo", installer="uv", tool="0.1.0"))

    report = setup.run(sync_only=True)

    assert dispatcher.mirrored == [("gold", "/repo")]
    assert host.ran("mainboard install default --profile gold")
    assert not host.ran("uv tool install")
    assert not host.ran("facts --json")
    assert not host.ran("mainboard --version")
    # The pixi is aligned here too, since a sync between two waves is exactly when a host that
    # moved would otherwise rewrite the shipped lock under the wave already queued against it.
    assert host.ran("pixi --version")
    assert (report.root, report.installer, report.tool, report.pixi) == (
        "/repo",
        "uv",
        "0.1.0",
        PIXI_VERSION,
    )


def test_sync_only_prefers_a_given_root_over_the_recorded_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with(rules=list(_HEALTHY))
    setup, dispatcher = onboarding(host, monkeypatch)
    dispatcher.cache.save_host(HostSetup(host="gold", root="/other", installer="uv"))
    report = setup.run(sync_only=True)
    assert report.root == "/repo"
    assert dispatcher.mirrored == [("gold", "/repo")]


def test_sync_only_resolves_the_plan_from_the_capabilities_the_setup_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sync probes nothing, so the platform the setup found is what fills the profile's gaps."""
    host = machine_with(rules=list(_HEALTHY))
    setup, dispatcher = onboarding(host, monkeypatch)
    found = Facts.parsed("gold", _CAPABILITIES)
    dispatcher.cache.save_host(HostSetup(host="gold", root="/repo", capabilities=found))
    assert setup.plan.profile.platform != found.pixi_platform
    setup.run(sync_only=True)
    assert setup.plan.profile.platform == found.pixi_platform


def test_the_probe_commands_run_the_environments_python_and_the_hosts_own_tool() -> None:
    assert gpus_command() == "mainboard gpus --json"


def test_sync_only_refuses_a_host_that_was_never_onboarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = machine_with()
    setup, _ = onboarding(host, monkeypatch)
    with pytest.raises(LookupError, match="'gold' has never been set up"):
        setup.run(sync_only=True)


def test_a_host_on_another_pixi_is_brought_to_the_pin_and_refused_when_it_stays_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock is pixi's file, so a host on another version rewrites it and builds elsewhere.

    The check this replaces refused an older pixi and let a newer one through, which is exactly
    the case that killed a Miyabi wave on 2026-09-05: the workstation solved on 0.77 and the
    host provisioned on 0.79, one artifact reached two addresses and every job of the wave found
    no built environment. Both directions are brought in line now, with pixi's own installer at
    the pinned version, and a host that will not move is refused before anything is provisioned.
    """

    class Drifted(RecordingMachine):
        """A host above the pin until the pinned installer has run here, on the pin after."""

        def answer(self, argv: list[str], *, stdin: str = "") -> tuple[int, str]:
            if "pixi --version" not in " ".join(argv):
                return super().answer(argv, stdin=stdin)
            self.calls.append(argv)
            settled = self.ran("pixi.sh/install.sh")
            return 0, f"pixi {PIXI_VERSION}\n" if settled else "pixi 0.80.0\n"

    drifted = Drifted(rules=list(_HEALTHY))
    setup, _ = onboarding(drifted, monkeypatch)

    assert setup.run().pixi == PIXI_VERSION
    assert drifted.ran(f"PIXI_VERSION={PIXI_VERSION}")

    for answer, named in ((0, "0.80.0"), (1, "none")):
        stuck = machine_with(rules=[("pixi --version", answer, "pixi 0.80.0\n"), *_HEALTHY])
        refused, _ = onboarding(stuck, monkeypatch)
        with pytest.raises(
            MissionError, match=f"still runs pixi {named} after installing {PIXI_VERSION}"
        ):
            refused.run()
        assert not stuck.ran("mainboard install default")


def test_a_dead_queue_daemon_is_started_once_and_refused_when_it_stays_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain ssh host dispatches through pueue, so its daemon has to answer before jobs do."""

    class Reviving(RecordingMachine):
        def answer(self, argv: list[str], *, stdin: str = "") -> tuple[int, str]:
            if "pueue status" in " ".join(argv):
                self.calls.append(argv)
                return (0 if self.ran("pueued -d") else 1), ""
            return super().answer(argv, stdin=stdin)

    revived = Reviving(rules=list(_HEALTHY))
    setup, _ = onboarding(revived, monkeypatch)
    assert setup.run().host
    assert revived.ran("pueued -d")
    installed = next(i for i, line in enumerate(revived.lines) if "mainboard install" in line)
    daemon = next(i for i, line in enumerate(revived.lines) if "pueued -d" in line)
    assert installed < daemon
    assert "activate.sh" in revived.lines[daemon]
    assert "</dev/null >/dev/null 2>&1" in revived.lines[daemon]

    dead = machine_with(rules=[("pueue status", 1, ""), *_HEALTHY])
    setup, _ = onboarding(dead, monkeypatch)
    with pytest.raises(MissionError, match="pueued is not answering.*pueued -d"):
        setup.run()

    scheduled = machine_with(rules=[("pueue status", 1, ""), *_HEALTHY])
    setup, _ = onboarding(scheduled, monkeypatch)
    monkeypatch.setattr(onboard_module, "pick", lambda profile: object())
    assert setup.run().host
    assert not scheduled.ran("pueue status")


@pytest.mark.parametrize("vendored", [True, False], ids=["from the source", "from an index"])
def test_the_extras_a_center_carries_ride_every_install_route(*, vendored: bool) -> None:
    """A center's tool plots, so its successor's is installed with the same extras."""
    shell = PosixShell(machine_with(), plan(), "/repo")
    routes = installers(shell, "packages/tool", vendored=vendored, extras=["plot", "wandb"])
    commands = [routes.select(name).command for name in routes.names if name != "present"]
    assert all("[plot,wandb]" in command for command in commands)
