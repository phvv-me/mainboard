import subprocess
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from plumbum import local

from mainboard import Board, ExecutionPlan, MissionError
from mainboard.context.plan import environment_prefix
from mainboard.dispatch import (
    Dialect,
    Facts,
    HostUnreachable,
    Onboarding,
    Order,
    Resources,
    Shipment,
    Source,
    SshTransport,
)
from mainboard.dispatch.agent import SshLink
from mainboard.dispatch.backends import Rentable
from mainboard.dispatch.hosts import landing
from mainboard.dispatch.hosts.landing import Landing
from mainboard.dispatch.hosts.onboard import Bootstrap, core
from mainboard.dispatch.hosts.rentals import Rental
from mainboard.dispatch.transport.core import Endpoint, Machine
from mainboard.dispatch.transport.shells import HostShell
from mainboard.engines.compile import backend
from mainboard.manifest import HostProfile
from mainboard.probe import HostFacts

from .builders import sshd, stub
from .constants import HOST, PLAN, ROOT
from .fakes import Recorded

_MANIFEST = (
    f'[workspace]\nname = "it"\n[hosts.{HOST}]\nkind = "ssh"\nroot = "{ROOT}"\n'
    'platform = "linux-64"\n'
)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A login home whose system `python3`, `python` and `curl` are stubs that leave a trace."""
    for name in ("python3", "python", "curl"):
        stub(tmp_path / "system-bin" / name, f'echo "$0" >> "{tmp_path}/system"; exit 3')
    return tmp_path


@pytest.fixture
def launcher() -> str:
    """The launcher line, resolved after the autouse fixture entered the test's workspace."""
    return Dialect().python(
        environment_prefix("~/.mb-jobs", env="default"), host=HOST, python="3.14"
    )


@pytest.fixture
def board(locked: Path) -> Board:
    """A workspace declaring the fake host."""
    (locked / "mb.toml").write_text(_MANIFEST, encoding="utf-8", newline="\n")
    return Board(locked)


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every remote command dispatch hands ssh, each refused before it leaves."""
    lines: list[str] = []

    def spawn(self: SshLink, command: str) -> subprocess.Popen[bytes]:
        lines.append(command)
        raise HostUnreachable("recorded")

    def run(self: SshTransport, argv: Sequence[str], host: str, **_: str | Path) -> None:
        lines.append(argv[-1])
        raise HostUnreachable("recorded")

    monkeypatch.setattr(SshLink, "spawn", spawn)
    monkeypatch.setattr(SshTransport, "run", run)
    return lines


@pytest.fixture
def login(home: Path, monkeypatch: pytest.MonkeyPatch) -> HostShell:
    """A host shell whose commands run under the login `home`, which has only fake tools."""
    shell = HostShell(local, PLAN, str(home / ".mb-jobs"))

    def execute(line: str) -> tuple[int, str, str]:
        ran = sshd(home, line, program="sh")
        return ran.returncode, ran.stdout, ran.stderr

    monkeypatch.setattr(shell, "execute", execute)
    return shell


@pytest.fixture
def onboarding(board: Board, monkeypatch: pytest.MonkeyPatch) -> tuple[Onboarding, Recorded]:
    """An onboarding of the fake host whose every command lands on a recorded shell."""
    shell = Recorded(
        replies={
            "pixi --version": f"pixi {backend.PixiEngine.VERSION}",
            core.facts_command(): HostFacts().model_dump_json(),
        }
    )
    monkeypatch.setattr(core, "open_shell", lambda *args, **kwargs: shell)
    monkeypatch.setattr(
        core, "probe_capabilities", lambda host: Facts(name=host, platform="Linux x86_64")
    )
    monkeypatch.setattr(board.dispatcher, "mirror", lambda *args, **kwargs: [])
    return Onboarding(board.dispatcher, PLAN, watch=lambda text: None), shell


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """A recorded shell standing for a rented aarch64 machine, stopped before its environment."""
    shell = Recorded()

    def host_shell(remote: Machine, plan: ExecutionPlan, root: str) -> HostShell:
        shell.plan, shell.root = plan, root
        return shell

    def stop_at_environment(self: Bootstrap) -> None:
        raise MissionError("environment reached")

    facts = Facts(name="rental", home="/home/rental", platform="Linux aarch64")
    monkeypatch.setattr(landing, "connection", lambda *args: nullcontext(local))
    monkeypatch.setattr(landing, "probe_capabilities", lambda *args, **kwargs: facts)
    monkeypatch.setattr(landing, "HostShell", host_shell)
    monkeypatch.setattr(Bootstrap, "environment", stop_at_environment)
    return shell


@pytest.fixture
def rented(
    board: Board, machine: Recorded, monkeypatch: pytest.MonkeyPatch
) -> tuple[Landing, Rental, Recorded]:
    """A job about to equip a rental, with the recorded shell its commands will land on."""
    monkeypatch.setattr(board.dispatcher, "mirror", lambda *args, **kwargs: [])
    plan = PLAN.model_copy(update={"profile": HostProfile(kind="vast")})
    shipment = Shipment.of_command("true", source=Source(identity="fake", key="fake"), imports=())
    order = Order(plan=plan, shipment=shipment, resources=Resources())
    rental = Rental(handle="fake", endpoint=Endpoint(address="example.invalid"))
    return Landing(board.dispatcher, Mock(spec=Rentable), order), rental, machine
