# What a host became once set up, and the commands a host answers with its own facts.

from patos import FrozenModel

from ....core.errors import MissionError
from ....probe.readings.snapshot import HostFacts
from ...transport.targets import Facts
from .constants import TOOL


def gpus_command() -> str:
    """The command a remote host runs to say who holds each of its cards, as JSON."""
    return f"{TOOL} host list local --gpus --json"


def facts_command() -> str:
    """The command a machine answers with its own hardware snapshot as JSON."""
    return f"{TOOL} host list local --facts --json"


class HostSetup(FrozenModel):
    """What one machine became after an install: where its workspace is and how it activates.

    The record `Board.install` hands back and the dispatch cache keeps per host, so a later
    command reads what a host is without touching it, and a fallback install route stays a
    stated fact rather than a silent degradation.

    host: the alias onboarded, `local` for an install on this machine.
    root: the workspace root on that machine.
    activate: the activation script the workspace now carries.
    installer: the install route that won, `in-place` when the tool was already running here.
    rejected: the routes passed over, each with the reason it was not usable.
    tool: the tool version the machine reports once installed.
    pixi: the pixi version the machine runs, which every setup and sync brings to the fleet's
        one pinned version, so a host that quietly moved shows here instead of in a dead wave.
    capabilities: the host as the bootstrap probe found it, carrying the home a `~` root was
        placed under and never a root of its own, since `root` is the one set up there; None
        for an in-place install.
    hardware: the host's hardware snapshot, read back through the new activation.
    onboarded_at: ISO-8601 time the install finished.
    synced_at: ISO-8601 time the workspace was last mirrored here, empty until one lands after
        the onboarding that first mirrored it.
    digest: the manifest digest this host was last provisioned from, empty for a host onboarded
        before this field existed; `doctor` compares it with the manifest to spot drift.
    dotfiles: set up with `--dotfiles`, which every later sync keeps.
    requires: the digest of the tool's declared requirements, the lock's releases of them and
        the solved Python minor as installed there (`Bootstrap.requirements`); a sync changing
        any reinstalls it, since an editable install follows the mirrored source but never gains
        a dependency the source added or moves one the lock moved. Empty for an older setup.
    """

    host: str
    root: str
    env: str = "default"
    activate: str = ""
    installer: str = ""
    rejected: tuple[tuple[str, str], ...] = ()
    tool: str = ""
    pixi: str = ""
    capabilities: Facts | None = None
    hardware: HostFacts | None = None
    onboarded_at: str = ""
    synced_at: str = ""
    digest: str = ""
    dotfiles: bool = False
    requires: str = ""

    @property
    def mirrored_at(self) -> str:
        """When this host's workspace copy was last brought up to date.

        That is the latest mirror, else the onboarding: the watermark a transfer set measures a
        delta against.
        """
        return max(self.synced_at, self.onboarded_at)


def read_facts(text: str) -> HostFacts:
    """The `HostFacts` in a remote command's output, read from its first `{`.

    Shell chatter above it is ignored.
    """
    start = text.find("{")
    if start < 0:
        raise MissionError(
            f"no host facts in the probe output: {text.strip()[-240:] or '(empty)'}"
        )
    return HostFacts.model_validate_json(text[start:])
