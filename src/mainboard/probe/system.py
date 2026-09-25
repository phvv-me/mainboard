from pathlib import Path

from packaging.version import InvalidVersion, Version
from patos import FrozenOpenModel

from ..core.host import pixi_platform
from .census import Census


class Card(FrozenOpenModel):
    """One NVIDIA card as its driver describes it.

    name: the marketing name, `NVIDIA GeForce RTX 5080` say.
    capability: the compute capability, `12.0` for Blackwell consumer cards.
    vram_mb: its memory in MiB, the system's for a unified card.
    unified: whether the card has no memory of its own and shares the system's, as a GB10 does.
    """

    name: str
    driver: str = ""
    capability: str = ""
    vram_mb: int = 0
    unified: bool = False


class System(FrozenOpenModel):
    """A machine's operating system, filesystem and software, as the census read it.

    The software half of a host's facts, beside the hardware inventory, and what every machine
    finding is judged from. It is an open model for the same reason `HostFacts` is: a reader on
    an older tool keeps parsing what a newer census adds.

    system: the kernel as `platform.system()` spells it, empty before any census ran.
    version: the operating system's own name for its version, e.g. `Ubuntu 24.04.1 LTS`.
    arch: the machine architecture as `platform.machine()` spells it.
    python: the interpreter the census ran under.
    shells: every shell found, by name, with its path.
    root: the directory the filesystem questions were asked of.
    case_sensitive: whether names differing only in case are different files there.
    symlinks: why a symbolic link cannot be made there, empty when it can.
    long_paths: whether paths past 260 characters open, always true off Windows.
    developer_mode: whether Windows lets this account make symbolic links.
    free_bytes: free space where the workspace lives.
    git: the global git settings a clone inherits, empty values for unset ones.
    tools: every tool that answered with a version, by name.
    cuda: the maximum CUDA version the NVIDIA driver supports, empty without one.
    """

    system: str = ""
    release: str = ""
    version: str = ""
    arch: str = ""
    python: str = ""
    shells: dict[str, str] = {}
    root: str = ""
    case_sensitive: bool = True
    symlinks: str = ""
    long_paths: bool = True
    developer_mode: bool = False
    free_bytes: int = 0
    git: dict[str, str] = {}
    tools: dict[str, str] = {}
    cuda: str = ""
    gpus: tuple[Card, ...] = ()

    @classmethod
    def collected(cls, root: Path) -> System:
        """This machine's census, the filesystem measured where the workspace `root` is or goes."""
        return cls.model_validate(Census().survey(str(root)))

    @property
    def surveyed(self) -> bool:
        """Whether a census filled this in, since a host onboarded by an older tool has none."""
        return bool(self.system)

    @property
    def windows(self) -> bool:
        """Whether this is a Windows machine."""
        return self.system == "Windows"

    @property
    def platform(self) -> str:
        """This machine as a pixi platform string, empty before a census ran."""
        return pixi_platform(self.system, self.arch) if self.surveyed else ""

    @property
    def vram_mb(self) -> int:
        """The largest card's memory in MiB, 0 without a card."""
        return max((card.vram_mb for card in self.gpus), default=0)

    @property
    def unified(self) -> bool:
        """Whether a card shares the system's memory rather than holding its own."""
        return any(card.unified for card in self.gpus)

    @property
    def capability(self) -> Version | None:
        """The newest card's compute capability, None without a card that reports one."""
        return max(filter(None, (_version(card.capability) for card in self.gpus)), default=None)

    @property
    def driver_cuda(self) -> Version | None:
        """The newest CUDA the driver supports, None without an NVIDIA driver."""
        return _version(self.cuda)

    def summary(self) -> str:
        """One line naming the operating system, its architecture and the driver's CUDA."""
        cuda = f", driver CUDA {self.cuda}" if self.cuda else ""
        return f"{self.version or self.system} {self.arch}{cuda}"


def _version(text: str) -> Version | None:
    """`text` as a comparable version, None when it is empty or unreadable."""
    try:
        return Version(text)
    except InvalidVersion:
        return None
