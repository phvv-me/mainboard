# Targets from `~/.ssh/config` and the over-ssh bootstrap probe that describes each host.

from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..manifest.schema.host import HostProfile
    from .transport import Machine

# The user's ssh client config; its concrete `Host` aliases are dispatch targets.
_SSH_CONFIG = Path.home() / ".ssh" / "config"

# Where to put the workspace: an HPC `/work` area if there is one, else `~/projects`.
_ROOT_FINDER = (
    'w=$(ls -d /work/*/"$USER"/projects 2>/dev/null | head -1); echo "${w:-$HOME/projects}"'
)

# Stock-tools capability probe: prints workspace root, scheduler, GPU, memory, group, queue,
# platform and the engines onboarding would otherwise have to install, as `key=value` lines,
# run in a login shell so the HPC scheduler is on PATH. A per-user engine is looked for in its
# install directory as well as on PATH, since a non-interactive login shell often never reads
# the `.bashrc` line its installer appended.
CAPABILITIES = "\n".join(
    (
        f"root=$({_ROOT_FINDER})",
        "if command -v sbatch >/dev/null 2>&1; then kind=slurm;"
        " elif command -v qsub >/dev/null 2>&1; then kind=pbs; else kind=ssh; fi",
        "gpu=$(nvidia-smi --query-gpu=name,memory.total"
        " --format=csv,noheader,nounits 2>/dev/null | head -1)",
        r"mem=$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo 2>/dev/null)",
        "queue=$(qstat -Q 2>/dev/null | awk 'NR>2 && tolower($1) ~ /interact/ {print $1; exit}')",
        # Some qstat wrappers reject -Q, so fall back to its --rsc tree and take the top-level
        # interactive router, the queue `qsub -I` accepts.
        '[ -z "$queue" ] && queue=$(qstat --rsc 2>/dev/null'
        " | awk '/^interact/ && tolower($1) !~ /mig/ {print $1; exit}')",
        'pixi=$(command -v pixi || ls "$HOME"/.pixi/bin/pixi 2>/dev/null)',
        'uv=$(command -v uv || ls "$HOME"/.local/bin/uv 2>/dev/null)',
        "platform=$(uname -sm)",
        "printf 'root=%s\\nkind=%s\\ngpu=%s\\nmem=%s\\naccount=%s\\nqueue=%s\\n"
        "pixi=%s\\nuv=%s\\nplatform=%s\\n'"
        ' "$root" "$kind" "$gpu" "$mem" "$(id -gn)" "$queue" "$pixi" "$uv" "$platform"',
    )
)


class Facts(FrozenModel):
    """One host's bootstrap-probed capabilities, before any manifest override.

    name: the ssh alias probed.
    root: the workspace root the host resolved (an HPC `/work` area, else `~/projects`).
    kind: the scheduler the login node's PATH exposes (`slurm` / `pbs` / `ssh`).
    account: the user's primary group, PBS `group_list`'s natural default.
    queue: the host's interactive queue, when one was discovered.
    gpu_name: the login node's GPU name, when it has one.
    gpu_mem_mb: that GPU's memory in MiB, when reported.
    sysmem_gb: system memory in GiB.
    platform: the host's `uname -sm` string, the kernel and machine architecture.
    pixi: the pixi binary already on the host, empty when onboarding has to install one.
    uv: the uv binary already on the host, empty when onboarding has to install one.
    """

    name: str
    root: str = "~/projects"
    kind: str = "ssh"
    account: str = ""
    queue: str = ""
    gpu_name: str | None = None
    gpu_mem_mb: int | None = None
    sysmem_gb: int | None = None
    platform: str = ""
    pixi: str = ""
    uv: str = ""

    @property
    def vram_gb(self) -> float | None:
        """Usable memory in GB: GPU VRAM if present, else system memory."""
        if self.gpu_mem_mb is not None:
            return self.gpu_mem_mb / 1024
        if self.sysmem_gb is not None:
            return float(self.sysmem_gb)
        return None

    def fits(self, needs_gb: float) -> bool:
        """Whether this host's usable memory satisfies `needs_gb`."""
        return self.vram_gb is not None and self.vram_gb >= needs_gb


def ssh_hosts(config_path: Path = _SSH_CONFIG) -> list[str]:
    """Concrete `Host` aliases from `~/.ssh/config`, in file order.

    Splits each line on any whitespace (ssh accepts tabs as well as spaces after `Host`), splits
    multi-alias `Host a b` lines, and drops any pattern token (one containing `*` or `?`, e.g.
    `Host *` or `dl*`), so the result is the list of real, connectable destinations. Non-`Host`
    directives (`Include`, `HostName`, ...) are skipped.
    """
    if not config_path.exists():
        return []
    hosts: list[str] = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        keyword, *aliases = line.split() or [""]
        if keyword.lower() != "host":
            continue
        concrete = (a for a in aliases if "*" not in a and "?" not in a)
        hosts.extend(alias for alias in concrete if alias not in hosts)
    return hosts


def find_root(remote: Machine) -> str:
    """The workspace root to use on the host (an HPC `/work` area, else `~/projects`)."""
    return str(remote["bash"][["-lc", _ROOT_FINDER]]()).strip()


def probe_capabilities(remote: Machine, alias: str) -> Facts:
    """Probe `alias` over ssh without syncing or installing, as `Facts`.

    Runs the stock-tool `CAPABILITIES` script in a login shell and parses its `key=value`
    lines, so it needs nothing on the host, available before a single byte is shipped.
    """
    raw = remote["bash"][["-lc", CAPABILITIES]]()
    fields = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    name, _, vram = fields["gpu"].partition(",")
    sysmem_kb = fields["mem"]
    return Facts(
        name=alias,
        root=fields["root"],
        kind=fields["kind"],
        account=fields["account"],
        queue=fields["queue"],
        gpu_name=name.strip() or None,
        gpu_mem_mb=int(vram) if vram.strip().isdigit() else None,
        sysmem_gb=round(int(sysmem_kb) / 1024**2) if sysmem_kb.isdigit() else None,
        platform=fields["platform"],
        pixi=fields["pixi"],
        uv=fields["uv"],
    )


def resolve(profile: HostProfile, facts: Facts) -> HostProfile:
    """`profile` with any field it left at its `auto`/unset default filled from `facts`.

    The manifest is the declared source of truth; a probed fact only ever fills a gap the
    manifest left open (`kind: "auto"`, an empty `root`, an empty `account`), so an explicit
    manifest value always wins and the manifest schema (not this function) owns validation.
    """
    updates: dict[str, str] = {}
    if profile.kind == "auto":
        updates["kind"] = facts.kind
    if not profile.root:
        updates["root"] = facts.root
    if not profile.account:
        updates["account"] = facts.account
    return profile.model_copy(update=updates) if updates else profile


def smallest_fit(candidates: Sequence[Facts], needs_gb: float) -> Facts:
    """Smallest-VRAM candidate that still satisfies `needs_gb` (keeps big iron free).

    candidates: probed hosts to route among.
    needs_gb: requested memory in GB.
    """
    fitting = sorted((c for c in candidates if c.fits(needs_gb)), key=lambda c: c.vram_gb or 0.0)
    if not fitting:
        have = ", ".join(f"{c.name}={c.vram_gb}" for c in candidates)
        raise LookupError(f"no target fits {needs_gb} GB; have: {have}")
    return fitting[0]
