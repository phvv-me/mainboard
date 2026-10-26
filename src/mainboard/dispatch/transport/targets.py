# Targets from `~/.ssh/config` and the over-ssh bootstrap probe that describes each host.

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Self

from patos import FrozenModel

from ...core.errors import MissionError
from ...core.execution.host import pixi_platform
from ...core.project import Project
from .core import SshTransport

if TYPE_CHECKING:
    from ...manifest.schema.compute.host import HostProfile
    from .core import Machine

# The user's ssh client config; its concrete `Host` aliases are dispatch targets.
_SSH_CONFIG = Path.home() / ".ssh" / "config"

# The tool's jobs folders under a login home, primary first, as the probes look for them.
_FOLDERS = tuple(root.removeprefix("~/") for root in Project().jobs_roots)

# Stock-tools capability probe printing `key=value` lines, run in a login shell so the HPC
# scheduler is on PATH. A per-user engine is also looked for in its install directory, since a
# non-interactive login shell often never reads the `.bashrc` line its installer appended.
_CAPABILITIES = "\n".join(
    (
        "if command -v sbatch >/dev/null 2>&1; then kind=slurm;"
        " elif command -v qsub >/dev/null 2>&1; then kind=pbs; else kind=ssh; fi",
        "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)",
        "queue=$(qstat -Q 2>/dev/null | awk 'NR>2 && tolower($1) ~ /interact/ {print $1; exit}')",
        # Some qstat wrappers reject -Q: take the top-level interactive router from --rsc, the
        # queue `qsub -I` accepts.
        '[ -z "$queue" ] && queue=$(qstat --rsc 2>/dev/null'
        " | awk '/^interact/ && tolower($1) !~ /mig/ {print $1; exit}')",
        'pixi=$(command -v pixi || ls "$HOME"/.pixi/bin/pixi 2>/dev/null)',
        'uv=$(command -v uv || ls "$HOME"/.local/bin/uv 2>/dev/null)',
        "platform=$(uname -sm)",
        # The first of the tool's own jobs folders the home already holds, under any name.
        f"jobs=$(for d in {' '.join(_FOLDERS)}; do"
        ' [ -d "$HOME/$d" ] && { printf "~/%s" "$d"; break; }; done)',
        "printf 'home=%s\\nkind=%s\\ngpu=%s\\naccount=%s\\nqueue=%s\\n"
        "pixi=%s\\nuv=%s\\nplatform=%s\\njobs=%s\\n'"
        ' "$HOME" "$kind" "$gpu" "$(id -gn)" "$queue" "$pixi" "$uv" "$platform" "$jobs"',
    )
)


class Facts(FrozenModel):
    """One host's bootstrap-probed capabilities, before any manifest override.

    home: the login home, which a profile's `~` root is placed under.
    kind: the scheduler the login node's PATH exposes (`slurm` / `pbs` / `ssh`).
    account: the user's primary group, PBS `group_list`'s natural default.
    queue: the host's interactive queue, when one was discovered.
    gpu_name: the login node's GPU, when reported.
    platform: the host's `uname -sm` string.
    pixi / uv: the binary already on the host, empty when onboarding has to install one.
    jobs: the tool's jobs folder the home already holds (`~/.mainboard-jobs`), under whichever
        name the release that set it up used; empty when none, or from a probe (or a record of
        one) that did not look.
    """

    name: str
    home: str = ""
    kind: str = "ssh"
    account: str = ""
    queue: str = ""
    gpu_name: str | None = None
    platform: str = ""
    pixi: str = ""
    uv: str = ""
    jobs: str = ""

    @classmethod
    def parsed(cls, name: str, *, text: str) -> Self:
        """The facts inside a probe's `key=value` lines."""
        fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        return cls(
            name=name,
            home=fields["home"],
            kind=fields["kind"],
            account=fields["account"],
            queue=fields["queue"],
            gpu_name=fields["gpu"].strip() or None,
            platform=fields["platform"].strip(),
            pixi=fields["pixi"],
            uv=fields["uv"],
            jobs=fields.get("jobs", ""),
        )

    @property
    def pixi_platform(self) -> str:
        """The host as a pixi platform string, empty before it was probed."""
        system, _, machine = self.platform.partition(" ")
        return pixi_platform(system, machine=machine) if system else ""


def ssh_hosts(config_path: Path = _SSH_CONFIG) -> list[str]:
    """Concrete `Host` aliases from `~/.ssh/config`, in file order, deduplicated.

    Splits on any whitespace (ssh accepts tabs after `Host`) and across multi-alias lines, and
    drops pattern tokens containing `*` or `?`, leaving only connectable destinations.
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


def home_of(remote: Machine) -> str:
    """The login home of a POSIX host already connected, for a rental that no setup probed."""
    return remote["bash"][["-lc", 'printf %s "$HOME"']]().strip()


def placed(root: str, *, home: str) -> str:
    """`root` with a leading `~` spelled as the host's `home`.

    Every consumer of a host's root shares this one expansion, so the mirror, the runner and every
    shell line agree on one path. Any other root, or a `~` whose home is still unknown, is returned
    as written.
    """
    head, slash, rest = root.partition("/")
    return f"{home}{slash}{rest}" if head == "~" and home else root


def probe_capabilities(host: str, *, ssh: SshTransport | None = None) -> Facts:
    """Probe `host` over ssh with stock tools only, before a byte is shipped.

    ssh: the bounded SSH policy; the default policy when omitted.
    """
    policy = ssh or SshTransport()
    probe = ("bash", "-lc", shlex.quote(_CAPABILITIES.replace("\n", "; ")))
    _, out, err = policy.invoke((*policy.command(host), *probe), host, operation="probe")
    if "platform=" not in out:
        raise MissionError(f"{host!r} did not answer the bash probe: {err.strip()[-200:]}")
    return Facts.parsed(host, text=out)


def rooted(profile: HostProfile, *, host: str) -> str:
    """`profile`'s root on `host`, refusing a `~` no probe of the host has placed yet.

    No shell line, scheduler directive or snapshot path could spell such a root.
    """
    if profile.root.startswith("~"):
        raise MissionError(
            f"{host!r} has no probed home to place {profile.root} under; "
            f"run `{Project().name} host setup {host}`"
        )
    return profile.root


def resolve(profile: HostProfile, facts: Facts, *, recorded: str = "") -> HostProfile:
    """`profile` with the gaps it left open filled from `facts`.

    The gaps are `kind: "auto"`, a `~` root, `account` and `platform`. An explicit manifest value
    always wins, and the manifest schema owns validation.

    A root the manifest leaves unset is one of the tool's jobs folders, chosen so a host never
    moves: the one its last setup placed when that was one of them, else the one its home
    already holds, and only on a host holding none the primary name's. So a host set up under a
    legacy name keeps its folder, its mirror and its built environments, and a renamed tool
    never builds a second copy beside the first.

    recorded: the root the host's last recorded setup placed, empty when it has none.
    """
    root = placed(_kept_root(profile, facts, recorded=recorded), home=facts.home)
    wanted = {
        "kind": (profile.kind == "auto", facts.kind),
        "root": (root != profile.root, root),
        "account": (not profile.account, facts.account),
        "platform": (not profile.platform and bool(facts.pixi_platform), facts.pixi_platform),
    }
    updates = {name: value for name, (needed, value) in wanted.items() if needed}
    return profile.model_copy(update=updates) if updates else profile


def _kept_root(profile: HostProfile, facts: Facts, *, recorded: str) -> str:
    """The root `resolve` keeps: the manifest's, else the recorded jobs folder, else the probed."""
    if "root" in profile.model_fields_set:
        return profile.root
    folders = [placed(folder, home=facts.home) for folder in Project().jobs_roots]
    return recorded if recorded in folders else facts.jobs or profile.root
