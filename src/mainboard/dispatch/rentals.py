# A machine rented for one job, and the handshake between its entrypoint and the landing.
#
# A provider rents a bare container: no workspace, no tool, no environment, which is why a
# dispatched `mainboard run` died on vast instance 49861190 with `bash: line 2: mainboard: command
# not found`, exit 127, three times over, billing every boot. So a rental is an ssh host that
# exists for one job: the dispatch mirrors the workspace, installs the tool from the mirror,
# provisions from the lock this workspace already solved and pins the tree, as `mainboard setup`
# does on gold.
#
# That happens on the meter after the boot. The entrypoint waits for a launch script, the landing
# writes it once the machine can run it, and the entrypoint runs the job in the foreground, so
# the log, the exit marker and the cancel sit where each provider backend already reads them. The
# wait is bounded because it is billed: a workstation that dies mid-landing ends the entrypoint
# with a status the ordinary settle-and-cancel path acts on.

import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from .shared import logger
from .transport import Endpoint, HostUnreachable, SshTransport

if TYPE_CHECKING:
    from collections.abc import Callable

# The launch script and the neighbouring name it is written under first. `/tmp` is the one
# writable directory every image agrees on, whichever user the entrypoint runs as.
LAUNCH = "/tmp/mainboard-launch.sh"
_PENDING = "/tmp/mainboard-launch.part"

# How long a rental waits for a landing, which the spend cap assumes every rental bills before its
# job starts. The slow step is the environment (a whole CUDA stack on a cold container), not the
# mirror (288 MB measured 2026-09-04). Generous on purpose: overrunning throws away a nearly ready
# rental, while a landing done in ten minutes never pays for the other fifty.
LANDING_SECONDS = 3600
_WAIT_SECONDS = 5
# `EX_TEMPFAIL` when no dispatch landed: the machine was fine and the workstation was not, and a
# distinct code tells that apart from the job failing when the log is all anyone has.
_NO_LANDING = 75
# The standard pairs under `~/.ssh`, newest algorithm first.
_KEY_NAMES = ("id_ed25519", "id_ecdsa", "id_rsa")
# Minutes rather than seconds: a provider that publishes an address early can refuse the knock
# for a while after its status turns running.
_SSH_ATTEMPTS = 40
_SSH_SECONDS = 5.0


class Identity(FrozenModel):
    """The key pair a rental is opened with.

    private: the private key file this machine connects with.
    public: the public key text handed to the provider at create time.
    """

    private: str
    public: str


class Rental(FrozenModel):
    """A machine rented for one job.

    handle: the provider's opaque run id, which every later state, log and cancel addresses.
    endpoint: the ssh target the dispatch lands on, valid only for this rental's lifetime.
    """

    handle: str
    endpoint: Endpoint


def identity(declared: str = "") -> Identity:
    """The key pair this workspace opens a rental with, refusing when it holds none usable.

    A fresh rental is reachable only through a public key given at create time whose private half
    this machine holds, so the pair is read here rather than from the provider's key list: a key
    registered from another laptop looks registered and refuses every connection.

    declared: `[hosts.<name>.vars] ssh-key`; empty takes the first standard pair under `~/.ssh`.
    """
    candidates = (
        [Path(declared).expanduser()]
        if declared
        else [Path.home() / ".ssh" / name for name in _KEY_NAMES]
    )
    paired = [private for private in candidates if Path(f"{private}.pub").is_file()]
    for private in paired:
        if unlocked(private):
            return Identity(private=str(private), public=published(private))
    if paired:
        raise MissionError(
            f"the ssh key {paired[0]} needs a passphrase and no agent holds it, and a rental is "
            "opened without a terminal to ask on, so every connection would be refused on a "
            "machine already billing. Load it with `ssh-add` (`ssh-add --apple-use-keychain` on "
            "macOS), or point [hosts.<host>.vars] ssh-key at a key that needs none."
        )
    named = f"{declared} " if declared else ""
    raise MissionError(
        f"no ssh key pair {named}to open a rental with; a rented machine is reached over ssh and "
        "is created with the public half of a key this machine holds. Generate one with "
        "`ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519`, or point [hosts.<host>.vars] ssh-key at "
        "the private key of a pair you already have."
    )


def _derived(private: Path) -> subprocess.CompletedProcess[str]:
    """`ssh-keygen -y` on `private` with an empty passphrase, never prompting."""
    return subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=a constant argv over a path this module chose since=2026-09-21
        ["ssh-keygen", "-y", "-P", "", "-f", str(private)],
        capture_output=True,
        text=True,
        check=False,
    )


def published(private: Path) -> str:
    """The public key ssh will actually present for `private`, derived from the private half.

    ssh never reads the `.pub`, so a stale or mangled one goes unnoticed until a create sends it:
    an `id_rsa.pub` wrapped over three lines since 2022 gave three rentals a key no sshd could
    parse, each refusing every knock with `Permission denied (publickey)` while billing
    (2026-09-21). A locked key an agent holds cannot be derived; there the `.pub` is all we have.
    """
    derived = _derived(private)
    if derived.returncode == 0 and derived.stdout.strip():
        return derived.stdout.strip()
    return Path(f"{private}.pub").read_text(encoding="utf-8").strip()


def unlocked(private: Path) -> bool:
    """Whether ssh can use `private` with nobody to type: no passphrase, or an agent holding it.

    A rental is knocked on in batch mode, so a locked pair goes out with the create and refuses
    every knock: an RTX 5090 billed five minutes on 2026-09-21 behind a locked `id_ed25519` an
    empty agent could not open, beside an unlocked `id_rsa`. A locked pair is passed over.
    """
    if _derived(private).returncode == 0:
        return True
    printed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=a constant argv over a path this module chose since=2026-09-21
        ["ssh-keygen", "-lf", f"{private}.pub"], capture_output=True, text=True, check=False
    )
    held = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=a constant argv, no input since=2026-09-21
        ["ssh-add", "-l"], capture_output=True, text=True, check=False
    )
    fingerprint = printed.stdout.split()[1:2]
    return bool(fingerprint) and fingerprint[0] in held.stdout


def seeded(public: str) -> str:
    """The shell lines that put `public` into root's authorized keys with the modes sshd wants.

    A host that injects the account key with the wrong owner or mode, or never, leaves the
    machine billing behind an sshd that refuses every knock (RTX 5090 host 206415, 2026-09-12),
    so the entrypoint writes the landing's own key itself.
    """
    quoted = shlex.quote(public.strip())
    return "\n".join(
        (
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh",
            f"grep -qxF {quoted} /root/.ssh/authorized_keys 2>/dev/null || "
            f"echo {quoted} >> /root/.ssh/authorized_keys",
            "chmod 600 /root/.ssh/authorized_keys && "
            "chown root:root /root/.ssh /root/.ssh/authorized_keys",
        )
    )


def waiting() -> str:
    """The entrypoint shell of a rented machine: hold still for the landing, then run the job.

    Leaves the job's exit status in `$status` for each backend's own reporting line (vast's log
    marker, hpc-ai's sentinel file), so a landed rental settles through the raw one's channel.
    """
    launch, waited = shlex.quote(LAUNCH), "$mb_waited"
    return "\n".join(
        (
            "mb_waited=0",
            f"while [ ! -f {launch} ]; do",
            f'  if [ "{waited}" -ge {LANDING_SECONDS} ]; then break; fi',
            f"  sleep {_WAIT_SECONDS}",
            f"  mb_waited=$(({waited} + {_WAIT_SECONDS}))",
            "done",
            f"if [ -f {launch} ]; then",
            f"  bash {launch}",
            "  status=$?",
            "else",
            f'  echo "mainboard: no dispatch landed within {LANDING_SECONDS}s"',
            f"  status={_NO_LANDING}",
            "fi",
        )
    )


def handoff() -> str:
    """The shell a landing runs to hand the waiting entrypoint its launch script over stdin.

    Written under a neighbouring name and moved into place, since the entrypoint polls for the
    real one and a half-written file is a job that runs half a line.
    """
    return f"cat > {shlex.quote(_PENDING)} && mv {shlex.quote(_PENDING)} {shlex.quote(LAUNCH)}"


def reachable(
    endpoint: Endpoint, *, sleeper: Callable[[float], None], attempts: int = _SSH_ATTEMPTS
) -> Endpoint:
    """`endpoint` once ssh actually answers on it.

    A provider publishes the address before its sshd answers, so knocking is the only honest
    test. A machine that never answers is refused here, where the rental can still be ended,
    rather than a mirror hanging against a box nobody can log into.

    sleeper: waits between knocks, injected so a test drives it without real time.
    """
    policy = SshTransport(endpoint=endpoint)
    said = "nothing"
    for _ in range(attempts):
        try:
            policy.warm(endpoint.destination)
        except HostUnreachable as refused:
            said = str(refused)
            logger.debug("%s not answering ssh yet: %s", endpoint.destination, refused)
            sleeper(_SSH_SECONDS)
        else:
            return endpoint
    # The last word goes in the refusal because the remedies are opposite: a timeout is a machine
    # or a network, `Permission denied (publickey)` a key sent wrong, which three rentals hid
    # behind "never answered" on 2026-09-21.
    raise MissionError(
        f"ssh never answered at {endpoint.destination} after {attempts} attempts; the rental is "
        f"up but unreachable, so nothing was landed on it. The last knock said: {said}"
    )
