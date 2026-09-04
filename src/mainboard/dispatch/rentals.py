# What a machine rented for one job is in this workspace, and the handshake its two halves keep.
#
# A provider rents a container, not a host. It comes up with no workspace, no tool and no
# environment, which is why a dispatched `mainboard run` died on vast instance 49861190 with
# `bash: line 2: mainboard: command not found`, exit 127, three times over, and billed for every
# boot. So a rental is treated as an ssh host that exists for one job: the dispatch mirrors the
# workspace onto it, installs the tool from that mirror, provisions the environment from the lock
# this workspace already solved and pins the tree the job runs from, exactly as `mainboard setup`
# does on gold.
#
# All of that happens after the boot and on the meter, which is what the handshake here is for.
# The container's entrypoint waits for a launch script instead of running the command straight
# away; the landing writes that script once the machine can actually run it; the entrypoint then
# runs the job in the foreground, which leaves the log, the exit marker and the cancel that stops
# the meter exactly where each provider backend already reads them. The wait is bounded, because
# it is billed: a workstation that dies mid-landing ends the entrypoint with a status the ordinary
# settle-and-cancel path can act on, rather than leaving a container waiting for a script nobody
# will ever write.

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from .shared import logger
from .transport import Endpoint, HostUnreachable, SshTransport

if TYPE_CHECKING:
    from collections.abc import Callable

# Where the landing leaves the script the waiting entrypoint runs, and the neighbouring name it
# is written under first. `/tmp` because it is the one writable directory every image agrees on,
# whichever user the provider's entrypoint happens to run as.
LAUNCH = "/tmp/mainboard-launch.sh"
_PENDING = "/tmp/mainboard-launch.part"

# How long a rented machine waits for a dispatch to land on it, and how long the spend cap has to
# assume every rental bills before its job starts. An hour, because the slow step is not the
# mirror (288 MB of this workspace measured 2026-09-04) but the environment: a workspace whose
# default environment is a whole CUDA stack downloads and links gigabytes on a cold container. The
# ceiling is generous rather than tight on purpose, since overrunning it throws away a rental that
# was nearly ready, while a landing that finishes in ten minutes never pays for the other fifty.
LANDING_SECONDS = 3600

# How often the entrypoint looks for the launch script while it waits.
_WAIT_SECONDS = 5

# What the entrypoint exits when no dispatch ever landed. `EX_TEMPFAIL`, since the machine was
# fine and the workstation was not, and a distinct code is what tells that apart from the job
# itself failing when the log is all anyone has left.
_NO_LANDING = 75

# The standard key pairs under `~/.ssh`, newest algorithm first, so a workspace that never
# declared one still opens a rental with the key it already uses everywhere else.
_KEY_NAMES = ("id_ed25519", "id_ecdsa", "id_rsa")

# How long to keep knocking before calling a rented machine unreachable. A container answers ssh
# seconds after its status turns running, but a provider that publishes an address early can leave
# the knock refused for a while, so the budget is minutes rather than seconds.
_SSH_ATTEMPTS = 40
_SSH_SECONDS = 5.0


class Identity(FrozenModel):
    """The key pair a rental is opened with: the public half attached, the private half used.

    private: the private key file this machine connects with.
    public: the public key text handed to the provider at create time.
    """

    private: str
    public: str


class Rental(FrozenModel):
    """A machine rented for one job: the provider's handle, and where ssh reaches it now.

    handle: the provider's own opaque run id, which every later state, log and cancel addresses.
    endpoint: the ssh target the dispatch lands on, valid only for this rental's lifetime.
    """

    handle: str
    endpoint: Endpoint


def identity(declared: str = "") -> Identity:
    """The key pair this workspace opens a rental with, refusing when it holds none.

    A provider hands out a machine rather than an account, so the only way into a fresh rental is
    a public key given at create time, and it has to be one this machine also holds the private
    half of. That is why the pair is read here rather than from the provider's own key list: a key
    registered from another laptop looks registered and refuses every connection.

    declared: `[hosts.<name>.vars] ssh-key`, the private key to use; empty takes the first
        standard pair under `~/.ssh`.
    """
    candidates = (
        [Path(declared).expanduser()]
        if declared
        else [Path.home() / ".ssh" / name for name in _KEY_NAMES]
    )
    for private in candidates:
        public = Path(f"{private}.pub")
        if public.is_file():
            return Identity(
                private=str(private), public=public.read_text(encoding="utf-8").strip()
            )
    named = f"{declared} " if declared else ""
    raise MissionError(
        f"no ssh key pair {named}to open a rental with; a rented machine is reached over ssh and "
        "is created with the public half of a key this machine holds. Generate one with "
        "`ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519`, or point [hosts.<host>.vars] ssh-key at "
        "the private key of a pair you already have."
    )


def waiting() -> str:
    """The entrypoint shell of a rented machine: hold still for the landing, then run the job.

    Leaves the job's own exit status in `$status`, which is what each backend's own reporting
    line (vast's log marker, hpc-ai's sentinel file) then answers with, so a landed rental settles
    through exactly the channel a raw one already did.
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
    """The shell a landing runs to hand the waiting entrypoint its launch script, over stdin.

    Written under a neighbouring name and moved into place, since the entrypoint is polling for
    the real one and a half-written file is a job that runs half a line.
    """
    return f"cat > {shlex.quote(_PENDING)} && mv {shlex.quote(_PENDING)} {shlex.quote(LAUNCH)}"


def reachable(
    endpoint: Endpoint, *, sleeper: Callable[[float], None], attempts: int = _SSH_ATTEMPTS
) -> Endpoint:
    """`endpoint` once ssh actually answers on it, the wait every fresh rental needs.

    A provider publishes an address as soon as the container is up, and its ssh daemon answers
    some seconds later, so knocking is the only honest test of whether a landing can start. A
    machine that never answers is a refusal here, where the rental can still be ended, rather than
    a mirror that hangs against a box nobody can log into.

    endpoint: where the provider says the machine is.
    sleeper: waits between knocks, injected so a test drives it without real time.
    attempts: how many knocks before giving up.
    """
    policy = SshTransport(endpoint=endpoint)
    for _ in range(attempts):
        try:
            policy.warm(endpoint.destination)
        except HostUnreachable as refused:
            logger.debug("%s not answering ssh yet: %s", endpoint.destination, refused)
            sleeper(_SSH_SECONDS)
        else:
            return endpoint
    raise MissionError(
        f"ssh never answered at {endpoint.destination} after {attempts} attempts; the rental is "
        "up but unreachable, so nothing was landed on it"
    )
