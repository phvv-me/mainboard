from enum import IntEnum, StrEnum, auto


class Access(StrEnum):
    """How usable one compute path is right now."""

    HERE = auto()
    PROVISIONED = auto()
    REACHABLE = auto()
    UNREACHABLE = auto()
    KEYED = auto()
    UNKEYED = auto()
    RENTED = auto()


class Onboard(StrEnum):
    """How much of a host's onboarding runs: all of it, all of it with the workspace's dotfiles,
    or a sync (re-mirror and re-provision, the tool and the hardware probe left as they are,
    the dotfiles as the setup chose)."""

    MINIMAL = auto()
    DOTFILES = auto()
    SYNC = auto()


class Source(StrEnum):
    """Where an executable was found, in the order the search tries them."""

    INSTALLED = auto()
    ENVIRONMENT = auto()
    PATH = auto()


class Kind(StrEnum):
    """The four events a job's observability stream can carry."""

    started = auto()
    line = auto()
    sample = auto()
    ended = auto()


class Pricing(StrEnum):
    """Which offers a search takes: either kind, only interruptible ones, or only on-demand."""

    ANY = auto()
    ON_DEMAND = auto()
    SPOT = auto()

    def admits(self, *, spot: bool) -> bool:
        """Whether an offer that is interruptible (`spot`) or not answers this choice."""
        return self is Pricing.ANY or (self is Pricing.SPOT) == spot


class LockPolicy(StrEnum):
    """What a verb may do to the lock before it installs, spelled as pixi spells it.

    update: solve again when the lock no longer answers the manifest, pixi's default.
    locked: refuse a lock that no longer answers the manifest (`--locked`), what a host does,
        since a host installs what the center solved and never solves for itself.
    frozen: take the lock as it stands without asking whether it answers (`--frozen`).
    """

    UPDATE = auto()
    LOCKED = auto()
    FROZEN = auto()

    @property
    def flag(self) -> str:
        """The flag passing this policy on to another machine's install, empty for the default."""
        return "" if self is LockPolicy.UPDATE else f"--{self}"


class Verdict(StrEnum):
    """How one section came back: fit, fit with something worth saying, or broken."""

    PASS = auto()
    WARN = auto()
    FAIL = auto()


class Act(StrEnum):
    """What an unattended process may need the owner's leave for, and the owner's card limits.

    `CARD` is no leave: `serve grant card PROVIDER --usd N` declares the limit of the card paying a
    provider that bills one after the fact (`grants.Card`), without which that provider rents
    nothing. It has no expiry and nothing asks for it.
    """

    CARD = auto()
    LAUNCH = auto()
    PUSH = auto()
    RENT = auto()


class Status(StrEnum):
    """File kind; historical version-control labels remain readable, never gate execution."""

    SOURCE = auto()
    BUILT = auto()
    CLEAN = auto()
    MODIFIED = auto()
    UNTRACKED = auto()
    IGNORED = auto()
    UNVERSIONED = auto()


class JobFrame(StrEnum):
    """The events a job's host emits, named as their spool lines carry them."""

    started = "mb.job.started"
    heartbeat = "mb.job.heartbeat"
    settled = "mb.job.settled"


class JobReading(StrEnum):
    """The attributes of a job frame beside its identities: counters, then how the job ended.

    wall_s: seconds since the runner started.
    cpu_s: user and system seconds the command's living processes have used.
    rss_bytes: resident memory of the command and its living descendants.
    processes: how many of them there are.
    exit_code: the job's exit status.
    stop: why it ended, as a `Stop` spells it.
    """

    wall_s = "mb.job.wall_s"
    cpu_s = "mb.job.cpu_s"
    rss_bytes = "mb.job.rss_bytes"
    processes = "mb.job.processes"
    exit_code = "mb.job.exit_code"
    stop = "mb.job.stop"


class Stop(StrEnum):
    """Why a job ended.

    exited: the command left on its own, whatever its status.
    signaled: a signal reached the runner.
    walltime: the runner ended the command at its walltime.
    refused: the environment would not start the command.
    """

    exited = auto()
    signaled = auto()
    walltime = auto()
    refused = auto()


class Exit(IntEnum):
    """The exit statuses a command did not choose, spelled as the shell spells them.

    TIMED_OUT: GNU `timeout`'s status for a command it had to stop, so scripts branching on it
        still work.
    MISSING: the shell's status for a command it could not find.
    """

    TIMED_OUT = 124
    MISSING = 127


class Guardrail(StrEnum):
    """Protections applied when layering an env onto a base image.

    `unset_pip_constraint` clears the `PIP_CONSTRAINT` NGC images bake in (unwritable in a SIF).
    """

    UNSET_PIP_CONSTRAINT = auto()


class Topic(StrEnum):
    """Every routing key a batch publishes under, dotted the way a broker subject is."""

    OPENED = "batch.opened"
    PREPARED = "job.prepared"
    ESTIMATED = "job.estimated"
    SUBMITTED = "job.submitted"
    REFUSED = "job.refused"
    HELD = "job.held"
    SKIPPED = "job.skipped"
    STATE = "job.state"
    ATTESTED = "job.attested"
    SAMPLE = "job.sample"
    SETTLED = "job.settled"
    EVIDENCE = "job.evidence"
    COST = "job.cost"
    CLOSED = "batch.closed"


class Bound(StrEnum):
    """What stops a provider billing once nothing here is left to release its machine.

    No provider documents a hard account cap (cloud survey, 2026-10-28), so these three are all
    there is. `CARD`: the provider bills a card after the fact, so the limit of the card paying it
    is the cap, and nothing here can read that limit. `PREPAID`: the provider draws a balance paid
    in advance and ends its machines at zero, a cap only while automatic top-up stays off.
    `THRESHOLD`: the provider ends the machine itself at a spend or time threshold each rental
    carries (Shadeform's `auto_delete`, a Modal sandbox's timeout).
    """

    CARD = auto()
    PREPAID = auto()
    THRESHOLD = auto()


class Role(StrEnum):
    """What a machine is for: the center that holds the monorepo, or a target that runs jobs."""

    CENTER = auto()
    TARGET = auto()
