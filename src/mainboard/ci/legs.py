# One leg is the gate run on one machine: here, in place, or on a declared host of another
# platform, reached the way every dispatch reaches it. A leg runs its steps in order and stops at
# the first that fails, the way a CI job does, so what a leg reports is what GitHub's job for the
# same platform would have reported, step for step.
#
# A remote leg ships the package first, the working tree as it stands with uncommitted edits in
# it, into a directory of its own under the host's generated tree. The host's own mirror is left
# alone, since a job may be running from it, and the package's environment persists there between
# runs, so the second matrix only pays for what changed. Each step is then one ssh process of its
# own under the step's deadline, spelled in the host's own shell, PowerShell on Windows.

import os
import subprocess
import time
from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.host import current_platform
from ..core.project import Project
from ..dispatch.shells import dialect_for, plain_errors
from ..dispatch.targets import rooted
from ..dispatch.transport import HostUnreachable, SshTransport
from ..lint.process import MISSING, TIMED_OUT, Invocation
from .definition import Family, family_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from ..context.plan import ExecutionPlan
    from .definition import Step

# The leg that runs on this machine, in place.
HERE = "local"

# What the report calls a remote leg's first step, the package reaching the host.
SHIP = "ship"

# The variable `uv run` leaves pointing at the environment it ran from. A step names its own
# project's environment through uv, so an inherited one would only draw a mismatch warning.
_INHERITED_VENV = "VIRTUAL_ENV"


class Verdict(StrEnum):
    """How one step ended on one leg.

    `ok` passed, `failed` exited nonzero, `timed out` outlived its deadline, `missing` found no
    such program, `unreachable` never reached the host, and `not run` followed a failure.
    """

    OK = auto()
    FAILED = auto()
    TIMED_OUT = "timed out"
    MISSING = auto()
    UNREACHABLE = auto()
    NOT_RUN = "not run"

    @classmethod
    def of(cls, code: int) -> Verdict:
        """The verdict an exit code states, with the shell's words for a deadline and a miss."""
        return {0: cls.OK, TIMED_OUT: cls.TIMED_OUT, MISSING: cls.MISSING}.get(code, cls.FAILED)


class Result(FrozenModel):
    """One step of the gate on one leg.

    leg: `local`, or the host alias the leg ran on.
    os: the leg's family.
    seconds: wall time, as this machine measured it.
    output: everything the step printed, stdout and stderr together.
    """

    leg: str
    os: Family
    step: str
    verdict: Verdict
    seconds: float = 0.0
    output: str = ""

    @property
    def failed(self) -> bool:
        """Whether this step is why the leg did not pass."""
        return self.verdict not in {Verdict.OK, Verdict.NOT_RUN}

    @property
    def transcript(self) -> str:
        """A heading naming the step and how it ended, then its output; empty if it never ran."""
        if self.verdict is Verdict.NOT_RUN:
            return ""
        heading = f"{self.leg} [{self.os}] {self.step}: {self.verdict} in {self.seconds:.1f}s"
        return "\n".join(filter(None, [heading, self.output.rstrip()]))

    def row(self) -> dict[str, str | float]:
        """The table row, output left to the transcript that already printed it."""
        return self.model_dump(exclude={"seconds", "output"}) | {"seconds": round(self.seconds, 1)}


class Leg(ABC):
    """The gate on one machine, stopping at the first step that fails.

    name: what the report calls the leg.
    """

    def __init__(self, name: str, family: Family) -> None:
        self.name = name
        self.family = family

    def run(self, steps: Sequence[Step]) -> Iterator[Result]:
        """Each step's result as it settles, `not run` for everything after a failure."""
        stopped = False
        for step in steps:
            result = self.skipped(step.name) if stopped else self._step(step)
            stopped = stopped or result.failed
            yield result

    def result(self, step: str, verdict: Verdict, *, started: float, output: str) -> Result:
        """One settled step of this leg, timed from `started`."""
        seconds = time.monotonic() - started
        return Result(
            leg=self.name,
            os=self.family,
            step=step,
            verdict=verdict,
            seconds=seconds,
            output=output,
        )

    def skipped(self, step: str) -> Result:
        """The row for a step that never ran because an earlier one failed."""
        return Result(leg=self.name, os=self.family, step=step, verdict=Verdict.NOT_RUN)

    @abstractmethod
    def _step(self, step: Step) -> Result:
        """Run one step on this leg's machine."""


class LocalLeg(Leg):
    """The gate on this machine, run in place in the package directory.

    root: the package directory.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(HERE, family_of(current_platform()))
        self.root = root

    def _step(self, step: Step) -> Result:
        environment = {key: value for key, value in os.environ.items() if key != _INHERITED_VENV}
        outcome = Invocation(
            step=step.name,
            owner=".",
            cwd=self.root,
            argv=step.argv,
            env=HERE,
            timeout=step.timeout,
        ).run(environment)
        return Result(
            leg=self.name,
            os=self.family,
            step=step.name,
            verdict=Verdict.of(outcome.code),
            seconds=outcome.seconds,
            output=outcome.output,
        )


class RemoteLeg(Leg):
    """The gate on a declared host, the package shipped to a directory of its own there first.

    plan: the host's bare execution plan, whose profile names the platform and the root.
    package: the package directory, workspace-relative with forward slashes.
    ship: sends the working tree's `package` under a root on the plan's host.
    ssh: the bounded ssh policy every step rides.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        package: str,
        *,
        ship: Callable[[ExecutionPlan, str, str], None],
        ssh: SshTransport | None = None,
    ) -> None:
        if not plan.profile.platform:
            raise MissionError(
                f"a CI leg needs the host's platform; declare [hosts.{plan.host}] platform"
            )
        super().__init__(plan.host, family_of(plan.profile.platform))
        self.plan = plan
        self.mirror = rooted(plan.profile, host=plan.host)
        self.package = package
        self.ship = ship
        self.ssh = ssh or SshTransport()

    @property
    def root(self) -> str:
        """Where the leg's copy of the workspace lives on the host, beside the host's mirror."""
        return f"{self.mirror}/{Project().out_dir}/ci"

    def run(self, steps: Sequence[Step]) -> Iterator[Result]:
        """The package shipped, then its steps, every one `not run` when it never arrived."""
        started = time.monotonic()
        try:
            self.ship(self.plan, self.root, self.package)
        except (HostUnreachable, OSError, RuntimeError) as error:
            yield self.result(SHIP, Verdict.UNREACHABLE, started=started, output=str(error))
            yield from (self.skipped(step.name) for step in steps)
            return
        yield self.result(SHIP, Verdict.OK, started=started, output="")
        yield from super().run(steps)

    def _step(self, step: Step) -> Result:
        dialect = dialect_for(self.plan.profile)
        where = f"{self.root}/{self.package}"
        line = dialect.stage(
            self.plan, where, command=dialect.invocation(step.argv), activate=False
        )
        started = time.monotonic()
        try:
            code, out, err = self.ssh.invoke(
                dialect.one_shot(self.ssh, self.plan.host, line),
                self.plan.host,
                operation=f"ci {step.name}",
                timeout=step.timeout,
            )
        except HostUnreachable as error:
            timed_out = isinstance(error.__cause__, subprocess.TimeoutExpired)
            verdict = Verdict.TIMED_OUT if timed_out else Verdict.UNREACHABLE
            return self.result(step.name, verdict, started=started, output=str(error))
        output = out + plain_errors(err)
        return self.result(step.name, Verdict.of(code), started=started, output=output)
