# What a dispatched job is once it reaches the machine that runs it: data, not a shell script.
#
# The dispatching workstation decides everything about a job, the command, the tree it runs
# from, the environment it enters, what it exports and how long it may take, and writes those
# decisions down as one `Job`. The host's own installed tool reads that record and carries it
# out, so the job behaves the same under PBS, pueue, a rented box or Windows, and nothing about
# it is spelled in a shell grammar only some of those machines speak.

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from patos import FrozenModel
from pydantic import Field

from ..engines.compile.prefixes import ACTIVATION, STAMP
from .activation import Runtime, prepended
from .entry import Refusal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .entry import Entering


class ToolCall(FrozenModel):
    """One call of this tool a job makes around its command, provisioning or watching.

    args: the tool's own arguments, its name left out, since the runner calls the very tool
        that is running it rather than whichever one a PATH happens to name.
    cwd: where the call runs, the job's working directory when empty.
    credentials: a JSON file of variables the call alone receives on top of the job's
        environment, empty for none; a file that is not there adds nothing.
    """

    args: tuple[str, ...]
    cwd: str = ""
    credentials: str = ""


class PrefixActivation(FrozenModel):
    """Enter one built prefix, addressed by the content of the artifact it was built from.

    The prefix is refused unless its completion stamp names the digest its directory is named
    for, since a half-built environment and a finished one look the same from outside.

    prefix: the built environment's directory, whose last segment is its digest.
    env: the environment inside it.
    refusal: what the job says when the prefix is missing or incomplete.
    """

    kind: Literal["prefix"] = "prefix"
    prefix: str
    env: str
    refusal: str

    def entered(self, base: Mapping[str, str], *, cwd: str, how: Entering) -> dict[str, str]:
        """`base` inside the prefix, with the runtime step applied and its own `lib/` leading.

        The runtime step runs here as well as in the prefix's `activate.sh`, since a prefix built
        before that script called it is still entered, and every step only adds what is
        missing. The environment's own libraries lead the loader path last of all, so one it
        ships wins over the host's copy and over a wheel's.

        base: the environment the runner was started with.
        cwd: where the activation runs, the job's own tree.
        how: this machine's way of entering an environment.
        """
        prefix = Path(self.prefix)
        script = prefix / ACTIVATION
        if not (self._stamped(prefix) and how.ready(prefix, script)):
            raise Refusal(self.refusal)
        entered = how.activated(base, shard=prefix, script=script, env=self.env, cwd=cwd)
        installed = prefix / ".pixi" / "envs" / self.env
        Runtime(installed).apply(entered)
        prepended(entered, "LD_LIBRARY_PATH", [installed / "lib"])
        return entered

    @staticmethod
    def _stamped(prefix: Path) -> bool:
        """Whether `prefix` is a finished build of the digest its directory is named for."""
        try:
            return (prefix / STAMP).read_text(encoding="utf-8").strip() == prefix.name
        except OSError:
            return False


class WorkspaceActivation(FrozenModel):
    """Enter the workspace's own environment, the footing a job addressing no prefix stands on.

    script: the generated activation the workspace writes for the environment.
    prefix: the environment's installed prefix, whose executables are enough when no script was
        ever written.
    refusal: what the job says when neither exists.
    """

    kind: Literal["workspace"] = "workspace"
    script: str
    prefix: str
    refusal: str

    def entered(self, base: Mapping[str, str], *, cwd: str, how: Entering) -> dict[str, str]:
        """`base` inside the workspace's environment, or a `Refusal` when it has none.

        The generated activation when one was written, the prefix's executables when only those
        exist, and a refusal otherwise: a command that quietly runs on whatever interpreter the
        machine ships costs far more to discover than one that will not start.

        base: the environment the runner was started with.
        cwd: where the activation runs, the job's own tree.
        how: this machine's way of entering an environment.
        """
        installed, script = Path(self.prefix), Path(self.script)
        shard = installed.parents[2]
        if how.ready(shard, script):
            entered = how.activated(base, shard=shard, script=script, env=installed.name, cwd=cwd)
        else:
            executables = [path for path in how.executables(installed) if path.is_dir()]
            if not executables:
                raise Refusal(self.refusal)
            entered = dict(base)
            prepended(entered, "PATH", executables)
        Runtime(installed).apply(entered)
        return entered


type Activation = Annotated[PrefixActivation | WorkspaceActivation, Field(discriminator="kind")]


class Job(FrozenModel):
    """Everything the host needs to run one dispatched command, in the order it happens.

    command: the command line, run through `bash -c` on POSIX and split into an argv on Windows.
    root: the pinned tree the command runs from.
    activation: how the job's environment is entered.
    container: an argv that runs the command inside a container instead, empty for none.
    walltime: the `HH:MM:SS` cap this runner enforces, empty when a scheduler enforces it or the
        caller chose none.
    logs: the directory a PBS job appends its merged output and exit artifact to, empty for a
        backend that captures the output itself.
    pythonpath: the exact `PYTHONPATH` the command imports under, empty for none.
    isolate_pythonpath: drop an inherited `PYTHONPATH` when no exact one is given.
    variables: exported after activation, provenance first and the host profile's exports last.
    provide: builds the environment before it is entered, None when nothing has to.
    attestation: records the machine in the foreground right before the command, None for none.
    sampler: watches the machine beside the command until the command ends, None for none.
    """

    command: str
    root: str
    activation: Activation
    container: tuple[str, ...] = ()
    walltime: str = ""
    logs: str = ""
    pythonpath: str = ""
    isolate_pythonpath: bool = True
    variables: dict[str, str] = {}
    provide: ToolCall | None = None
    attestation: ToolCall | None = None
    sampler: ToolCall | None = None

    @classmethod
    def read(cls, given: str) -> Job:
        """The record `given` spells, or the one the job script at that path hands over.

        A POSIX script hands its record over inline, which is what a scheduler feeding the
        script to a shell on stdin still runs. Everything that can name the script instead, a
        Windows queue whose shell would mangle the record's quotes or someone rerunning a job by
        hand, names the file, and its last line is the handover carrying the record.

        given: the record as JSON, or the path of a rendered job script.
        """
        if given.lstrip().startswith("{"):
            return cls.model_validate_json(given)
        return cls.handed(Path(given).read_text(encoding="utf-8"))

    @classmethod
    def handed(cls, script: str) -> Job:
        """The record a rendered job script hands over, the last word of its last line.

        Lines are what a shell reads, newline-separated and nothing else, since a command may
        carry a character Python would also call a line break and the record keeps it raw.
        """
        handover = script.rstrip("\n").rpartition("\n")[2]
        return cls.model_validate_json(shlex.split(handover)[-1])


def walltime_seconds(walltime: str) -> int:
    """A `HH:MM:SS` walltime as whole seconds."""
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    return hours * 3600 + minutes * 60 + seconds
