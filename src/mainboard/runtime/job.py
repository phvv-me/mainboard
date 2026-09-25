# What a dispatched job is once it reaches the machine that runs it: data, not a shell script.
#
# The dispatching workstation decides everything about a job and writes it down as one `Job`; the
# host's own installed tool carries it out, so the job behaves the same under PBS, pueue, a rented
# box or Windows, and nothing is spelled in a shell grammar only some of them speak.

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

    args: the tool's arguments without its name, since the runner calls the very tool running
        it rather than whichever one a PATH names.
    cwd: where the call runs, the job's working directory when empty.
    credentials: a JSON file of variables only the call receives, empty or missing for none.
    """

    args: tuple[str, ...]
    cwd: str = ""
    credentials: str = ""


class PrefixActivation(FrozenModel):
    """Enter one built prefix, addressed by the content of the artifact it was built from.

    Refused unless its completion stamp names the digest its directory is named for, since a
    half-built environment looks finished from outside.

    prefix: the built environment's directory, whose last segment is its digest.
    refusal: what the job says when the prefix is missing or incomplete.
    """

    kind: Literal["prefix"] = "prefix"
    prefix: str
    env: str
    refusal: str

    def entered(self, base: Mapping[str, str], *, cwd: str, how: Entering) -> dict[str, str]:
        """`base` inside the prefix, with the runtime step applied and its own `lib/` leading.

        The runtime step runs here too, for a prefix whose `activate.sh` predates calling it. The
        environment's own libraries lead the loader path last, so one it ships wins over the
        host's copy and a wheel's.

        cwd: where the activation runs, the job's own tree.
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
        try:
            return (prefix / STAMP).read_text(encoding="utf-8").strip() == prefix.name
        except OSError:
            return False


class WorkspaceActivation(FrozenModel):
    """Enter the workspace's own environment, the footing a job addressing no prefix stands on.

    prefix: the installed prefix, whose executables are enough when no `script` was written.
    refusal: what the job says when neither exists.
    """

    kind: Literal["workspace"] = "workspace"
    script: str
    prefix: str
    refusal: str

    def entered(self, base: Mapping[str, str], *, cwd: str, how: Entering) -> dict[str, str]:
        """`base` inside the workspace's environment: its activation, else its executables.

        Otherwise a `Refusal`, since a command quietly running on the machine's own interpreter
        costs far more to discover than one that will not start.
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

    command: run through `bash -c` on POSIX and split into an argv on Windows.
    root: the pinned tree the command runs from.
    container: an argv running the command inside a container instead, empty for none.
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
        """The record `given` spells as JSON, or the one the job script at that path hands over.

        A POSIX script hands its record over inline, which a scheduler feeding it to a shell on
        stdin still runs. Anything that can name the script instead (a Windows queue whose shell
        would mangle the quotes, a rerun by hand) names the file.
        """
        if given.lstrip().startswith("{"):
            return cls.model_validate_json(given)
        return cls.handed(Path(given).read_text(encoding="utf-8"))

    @classmethod
    def handed(cls, script: str) -> Job:
        """The record a rendered job script hands over, the last word of its last line.

        Lines split on newline only, as a shell reads them, since the record may carry raw
        characters Python would also call a line break.
        """
        handover = script.rstrip("\n").rpartition("\n")[2]
        return cls.model_validate_json(shlex.split(handover)[-1])


def walltime_seconds(walltime: str) -> int:
    """A `HH:MM:SS` walltime as whole seconds."""
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    return hours * 3600 + minutes * 60 + seconds
