# The one CI gate a package states, in its own pyproject.toml, for every place that runs it.
#
# The gate used to live twice, as YAML steps GitHub ran and as whatever a developer remembered to
# run before pushing, and the two drifted: a stale lock, a coverage hole and forty Windows-only
# failures each reached CI first. Now `[tool.mainboard.ci]` is the gate and `mainboard ci` alone
# runs it: from a developer's shell, with `--matrix` on a remote host of each other platform before
# a push, and from the GitHub workflow, which only checks out the code and calls that same verb.
#
# A step is a command line, never a shell script: Windows has no bash, and a gate on three
# platforms has to mean the same words on all three. A step runs on every platform the package
# supports unless it names the ones it is for, which is also how a platform that needs a different
# spelling (a coverage threshold Windows cannot meet) says so in the open.

import tomllib
from pathlib import Path
from shlex import split
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from ..core.base import Declared
from ..core.errors import MissionError
from ..core.host import platform_family
from ..core.project import Project

# The operating-system families a gate runs on, spelled the way pixi spells platform families,
# so a host profile's `platform` names its family without a second vocabulary.
type Family = Literal["linux", "osx", "win"]

FAMILIES: tuple[Family, ...] = ("linux", "osx", "win")

# The file a package declares its gate in.
PYPROJECT = "pyproject.toml"


def family_of(platform: str) -> Family:
    """The family a pixi platform string (`win-64`, `osx-arm64`) belongs to.

    platform: a pixi platform string, or a bare family.
    """
    family = platform_family(platform)
    for known in FAMILIES:
        if known == family:
            return known
    raise MissionError(f"{platform!r} names no platform family CI runs on; use one of {FAMILIES}")


class Step(Declared):
    """One command of the gate, run from the package directory and never through a shell.

    run: the command line, split the way a POSIX shell splits words and never handed to one.
    only: the families the step runs on, every family the package supports when empty.
    timeout: seconds before the step and everything it started are stopped.
    """

    name: str
    run: str
    only: tuple[Family, ...] = ()
    timeout: float = Field(default=1800.0, gt=0)

    @field_validator("run")
    @classmethod
    def splits(cls, run: str) -> str:
        """Refuse an empty command or one whose quoting never closes, at load rather than use."""
        if not split(run):
            raise ValueError("a CI step needs a command to run")
        return run

    @property
    def argv(self) -> tuple[str, ...]:
        """The command's words."""
        return tuple(split(self.run))

    def runs_on(self, family: Family) -> bool:
        """Whether the step belongs to the gate on `family`."""
        return not self.only or family in self.only


class Definition(Declared):
    """A package's `[tool.mainboard.ci]` table: where its gate must pass, and the gate itself.

    os: the families the package supports, each of which a full matrix covers.
    steps: the commands, in order; a leg stops at the first that fails, as a CI job does.
    """

    os: tuple[Family, ...] = FAMILIES
    steps: tuple[Step, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def only_supported(self) -> Self:
        """Refuse a step reserved for a family the package does not support at all."""
        for step in self.steps:
            if strays := sorted(set(step.only) - set(self.os)):
                raise ValueError(f"step {step.name!r} runs only on {strays}, outside os {self.os}")
        return self

    def on(self, family: Family) -> tuple[Step, ...]:
        """The gate as `family` runs it, refusing a family the package does not support.

        A GitHub runner of an unsupported platform is a workflow that drifted from this table,
        which is worth a red job rather than a green one that checked nothing.
        """
        if family not in self.os:
            raise MissionError(f"this package supports {list(self.os)}, not {family}")
        return tuple(step for step in self.steps if step.runs_on(family))


class Package(Declared):
    """A directory whose pyproject.toml declares a CI gate.

    root: the package directory, where every step runs.
    """

    root: Path
    definition: Definition

    @classmethod
    def found(cls, start: Path) -> Self:
        """The nearest package at or above `start` whose pyproject.toml declares a gate."""
        for directory in (start, *start.parents):
            try:
                text = (directory / PYPROJECT).read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            table = tomllib.loads(text).get("tool", {}).get(Project().name, {}).get("ci")
            if table is not None:
                return cls(root=directory, definition=Definition.model_validate(table))
        raise MissionError(
            f"no {PYPROJECT} declares [tool.{Project().name}.ci] from {start} upward"
        )
