from abc import ABC, abstractmethod
from pathlib import Path
from shlex import join
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.project import Project
from ..core.section import Section, Verdict

if TYPE_CHECKING:
    from ..board import Board

# The tool this workspace answers to, so no command below spells the binary's name.
_TOOL = Project().name


def _done(verdict: Verdict, detail: str) -> Section:
    return Section(section="fix", verdict=verdict, detail=detail)


class Remedy(FrozenModel, ABC):
    """One repair of a library clash, shown as a command and applied by the same code.

    Neither kind touches a system package or needs root: one edits the manifest, the other only
    the environment's own directory.
    """

    @property
    @abstractmethod
    def command(self) -> str:
        """The command that does what `apply` does, for a report to name."""

    @abstractmethod
    def apply(self, board: Board) -> Section:
        """Do the repair and answer the one row saying what changed."""


class Raise(Remedy):
    """Declare the newest release of the package that owns a library family.

    The family is incomplete in the release the lock pins (cuDNN 9.20's wheel omits
    `libcudnn_engines_tensor_ir`), so the loader falls through to the host's copy. A newer
    release that ships the whole family ends that, and the edit is an ordinary manifest change.

    package: the package to raise, as the manifest names it.
    ecosystem: `python` or `conda`, the table it is declared in.
    env: the environment the clash was found in, empty for the default one.
    """

    package: str
    ecosystem: str
    env: str = ""

    @property
    def command(self) -> str:
        flags = ["--lang", self.ecosystem, *(["--env", self.env] if self.env else [])]
        return join([_TOOL, "add", self.package, *flags])

    def apply(self, board: Board) -> Section:
        """Declare `package` at the newest release its index publishes, then re-lock and install.

        A solve the lock refuses (torch hard-pins its cuDNN) leaves the manifest as it was and
        says so; the way past a transitive pin is `[python.dependency-overrides]`.
        """
        try:
            moved = board.deps().add(
                self.package, ecosystem=self.ecosystem, env=self.env, install=True
            )[0]
        except MissionError as refusal:
            return _done(
                Verdict.WARN,
                f"could not raise {self.package}: {str(refusal).splitlines()[0]} "
                "(a transitive pin needs an entry under [python.dependency-overrides])",
            )
        return _done(
            Verdict.PASS,
            f"declared {moved.name} {moved.after} in {moved.where} (was {moved.before}), "
            "re-locked and installed",
        )


class Relink(Remedy):
    """Spell a library under the name something opens it by, inside the environment.

    cuDNN opens its sub-libraries by their full version (`libcudnn_graph.so.9.27.0`) while its
    wheel ships the soname (`.so.9`), so the host's copy under the full name answers instead.
    A link beside the environment's own file ends that. It lives in the environment's directory
    and is lost when the package is reinstalled, so `doctor --fix` makes it again.

    directory: the environment directory holding the file.
    name: the name to add.
    target: the environment's file it points at.
    """

    directory: Path
    name: str
    target: str

    @property
    def command(self) -> str:
        return join(["ln", "-s", self.target, str(self.directory / self.name)])

    def apply(self, board: Board) -> Section:
        del board
        link = self.directory / self.name
        try:
            link.symlink_to(self.target)
        except FileExistsError:
            return _done(Verdict.PASS, f"{link} already exists")
        return _done(Verdict.PASS, f"linked {link} -> {self.target}")
