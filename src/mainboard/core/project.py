from pathlib import Path

from patos import FrozenModel

from .membership import Membership

_PACKAGE = __name__.split(".")[0]


class Project(FrozenModel):
    """Every name the tool answers to, derived from the installed package name.

    Renaming the tool is renaming the module directory: everything follows `__name__`, and only
    pyproject.toml (distribution name and console script) spells the name literally.
    """

    name: str = _PACKAGE

    @property
    def manifest(self) -> str:
        """The workspace manifest filename."""
        return f"{self.name}.toml"

    @property
    def out_dir(self) -> str:
        """The generated-artifacts directory name at the workspace root."""
        return f".{self.name}"

    @property
    def jobs_root(self) -> str:
        """Where a dispatch target keeps the tool's code and state unless its profile says
        otherwise: one dedicated folder under the login home, never a human checkout."""
        return f"~/.{self.name}-jobs"

    @property
    def plugin_group(self) -> str:
        """The entry-point group third-party providers advertise under."""
        return f"{self.name}.providers"

    def activation(self, env: str = "default") -> str:
        """The activation script for `env`, relative to the workspace root.

        One per environment, since a shared file would activate whichever was provisioned last.
        The default keeps the bare `activate.sh` that onboarded hosts and hand-written job
        scripts already source.
        """
        suffix = "" if env == "default" else f"-{env}"
        return f"{self.out_dir}/activate{suffix}.sh"

    def find_root(self, start: Path) -> Path:
        """The workspace `start` lies in: the nearest manifest upward, or the one composing it.

        Inside a member, the ancestor whose `[workspace] members` claims that member is the
        workspace, the way cargo finds its workspace root, so a member's tasks are reachable
        from its own directory; cloned alone, the member's manifest is the nearest and only one.
        """
        nearest = self._nearest(start)
        for ancestor in nearest.parents:
            if (ancestor / self.manifest).is_file() and Membership.declared(
                ancestor, self.manifest
            ).claims(nearest):
                return ancestor
        return nearest

    def _nearest(self, start: Path) -> Path:
        """The nearest directory at or above `start` holding a manifest."""
        for directory in (start, *start.parents):
            if (directory / self.manifest).is_file():
                return directory
        raise FileNotFoundError(
            f"no {self.manifest} found from {start} upward; run inside a workspace"
        )

    def workspace(self, start: Path | None = None) -> Path:
        """The workspace `start` (default the cwd) belongs to, or `start` itself outside any.

        Generated state belongs to the workspace, not the directory a command was typed in, and
        a scratch tree under no manifest keeps its own rather than raising.
        """
        here = start or Path.cwd()
        try:
            return self.find_root(here)
        except FileNotFoundError:
            return here
