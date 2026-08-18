from pathlib import Path

from patos import FrozenModel

_PACKAGE = __name__.split(".")[0]


class Project(FrozenModel):
    """Every name the tool answers to, derived from the installed package name.

    Renaming the tool is renaming the module directory: the manifest filename,
    generated directory, and entry-point group all follow `__name__`, so no
    code names the tool literally. The only literal spellings left are the
    distribution metadata in pyproject.toml (name and console script), which a
    rename edits alongside the directory.
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
    def plugin_group(self) -> str:
        """The entry-point group third-party providers advertise under."""
        return f"{self.name}.providers"

    def activation(self, env: str = "default") -> str:
        """The generated activation script for `env`, relative to the workspace root.

        One script per environment, because a workspace installs several and a single file
        would activate whichever environment was provisioned last no matter which one the
        caller asked for. The default environment keeps the bare `activate.sh`, the name a
        host onboarded earlier already carries and a hand-written job script already sources.

        env: the environment the script activates.
        """
        suffix = "" if env == "default" else f"-{env}"
        return f"{self.out_dir}/activate{suffix}.sh"

    def find_root(self, start: Path) -> Path:
        """Walk up from `start` to the nearest directory holding the manifest.

        start: the directory the search begins in, usually the cwd.
        """
        for directory in (start, *start.parents):
            if (directory / self.manifest).is_file():
                return directory
        raise FileNotFoundError(
            f"no {self.manifest} found from {start} upward; run inside a workspace"
        )
