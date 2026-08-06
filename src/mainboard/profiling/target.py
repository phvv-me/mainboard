from __future__ import annotations

import runpy
import subprocess
import sys
from os import PathLike, environ, pathsep
from pathlib import Path

from ..models.base import FrozenModel, FrozenSequence


class Target(FrozenModel):
    """A Python module or script invocation that can be executed in the current process."""

    name: str
    module: bool
    args: FrozenSequence[str] = ()

    @classmethod
    def resolve(
        cls,
        target: str | PathLike[str],
        *,
        module: bool | None = None,
        args: tuple[str, ...] = (),
    ) -> Target:
        """Build a target and infer module mode when `module` is omitted."""
        name = str(target)
        is_module = not name.endswith(".py") if module is None else module
        return cls(name=name, module=is_module, args=args)

    def run(self) -> None:
        """Execute the target as `__main__` while preserving the caller's arguments."""
        previous = sys.argv
        sys.argv = [self.name, *self.args]
        try:
            if self.module:
                runpy.run_module(self.name, run_name="__main__")
            else:
                runpy.run_path(self.name, run_name="__main__")
        finally:
            sys.argv = previous

    def launch(
        self,
        executable: Path,
        *,
        timeout: float | None = None,
        import_paths: tuple[Path, ...] = (),
    ) -> None:
        """Execute the target in a bounded child of the selected Python interpreter."""
        command = [str(executable), *(("-m",) if self.module else ()), self.name, *self.args]
        subprocess.run(
            command,
            check=True,
            timeout=timeout,
            env=self.environment(import_paths),
        )

    @staticmethod
    def environment(import_paths: tuple[Path, ...]) -> dict[str, str] | None:
        """Prepend required import roots while preserving the caller's environment."""
        if not import_paths:
            return None
        current = environ.get("PYTHONPATH")
        python_path = pathsep.join(str(item) for item in import_paths)
        if current:
            python_path = pathsep.join((python_path, current))
        return environ | {"PYTHONPATH": python_path}
