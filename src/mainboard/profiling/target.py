import runpy
import sys
from os import PathLike

from ..models.base import FrozenModel


class Target(FrozenModel):
    """A Python module or script invocation that can be executed in the current process."""

    name: str
    module: bool
    args: tuple[str, ...] = ()

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
