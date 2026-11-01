import os
from typing import TYPE_CHECKING

from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Mapping


class Variable(FrozenModel):
    """One environment variable the tool trades with the processes it starts, under every name.

    Both ends may run different releases, so an export sets every name (an older reader knows
    only the legacy one) and a read takes the first name set, primary first.
    """

    names: tuple[str, ...]

    def exported(self, value: str) -> dict[str, str]:
        """`value` under every name, the mapping a child environment is updated with."""
        return dict.fromkeys(self.names, value)

    def present(self, environ: Mapping[str, str] | None = None) -> bool:
        """Whether `environ` (the process environment) sets the variable under any name."""
        source = os.environ if environ is None else environ
        return any(name in source for name in self.names)

    def read(self, environ: Mapping[str, str] | None = None) -> str:
        """The value under the first name `environ` (the process environment) sets, else empty."""
        source = os.environ if environ is None else environ
        return next((source[name] for name in self.names if name in source), "")
