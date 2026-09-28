# `mb`, the short name `mainboard` answers to as an import, as it does as a command.
#
# Not a copy: `import mb` is the `mainboard` package itself, and `import mb.trials` is the very
# `mainboard.trials` module, so no module is ever loaded twice under two names (two copies would
# mean two of every class, and an `isinstance` across them failing).

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType
from typing import TYPE_CHECKING

import mainboard

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mainboard import *  # noqa: F403 - what a type checker should see under this name


class _Alias(MetaPathFinder, Loader):
    """Finds `mb.<x>` as the already-importable `mainboard.<x>`."""

    def find_spec(
        self, name: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        return spec_from_loader(name, self) if name.startswith("mb.") else None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = import_module("mainboard" + spec.name.removeprefix("mb"))
        # The import system stamps this alias spec onto the module; keep the real one to restore.
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        if module.__spec__ is not None:
            module.__spec__ = module.__spec__.loader_state


sys.meta_path.insert(0, _Alias())
sys.modules[__name__] = mainboard
