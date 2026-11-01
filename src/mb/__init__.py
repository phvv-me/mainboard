# `mb`, the short name `mainboard` answers to as an import, as it does as a command.
#
# Not a copy: `import mb` is the `mainboard` package itself, and `import mb.trials` is the very
# `mainboard.trials` module, so no module is ever loaded twice under two names (two copies would
# mean two of every class, and an `isinstance` across them failing).

import sys
from typing import TYPE_CHECKING

import mainboard

from .finder import install

if TYPE_CHECKING:
    from mainboard import *  # noqa: F403 - what a type checker should see under this name

install()
sys.modules[__name__] = mainboard
