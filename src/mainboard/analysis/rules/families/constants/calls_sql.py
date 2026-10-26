"""The constants of `calls_sql` other modules share."""

import builtins
import sys

# mcmr names a call's standard-library roots from the interpreter's own list.
STANDARD_LIBRARY = sorted(sys.stdlib_module_names)

# Every builtin the interpreter names (mcmr's own list is a subset of it).
BUILTINS = sorted(name for name in dir(builtins) if not name.startswith("_"))
