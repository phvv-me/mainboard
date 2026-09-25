from functools import cache

from plumbum import local


@cache
def run(*command: str) -> str:
    """The stdout of `command`, e.g. `("clang", "--version")`.

    Cached by argv, so repeated identity probes run the process once.
    """
    program, *args = command
    return local[program](*args)
