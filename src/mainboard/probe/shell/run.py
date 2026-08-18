from functools import cache

from plumbum import local


@cache
def run(*command: str) -> str:
    """Run a command once and return its stdout.

    Results are cached by argv, so repeated identity probes (e.g. `--version`)
    execute the underlying process a single time.

    command: program name followed by its arguments, e.g. `("clang", "--version")`.
    """
    program, *args = command
    return local[program](*args)
