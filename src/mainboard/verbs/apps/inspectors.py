"""What `mb code app inspect` reads: a built bundle, or what a result bundle holds."""

from pathlib import Path
from typing import TYPE_CHECKING

from patos import value_dispatch

from ...core.errors import MissionError
from ...log import logger
from .enums import Kind

if TYPE_CHECKING:
    from ...apple import Apple, BundleChecks
    from ...render import Output


@value_dispatch
def inspect(
    path: Path, *, apple: Apple, checks: BundleChecks, output: Output, kind: Kind | None = None
) -> int:
    """Refuse a path of no known kind, naming the kinds."""
    del apple, checks, output, kind
    raise MissionError(
        f"{path} is of no kind to inspect; the kinds are {Kind.BUNDLE} (a built .app, .xcarchive "
        f"or any folder holding an Info.plist) and {Kind.RESULTS} (an .xcresult, or a folder "
        "holding them)"
    )


@inspect.register(Kind.BUNDLE)
def _bundle(path: Path, *, apple: Apple, checks: BundleChecks, output: Output) -> int:
    """Print a built bundle's metadata, resources and Mach-O files; 1 unless every check held."""
    report = apple.bundle(path, checks)
    output.print_record(report.document(), title="Apple bundle")
    return 0 if report.successful else 1


@inspect.register(Kind.RESULTS)
def _results(path: Path, *, apple: Apple, checks: BundleChecks, output: Output) -> int:
    """Print a result bundle's build diagnostics, then its test results; 1 unless both passed.

    A bundle with no readable test results (a build that never reached its tests) is judged by
    its build alone, and the unreadable results are warned.
    """
    if checks.model_dump(exclude_defaults=True):
        raise MissionError("bundle checks apply to a built bundle, not to a result bundle")
    build = apple.diagnostics(path)
    output.print_record(build.document(), title="Apple build diagnostics")
    try:
        tests = apple.results(path)
    except MissionError as fault:
        logger.warning(f"Apple test results unavailable: {fault}")
        return 0 if build.successful else 1
    output.print_record(tests.document(), title="Apple tests")
    return 0 if build.successful and tests.successful else 1
