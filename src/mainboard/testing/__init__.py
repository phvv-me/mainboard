# A pytest plugin any house package registers to catch, on a developer's own machine, the
# failures that used to surface only on a CI runner of another operating system.
#
# Registered the ordinary way, `pytest_plugins = ["mainboard.testing"]` in the rootdir conftest,
# and not as a `pytest11` entry point, for the reason `mainboard.trials.pytest_plugin` gives: an
# entry point is imported before pytest-cov starts, and would load into every pytest session on a
# machine this tool is installed beside.
#
# Three things, each one line for a test to ask for:
#
# - Every test is sealed off other machines (`seal.py`), and a test that reached for one fails
#   even when the code under test swallowed the refusal.
# - `windows_like` (a fixture, a marker, or `--windows-like` for the whole session) hides the
#   POSIX tools and renders paths as Windows does (`windows.py`).
# - `path_flavor` and `newline` parametrize a test over both path grammars and both line endings,
#   and `case_variants` names a file the ways a case-insensitive filesystem treats as one.
#
# EVERY NAME A HOOK OR A FIXTURE ANNOTATES IS IMPORTED AT RUNTIME, since pytest reads those
# signatures when it registers them.

from collections.abc import Iterator
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

import pytest

from .seal import REMOTE_TOOLS, RemoteReached, Seal, is_local, program
from .windows import POSIX_TOOLS, Spelling, WindowsLike, first_party

__all__ = [
    "POSIX_TOOLS",
    "REMOTE_TOOLS",
    "RemoteReached",
    "Seal",
    "Spelling",
    "WindowsLike",
    "case_variants",
    "first_party",
    "is_local",
    "program",
]

_MARKER = "windows_like"


def case_variants(name: str) -> tuple[str, ...]:
    """`name` spelled the ways a case-insensitive filesystem (Windows, macOS) reads as one file.

    name: a file or directory name.
    """
    return tuple(dict.fromkeys((name, name.lower(), name.upper(), name.swapcase())))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--windows-like",
        action="store_true",
        help="run every test as a bare Windows machine would: no POSIX tools on PATH, and "
        "paths rendered with backslashes when the project turns them into text",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", f"{_MARKER}: run this test as a bare Windows machine would answer it"
    )


@pytest.fixture(scope="session")
def seal_stand_ins(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The refusing stand-ins for every remote tool, written once for the session."""
    return Seal.made(tmp_path_factory.mktemp("sealed")).stand_ins


@pytest.fixture(autouse=True)
def sealed(seal_stand_ins: Path) -> Iterator[Seal]:
    """Refuse every reach for another machine, and fail the test that made one after it ends.

    The patch is held by its own `MonkeyPatch` rather than the `monkeypatch` fixture, so it is
    installed before any fixture a test asks for and undone only after all of them.
    """
    seal = Seal(seal_stand_ins)
    with pytest.MonkeyPatch.context() as monkeypatch:
        seal.install(monkeypatch)
        yield seal
    if breaches := seal.breaches():
        pytest.fail(f"the test reached another machine: {'; '.join(breaches)}", pytrace=False)


@pytest.fixture(autouse=True)
def windows_like_when_asked(request: pytest.FixtureRequest) -> None:
    """Put `windows_like` under a marked test, or under every test for `--windows-like`.

    Autouse, so the simulation is in place before any fixture the test asks for is built, the
    way everything a Windows runner does happens on Windows.
    """
    if request.config.getoption("--windows-like") or request.node.get_closest_marker(_MARKER):
        request.getfixturevalue(_MARKER)


@pytest.fixture(scope="session")
def windows_farms(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where the pruned PATHs are built, once per distinct PATH for the whole session."""
    return tmp_path_factory.mktemp("windows-path")


@pytest.fixture
def windows_like(windows_farms: Path) -> Iterator[None]:
    """This machine answering the test as a bare Windows box would, for the test's duration."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        WindowsLike(windows_farms).install(monkeypatch)
        yield


@pytest.fixture(params=[PurePosixPath, PureWindowsPath], ids=["posix", "windows"])
def path_flavor(request: pytest.FixtureRequest) -> type[PurePath]:
    """Each path grammar a center or a target speaks, one test run per grammar."""
    return request.param


@pytest.fixture(params=["\n", "\r\n"], ids=["lf", "crlf"])
def newline(request: pytest.FixtureRequest) -> str:
    """Each line ending a checkout can hand a reader, `core.autocrlf` Windows' included."""
    return request.param
