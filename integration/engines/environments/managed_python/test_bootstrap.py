import shlex
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from mainboard import Board, MissionError, Project
from mainboard.dispatch import Dialect, Onboarding
from mainboard.dispatch.hosts.landing import Landing
from mainboard.dispatch.hosts.onboard import Bootstrap
from mainboard.dispatch.hosts.rentals import Rental
from mainboard.dispatch.transport.shells import HostShell
from mainboard.engines.compile import Lockfile

from .builders import stub
from .constants import PLAN, PYTHON, PYTHON_INSTALL, ROOT, SYSTEM
from .fakes import Recorded

_MKDIR = f"mkdir -p {ROOT}"
_HAS_UV = "command -v uv"
_UPDATE = "uv self update --quiet"
_CURL, _INSTALLER = Dialect.uv_bootstrap
_UV = """echo "$@" >> "$HOME/asked"
case "$1 $2" in
  "self update") ;;
  "python install")
    case "$5" in '>=3.14.0,<3.15') wanted=3.14 ;; *) wanted=3.15 ;; esac
    if [ ! -f "$HOME/installed-python" ]; then
      echo "$wanted" > "$HOME/installed-python"
      echo downloaded >> "$HOME/downloads"
    fi
    ;;
  "tool install")
    while [ "$1" != --python ]; do shift; done
    [ "$2" = '>=3.14.0,<3.15' ] || { echo 'error: no cp315 wheel' >&2; exit 1; }
    ;;
  *) exit 2 ;;
esac"""
_KEPT_UV = """case "$1 $2" in
  "tool dir") echo "$HOME/tools" ;;
  "tool install") echo "$@" > "$HOME/reinstalled" ;;
  *) exit 1 ;;
esac"""
_CONTEXT = "resolver context " * 30
_DIAGNOSTIC = f"""error: No solution found when resolving dependencies
  cause: cuda-bindings has no wheel for cp315
    {_CONTEXT}
  hint: use the interpreter solved by the workspace
"""
_FAILING_UV = f"""echo "$@" >> "$HOME/asked"
[ "$1 $2" = "self update" ] && exit 0
printf %s {shlex.quote(_DIAGNOSTIC)} >&2
exit 1"""


@pytest.mark.parametrize(
    ("absent", "commands", "refusal"),
    [
        pytest.param(
            (_HAS_UV,),
            [_MKDIR, _HAS_UV, _CURL, _INSTALLER, PYTHON_INSTALL],
            None,
            id="a-host-without-uv-fetches-it",
        ),
        pytest.param(
            (_UPDATE,),
            [_MKDIR, _HAS_UV, _UPDATE, _CURL, _INSTALLER, PYTHON_INSTALL],
            None,
            id="a-uv-too-old-to-know-a-final-release-is-replaced",
        ),
        pytest.param(
            (),
            [_MKDIR, _HAS_UV, _UPDATE, PYTHON_INSTALL],
            None,
            id="a-uv-that-updates-itself-is-kept",
        ),
        pytest.param(
            (_HAS_UV, _CURL),
            [_MKDIR, _HAS_UV, _CURL],
            "neither uv nor curl",
            id="a-host-with-neither-is-refused-before-it-runs-anything",
        ),
    ],
)
def test_onboarding_puts_uv_and_its_cpython_there_first_or_refuses(
    locked: Path, absent: tuple[str, ...], commands: list[str], refusal: str | None
) -> None:
    shell = Recorded(*absent)

    with pytest.raises(MissionError, match=refusal) if refusal else nullcontext():
        Bootstrap(shell, workspace=locked).python()

    assert shell.commands == commands


@pytest.mark.parametrize(
    "vendored", [pytest.param(True, id="vendored"), pytest.param(False, id="indexed")]
)
def test_the_tool_installs_through_uv_alone(locked: Path, vendored: bool) -> None:
    routes = Bootstrap(Recorded(), workspace=locked, floor="0.5.4").routes(vendored=vendored)
    commands = [routes.select(name).command for name in routes.names]
    installs = [command for command in commands if command != Dialect.noop]

    assert installs
    assert all("--managed-python" in command for command in installs)
    assert all(command.startswith("uv tool install") for command in installs)
    assert not any(SYSTEM.search(command) for command in commands)


def _host_with_uv(home: Path, login: HostShell, *, preinstalled: str | None, source: str) -> None:
    """A host whose uv is a fake, with a CPython installed already or not.

    preinstalled: the minor uv already has, None for a host that has to download one.
    source: `editable` when the mirror carries the tool's source, `index` when it does not.
    """
    stub(home / ".local/bin/uv", _UV)
    if preinstalled:
        (home / "installed-python").write_text(f"{preinstalled}\n", encoding="utf-8")
    if source == "editable":
        (Path(login.root) / "packages/mainboard").mkdir(parents=True)


@pytest.mark.parametrize(
    "preinstalled", [pytest.param(None, id="fresh"), pytest.param("3.14", id="reuse")]
)
@pytest.mark.parametrize(
    ("source", "winner"),
    [pytest.param("index", "uv-index", id="index"), pytest.param("editable", "uv", id="editable")],
)
def test_python_and_the_tool_install_on_the_solved_minor(
    locked: Path, home: Path, login: HostShell, preinstalled: str | None, source: str, winner: str
) -> None:
    """A fake uv would pick 3.15 from an open requirement, or reuse an installed 3.14."""
    _host_with_uv(home, login, preinstalled=preinstalled, source=source)
    bootstrap = Bootstrap(login, workspace=locked)
    bootstrap.python()

    assert bootstrap.tool().winner == winner
    asked = (home / "asked").read_text(encoding="utf-8").splitlines()
    assert asked[:2] == ["self update --quiet", f"python install --upgrade --no-bin {PYTHON}"]
    assert (f"--python {PYTHON}" in asked[2], "--editable" in asked[2]) == (
        True,
        source == "editable",
    )
    assert "--force --reinstall --managed-python" in asked[2]
    assert (home / "installed-python").read_text(encoding="utf-8") == "3.14\n"
    assert (home / "downloads").exists() is (preinstalled is None)


def _solved_python(workspace: Path, *, later: int) -> None:
    """Solve the default environment's Python as the running interpreter's minor, `later` on."""
    lock = Lockfile(workspace)
    solved = lock.environments()["default"]
    python = f"python-{sys.version_info.major}.{sys.version_info.minor + later}.8"
    text = solved.lock.replace("python-3.14.8", python)
    lock.path.write_text(
        lock.render({"default": solved.model_copy(update={"lock": text})}), encoding="utf-8"
    )


def _runs_a_tool(home: Path, *, version: str) -> None:
    """A host whose `mainboard` reports `version` and runs on this very interpreter."""
    stub(home / ".local/bin/mainboard", f"echo mainboard {version}")
    stub(home / ".local/bin/uv", _KEPT_UV)
    (home / "tools/mainboard/bin").mkdir(parents=True)
    (home / "tools/mainboard/bin/python").symlink_to(sys.executable)


@pytest.mark.parametrize(
    ("floor", "version", "later", "winner"),
    [
        pytest.param(">=0.5", "0.5.6", 0, "present", id="a-tool-meeting-the-floor-is-kept"),
        pytest.param("0.5.4", "0.5.4", 0, "present", id="a-bare-floor-names-that-release"),
        pytest.param("0.5.4", "0.5.6", 0, "uv-index", id="a-bare-floor-names-that-release-alone"),
        pytest.param("*", "0.5.6", 0, "present", id="no-declared-floor-keeps-any-tool"),
        pytest.param(">=0.6", "0.5.6", 0, "uv-index", id="a-tool-below-the-floor-is-replaced"),
        pytest.param(">=0.5", "dev", 0, "uv-index", id="an-unreadable-version-is-replaced"),
        pytest.param("0.5.4", "0.5.4", 1, "uv-index", id="a-tool-on-another-minor-is-replaced"),
    ],
)
def test_a_tool_the_host_runs_is_kept_only_when_it_meets_the_floor_on_the_solved_minor(
    locked: Path, home: Path, login: HostShell, floor: str, version: str, later: int, winner: str
) -> None:
    _solved_python(locked, later=later)
    Path(login.root).mkdir()
    _runs_a_tool(home, version=version)

    result = Bootstrap(login, workspace=locked, floor=floor).tool()

    assert result.winner == winner
    assert (home / "reinstalled").exists() is (winner != "present")


@pytest.mark.parametrize(
    ("step", "verbs"),
    [
        pytest.param("python", ["self update", "python install"], id="python"),
        pytest.param("tool", ["tool install"], id="tool"),
    ],
)
def test_a_failed_uv_preserves_its_complete_diagnostic_and_nothing_runs_after_it(
    locked: Path, home: Path, login: HostShell, step: str, verbs: list[str]
) -> None:
    stub(home / ".local/bin/uv", _FAILING_UV)
    Path(login.root).mkdir()
    bootstrap = Bootstrap(login, workspace=locked)

    with pytest.raises(MissionError) as refused:
        getattr(bootstrap, step)()

    assert _DIAGNOSTIC.strip() in str(refused.value) and "failed on 'fakehost'" in str(
        refused.value
    )
    asked = (home / "asked").read_text(encoding="utf-8").splitlines()
    assert [" ".join(line.split()[:2]) for line in asked] == verbs
    assert not (home / "system").exists()


def test_setup_installs_python_and_the_tool_on_the_solved_minor_held_to_the_locks_releases(
    onboarding: tuple[Onboarding, Recorded],
) -> None:
    onboarder, shell = onboarding

    onboarder.run()

    [(constraints, pinned)] = shell.written.items()
    assert pinned == "duckdb==2.0.0\npytz==2026.1\n" and shell.installed(PYTHON) == (True, True)
    assert any(f"--constraints {constraints}" in command for command in shell.commands)
    assert "mainboard install default --locked --profile fakehost" in shell.commands
    assert "mainboard lake upgrade" in shell.commands


def _synced(board: Board, onboarder: Onboarding, shell: Recorded) -> str:
    """Sync the host again with a clean record and no lake catalog on it, answering its address."""
    catalog = f"{ROOT}/{Project().out_dir(board.root)}/lake.sqlite"
    shell.reset(Dialect().is_file(catalog))
    return onboarder.run(sync_only=True).requires


@pytest.mark.parametrize(
    ("old", "new", "python", "pinned"),
    [
        pytest.param(
            "duckdb-2.0.0", "duckdb-2.0.1", PYTHON, "duckdb==2.0.1\npytz==2026.1\n", id="a-release"
        ),
        pytest.param(
            "3.14.8",
            "3.15.0",
            ">=3.15.0,<3.16",
            "duckdb==2.0.0\npytz==2026.1\n",
            id="a-python-minor",
        ),
    ],
)
def test_a_sync_reinstalls_the_tool_only_when_the_lock_moved_what_it_runs_on(
    board: Board,
    onboarding: tuple[Onboarding, Recorded],
    old: str,
    new: str,
    python: str,
    pinned: str,
) -> None:
    onboarder, shell = onboarding
    setup = onboarder.run().requires
    assert _synced(board, onboarder, shell) == setup
    assert shell.installed(PYTHON) == (True, False)

    lock = Lockfile(board.root)
    lock.path.write_text(lock.path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    assert _synced(board, onboarder, shell) != setup
    assert shell.installed(python) == (True, True) and list(shell.written.values()) == [pinned]
    assert "mainboard lake upgrade" not in shell.commands


def test_a_rental_uses_its_probed_platform_pin(rented: tuple[Landing, Rental, Recorded]) -> None:
    job, rental, shell = rented

    with pytest.raises(MissionError, match="environment reached"):
        job.equip(rental)

    assert job.plan.profile.platform == "linux-aarch64"
    assert shell.installed(">=3.15.0,<3.16") == (True, True)
    assert list(shell.written.values()) == [""]


def test_the_tool_runs_on_the_default_environments_python_and_a_host_without_one_is_refused(
    locked: Path,
) -> None:
    shell = Recorded()
    shell.plan = PLAN.model_copy(update={"env": "training"})

    assert Bootstrap(shell, workspace=locked).python_request == PYTHON

    Lockfile(locked).path.unlink()
    with pytest.raises(MissionError, match="lock default"):
        Bootstrap(shell, workspace=locked)
    assert shell.commands == []
