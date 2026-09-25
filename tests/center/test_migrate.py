import json
import platform
import subprocess
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import TracebackType

import pytest
from patos import Resolution

from mainboard import Board
from mainboard.center import migrate
from mainboard.center.migrate import Migration, github_token
from mainboard.center.state import claude_key
from mainboard.core.errors import MissionError
from mainboard.core.host import current_platform, pixi_platform
from mainboard.core.section import Section, Verdict
from mainboard.dispatch.onboard import Bootstrap, Onboarding
from mainboard.dispatch.shells import Posix
from mainboard.dispatch.targets import Facts
from mainboard.probe.system import System

from .conftest import LocalTransport, Workspace

# The destination's alias, a machine no manifest declares yet.
_DESTINATION = "pedro-home"

# A census of a machine that runs this platform, with every tool a center needs, so the only
# findings are the ones a test sets up.
_FIT = System(
    system=platform.system(),
    arch=platform.machine(),
    shells={"bash": "/bin/bash"},
    tools={"git": "2.51.0", "git-lfs": "3.7.0", "gh": "2.80.0", "ssh": "9.9"},
    free_bytes=10**12,
    root="/dest",
)

# What the destination's own `center verify` prints, chatter above the JSON it answers with.
_VERIFIED = "Loading...\n" + json.dumps(
    [{"section": "smoke", "verdict": "pass", "detail": "torch 2.14 on CPU", "fix": ""}]
)


class Shell:
    """A destination shell that runs nothing: it keeps each command and answers fixed words.

    verified: what `center verify` prints there.
    """

    dialect = Posix()

    def __init__(self, verified: str = _VERIFIED) -> None:
        self.verified = verified
        self.ran: list[str] = []

    def __enter__(self) -> Shell:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        return None

    def run(self, command: str, *, activate: bool = False) -> str:
        self.ran.append(command)
        return "installed default\n"

    def stage(self, command: str, *, activate: bool) -> str:
        return command

    def execute(self, line: str) -> tuple[int, str, str]:
        self.ran.append(line)
        return 1, self.verified, ""


class Moving:
    """Everything one migration test arranges: the tree, both homes, the far side, the shell.

    tree: the git fixture checkout, this center's workspace.
    home: this center's home.
    destination: where the workspace goes on the far side.
    far: the far side's home.
    shell: the destination shell the install and verify steps go through.
    probes: what each capabilities probe answers, in order.
    """

    def __init__(self, tree: Workspace, home: Path, tmp_path: Path) -> None:
        self.tree = tree
        self.home = home
        self.destination = tmp_path / "dest"
        self.far = tmp_path / "far"
        self.far.mkdir()
        self.shell = Shell()
        self.probes: list[Facts] = [self.facts(uv="uv")]
        self.canned = {"census": _FIT.model_dump_json(), "login": '{"signed": true}'}

    def facts(self, *, uv: str) -> Facts:
        """The destination as the stock probe finds it."""
        return Facts(name=_DESTINATION, home=str(self.far), uv=uv, platform="Linux x86_64")

    def migration(self, *, token: str = "gho_secret") -> Migration:
        """A migration of this tree to the far side, with this machine's GitHub token."""
        transport = LocalTransport(home=self.far, canned=self.canned)
        return Migration(
            Board(self.tree.path),
            _DESTINATION,
            root=str(self.destination),
            transport=transport,
            watch=lambda said: None,
            home=self.home,
            token=lambda: token,
        )


@pytest.fixture
def moving(tree: Workspace, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Moving:
    """A center with state worth carrying, and a destination reached through local stand-ins."""
    arranged = Moving(tree, home, tmp_path)
    Board(tree.path).dispatcher.cache.connection.close()
    (tree.path / ".env").write_text("EXA_API_KEY=secret\n", encoding="utf-8")
    ledger = tree.path / ".mainboard" / "costs" / "costs.ndjson"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"usd": 1}\n', encoding="utf-8")
    memory = home / ".claude" / "projects" / claude_key(str(tree.path)) / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("- a note\n", encoding="utf-8")
    trusted = {"projects": {str(tree.path): {"hasTrustDialogAccepted": True, "lastCost": 3}}}
    (home / ".claude.json").write_text(json.dumps(trusted), encoding="utf-8")

    def probe(host: str, *, ssh: LocalTransport) -> Facts:
        return arranged.probes.pop(0) if len(arranged.probes) > 1 else arranged.probes[0]

    monkeypatch.setattr(migrate, "probe_capabilities", probe)
    monkeypatch.setattr(migrate, "open_shell", lambda plan, root, ssh: arranged.shell)
    monkeypatch.setattr(
        Bootstrap, "tool", lambda self: Resolution(winner="uv", implementation=None, rejected=())
    )
    monkeypatch.setattr(Onboarding, "align_pixi", lambda self, shell, host: "0.79.0")
    return arranged


def _sections(report: Sequence[Section]) -> dict[str, Section]:
    """The report keyed by section, the last row of a name winning."""
    return {row.section: row for row in report}


def test_a_migration_clones_carries_installs_and_verifies_then_changes_nothing_again(
    moving: Moving,
) -> None:
    """The whole move lands, and running it again converges instead of repeating it.

    The clone sits at this HEAD with the owned submodule at its pointer and the foreign one left
    to fetch, the state git does not hold is on the far side with Claude Code's memory re-keyed
    to the new path, the tool and environment were installed, and the destination's own verify
    is the report's last word before what stays behind.
    """
    first = _sections(moving.migration().run())
    assert first["github"].verdict is Verdict.PASS
    assert first["clone ."].verdict is Verdict.PASS
    assert first["clone packages/lib"].verdict is Verdict.PASS
    assert "foreign" in first["clone references"].detail
    assert first["verify: smoke"].detail == "torch 2.14 on CPU"
    assert first["install tool"].detail == "uv"
    assert first["install pixi"].detail == "0.79.0"
    assert "left: pins" in first
    head = moving.tree.head(moving.tree.path)
    assert moving.tree.head(moving.destination) == head
    assert (moving.destination / "packages" / "lib" / "lib.txt").is_file()
    assert not (moving.destination / "references" / "ref" / "ref.txt").exists()
    assert (moving.destination / ".env").read_text(encoding="utf-8") == "EXA_API_KEY=secret\n"
    assert (moving.destination / ".mainboard" / "costs" / "costs.ndjson").is_file()
    rekeyed = moving.far / ".claude" / "projects" / claude_key(str(moving.destination))
    assert (rekeyed / "memory" / "MEMORY.md").is_file()
    projects = json.loads((moving.far / ".claude.json").read_text(encoding="utf-8"))["projects"]
    assert projects == {str(moving.destination): {"hasTrustDialogAccepted": True}}
    assert "mainboard install" in moving.shell.ran

    again = _sections(moving.migration().run())
    assert "0 written" in again["carry workspace"].detail
    assert "0 written" in again["carry agents"].detail
    assert "0 changed" in again["carry claude project"].detail
    assert again["clone ."].verdict is Verdict.PASS


def test_a_head_no_remote_holds_stops_the_move_before_the_destination_is_touched(
    moving: Moving,
) -> None:
    """A clone cannot fetch a commit no remote branch reaches, so nothing is attempted."""
    moving.tree.forge.commit(moving.tree.path, "local only", {"local.txt": "x\n"})
    report = moving.migration().run()
    assert [row.section for row in report] == ["publish ."]
    assert report[0].verdict is Verdict.FAIL
    assert not moving.destination.exists()


def test_a_platform_the_workspace_cannot_serve_stops_the_move_after_the_census(
    moving: Moving,
) -> None:
    """The install would only rediscover it after the clone and the copy spent their time.

    The workspace declares no platforms, so it serves this machine's alone, and the destination
    is whichever of two others this machine is not.
    """
    system, arch = next(
        pair
        for pair in [("Linux", "aarch64"), ("Darwin", "x86_64")]
        if pixi_platform(*pair) != current_platform()
    )
    alien = _FIT.model_copy(update={"system": system, "arch": arch})
    moving.canned["census"] = alien.model_dump_json()
    report = _sections(moving.migration().run())
    assert report["destination: platform"].verdict is Verdict.FAIL
    assert "github" not in report
    assert not moving.destination.exists()


@pytest.mark.parametrize(
    ("token", "login", "verdict"),
    [
        ("", '{"signed": true}', Verdict.WARN),
        ("gho_secret", '{"signed": false, "detail": "gh could not run"}', Verdict.FAIL),
    ],
    ids=["no login here to carry", "the destination's gh refused it"],
)
def test_the_github_row_says_what_became_of_the_login(
    moving: Moving, token: str, login: str, verdict: Verdict
) -> None:
    """A login that did not land is named with its fix, and the move goes on to report the rest."""
    moving.canned["login"] = login
    report = _sections(moving.migration(token=token).run())
    assert report["github"].verdict is verdict
    assert report["github"].fix


def test_a_destination_without_uv_gets_it_before_anything_else(moving: Moving) -> None:
    """uv is what every later call runs its Python through, so it lands first."""
    moving.probes = [moving.facts(uv=""), moving.facts(uv="uv")]
    moving.migration().run()
    assert moving.shell.ran[0] == Posix().uv_bootstrap[1]


def test_a_uv_installer_that_puts_nothing_there_is_refused_by_name(moving: Moving) -> None:
    """A second probe still without uv is the destination refusing, not a reason to go on."""
    moving.probes = [moving.facts(uv=""), moving.facts(uv=""), moving.facts(uv="")]
    with pytest.raises(MissionError, match="still has no uv"):
        moving.migration().run()


def test_a_destination_root_holding_something_else_is_held_and_nothing_is_installed(
    moving: Moving,
) -> None:
    """Nothing here overwrites a directory it did not make; the row names the way out."""
    moving.destination.mkdir()
    (moving.destination / "mine.txt").write_text("keep\n", encoding="utf-8")
    report = _sections(moving.migration().run())
    assert report["clone ."].verdict is Verdict.FAIL
    assert "--root" in report["clone ."].fix
    assert "install tool" not in report
    assert (moving.destination / "mine.txt").read_text(encoding="utf-8") == "keep\n"


def test_an_install_step_that_fails_is_the_last_one_tried(
    moving: Moving, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment cannot install on a pixi that is not the fleet's."""

    def refuse(self: Onboarding, shell: Shell, host: str) -> str:
        raise MissionError("still runs pixi 0.70\nmore")

    monkeypatch.setattr(Onboarding, "align_pixi", refuse)
    report = _sections(moving.migration().run())
    assert report["install pixi"].verdict is Verdict.FAIL
    assert report["install pixi"].detail == "still runs pixi 0.70"
    assert "install default" not in report


def test_a_destination_whose_verify_says_nothing_readable_fails_that_row(moving: Moving) -> None:
    """No report is not a clean report."""
    moving.shell.verified = "mainboard: command not found"
    report = _sections(moving.migration().run())
    assert report["verify"].verdict is Verdict.FAIL
    assert "command not found" in report["verify"].detail


def test_a_center_with_nothing_optional_carries_nothing_for_it(moving: Moving) -> None:
    """No record of this workspace in `~/.claude.json` and no foreign checkout are not errors.

    Each is simply one row fewer to act on: nothing to merge, nothing to fetch on demand.
    """
    (moving.home / ".claude.json").unlink()
    moving.tree.git(moving.tree.path, "submodule", "deinit", "-q", "references/ref")
    moving.tree.git(moving.tree.lib, "submodule", "deinit", "-q", "vendor/dep")
    report = _sections(moving.migration().run())
    assert report["carry claude project"].detail == "nothing to carry"
    assert "clone references" not in report


def test_the_extras_carried_are_the_ones_this_center_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The destination gets the same tool, plotting included where it was here."""

    def installed(name: str) -> None:
        if name == "wandb":
            raise PackageNotFoundError(name)

    monkeypatch.setattr(migrate, "distribution", installed)
    assert Migration.extras() == ["plot"]


@pytest.mark.parametrize(
    ("status", "output", "token"),
    [(0, "gho_abc\n", "gho_abc"), (1, "not logged in", ""), (None, "", "")],
    ids=["signed in", "signed out", "no gh here"],
)
def test_the_github_token_is_whatever_gh_holds_and_nothing_otherwise(
    monkeypatch: pytest.MonkeyPatch, status: int | None, output: str, token: str
) -> None:
    """An absent or signed-out gh (`None`: it cannot start) is a missing login, never a raise."""

    def run(argv: Sequence[str], **options: bool | int) -> subprocess.CompletedProcess[str]:
        if status is None:
            raise FileNotFoundError(argv[0])
        return subprocess.CompletedProcess(argv, status, output, "")

    monkeypatch.setattr(migrate.subprocess, "run", run)
    assert github_token() == token
