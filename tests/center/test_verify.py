import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from mainboard import Board, Project
from mainboard.center.verify import Verification, spawned
from mainboard.core.errors import MissionError
from mainboard.core.section import Section, Verdict
from mainboard.engines.compile.backend.result import CommandResult
from mainboard.engines.compile.provisioner import Provisioner
from mainboard.probe.system import Card, System
from mainboard.workstation import Readiness, Workstation

from ..git.conftest import Workspace

# What a healthy default environment's smoke run prints, CUDA or not.
_CPU = {"torch": "2.9.0", "cuda": "", "available": False, "device": ""}
_CUDA = {"torch": "2.9.0", "cuda": "13.0", "available": True, "device": "NVIDIA RTX 4090"}

# A card the driver lists, which makes a smoke run that sees no CUDA a failure.
_CARD = Card(name="NVIDIA GeForce RTX 4090", driver="580.1", capability="8.9", vram_mb=24564)

# Lint tools whose programs this test machine never has on its own PATH.
_LINT = """
[lint]
exclude = ["vendor/"]

[lint.tools.fmt]
check = "fmtx-mainboard-test --check {files}"
fix = "fixx-mainboard-test {files}"
files = ["*.py"]

[lint.tools.types]
check = "typesx-mainboard-test"
files = ["*.py"]
"""


class Examined(Workstation):
    """A workstation whose checks already ran, answering exactly `found`.

    found: the rows its examination reports.
    """

    def __init__(self, root: Path, found: Sequence[Readiness] = ()) -> None:
        super().__init__(root, system="Linux", linking=lambda: "")
        self.found = list(found)

    def examine(self) -> list[Readiness]:
        return self.found


class Doctored:
    """The workspace doctor, reduced to the one row it reports."""

    def sections(self) -> list[Section]:
        return [Section(section="doctor", verdict=Verdict.PASS, detail="healthy")]


class Shells:
    """Shells that each report every name they are asked about at `where`, and record calls.

    where: the path every name resolves to, by shell path.
    """

    def __init__(self, where: Mapping[str, str]) -> None:
        self.where = dict(where)
        self.ran: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str], environment: Mapping[str, str]) -> tuple[int, str]:
        self.ran.append(tuple(command))
        folder = self.where[command[0]]
        names = command[-1].split("for t in ", 1)[1].split(";", 1)[0].split()
        return 0, "".join(f"{name}\tApplication\t{folder}/{name}\n" for name in names)


def verification(
    board: Board,
    home: Path,
    *,
    system: System | None = None,
    found: Sequence[Readiness] = (),
    smoke: tuple[int, str] = (0, json.dumps(_CPU)),
    spawn: Shells | None = None,
) -> Verification:
    """The readiness suite over `board`, every machine seam answered by a stand-in."""
    return Verification(
        board,
        workstation=Examined(board.root, found),
        system=system or System(system="Linux", arch="x86_64"),
        home=home,
        spawn=spawn or Shells({}),
        smoke=lambda: smoke,
    )


def prefix(board: Board) -> Path:
    """The default environment's prefix in `board`, where an install would put it."""
    return Provisioner(board.root, board.manifest).pixi_for("default").env_prefix("default")


def installed(board: Board, packages: Mapping[str, Sequence[str]]) -> Path:
    """A default environment installed with each package's files, its prefix answered."""
    root = prefix(board)
    (root / "conda-meta").mkdir(parents=True)
    for name, files in packages.items():
        record = {"name": name, "files": list(files)}
        (root / "conda-meta" / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
        for file in files:
            (root / file).parent.mkdir(parents=True, exist_ok=True)
            (root / file).touch(mode=0o755)
    return root


@pytest.fixture
def station(workspace: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> Board:
    """The fixture workspace as a board whose doctor reports one healthy row."""
    monkeypatch.setattr(Board, "doctor", lambda self, env="": Doctored())
    return Board(workspace)


def test_every_readiness_question_is_asked_in_order_and_each_tooling_row_keeps_its_weight(
    station: Board, home: Path
) -> None:
    """Tooling comes first since every repair after it is a git command, and nothing is dropped.

    A broken tool fails, a fix still owed warns, and a settled check passes, so the report's
    exit status follows the worst of them. Machine findings are marked as such, the doctor's
    rows follow unchanged, and a workspace with no environment installed is one path warning.
    """
    found = [
        Readiness(check="git", detail="git 2.51"),
        Readiness(check="git-lfs", detail="absent", fix="brew install git-lfs"),
        Readiness(check="credentials", broken=True, detail="none", fix="gh auth login"),
    ]

    rows = verification(station, home, found=found).sections()

    names = [row.section for row in rows]
    assert [(row.section, row.verdict) for row in rows[:3]] == [
        ("workstation: git", Verdict.PASS),
        ("workstation: git-lfs", Verdict.WARN),
        ("workstation: credentials", Verdict.FAIL),
    ]
    doctor = names.index("doctor")
    assert doctor > 3
    assert all(name.startswith("machine: ") for name in names[3:doctor])
    assert names[doctor:] == [
        "doctor",
        "check",
        "smoke",
        "lint",
        "git: status",
        "agents: links",
        "agents: instructions",
        "agents: .mcp.json",
        "agents: opencode.json",
        "agents: .codex/config.toml",
        "agents: memory",
        "agents: logins",
        "path",
        "portable",
    ]
    path = rows[names.index("path")]
    assert (path.verdict, path.fix) == (Verdict.WARN, "mainboard install")
    assert rows[names.index("check")].detail == "local runs default bare, 1 tasks declared"


def test_a_plan_the_manifest_cannot_resolve_here_is_a_failure_naming_why(
    station: Board, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan is what every local run executes under, so a refusal to resolve one fails."""

    def refused(self: Board, *, env: str = "", container: str = "") -> None:
        raise MissionError("environment 'default' is not declared for osx-arm64")

    monkeypatch.setattr(Board, "plan", refused)

    row = verification(station, home).plan()

    assert (row.verdict, row.detail, row.fix) == (
        Verdict.FAIL,
        "environment 'default' is not declared for osx-arm64",
        "mainboard check",
    )


@pytest.mark.parametrize(
    ("status", "said", "gpus", "verdict", "detail"),
    [
        (
            1,
            "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'\n\n",
            (),
            Verdict.FAIL,
            "python -c 'import torch' failed: ModuleNotFoundError: No module named 'torch'",
        ),
        (137, "", (), Verdict.FAIL, "python -c 'import torch' failed: 137"),
        (0, "warming up\n", (), Verdict.FAIL, "python -c 'import torch' failed: warming up"),
        (
            0,
            json.dumps(_CPU),
            (_CARD,),
            Verdict.FAIL,
            "torch 2.9.0 sees no CUDA device, though the driver lists NVIDIA GeForce RTX 4090",
        ),
        (
            0,
            f"{{ banner\nloading\n{json.dumps(_CUDA)}\nbye\n",
            (_CARD,),
            Verdict.PASS,
            "torch 2.9.0 on NVIDIA RTX 4090 (CUDA 13.0)",
        ),
        (0, json.dumps(_CPU), (), Verdict.PASS, "torch 2.9.0 on CPU"),
    ],
    ids=["no-torch", "killed", "no-json", "gpu-unseen", "gpu", "cpu"],
)
def test_the_smoke_run_passes_only_when_torch_sees_every_card_the_driver_lists(
    station: Board,
    home: Path,
    status: int,
    said: str,
    gpus: tuple[Card, ...],
    verdict: Verdict,
    detail: str,
) -> None:
    """The last JSON line is the answer; a crash is named by its last words or its status.

    A machine with a card whose torch cannot see it is a CPU-only build on a GPU box, which
    every training run would silently pay for, so that is a failure rather than a pass.
    """
    system = System(system="Linux", gpus=gpus)

    row = verification(station, home, system=system, smoke=(status, said)).smoke()

    assert (row.verdict, row.detail) == (verdict, detail)


def test_every_lint_tool_must_start_from_the_environment_or_this_machine(
    workspace: Path, home: Path
) -> None:
    """A lint tool is found in the default environment first, so installing it there is enough.

    Both the read-only check and the fix a writer declares must start, since `lint` runs one
    and `lint --check` the other.
    """
    manifest = workspace / Project().manifest
    manifest.write_text(manifest.read_text(encoding="utf-8") + _LINT, encoding="utf-8")
    board = Board(workspace)
    suffix = ".exe" if sys.platform == "win32" else ""
    checks = verification(board, home)

    missing = checks.lint()
    installed(
        board,
        {"fmt": [f"bin/fmtx-mainboard-test{suffix}", f"bin/fixx-mainboard-test{suffix}"]},
    )
    one = checks.lint()
    installed_types = prefix(board) / "bin" / f"typesx-mainboard-test{suffix}"
    installed_types.touch(mode=0o755)
    both = checks.lint()

    assert (missing.verdict, missing.detail) == (
        Verdict.FAIL,
        "lint runs fixx-mainboard-test, fmtx-mainboard-test, typesx-mainboard-test, "
        "which the environment does not provide",
    )
    assert one.detail == (
        "lint runs typesx-mainboard-test, which the environment does not provide"
    )
    assert (both.verdict, both.detail) == (Verdict.PASS, "2 lint tools can start")


def test_the_tree_rows_name_absent_behind_and_unsaved_owned_repositories(
    tree: Workspace, home: Path
) -> None:
    """A fresh clone is one passing row; each way a tree falls out of step adds its own.

    An owned submodule never checked out and a repository behind its upstream both warn with
    the pull that fixes them, while unsaved work is only noted, since it is the work itself.
    """
    board = Board(tree.path)
    checks = verification(board, home)

    fresh = checks.tree()
    tree.git(tree.path, "submodule", "deinit", "-q", "-f", "packages/lib")
    seed = tree.forge.seed("Pedrexus", "projects")
    tree.forge.commit(seed, "upstream moved", {"news.txt": "news\n"})
    tree.git(seed, "push", "-q", "origin", "main")
    tree.git(tree.path, "fetch", "-q", "origin")
    (tree.path / "scratch.txt").write_text("draft\n", encoding="utf-8")
    moved = checks.tree()

    assert [(row.section, row.verdict, row.detail) for row in fresh] == [
        ("git: status", Verdict.PASS, "2 owned repositories, all saved")
    ]
    assert [(row.section, row.verdict, row.detail) for row in moved] == [
        ("git: checkout", Verdict.WARN, "owned submodules not checked out: packages/lib"),
        ("git: behind", Verdict.WARN, "behind upstream: ."),
        ("git: status", Verdict.PASS, "1 owned repositories; unsaved work in ."),
    ]


def test_an_installed_environment_is_put_on_every_shell_s_path_and_proved_there(
    station: Board, home: Path
) -> None:
    """Only commands of declared packages are asked about, from every shell the census found."""
    root = installed(station, {"pueue": ["bin/pueue"], "undeclared": ["bin/stray"]})
    shells = Shells({"/bin/zsh": str(root / "bin"), "/bin/bash": str(root / "bin")})
    system = System(system="Linux", shells={"zsh": "/bin/zsh", "bash": "/bin/bash"})

    rows = verification(station, home, system=system, spawn=shells).path()

    assert [(row.section, row.verdict) for row in rows] == [
        ("path", Verdict.PASS),
        ("path bash", Verdict.PASS),
        ("path zsh", Verdict.PASS),
    ]
    assert rows[1].detail == "1 of 1 resolve into the environment"
    assert all("pueue" in command[-1] and "stray" not in command[-1] for command in shells.ran)
    assert (home / ".zshenv").is_file()


def test_left_alone_the_suite_reads_this_machine_and_smokes_the_real_environment(
    station: Board, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every seam defaults to the real one: the census, git tooling, home, processes and pixi.

    The smoke run goes through the provisioner, bounded, and reads stdout and stderr together.
    """
    captured: list[tuple[list[str], str, float | None]] = []

    def capture(
        self: Provisioner, command: list[str], env: str = "default", *, timeout: float | None
    ) -> CommandResult:
        captured.append((command, env, timeout))
        return CommandResult(0, f"{json.dumps(_CPU)}\n", "UserWarning: no NUMA\n")

    monkeypatch.setattr("mainboard.center.verify.System.collected", lambda root: System())
    monkeypatch.setattr(Provisioner, "capture", capture)

    checks = Verification(station)

    assert isinstance(checks.workstation, Workstation)
    assert checks.workstation.root == station.root
    assert (checks.home, checks.spawn) == (home, spawned)
    assert checks.smoke().detail == "torch 2.9.0 on CPU"
    ((command, env, timeout),) = captured
    assert (command[:2], env, timeout) == (["python", "-c"], "default", 600.0)
    assert "torch.cuda.is_available()" in command[2]


def test_a_spawned_command_answers_its_status_and_every_word_it_said() -> None:
    """The process seam runs under exactly the environment it is handed, output joined.

    A program that cannot start is the shell's own 127, never an exception in a report.
    """
    script = "import os, sys; print(os.environ['PROBE']); sys.stderr.write('err\\n'); sys.exit(3)"

    assert spawned([sys.executable, "-c", script], {**os.environ, "PROBE": "seen"}) == (
        3,
        "seen\nerr\n",
    )
    assert spawned(["/no/such/program-anywhere"], os.environ) == (127, "")
