import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from mainboard import Project
from mainboard.core.section import Section, Verdict
from mainboard.engines.compile.provisioner import Provisioner
from mainboard.fitness import Fitness, Role
from mainboard.manifest.loading import load
from mainboard.probe.system import Card, System
from mainboard.workstation import install_command

# A workspace declaring three platforms, a CUDA floor of its own and a serving environment that
# raises it, a job host that asks for one 16 GB card, and the two scheduler kinds whose ssh
# endpoint is a login node.
_MANIFEST = """
[workspace]
name = "w"
{platforms}

[system]
cuda = "12.0"

[envs.serving]
system = {{ cuda = "13.0" }}

[hosts.gold]
kind = "ssh"
env = "serving"

[hosts.gold.defaults]
gpus = 1
vram-gb = 16

[hosts.miyabi]
kind = "pbs"

[hosts.cluster]
kind = "slurm"
"""
_DECLARED = 'platforms = ["linux-64", "win-64", "osx-arm64"]'

# Locked builds as pixi's lock spells their locations, CUDA-built and not.
_PLAIN = "conda: https://conda.anaconda.org/conda-forge/linux-64/bzip2-1.0.8-hda65f42_10.conda"
_CU128 = "pypi: https://download.pytorch.org/whl/torch-2.9.0+cu128-cp314-linux_x86_64.whl"
_CU124 = "pypi: https://download.pytorch.org/whl/torch-2.6.0+cu124-cp314-linux_x86_64.whl"
_CU118 = "pypi: https://download.pytorch.org/whl/cu118/torch-2.4.0-cp314-linux_x86_64.whl"
_CU126 = "pypi: https://download.pytorch.org/whl/torch-2.7.0+cu126-cp314-linux_x86_64.whl"
_CONDA_12_8 = (
    "conda: https://conda.anaconda.org/conda-forge/noarch/cuda-version-12.8-h5d125a7_3.conda"
)
_CONDA_12_4 = (
    "conda: https://conda.anaconda.org/conda-forge/noarch/cuda-version-12.4-h3060b56_3.conda"
)

# A machine every question passes on: declared platform, a driver above every floor, an Ada
# card with room for any job here, a roomy disk, every center tool and a case-sensitive disk.
_FIT = System(
    system="Linux",
    version="Ubuntu 24.04",
    arch="x86_64",
    shells={"bash": "/bin/bash"},
    root="/work",
    free_bytes=100 * 10**9,
    tools={
        "git": "2.51.0",
        "git-lfs": "3.7.0",
        "gh": "2.80.0",
        "ssh": "10.0",
        "rsync": "3.4.1",
        "tar": "1.35",
    },
    cuda="13.0",
    gpus=(Card(name="NVIDIA GeForce RTX 4090", capability="8.9", vram_mb=24564),),
)

# The sections each role is asked, in the order a report prints them.
_TARGET = ["platform", "lock", "driver", "cuda-builds", "memory", "disk"]
_CENTER = [*_TARGET, "tools", "sync", "links", "case", "line-endings", "shells"]

type Change = str | int | bool | dict[str, str] | tuple[Card, ...]


def machine(**changes: Change) -> System:
    """The fit machine with `changes` applied, so each test names only what it breaks."""
    return _FIT.model_copy(update=changes)


def fitness_of(root: Path, platforms: str = _DECLARED) -> Fitness:
    """A judge over a workspace at `root` holding the fixture manifest."""
    manifest = root / Project().manifest
    manifest.write_text(_MANIFEST.format(platforms=platforms), encoding="utf-8")
    return Fitness(root, load(manifest))


def lock(
    fitness: Fitness, builds: Mapping[str, Sequence[str]], environment: str = "default"
) -> Path:
    """Write `environment`'s lock holding `builds` per subdirectory, each under a named label.

    The labels are the `-system` names pixi gives a floor-carrying platform, so every read has
    to go through the roster back to the subdirectory a machine is judged by.
    """
    roster = "".join(f"- name: {subdir}-system\n  subdir: {subdir}\n" for subdir in builds)
    mapping = "".join(
        f"      {subdir}-system:\n" + "".join(f"      - {location}\n" for location in located)
        for subdir, located in builds.items()
    )
    path = Provisioner(fitness.root, fitness.manifest).pixi_for(environment).lock
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"version: 6\nplatforms:\n{roster}environments:\n  default:\n    packages:\n{mapping}"
        "packages: []\n",
        encoding="utf-8",
    )
    return path


def row(sections: list[Section], name: str) -> Section:
    """The one row named `name` in a report."""
    (found,) = [section for section in sections if section.section == name]
    return found


@pytest.fixture
def fitness(tmp_path: Path) -> Fitness:
    """A judge over the fixture workspace with a CUDA 12.8 lock for Linux in both environments."""
    judge = fitness_of(tmp_path)
    for environment in ("default", "serving"):
        lock(judge, {"linux-64": [_PLAIN, _CU128]}, environment)
    return judge


@pytest.mark.parametrize(("host", "fix"), [("local", ""), ("gold", "mainboard setup gold")])
def test_a_machine_no_census_described_is_one_row_saying_so(
    fitness: Fitness, host: str, fix: str
) -> None:
    """Every other question would only repeat that absence, so it is the whole report.

    A remote host gets the setup that records a census, and this machine, which can always be
    asked again, gets nothing to run.
    """
    assert fitness.judge(System(), host=host) == [
        Section(
            section="census",
            verdict=Verdict.WARN,
            detail="no software census recorded for this machine",
            fix=fix,
        )
    ]


@pytest.mark.parametrize("host", ["local", "gold"])
@pytest.mark.parametrize(("role", "sections"), [(Role.TARGET, _TARGET), (Role.CENTER, _CENTER)])
def test_a_fit_machine_passes_every_question_its_role_asks(
    fitness: Fitness, host: str, role: Role, sections: list[str]
) -> None:
    """A center is asked about its tooling and checkout on top of what any machine is asked.

    A report with nothing wrong also has nothing to run, so no passing row carries a fix.
    """
    found = fitness.judge(machine(), host=host, role=role)
    assert [section.section for section in found] == sections
    assert {(section.verdict, section.fix) for section in found} == {(Verdict.PASS, "")}


@pytest.mark.parametrize(
    ("platforms", "verdict", "detail", "fix"),
    [
        (
            _DECLARED,
            Verdict.FAIL,
            "linux-aarch64 is not among the declared platforms "
            "['linux-64', 'win-64', 'osx-arm64']",
            'add "linux-aarch64" to [workspace] platforms, then mainboard install --resolve',
        ),
        (
            "",
            Verdict.PASS,
            "Ubuntu 24.04 aarch64, driver CUDA 13.0 is linux-aarch64, "
            "which the workspace declares",
            "",
        ),
    ],
    ids=["an undeclared platform", "no roster, so the machine the workspace lives on"],
)
def test_a_platform_is_fit_only_when_the_workspace_declares_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platforms: str,
    verdict: Verdict,
    detail: str,
    fix: str,
) -> None:
    """A workspace declaring no platforms solves for the machine it lives on and nothing else.

    An undeclared platform fails with the one line that adds it and the solve that follows.
    """
    monkeypatch.setattr("mainboard.fitness.current_platform", lambda: "linux-aarch64")
    found = fitness_of(tmp_path, platforms).platform(machine(arch="aarch64"))
    assert (found.verdict, found.detail, found.fix) == (verdict, detail, fix)


@pytest.mark.parametrize(
    ("builds", "verdict", "detail"),
    [
        (None, Verdict.WARN, "default: nothing solved yet"),
        ({"win-64": [_PLAIN]}, Verdict.FAIL, "default: the lock holds no linux-64 builds"),
        ({"linux-64": [_PLAIN, _CU128]}, Verdict.PASS, "default: 2 linux-64 builds locked"),
    ],
    ids=["never solved", "solved for other platforms", "solved for this one"],
)
def test_the_lock_must_hold_builds_for_this_machines_platform(
    tmp_path: Path, builds: dict[str, list[str]] | None, verdict: Verdict, detail: str
) -> None:
    """A lock solved elsewhere installs nothing here, so the fix is a solve that includes it."""
    judge = fitness_of(tmp_path)
    if builds is not None:
        lock(judge, builds)
    found = judge.lock(machine(), "default")
    assert (found.verdict, found.detail) == (verdict, detail)
    assert found.fix == ("" if verdict is Verdict.PASS else "mainboard install default --resolve")


def test_a_lock_is_read_once_however_many_questions_ask_about_it(fitness: Fitness) -> None:
    """The lock and the CUDA builds both read one file, so a judge reads it only once.

    An environment never solved is remembered as such too, rather than asked about again.
    """
    solved = lock(fitness, {"linux-64": [_CU128]})
    solved.unlink()
    assert fitness.lock(machine(), "default").verdict is Verdict.WARN
    lock(fitness, {"linux-64": [_CU128]})
    assert solved.is_file()
    assert (
        fitness.builds(machine(), "default").detail == "no CUDA builds locked for this card to run"
    )
    assert fitness.locks == {"default": None}


@pytest.mark.parametrize("host", ["miyabi", "cluster"])
def test_a_scheduler_login_node_answers_its_cards_once_with_the_caveat(
    fitness: Fitness, host: str
) -> None:
    """A login node's cards say nothing about the compute node a job lands on.

    So a machine with no card and no driver is not flagged, however high the CUDA floor.
    """
    found = fitness.judge(machine(cuda="", gpus=()), host=host)
    assert [section.section for section in found] == ["platform", "lock", "cards", "disk"]
    assert row(found, "cards").verdict is Verdict.PASS
    assert "login node" in row(found, "cards").detail


@pytest.mark.parametrize(
    ("host", "changes", "verdict", "detail"),
    [
        ("local", {}, Verdict.PASS, "driver CUDA 13.0 meets the floor 12.0"),
        (
            "gold",
            {"cuda": "12.8"},
            Verdict.FAIL,
            "the driver supports CUDA 12.8, below the workspace floor 13.0",
        ),
        (
            "local",
            {"cuda": ""},
            Verdict.WARN,
            "no NVIDIA driver answered, and the workspace solves CUDA 12.0 builds",
        ),
        (
            "local",
            {"system": "Darwin", "arch": "arm64"},
            Verdict.PASS,
            "no CUDA floor on osx-arm64",
        ),
    ],
    ids=["meets the workspace floor", "below the environment's own", "no driver", "a mac"],
)
def test_the_driver_is_judged_against_the_floor_of_the_environment_the_host_runs(
    fitness: Fitness, host: str, changes: dict[str, str], verdict: Verdict, detail: str
) -> None:
    """An environment's own floor overrides the workspace's, and a Mac carries no CUDA floor.

    Every row short of a pass names the one fix there is, a newer driver.
    """
    found = row(fitness.judge(machine(**changes), host=host), "driver")
    assert (found.verdict, found.detail) == (verdict, detail)
    assert ("nvidia.com" in found.fix) is (verdict is not Verdict.PASS)


@pytest.mark.parametrize(
    ("locations", "changes", "verdict", "detail"),
    [
        ([_PLAIN], {}, Verdict.PASS, "no CUDA builds locked for this card to run"),
        ([_CU128], {"gpus": ()}, Verdict.PASS, "no CUDA builds locked for this card to run"),
        (
            [_CU128],
            {"cuda": "12.4"},
            Verdict.FAIL,
            "builds locked for CUDA 12.8, a driver supporting 12.4",
        ),
        (
            [_CU124],
            {"cuda": "12.8", "gpus": (Card(name="RTX 5080", capability="12.0"),)},
            Verdict.FAIL,
            "compute capability 12.0 needs CUDA 12.8 builds, the lock holds CUDA 12.4",
        ),
        ([_CU118], {"cuda": "12.0"}, Verdict.PASS, "CUDA 11.8 builds run on this driver and card"),
        (
            [_CONDA_12_8],
            {"cuda": "", "gpus": (Card(name="RTX 5080", capability="12.0"),)},
            Verdict.PASS,
            "CUDA 12.8 builds run on this driver and card",
        ),
        (
            [_CU126, _CONDA_12_4],
            {"gpus": (Card(name="T4", capability="7.5"),)},
            Verdict.PASS,
            "CUDA 12.6 builds run on this driver and card",
        ),
        (
            [_CU124],
            {"gpus": (Card(name="card", capability=""),)},
            Verdict.PASS,
            "CUDA 12.4 builds run on this driver and card",
        ),
    ],
    ids=[
        "nothing built for cuda",
        "no card to run them",
        "a driver below the builds",
        "blackwell on builds that carry no kernels for it",
        "a wheel index path spelling",
        "a conda pin with no driver to compare",
        "the newest build of either spelling, on a card older than every generation",
        "a card that reports no capability",
    ],
)
def test_locked_cuda_builds_must_run_on_this_driver_and_carry_kernels_for_this_card(
    fitness: Fitness,
    locations: list[str],
    changes: dict[str, str | tuple[Card, ...]],
    verdict: Verdict,
    detail: str,
) -> None:
    """A Blackwell card on CUDA 12.4 builds imports and then fails its first kernel launch.

    So the newest CUDA any build was compiled against, in either the wheel or the conda
    spelling, has to be both within the driver and new enough for the card's generation.
    """
    lock(fitness, {"linux-64": locations})
    found = fitness.builds(machine(**changes), "default")
    assert (found.verdict, found.detail) == (verdict, detail)
    if verdict is Verdict.FAIL:
        assert found.fix


@pytest.mark.parametrize(
    ("host", "gpus", "verdict", "detail"),
    [
        ("gold", (), Verdict.FAIL, "jobs here ask for 1 cards, the machine has 0"),
        (
            "gold",
            (Card(name="RTX 4060", vram_mb=8192),),
            Verdict.FAIL,
            "jobs here need 16 GB of card memory, the largest card holds 8",
        ),
        ("gold", _FIT.gpus, Verdict.PASS, "1 cards, largest 24 GB, jobs need 16"),
        ("local", _FIT.gpus, Verdict.PASS, "1 cards, largest 24 GB"),
    ],
    ids=["too few cards", "too little memory", "enough of both", "no declared need"],
)
def test_the_cards_must_hold_what_the_hosts_jobs_declare_they_need(
    fitness: Fitness, host: str, gpus: tuple[Card, ...], verdict: Verdict, detail: str
) -> None:
    """A job that runs out of card memory finds out late, so the declared need is checked now."""
    found = fitness.memory(machine(gpus=gpus), fitness.manifest.profile(host))
    assert (found.verdict, found.detail) == (verdict, detail)


@pytest.mark.parametrize(
    ("role", "verdict", "detail"),
    [
        (Role.CENTER, Verdict.WARN, "30 GB free at /work, a center wants 60"),
        (Role.TARGET, Verdict.PASS, "30 GB free at /work"),
    ],
)
def test_a_center_needs_room_for_the_whole_tree_and_a_target_for_one_mirror(
    fitness: Fitness, role: Role, verdict: Verdict, detail: str
) -> None:
    """The same free space can be plenty for a mirror and too little for every environment."""
    found = fitness.disk(machine(free_bytes=30 * 10**9), role)
    assert (found.verdict, found.detail) == (verdict, detail)


@pytest.mark.parametrize(
    ("system", "tools", "missing", "found"),
    [
        (
            "Windows",
            {"git": "2.51.0", "git-lfs": "3.7.0"},
            ["gh", "ssh"],
            "git 2.51.0, git-lfs 3.7.0",
        ),
        ("Linux", {}, ["git", "git-lfs", "gh", "ssh"], "nothing"),
    ],
    ids=["some missing", "nothing at all"],
)
def test_a_center_missing_a_tool_fails_with_each_install_this_platform_uses(
    fitness: Fitness, system: str, tools: dict[str, str], missing: list[str], found: str
) -> None:
    """Every missing tool is named with its own install, in one line a person can run through."""
    judged = fitness.tools(machine(system=system, tools=tools))
    assert judged.verdict is Verdict.FAIL
    assert judged.detail == f"missing {', '.join(missing)}; found {found}"
    assert judged.fix == "; ".join(install_command(system, tool) for tool in missing)


@pytest.mark.parametrize(
    ("tools", "verdict", "detail"),
    [
        ({"rsync": "3.4.1", "tar": "1.35"}, Verdict.PASS, "mirrors ship through rsync"),
        ({"tar": "1.35"}, Verdict.PASS, "mirrors ship through tar"),
        ({}, Verdict.FAIL, "neither rsync nor tar answers, so no mirror can ship"),
    ],
    ids=["rsync", "tar standing in", "neither"],
)
def test_a_mirror_ships_through_rsync_or_the_tar_that_replaces_it(
    fitness: Fitness, tools: dict[str, str], verdict: Verdict, detail: str
) -> None:
    """Rsync is preferred, and tar is enough, so only a machine with neither fails."""
    found = fitness.sync(machine(tools=tools))
    assert (found.verdict, found.detail) == (verdict, detail)
    assert found.fix == ("" if tools else install_command("Linux", "tar"))


@pytest.mark.parametrize(
    ("system", "symlinks", "long_paths", "notes"),
    [
        ("Linux", "refused", False, []),
        ("Windows", "", True, []),
        ("Windows", "[WinError 1314]", True, ["cannot create symbolic links"]),
        ("Windows", "", False, ["260 characters"]),
        ("Windows", "[WinError 1314]", False, ["cannot create symbolic links", "260 characters"]),
    ],
    ids=["not windows", "windows with both", "no links", "no long paths", "neither"],
)
def test_windows_links_and_long_paths_each_warn_with_their_own_switch(
    fitness: Fitness, system: str, symlinks: str, long_paths: bool, notes: list[str]
) -> None:
    """Only Windows turns either off, and each refusal is named with the switch that lifts it.

    A checkout still completes without them, links as junctions and copies, so it warns.
    """
    found = fitness.links(machine(system=system, symlinks=symlinks, long_paths=long_paths))
    assert found.verdict is (Verdict.WARN if notes else Verdict.PASS)
    assert [
        note for note in ("cannot create symbolic links", "260 characters") if note in found.detail
    ] == notes
    assert ("Developer Mode" in found.fix) is ("cannot create symbolic links" in notes)
    assert ("LongPathsEnabled" in found.fix) is ("260 characters" in notes)


@pytest.fixture
def tracked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[Sequence[str]], None]:
    """Track `paths` in a real repository at the workspace root, whatever this disk folds.

    The entries go straight into the index, since a case-insensitive disk could not hold two
    files differing only in case, and git reads none of this machine's configuration.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def git(*args: str, stdin: str = "") -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def track(paths: Sequence[str]) -> None:
        git("init", "--quiet")
        blob = git("hash-object", "-w", "--stdin")
        for path in paths:
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")

    return track


@pytest.mark.parametrize(
    ("paths", "verdict", "detail"),
    [
        (
            ["README.md", "src/main.py"],
            Verdict.PASS,
            "case-insensitive filesystem, and no tracked paths differ only in case",
        ),
        (
            ["README.md", "readme.md", "src/main.py"],
            Verdict.FAIL,
            "case-insensitive filesystem, and 2 tracked paths differ only in case: "
            "README.md, readme.md",
        ),
        (
            ["A", "B", "a", "b", "c"],
            Verdict.FAIL,
            "case-insensitive filesystem, and 4 tracked paths differ only in case: A, B, a "
            "and 1 more",
        ),
    ],
    ids=["no collisions", "one colliding pair", "more than a line names"],
)
def test_a_case_insensitive_disk_fails_on_tracked_paths_that_differ_only_in_case(
    fitness: Fitness,
    tracked: Callable[[Sequence[str]], None],
    paths: list[str],
    verdict: Verdict,
    detail: str,
) -> None:
    """Two such paths check out as one file there, silently losing the other's content."""
    tracked(paths)
    found = fitness.case(machine(case_sensitive=False))
    assert (found.verdict, found.detail) == (verdict, detail)
    assert bool(found.fix) is (verdict is Verdict.FAIL)


def test_a_case_sensitive_disk_never_asks_git(fitness: Fitness) -> None:
    """The fixture root is no repository at all, so a git question there would have failed."""
    assert fitness.case(machine()).detail == "case-sensitive filesystem"


@pytest.mark.parametrize(
    ("autocrlf", "attributes", "detail"),
    [
        ("true", None, ""),
        ("true", "* text=auto\n", ""),
        ("true", "* text=auto eol=lf\n", "newlines follow the repository's .gitattributes"),
        ("", None, "newlines follow core.autocrlf=unset"),
        ("input", None, "newlines follow core.autocrlf=input"),
    ],
    ids=["autocrlf alone", "attributes without eol", "eol pinned", "unset", "input"],
)
def test_text_checks_out_with_the_newline_the_repository_stores(
    fitness: Fitness, autocrlf: str, attributes: str | None, detail: str
) -> None:
    """`core.autocrlf=true` rewrites every text file unless `.gitattributes` pins the newline.

    An empty expected detail marks the one warning, which names the setting that stops it.
    """
    if attributes is not None:
        (fitness.root / ".gitattributes").write_text(attributes, encoding="utf-8")
    found = fitness.line_endings(machine(git={"core.autocrlf": autocrlf} if autocrlf else {}))
    warned = Section(
        section="line-endings",
        verdict=Verdict.WARN,
        detail="core.autocrlf=true rewrites every text file to CRLF on checkout",
        fix="git config --global core.autocrlf input",
    )
    passed = Section(section="line-endings", verdict=Verdict.PASS, detail=detail)
    assert found == (passed if detail else warned)


@pytest.mark.parametrize(
    ("system", "shells", "verdict"),
    [
        ("Windows", {"pwsh": "pwsh.exe", "cmd": "cmd.exe"}, Verdict.WARN),
        ("Windows", {"bash": "C:/Program Files/Git/bin/bash.exe"}, Verdict.PASS),
        ("Linux", {"zsh": "/bin/zsh"}, Verdict.PASS),
    ],
    ids=["windows without git bash", "windows with it", "not windows"],
)
def test_windows_without_git_bash_warns_since_the_agents_run_commands_in_it(
    fitness: Fitness, system: str, shells: dict[str, str], verdict: Verdict
) -> None:
    """Every shell found is listed, and a missing Bash is fixed by the Git that ships it."""
    found = fitness.shell(machine(system=system, shells=shells))
    assert found.verdict is verdict
    assert all(name in found.detail for name in shells)
    assert found.fix == (install_command(system, "git") if verdict is Verdict.WARN else "")


@pytest.mark.parametrize(
    ("environment", "floors"),
    [("serving", {"cuda": "13.0"}), ("default", {"cuda": "12.0"}), ("ghost", {"cuda": "12.0"})],
    ids=["its own", "none of its own", "an unknown environment"],
)
def test_an_environments_floors_are_its_own_over_the_workspaces(
    fitness: Fitness, environment: str, floors: dict[str, str]
) -> None:
    """A name the manifest does not declare falls back to the workspace floors, never raising."""
    assert fitness._floors(environment) == floors
