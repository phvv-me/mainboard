import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.probe import census
from mainboard.probe.census import GIT_SETTINGS, SHELLS, TOOLS, Census, main, run
from mainboard.probe.system import System

from ..strategies import WORDS

# What a command nobody scripted answers: it ran and failed without a word.
_SILENT = (1, "")

# The one file a Linux census reads its distribution name from.
_OS_RELEASE = Path("/etc/os-release")

# The two queries the NVIDIA census makes, the plain banner and the card listing.
_BANNER = ("nvidia-smi",)
_LISTING = (
    "nvidia-smi",
    "--query-gpu=name,driver_version,compute_cap,memory.total",
    "--format=csv,noheader,nounits",
)


class Shell:
    """A scripted machine: each argv answered from a table, every command kept.

    answers: what each exact command line answers, anything else failing silently.
    """

    def __init__(self, answers: Mapping[tuple[str, ...], tuple[int, str]]) -> None:
        self.answers = dict(answers)
        self.ran: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> tuple[int, str]:
        self.ran.append(tuple(command))
        return self.answers.get(tuple(command), _SILENT)


def census_of(
    system: str,
    answers: Mapping[tuple[str, ...], tuple[int, str]] | None = None,
    on_path: Mapping[str, str] | None = None,
) -> tuple[Census, Shell]:
    """A census of `system` over a scripted shell, with `on_path` the programs PATH holds."""
    shell = Shell(answers or {})
    found = dict(on_path or {})
    return Census(runner=shell, finder=found.get, system=system), shell


@pytest.fixture
def os_release(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """`/etc/os-release` stood in for: its text when the dict holds one, absent when it is empty.

    Only that one path is intercepted, so every other read in the test still reaches disk.
    """
    held: dict[str, str] = {}
    real = Path.read_text

    def read_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if self != _OS_RELEASE:
            return real(self, encoding=encoding, errors=errors)
        try:
            return held["text"]
        except KeyError:
            raise FileNotFoundError(self) from None

    monkeypatch.setattr(Path, "read_text", read_text)
    return held


def test_a_survey_is_plain_json_a_system_model_reads_back_whole(
    tmp_path: Path, os_release: dict[str, str]
) -> None:
    """The census crosses an ssh pipe as JSON, so every answer has to survive `json.dumps`.

    The root handed in need not exist yet, as for a clone about to be made, so the filesystem
    questions are asked of its nearest existing ancestor, and the model on the far side parses
    every key the census wrote without dropping one.
    """
    os_release["text"] = 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\n'
    machine, _ = census_of("Linux", on_path={"bash": "/bin/bash"})

    surveyed = machine.survey(str(tmp_path / "not" / "cloned" / "yet"))

    assert json.loads(json.dumps(surveyed)) == surveyed
    assert surveyed["root"] == str(tmp_path)
    assert surveyed["version"] == "Ubuntu 24.04.1 LTS"
    assert (surveyed["cuda"], surveyed["gpus"], surveyed["tools"]) == ("", [], {})
    assert set(System.model_validate(surveyed).model_dump()) == set(surveyed)


@pytest.mark.parametrize(
    ("system", "release", "expected"),
    [
        ("Darwin", None, "macOS 26.1"),
        ("Windows", None, "Windows 11 build 26100"),
        ("Linux", 'NAME=Arch\nPRETTY_NAME="Arch Linux"\n', "Arch Linux"),
        ("Linux", "NAME=Alpine\nbroken line\n", "#1 SMP kernel"),
        ("Linux", None, "#1 SMP kernel"),
    ],
    ids=["macos", "windows", "a distribution", "a release without a pretty name", "no release"],
)
def test_the_version_is_the_operating_systems_own_name_for_itself(
    monkeypatch: pytest.MonkeyPatch,
    os_release: dict[str, str],
    system: str,
    release: str | None,
    expected: str,
) -> None:
    """A report names `Ubuntu 24.04`, not a kernel build string, wherever the OS says one.

    Only when no distribution names itself does the kernel's own version stand in.
    """
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.1", ("", "", ""), "arm64"))
    monkeypatch.setattr(platform, "win32_ver", lambda: ("11", "26100", "", "Multiprocessor"))
    monkeypatch.setattr(platform, "version", lambda: "#1 SMP kernel")
    if release is not None:
        os_release["text"] = release
    machine, _ = census_of(system)
    assert machine.version() == expected


@given(found=st.sets(st.sampled_from(SHELLS)), system=st.sampled_from(["Linux", "Darwin"]))
def test_off_windows_the_shells_are_exactly_the_ones_on_path(found: set[str], system: str) -> None:
    """Whatever PATH holds is reported by name with its path, and nothing PATH lacks."""
    machine, _ = census_of(system, on_path={name: f"/bin/{name}" for name in found})
    assert machine.shells() == {name: f"/bin/{name}" for name in found}


@pytest.mark.parametrize("layout", ["beside git", "under program files", "nowhere"], ids=str)
def test_windows_bash_is_git_for_windows_never_the_wsl_launcher_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    """System32's `bash.exe` answers to the name but runs WSL, so PATH's `bash` is dropped.

    Git for Windows' Bash is found beside the `git` on PATH or under Program Files, and a
    machine with neither reports no bash at all rather than the launcher.
    """
    for name in ("ProgramFiles", "ProgramW6432"):
        monkeypatch.delenv(name, raising=False)
    on_path = {"bash": "C:/Windows/System32/bash.exe", "cmd": "C:/Windows/System32/cmd.exe"}
    if layout == "beside git":
        git = tmp_path / "Git" / "cmd" / "git.exe"
        bash = tmp_path / "Git" / "bin" / "bash.exe"
        on_path["git"] = str(git)
        git.parent.mkdir(parents=True)
        git.touch()
    else:
        monkeypatch.setenv("ProgramW6432", str(tmp_path / "Program Files"))
        bash = tmp_path / "Program Files" / "Git" / "bin" / "bash.exe"
    installed = layout != "nowhere"
    if installed:
        bash.parent.mkdir(parents=True, exist_ok=True)
        bash.touch()
    machine, _ = census_of("Windows", on_path=on_path)

    found = machine.shells()

    assert found.keys() == ({"cmd", "bash"} if installed else {"cmd"})
    assert found["cmd"] == on_path["cmd"]
    if installed:
        assert Path(found["bash"]).resolve() == bash.resolve()


def test_the_filesystem_questions_answer_what_this_directory_does(tmp_path: Path) -> None:
    """Case sensitivity is whatever two names differing in case turn out to be here.

    The probe cleans up after itself, so asking leaves the workspace directory as it found it.
    """
    (tmp_path / "Mixed").touch()
    sensitive = not (tmp_path / "mixed").exists()
    (tmp_path / "Mixed").unlink()

    assert Census.case_sensitive(tmp_path) is sensitive
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("refusal", "expected"),
    [
        (None, ""),
        (OSError(1314, "A required privilege is not held"), "privilege"),
        (OSError(), "OSError"),
    ],
    ids=["links work", "a refusal with words", "a refusal without any"],
)
def test_a_symlink_refusal_is_reported_in_words_and_never_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: OSError | None, expected: str
) -> None:
    """An empty answer means links work, so a refusal that says nothing still names its type."""

    def link(self: Path, target: Path) -> None:
        if refusal:
            raise refusal

    monkeypatch.setattr(Path, "symlink_to", link)
    said = Census.symlinks(tmp_path)
    assert (expected in said) and (bool(said) is bool(refusal))


def test_a_directory_that_cannot_be_written_answers_the_safe_defaults(tmp_path: Path) -> None:
    """A probe that cannot even make its scratch directory must still answer, not raise.

    Case-sensitive is the assumption that raises no alarm, and the link question answers the
    OS's own refusal.
    """
    gone = tmp_path / "gone"
    assert Census.case_sensitive(gone) is True
    assert Census.symlinks(gone)


@given(
    system=st.sampled_from(["Windows", "Linux", "Darwin"]),
    longs=st.integers(min_value=0, max_value=2**32 - 1),
    developer=st.integers(min_value=0, max_value=2**32 - 1),
    status=st.sampled_from([0, 1]),
    spelled=st.sampled_from(["{:x}", "{:X}"]),
)
def test_the_windows_switches_are_the_registry_dword_being_exactly_one(
    system: str, longs: int, developer: int, status: int, spelled: str
) -> None:
    """Only a readable DWORD of 1 switches either on; off Windows the registry is never asked.

    Long paths are a given everywhere but Windows, Developer Mode is a Windows-only notion, and
    a `reg query` that fails reads as the unset switch it most likely is.
    """
    answers: dict[tuple[str, ...], tuple[int, str]] = {
        ("reg", "query", key, "/v", value): (status, f"    {value}    REG_DWORD    0x{word}\n")
        for (key, value), word in [
            (census._LONG_PATHS, spelled.format(longs)),
            (census._DEVELOPER_MODE, spelled.format(developer)),
        ]
    }
    machine, shell = census_of(system, answers)
    windows = system == "Windows"

    assert machine.long_paths() is (not windows or (status == 0 and longs == 1))
    assert machine.developer_mode() is (windows and status == 0 and developer == 1)
    assert bool(shell.ran) is windows


def test_a_registry_answer_without_a_dword_reads_as_unset() -> None:
    """A value stored under another type is not a switch this census can read."""
    key, value = census._LONG_PATHS
    machine, _ = census_of("Windows", {("reg", "query", key, "/v", value): (0, "REG_SZ yes")})
    assert machine.registry(key, value) == 0


@given(
    settings=st.dictionaries(
        st.sampled_from(GIT_SETTINGS), st.tuples(st.sampled_from([0, 1]), WORDS)
    )
)
def test_git_settings_are_every_global_key_with_unset_ones_empty(
    settings: dict[str, tuple[int, str]],
) -> None:
    """Every setting a checkout depends on is always reported, so a reader never guesses.

    A key git could not read is empty, and a set one is its value without git's newline.
    """
    answers: dict[tuple[str, ...], tuple[int, str]] = {
        ("git", "config", "--global", "--get", name): (status, f"{value}\n")
        for name, (status, value) in settings.items()
    }
    machine, _ = census_of("Linux", answers)
    assert machine.git() == {
        name: said if status == 0 else ""
        for name in GIT_SETTINGS
        for status, said in [settings.get(name, (1, ""))]
    }


def test_a_tool_counts_only_when_it_is_on_path_and_answers_with_a_version() -> None:
    """A tool that is missing, fails, or answers without a number is left out, never guessed.

    `git lfs` is asked through git, so it is found whenever git is, and its version is its own.
    """
    answers = {
        TOOLS["git"]: (0, "git version 2.51.0\n"),
        TOOLS["git-lfs"]: (0, "git-lfs/3.7.0 (GitHub; darwin arm64)\n"),
        TOOLS["ssh"]: (0, "OpenSSH_10.0p2, LibreSSL 3.3.6\n"),
        TOOLS["uv"]: (0, "uv (no version here)\n"),
        TOOLS["pixi"]: (2, "pixi 0.55.0\n"),
    }
    on_path = {name: f"/bin/{name}" for name in ("git", "ssh", "uv", "pixi")}
    machine, shell = census_of("Darwin", answers, on_path)

    assert machine.tools() == {"git": "2.51.0", "git-lfs": "3.7.0", "ssh": "10.0"}
    assert {command[0] for command in shell.ran} == set(on_path)


@pytest.mark.parametrize(
    ("banner", "listing", "cuda", "cards"),
    [
        (
            "| NVIDIA-SMI 580.82  Driver Version: 580.82  CUDA Version: 13.0 |",
            (
                0,
                "NVIDIA GeForce RTX 5080, 580.82, 12.0, 16303\n"
                "NVIDIA A100, 580.82, 8.0, [N/A]\n"
                "a line that is not a row\n",
            ),
            "13.0",
            [
                {
                    "name": "NVIDIA GeForce RTX 5080",
                    "driver": "580.82",
                    "capability": "12.0",
                    "vram_mb": 16303,
                },
                {"name": "NVIDIA A100", "driver": "580.82", "capability": "8.0", "vram_mb": 0},
            ],
        ),
        (
            "NVIDIA-SMI has failed because it couldn't communicate with the driver",
            (9, "NVIDIA GeForce RTX 5080, 580.82, 12.0, 16303\n"),
            "",
            [],
        ),
    ],
    ids=["a driver with two cards", "a driver that does not answer"],
)
def test_the_driver_names_its_cuda_and_every_card_it_lists(
    banner: str, listing: tuple[int, str], cuda: str, cards: list[dict[str, str | int]]
) -> None:
    """A memory field the driver cannot read is 0, not a crash, and a failed query lists none.

    A card listing from a query that failed is not trusted, however well formed it looks.
    """
    machine, _ = census_of(
        "Linux", {_BANNER: (0, banner), _LISTING: listing}, {"nvidia-smi": "/bin/nvidia-smi"}
    )
    assert machine.nvidia() == (cuda, cards)


def test_a_machine_without_nvidia_smi_is_never_asked_about_cards() -> None:
    """No driver means no CUDA and no cards, and not one query spent finding that out."""
    machine, shell = census_of("Linux")
    assert machine.nvidia() == ("", [])
    assert shell.ran == []


@pytest.mark.parametrize(
    ("depth", "exists"), [(0, True), (3, False)], ids=["an existing root", "a root to be made"]
)
def test_the_anchor_is_the_nearest_existing_directory_at_or_above_the_root(
    tmp_path: Path, depth: int, exists: bool
) -> None:
    """A clone lands on the filesystem of the deepest directory that already exists."""
    root = tmp_path.joinpath(*["deeper"] * depth)
    assert root.is_dir() is exists
    assert Census.anchor(str(root)) == tmp_path


@pytest.mark.parametrize(
    ("command", "answer"),
    [
        (
            [sys.executable, "-c", "import sys; print('out'); sys.stderr.write('err')"],
            (0, "out\nerr"),
        ),
        ([sys.executable, "-c", "raise SystemExit(3)"], (3, "")),
        (["a-program-that-is-not-installed-anywhere"], (127, "")),
    ],
    ids=["output and errors joined", "a failing status kept", "a missing program"],
)
def test_run_answers_status_and_output_and_never_raises(
    command: list[str], answer: tuple[int, str]
) -> None:
    """A census must survive every tool it asks, so a missing one answers like a shell would."""
    assert run(command) == answer


def test_a_tool_that_hangs_answers_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `--version` that never returns is a tool that does not answer, bounded by a deadline."""
    monkeypatch.setattr(census, "_SECONDS", 0.05)
    assert run([sys.executable, "-c", "import time; time.sleep(5)"]) == (127, "")


def test_main_prints_one_json_line_of_the_census_of_the_root_it_was_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A remote caller reads back exactly one line, so the census is one JSON document."""
    asked: list[str] = []

    def survey(self: Census, root: str) -> dict[str, census.Json]:
        asked.append(root)
        return {"system": self.system, "shells": {"sh": "/bin/sh"}}

    monkeypatch.setattr(Census, "survey", survey)
    main("/work/projects")

    printed = capsys.readouterr().out
    assert printed.count("\n") == 1
    assert json.loads(printed)["shells"] == {"sh": "/bin/sh"}
    assert asked == ["/work/projects"]
