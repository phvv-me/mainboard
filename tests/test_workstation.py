from collections.abc import Mapping, Sequence
from pathlib import Path
from shlex import join

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.workstation import Workstation, install_command, refusal_to_link

from .strategies import WORDS

_ORIGIN = "https://github.com/phvv-me/mainboard"

# The arguments that make a git command line a question rather than a change.
_READS = {"--get", "--get-urlmatch", "--version", "version", "--exec-path", "remote", "ls-files"}

# A git answering every question with its standing default: a command that fails and says
# nothing, which is what `git config --get` says about a key nobody set.
_UNSET = (1, "")

# Every command a workstation check may run and still leave this machine's software alone. A
# read, a local git setting, or the one filter install git-lfs scopes to the global config.
_SAFE_WRITES = (
    ("git", "lfs", "install", "--skip-repo"),
    ("git", "config", "--global", "credential.helper", "manager"),
    ("git", "config", "--global", "credential.helper", "osxkeychain"),
    ("git", "config", "--global", "core.longpaths", "true"),
)


# The tools Windows and macOS each install with a command of their own; Windows alone also
# names how its bundled tar arrives.
_NATIVE = ("git", "git-lfs", "gh", "ssh")


class Git:
    """A scripted git: each argv answered from a table, and every command it was handed kept.

    answers: what each exact command line answers, anything else failing silently.
    """

    def __init__(self, answers: Mapping[tuple[str, ...], tuple[int, str]]) -> None:
        self.answers = dict(answers)
        self.ran: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> tuple[int, str]:
        self.ran.append(tuple(command))
        return self.answers.get(tuple(command), _UNSET)


def local(root: Path, *command: str) -> tuple[str, ...]:
    """A git command line scoped to the workspace repository at `root`."""
    return ("git", "-C", str(root), *command)


def fit(root: Path) -> dict[tuple[str, ...], tuple[int, str]]:
    """A workstation with nothing left to set up: every check here passes against it."""
    return {
        ("git", "--version"): (0, "git version 2.51.0\n"),
        ("git", "lfs", "version"): (0, "git-lfs/3.7.0\n"),
        local(root, "config", "--get", "filter.lfs.process"): (0, "git-lfs filter-process\n"),
        local(root, "remote", "-v"): (
            0,
            f"origin\t{_ORIGIN} (fetch)\norigin\t{_ORIGIN} (push)\nmirror\tgit@gold:lab (fetch)\n",
        ),
        local(root, "config", "--get-urlmatch", "credential.helper", _ORIGIN): (0, "manager\n"),
        local(root, "config", "--type=bool", "--get", "core.symlinks"): (0, "true\n"),
        local(root, "ls-files", "-s", "-z"): (0, "100644 1a2b 0\tREADME.md\0"),
        local(root, "config", "--type=bool", "--get", "core.longpaths"): (0, "true\n"),
    }


def station(
    root: Path,
    system: str,
    answers: Mapping[tuple[str, ...], tuple[int, str]],
    refusal: str = "",
) -> tuple[Workstation, Git]:
    """A workstation of `system` over a scripted git, and that git to read its commands back."""
    git = Git(answers)
    return Workstation(root, shell=git, system=system, linking=lambda: refusal), git


def unset(root: Path, *keys: str) -> dict[tuple[str, ...], tuple[int, str]]:
    """The fit workstation with each command line in `keys` answering as never configured."""
    answers = fit(root)
    for key in keys:
        del answers[local(root, "config", *key.split())]
    return answers


@given(system=st.sampled_from(["Windows", "Darwin", "Linux"]) | WORDS)
def test_a_fit_workstation_passes_every_check_its_platform_asks_and_changes_nothing(
    tmp_path: Path, system: str
) -> None:
    """Windows alone checks links and path lengths, since only its git turns either off."""
    workstation, git = station(tmp_path, system, fit(tmp_path))

    found = workstation.examine()

    windows = ["symlinks", "longpaths"] if system == "Windows" else []
    assert [row.check for row in found] == ["git", "git-lfs", "credentials", *windows]
    assert not any(row.broken or row.fix for row in found)
    assert not set(_SAFE_WRITES) & set(git.ran)


@given(
    system=st.sampled_from(["Windows", "Darwin", "Linux"]),
    failing=st.sets(st.integers(min_value=0, max_value=7)),
    keychain=st.booleans(),
)
def test_no_state_of_the_machine_makes_a_check_run_an_install_or_leave_a_break_unexplained(
    tmp_path: Path, system: str, failing: set[int], keychain: bool
) -> None:
    """Installs and administrator switches are named, never run, whatever git answers.

    Every command a check runs is git, and the only ones that write are the local settings
    this module applies in place. A broken row always carries the command that repairs it, so
    the verdict a report derives from a row is always explainable.
    """
    answers = fit(tmp_path)
    probes = list(answers)
    for index in failing:
        del answers[probes[index]]
    (tmp_path / "with").mkdir(exist_ok=True)
    (tmp_path / "with" / "git-credential-osxkeychain").touch()
    answers[("git", "--exec-path")] = (0, str(tmp_path / ("with" if keychain else "without")))
    answers[("git", "credential-manager", "--version")] = (0 if keychain else 1, "")
    answers |= {command: (0, "") for command in _SAFE_WRITES}
    workstation, git = station(tmp_path, system, answers)

    found = workstation.examine()

    assert all(row.fix for row in found if row.broken)
    assert all(command[0] == "git" for command in git.ran)
    allowed = {*_SAFE_WRITES, local(tmp_path, "config", "core.symlinks", "true")}
    assert {command for command in git.ran if not _READS & set(command)} <= allowed


@pytest.mark.parametrize(
    ("system", "git_fix", "lfs_fix"),
    [
        ("Windows", "winget install --id Git.Git -e", "winget install --id GitHub.GitLFS -e"),
        ("Darwin", "xcode-select --install", "brew install git-lfs"),
        (
            "Linux",
            "sudo apt install git (or the git package of this distribution)",
            "sudo apt install git-lfs (or the git-lfs package of this distribution)",
        ),
    ],
)
def test_a_missing_git_or_git_lfs_is_broken_and_names_this_platforms_installer(
    tmp_path: Path, system: str, git_fix: str, lfs_fix: str
) -> None:
    """Nothing that installs software is run here; the exact command is named instead.

    Without git every later check would report the same absence under another name, so that
    one row is the whole answer.
    """
    missing, _ = station(tmp_path, system, {("git", "--version"): (1, "not installed here")})
    found = missing.examine()
    assert [(row.check, row.broken, row.fix) for row in found] == [("git", True, git_fix)]
    assert found[0].detail == "git does not run here: not installed here"

    answers = fit(tmp_path)
    del answers[("git", "lfs", "version")]
    without_lfs, _ = station(tmp_path, system, answers)
    lfs = without_lfs.lfs()
    assert (lfs.broken, lfs.fix) == (True, lfs_fix)
    assert "pointer text" in lfs.detail


@pytest.mark.parametrize(
    ("installed", "broken", "fix", "fragment"),
    [
        (True, False, "", "installed its filters into the global git config"),
        (False, False, "git lfs install --skip-repo", "could not apply it here: locked"),
    ],
    ids=["filters installed in place", "an install git refused is left to run by hand"],
)
def test_git_lfs_without_its_filters_gets_them_installed_into_the_global_config_only(
    tmp_path: Path, installed: bool, broken: bool, fix: str, fragment: str
) -> None:
    """`--skip-repo` keeps the repair to config: the repository's own hooks are left alone."""
    answers = unset(tmp_path, "--get filter.lfs.process")
    answers[("git", "lfs", "install", "--skip-repo")] = (0, "") if installed else (1, "locked\n")
    workstation, git = station(tmp_path, "Linux", answers)

    found = workstation.lfs()

    assert (found.broken, found.fix) == (broken, fix)
    assert fragment in found.detail
    assert ("git", "lfs", "install", "--skip-repo") in git.ran


@pytest.mark.parametrize(
    "remotes",
    [(0, "mirror\tgit@gold:lab (fetch)\n"), (128, "fatal: not a git repository\n"), (0, "\n")],
    ids=["only ssh remotes", "no repository at all", "no remotes"],
)
def test_a_workspace_with_no_https_remote_needs_no_credential_helper(
    tmp_path: Path, remotes: tuple[int, str]
) -> None:
    answers = fit(tmp_path) | {local(tmp_path, "remote", "-v"): remotes}
    workstation, _ = station(tmp_path, "Linux", answers)
    found = workstation.credentials()
    assert (found.broken, found.fix) == (False, "")


@pytest.mark.parametrize(
    ("system", "keychain", "fix", "written"),
    [
        ("Windows", True, "", ("git", "config", "--global", "credential.helper", "manager")),
        (
            "Windows",
            False,
            "winget install --id Git.Git -e; git config --global credential.helper manager",
            None,
        ),
        ("Darwin", True, "", ("git", "config", "--global", "credential.helper", "osxkeychain")),
        (
            "Darwin",
            False,
            "brew install git; git config --global credential.helper osxkeychain",
            None,
        ),
        ("Linux", True, "git config --global credential.helper store", None),
    ],
    ids=[
        "windows sets the credential manager its git ships",
        "windows without it names the git install that brings it",
        "macos sets the keychain helper its git ships",
        "macos without it names the install that brings it",
        "linux has no standard helper, so the choice is named with its tradeoff",
    ],
)
def test_an_https_remote_without_a_helper_gets_the_platforms_own_only_when_it_is_verifiably_there(
    tmp_path: Path, system: str, keychain: bool, fix: str, written: tuple[str, ...] | None
) -> None:
    """Setting a helper that is not installed would turn a password prompt into a hard failure."""
    answers = unset(tmp_path, f"--get-urlmatch credential.helper {_ORIGIN}")
    answers[("git", "--exec-path")] = (0, f"{tmp_path}\n")
    if keychain:
        answers[("git", "credential-manager", "--version")] = (0, "2.6.1\n")
        (tmp_path / "git-credential-osxkeychain").touch()
    if written:
        answers[written] = (0, "")
    workstation, git = station(tmp_path, system, answers)

    found = workstation.credentials()

    assert (found.broken, found.fix) == (False, fix)
    assert found.detail.count(_ORIGIN) == 1
    assert (written in git.ran) if written else not any("--global" in cmd for cmd in git.ran)
    if system == "Linux":
        assert "plain text" in found.detail and "git-credential-manager configure" in found.detail


def test_windows_refusing_symbolic_links_is_broken_until_developer_mode_is_on(
    tmp_path: Path,
) -> None:
    """Without the privilege git writes every link as a plain file holding its target's path."""
    workstation, git = station(
        tmp_path, "Windows", fit(tmp_path), refusal="[WinError 1314] A required privilege"
    )
    found = workstation.symlinks()
    assert found.broken
    assert "WinError 1314" in found.detail
    assert found.fix.startswith("enable Developer Mode: Settings > System > For developers")
    assert "AllowDevelopmentWithoutDevLicense /d 1` as administrator" in found.fix
    assert not any("config" in command and "true" in command for command in git.ran)


@pytest.mark.parametrize(
    ("configured", "fix", "detail"),
    [
        (True, "", "set core.symlinks=true in this repository; links work"),
        (False, "config core.symlinks true", "could not apply it here: locked"),
    ],
    ids=["set in this repository", "a setting git refused is left to run by hand"],
)
def test_a_repository_that_does_not_ask_for_links_is_switched_on_in_place(
    tmp_path: Path, configured: bool, fix: str, detail: str
) -> None:
    """A local setting of this repository alone, never the global one."""
    answers = unset(tmp_path, "--type=bool --get core.symlinks")
    setting = local(tmp_path, "config", "core.symlinks", "true")
    answers[setting] = (0, "") if configured else (1, "locked\n")
    workstation, git = station(tmp_path, "Windows", answers)

    found = workstation.symlinks()

    assert setting in git.ran
    assert not found.broken
    assert found.fix.endswith(fix)
    assert found.detail == detail


@given(links=st.lists(WORDS, min_size=1, max_size=6, unique=True), switched=st.booleans())
def test_links_checked_out_while_they_were_off_are_named_with_the_checkout_that_makes_them_links(
    tmp_path: Path, links: list[str], switched: bool
) -> None:
    """The re-checkout rewrites the working tree, so it is named, path by path, never run.

    Naming the paths keeps the command from touching anything else a person has edited, which
    a whole-tree checkout would throw away.
    """
    answers = fit(tmp_path)
    entries = "".join(f"120000 9f{index} 0\t{link}\0" for index, link in enumerate(links))
    answers[local(tmp_path, "ls-files", "-s", "-z")] = (0, f"100644 1a 0\tREADME.md\0{entries}")
    if switched:
        del answers[local(tmp_path, "config", "--type=bool", "--get", "core.symlinks")]
        answers[local(tmp_path, "config", "core.symlinks", "true")] = (0, "")
    workstation, git = station(tmp_path, "Windows", answers)

    found = workstation.symlinks()

    assert not found.broken
    assert found.fix == join(local(tmp_path, "checkout", "--", *links))
    assert f"{len(links)} links were checked out as plain files" in found.detail
    assert ("more" in found.detail) is (len(links) > 3)
    assert found.detail.startswith("set core.symlinks=true") is switched
    assert not any("checkout" in command for command in git.ran)


@pytest.mark.skipif(bool(refusal_to_link()), reason="this account cannot create symbolic links")
def test_a_link_that_is_already_a_link_on_disk_is_not_named(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to(tmp_path / "README.md")
    answers = fit(tmp_path)
    answers[local(tmp_path, "ls-files", "-s", "-z")] = (0, "120000 9f 0\tlink\0")
    workstation, _ = station(tmp_path, "Windows", answers)
    assert workstation.symlinks().detail == "core.symlinks=true; links work"


@pytest.mark.parametrize(
    ("configured", "fix"),
    [(True, ""), (False, "git config --global core.longpaths true")],
    ids=["set globally in place", "a setting git refused is left to run by hand"],
)
def test_windows_git_without_long_paths_gets_them_switched_on_globally(
    tmp_path: Path, configured: bool, fix: str
) -> None:
    answers = unset(tmp_path, "--type=bool --get core.longpaths")
    setting = ("git", "config", "--global", "core.longpaths", "true")
    answers[setting] = (0, "") if configured else (1, "locked\n")
    workstation, git = station(tmp_path, "Windows", answers)

    found = workstation.longpaths()

    assert setting in git.ran
    assert (found.broken, found.fix) == (False, fix)
    if configured:
        assert found.detail == "set core.longpaths=true in the global git config"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [(None, ""), (OSError(1314, "A required privilege is not held by the client"), "privilege")],
    ids=["a link made here", "the refusal windows gives an account without developer mode"],
)
def test_the_link_probe_answers_the_os_refusal_or_nothing(
    monkeypatch: pytest.MonkeyPatch, outcome: OSError | None, expected: str
) -> None:
    """The probe is one real link in a scratch directory, stood in for so every OS takes both."""
    seen: list[Path] = []

    def link(self: Path, target: Path) -> None:
        seen.append(self)
        if outcome:
            raise outcome

    monkeypatch.setattr(Path, "symlink_to", link)
    refusal = refusal_to_link()
    assert (expected in refusal) and (bool(refusal) is bool(outcome))
    assert [path.name for path in seen] == ["link"]
    assert not seen[0].parent.exists()


@given(
    system=st.sampled_from(["Windows", "Darwin", "Linux"]) | WORDS,
    package=st.sampled_from([*_NATIVE, "tar"]) | WORDS,
)
def test_every_install_is_a_named_command_and_a_distributions_wherever_none_is_known(
    system: str, package: str
) -> None:
    """A fix line always names something to run, and the installer is the platform's own.

    Windows and macOS each have one installer, so a known package there names it; everything
    else, a Linux distribution or a package this table never heard of, gets the distribution
    install spelled with that package's name.
    """
    distribution = f"sudo apt install {package} (or the {package} package of this distribution)"
    native = {("Windows", tool) for tool in (*_NATIVE, "tar")} | {
        ("Darwin", tool) for tool in _NATIVE
    }
    assert (install_command(system, package) == distribution) is ((system, package) not in native)


def test_the_default_workstation_is_this_machine(tmp_path: Path) -> None:
    """Left alone it names this machine's platform, so a report is about where it was asked."""
    workstation = Workstation(tmp_path)
    assert workstation.system
    assert workstation.linking is refusal_to_link
