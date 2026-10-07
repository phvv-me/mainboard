# The center workstation's git tooling, on macOS or Linux: whether git runs, whether large files
# arrive as files rather than pointer text, and whether an https remote authenticates without a
# password prompt nobody is there to answer.
#
# A repair that is a local git setting is applied here, since printing it for a person to type
# would only make that person the slowest step. A repair that needs an installer or an
# administrator is never run: it is named, exactly, per platform.

import platform
from pathlib import Path
from shlex import join
from typing import TYPE_CHECKING

from patos import FrozenModel

from .durable import Shell, locally

if TYPE_CHECKING:
    from collections.abc import Sequence

# Installs keyed by package, then by `platform.system()`. A system not named is read as a Linux
# distribution, the one family with no single installer.
_INSTALLS = {
    "git": {"Darwin": "xcode-select --install"},
    "git-lfs": {"Darwin": "brew install git-lfs"},
    "gh": {"Darwin": "brew install gh"},
    "ssh": {"Darwin": "ssh ships with macOS; restore it with xcode-select --install"},
}
_DISTRIBUTION = "sudo apt install {package} (or the {package} package of this distribution)"

# The credential helper macOS ships with its git, as git names it after `credential-`, and how
# it arrives when missing. Linux ships none.
_KEYCHAIN = "osxkeychain"
_KEYCHAIN_INSTALL = "brew install git"
_PLAINTEXT_HELPER = "git config --global credential.helper store"

# How many names a detail line lists before it only counts the rest.
_NAMED = 3


def install_command(system: str, package: str) -> str:
    """The command that installs `package` on a `system` machine, a distribution's when unknown.

    system: the platform as `platform.system()` spells it.
    """
    return _INSTALLS.get(package, {}).get(system, _DISTRIBUTION.format(package=package))


def abbreviated(names: Sequence[str], named: int = _NAMED) -> str:
    """The first `named` of `names`, then only a count of the rest."""
    rest = len(names) - named
    return ", ".join(names[:named]) + (f" and {rest} more" if rest > 0 else "")


class Readiness(FrozenModel):
    """One question about this workstation's tooling, answered after any safe repair ran.

    broken: whether work in this workspace goes wrong until the fix is run.
    detail: the one line behind the answer, naming any setting this check just changed.
    fix: the command a person still has to run, empty when nothing is left to do.
    """

    check: str
    broken: bool = False
    detail: str
    fix: str = ""


class Workstation:
    """The center machine's git tooling, probed through one bounded seam and repaired in place.

    Every probe is one git command run through `shell`, under that shell's own deadline, so a
    test hands in a scripted git. The checks run one after another because three of them may
    write the same global git config, and git refuses a second writer on a locked config rather
    than waiting for the first.

    root: the workspace repository the local settings and remotes belong to.
    system: the platform as `platform.system()` spells it, this machine's when empty.
    """

    def __init__(self, root: Path, *, shell: Shell = locally, system: str = "") -> None:
        self.root = root
        self.shell = shell
        self.system = system or platform.system()

    def examine(self) -> list[Readiness]:
        """Every check this platform needs, or git's alone when git does not run.

        Every later check is a git command, so it would only report the same absence.
        """
        git = self.git()
        if git.broken:
            return [git]
        macos = [self.precomposed()] if self.system == "Darwin" else []
        return [git, self.lfs(), self.credentials(), *macos]

    def git(self) -> Readiness:
        """Whether git runs here at all."""
        status, said = self.shell(("git", "--version"))
        if status:
            return Readiness(
                check="git",
                broken=True,
                detail=f"git does not run here: {said.strip()}",
                fix=install_command(self.system, "git"),
            )
        return Readiness(check="git", detail=said.strip())

    def lfs(self) -> Readiness:
        """Whether large files check out as their contents rather than as pointer text.

        The binary alone is not enough: without its filters in some git config, a clone holds
        the pointer files and nothing says so until one of them is opened. A system-wide install
        puts those filters in the system config, so the effective value is what is read here.
        """
        status, said = self.shell(("git", "lfs", "version"))
        if status:
            return Readiness(
                check="git-lfs",
                broken=True,
                detail="git-lfs is not installed, so large files check out as pointer text",
                fix=install_command(self.system, "git-lfs"),
            )
        if self._config("--get", "filter.lfs.process"):
            return Readiness(check="git-lfs", detail=f"{said.strip()}, filters installed")
        return self._applied(
            "git-lfs",
            ("git", "lfs", "install", "--skip-repo"),
            f"{said.strip()}, installed its filters into the global git config",
        )

    def credentials(self) -> Readiness:
        """Whether every https remote reaches a credential helper instead of a password prompt.

        Git itself matches each remote against the generic and the url-specific helpers, so an
        uncovered remote is exactly the one a fetch would stop and ask about. The macOS helper is
        set only once this machine verifiably has it; elsewhere the fix is named.
        """
        uncovered = [
            url
            for url in self._https_remotes()
            if not self._config("--get-urlmatch", "credential.helper", url)
        ]
        if not uncovered:
            return Readiness(check="credentials", detail="every https remote has a helper")
        remotes = ", ".join(uncovered)
        if self.system != "Darwin":
            return Readiness(
                check="credentials",
                detail=(
                    f"no credential helper for {remotes}; `store` keeps the token in plain "
                    f"text in ~/.git-credentials, while Git Credential Manager "
                    f"(`git-credential-manager configure`) keeps it in the system keyring"
                ),
                fix=_PLAINTEXT_HELPER,
            )
        if self._keychain():
            return self._applied(
                "credentials",
                ("git", "config", "--global", "credential.helper", _KEYCHAIN),
                f"set credential.helper={_KEYCHAIN} for {remotes}",
            )
        return Readiness(
            check="credentials",
            detail=(
                f"no credential helper for {remotes}, and git-credential-{_KEYCHAIN} is missing"
            ),
            fix=f"{_KEYCHAIN_INSTALL}; git config --global credential.helper {_KEYCHAIN}",
        )

    def precomposed(self) -> Readiness:
        """Whether git on macOS reads a decomposed file name as the composed one it tracks.

        A repository cloned on another system lacks the `core.precomposeunicode` a macOS clone
        sets, so every tracked `ä` or `ã` the filesystem hands back decomposed lists twice: the
        tracked file, and an untracked twin a sweep would commit.
        """
        if self._config("--type=bool", "--get", "core.precomposeunicode") == "true":
            return Readiness(check="precomposeunicode", detail="core.precomposeunicode=true")
        return self._applied(
            "precomposeunicode",
            ("git", "config", "--global", "core.precomposeunicode", "true"),
            "set core.precomposeunicode=true in the global git config",
        )

    def _applied(self, check: str, command: Sequence[str], done: str) -> Readiness:
        """Run one safe local git setting, answering `done` or the command left to run."""
        status, said = self.shell(command)
        if status:
            return Readiness(
                check=check,
                detail=f"could not apply it here: {said.strip()}",
                fix=join(command),
            )
        return Readiness(check=check, detail=done)

    def _config(self, *query: str) -> str:
        """The effective value of one git setting in this repository, empty when unset."""
        status, said = self.shell(("git", "-C", str(self.root), "config", *query))
        return "" if status else said.strip()

    def _https_remotes(self) -> list[str]:
        """Every https url this repository fetches from or pushes to, each once."""
        _, said = self.shell(("git", "-C", str(self.root), "remote", "-v"))
        urls = [line.split()[1] for line in said.splitlines() if len(line.split()) > 1]
        return list(dict.fromkeys(url for url in urls if url.startswith("https://")))

    def _keychain(self) -> bool:
        """Whether this Mac's git carries the Keychain credential helper."""
        status, place = self.shell(("git", "--exec-path"))
        return not status and Path(place.strip(), f"git-credential-{_KEYCHAIN}").is_file()
