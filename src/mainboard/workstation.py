# The center workstation's own tooling: the one machine that holds the monorepo and runs this
# tool, which can be macOS, Linux or a native Windows box with no WSL in sight. Every question
# here is about git on this machine rather than about the workspace, whether git runs at all,
# whether large files arrive as files instead of as pointer text, whether an https remote can
# authenticate without a password prompt nobody is there to answer, and on Windows whether the
# repository's symbolic links and deep paths survive a checkout.
#
# A repair that is a local git setting is applied here, since knowing the exact command and
# printing it for a person to type would only make that person the slowest step. A repair that
# needs an installer or an administrator is never run: it is named, exactly, per platform.

import platform
from pathlib import Path
from shlex import join
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from patos import FrozenModel

from .durable import Shell, locally

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# The installs this machine may be missing, keyed by `platform.system()`. A system not named
# here is read as a Linux distribution, the one family left that has no single installer.
_GIT_INSTALL = {
    "Windows": "winget install --id Git.Git -e",
    "Darwin": "xcode-select --install",
}
_LFS_INSTALL = {
    "Windows": "winget install --id GitHub.GitLFS -e",
    "Darwin": "brew install git-lfs",
}
_DISTRIBUTION = "sudo apt install {package} (or the {package} package of this distribution)"

# The credential helper each platform ships with its git, as git names it after `credential-`.
_KEYCHAIN = {"Windows": "manager", "Darwin": "osxkeychain"}

# How the standard helper arrives when it is missing, and what stands in for one on Linux.
_KEYCHAIN_INSTALL = {"Windows": _GIT_INSTALL["Windows"], "Darwin": "brew install git"}
_PLAINTEXT_HELPER = "git config --global credential.helper store"

# The one switch that lets an unprivileged account create symbolic links on Windows.
_DEVELOPER_MODE = (
    "enable Developer Mode: Settings > System > For developers > Developer Mode (or run "
    "`reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock /t REG_DWORD "
    "/f /v AllowDevelopmentWithoutDevLicense /d 1` as administrator)"
)

# The mode git records a symbolic link under in its index.
_LINK_MODE = "120000"

# How many unmaterialized links a detail line names before it only counts the rest.
_NAMED = 3

# Why this process cannot create a symbolic link, empty when it can.
type Linking = Callable[[], str]


def refusal_to_link() -> str:
    """Try one symbolic link in a scratch directory, answering the OS's refusal or nothing.

    Windows refuses an account without Developer Mode (or elevation) with WinError 1314, and a
    git that cannot link checks every link out as a plain file holding its target's path.
    """
    with TemporaryDirectory() as scratch:
        try:
            Path(scratch, "link").symlink_to(Path(scratch, "target"))
        except OSError as refusal:
            return str(refusal)
    return ""


class Readiness(FrozenModel):
    """One question about this workstation's tooling, answered after any safe repair ran.

    check: the area asked about, the row's name in a report.
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
    test hands in a scripted git and no answer here ever depends on the machine running it. The
    checks run one after another because three of them may write the same global git config,
    and git refuses a second writer on a locked config rather than waiting for the first.
    """

    def __init__(
        self,
        root: Path,
        *,
        shell: Shell = locally,
        system: str = "",
        linking: Linking = refusal_to_link,
    ) -> None:
        """root: the workspace repository the local settings and remotes belong to.

        shell: runs one command and answers with its status and output, this machine's bounded
            runner when left alone.
        system: the platform as `platform.system()` spells it, this machine's when empty.
        linking: says why a symbolic link cannot be created here, empty when it can.
        """
        self.root = root
        self.shell = shell
        self.system = system or platform.system()
        self.linking = linking

    def examine(self) -> list[Readiness]:
        """Every check this platform needs, git first since each later one is a git command.

        A machine without git is that one answer, because every other check would only report
        the same absence under a different name.
        """
        git = self.git()
        if git.broken:
            return [git]
        windows = [self.symlinks(), self.longpaths()] if self.system == "Windows" else []
        return [git, self.lfs(), self.credentials(), *windows]

    def git(self) -> Readiness:
        """Whether git runs here at all."""
        status, said = self.shell(("git", "--version"))
        if status:
            return Readiness(
                check="git",
                broken=True,
                detail=f"git does not run here: {said.strip()}",
                fix=self._install(_GIT_INSTALL, "git"),
            )
        return Readiness(check="git", detail=said.strip())

    def lfs(self) -> Readiness:
        """Whether large files check out as their contents rather than as pointer text.

        The binary alone is not enough: without its filters in some git config, a clone holds
        the pointer files and nothing says so until one of them is opened. Git for Windows
        installs those filters system-wide, so the effective value is what is read here.
        """
        status, said = self.shell(("git", "lfs", "version"))
        if status:
            return Readiness(
                check="git-lfs",
                broken=True,
                detail="git-lfs is not installed, so large files check out as pointer text",
                fix=self._install(_LFS_INSTALL, "git-lfs"),
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
        uncovered remote is exactly the one a fetch would stop and ask about. The platform's
        own helper is set only once this machine verifiably has it; elsewhere the fix is named.
        """
        uncovered = [
            url
            for url in self._https_remotes()
            if not self._config("--get-urlmatch", "credential.helper", url)
        ]
        if not uncovered:
            return Readiness(check="credentials", detail="every https remote has a helper")
        remotes = ", ".join(uncovered)
        helper = self._keychain()
        if helper:
            return self._applied(
                "credentials",
                ("git", "config", "--global", "credential.helper", helper),
                f"set credential.helper={helper} for {remotes}",
            )
        standard = _KEYCHAIN.get(self.system)
        if standard is None:
            return Readiness(
                check="credentials",
                detail=(
                    f"no credential helper for {remotes}; `store` keeps the token in plain "
                    f"text in ~/.git-credentials, while Git Credential Manager "
                    f"(`git-credential-manager configure`) keeps it in the system keyring"
                ),
                fix=_PLAINTEXT_HELPER,
            )
        return Readiness(
            check="credentials",
            detail=f"no credential helper for {remotes}, and git-credential-{standard} is missing",
            fix=(
                f"{_KEYCHAIN_INSTALL[self.system]}; "
                f"git config --global credential.helper {standard}"
            ),
        )

    def symlinks(self) -> Readiness:
        """Whether the repository's symbolic links are links on this Windows disk.

        Three things have to hold. This account must be allowed to create a link at all, which
        is Developer Mode; the repository must ask git for links, which is `core.symlinks` and
        is set here when missing; and every link already checked out while it was off is still
        a plain file holding its target's path, which only a re-checkout of those paths turns
        into a link, named rather than run since it rewrites the working tree.
        """
        refusal = self.linking()
        if refusal:
            return Readiness(
                check="symlinks",
                broken=True,
                detail=f"this account cannot create symbolic links: {refusal}",
                fix=_DEVELOPER_MODE,
            )
        changed = ""
        if self._config("--type=bool", "--get", "core.symlinks") != "true":
            configured = self._applied(
                "symlinks",
                ("git", "-C", str(self.root), "config", "core.symlinks", "true"),
                "set core.symlinks=true in this repository",
            )
            if configured.fix:
                return configured
            changed = f"{configured.detail}; "
        flat = self._flattened_links()
        if flat:
            unnamed = len(flat) - _NAMED
            named = ", ".join(flat[:_NAMED]) + (f" and {unnamed} more" if unnamed > 0 else "")
            return Readiness(
                check="symlinks",
                detail=(
                    f"{changed}{len(flat)} links were checked out as plain files ({named}); "
                    f"checking those paths out again makes them links"
                ),
                fix=join(("git", "-C", str(self.root), "checkout", "--", *flat)),
            )
        return Readiness(check="symlinks", detail=f"{changed or 'core.symlinks=true; '}links work")

    def longpaths(self) -> Readiness:
        """Whether git on Windows checks out paths past the 260 character limit."""
        if self._config("--type=bool", "--get", "core.longpaths") == "true":
            return Readiness(check="longpaths", detail="core.longpaths=true")
        return self._applied(
            "longpaths",
            ("git", "config", "--global", "core.longpaths", "true"),
            "set core.longpaths=true in the global git config",
        )

    def _applied(self, check: str, command: Sequence[str], done: str) -> Readiness:
        """Run one safe local git setting, answering `done` or the command left to run.

        check: the row the setting belongs to.
        command: the git invocation that makes the setting.
        done: the detail once it took.
        """
        status, said = self.shell(command)
        if status:
            return Readiness(
                check=check,
                detail=f"could not apply it here: {said.strip()}",
                fix=join(command),
            )
        return Readiness(check=check, detail=done)

    def _config(self, *query: str) -> str:
        """The effective value of one git setting in this repository, empty when unset.

        query: the `git config` arguments that read it, `--get` or `--get-urlmatch` among them.
        """
        status, said = self.shell(("git", "-C", str(self.root), "config", *query))
        return "" if status else said.strip()

    def _https_remotes(self) -> list[str]:
        """Every https url this repository fetches from or pushes to, each once."""
        _, said = self.shell(("git", "-C", str(self.root), "remote", "-v"))
        urls = [line.split()[1] for line in said.splitlines() if len(line.split()) > 1]
        return list(dict.fromkeys(url for url in urls if url.startswith("https://")))

    def _keychain(self) -> str:
        """The platform's standard credential helper when this machine verifiably has it."""
        match self.system:
            case "Windows":
                status, _ = self.shell(("git", "credential-manager", "--version"))
                return "" if status else "manager"
            case "Darwin":
                status, place = self.shell(("git", "--exec-path"))
                found = not status and Path(place.strip(), "git-credential-osxkeychain").is_file()
                return "osxkeychain" if found else ""
        return ""

    def _flattened_links(self) -> list[str]:
        """Every link the index records that sits on disk as something other than a link."""
        _, said = self.shell(("git", "-C", str(self.root), "ls-files", "-s", "-z"))
        entries = [entry.split("\t", 1) for entry in said.split("\0") if "\t" in entry]
        return [
            path
            for stage, path in entries
            if stage.startswith(_LINK_MODE) and not (self.root / path).is_symlink()
        ]

    def _install(self, known: dict[str, str], package: str) -> str:
        """The install command for `package` on this platform, a distribution's when unknown."""
        return known.get(self.system, _DISTRIBUTION.format(package=package))
