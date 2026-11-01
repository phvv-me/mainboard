# What lets the destination fetch the clone and open the workspace as this center did: the GitHub
# login, the ssh host keys and Claude Code's per-project trust.

import re
import subprocess
from typing import TYPE_CHECKING

from patos import FrozenModel

from ....core import Project, Section, Verdict, captured

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..carrier import Carrier
    from ..state import Carried

# A remote reached over ssh at GitHub, `git@github.com:owner/name` or `ssh://git@github.com/...`.
_GITHUB_SSH = re.compile(r"(ssh://)?([^@/:]+@)?github\.com[:/]")


class Merged(FrozenModel):
    """How many entries a JSON merge changed."""

    changed: int


class Signed(FrozenModel):
    """What the destination's gh said to the login it was handed."""

    signed: bool
    detail: str = ""


class Trusted(FrozenModel):
    """How many ssh host blocks and known host lines the destination took."""

    blocks: int
    keys: int


class Credentials:
    """The logins and trust handed to the destination.

    destination: the ssh alias of the machine becoming the center.
    """

    @staticmethod
    def _gh(*arguments: str) -> str:
        """What `gh arguments` prints, empty when gh is absent or refuses."""
        try:
            done = captured(["gh", *arguments], timeout=30)
        except OSError, subprocess.SubprocessError:
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    @staticmethod
    def claude(carrier: Carrier, carried: Carried) -> Section:
        """This workspace's Claude Code project settings, merged in under the new path."""
        entries = carried.claude_project()
        if not entries:
            return Section(
                section="carry claude project", verdict=Verdict.PASS, detail="nothing to carry"
            )
        answer = carrier.call(
            "merge", {"path": ".claude.json", "key": "projects", "entries": entries}, Merged
        )
        return Section(
            section="carry claude project",
            verdict=Verdict.PASS,
            detail=f"trust and tool settings under the new path, {answer.changed} changed",
        )

    def __init__(self, destination: str) -> None:
        self.destination = destination

    def sign_in(self, carrier: Carrier) -> Section:
        """Hand this machine's GitHub login to gh there, which git then uses for https."""
        # Read into memory for the one call that hands it on over ssh's stdin, never printed.
        token = self._gh("auth", "token")
        if not token:
            return Section(
                section="github",
                verdict=Verdict.WARN,
                detail="no gh login here to carry",
                fix=f"gh auth login && gh auth setup-git (on {self.destination})",
            )
        answer = carrier.call("login", {"token": token}, Signed)
        if answer.signed:
            return Section(
                section="github", verdict=Verdict.PASS, detail="gh signed in, git uses it"
            )
        return Section(
            section="github",
            verdict=Verdict.FAIL,
            detail=f"gh did not take the login: {answer.detail}",
            fix="install gh there (winget install --id GitHub.cli -e), then migrate again",
        )

    def trust(self, carrier: Carrier, carried: Carried, urls: Sequence[str]) -> Section:
        """The ssh host blocks and known host lines the destination lacks, added beside its own.

        GitHub's host keys come from GitHub's API over the gh login when a clone reaches GitHub
        over ssh and this machine never recorded them, since a first connection must not be what
        vouches for them.

        urls: the remotes the clone fetches from.
        """
        known = carried.known_hosts()
        wanted = any(_GITHUB_SSH.match(url) for url in urls) and not any(
            carried.GITHUB in line.split()[0].split(",") for line in known
        )
        # GitHub's ssh host keys as its API publishes them, none when unreachable.
        fetched = self._gh("api", "meta", "--jq", ".ssh_keys[]").splitlines() if wanted else []
        known += [f"{carried.GITHUB} {key}" for key in fetched]
        added = carrier.call("ssh", {"blocks": carried.ssh_config(), "known": known}, Trusted)
        detail = f"{added.blocks} host blocks and {added.keys} known host lines added"
        if wanted and not fetched:
            return Section(
                section="carry ssh config",
                verdict=Verdict.WARN,
                detail=f"{detail}; GitHub's host keys unknown here and gh could not fetch them",
                fix=f"gh auth login, then {Project().name} host setup {self.destination} --center",
            )
        return Section(section="carry ssh config", verdict=Verdict.PASS, detail=detail)
