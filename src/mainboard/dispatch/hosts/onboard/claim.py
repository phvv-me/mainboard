# Whose workspace a host root holds, read from the manifest a mirror left in it.

import shlex
import tomllib
from typing import TYPE_CHECKING

from patos import FrozenOpenModel
from pydantic import ValidationError

from ....core.errors import MissionError
from ....core.project import Project

if TYPE_CHECKING:
    from pathlib import Path

    from ...transport.shells import HostShell


class _Header(FrozenOpenModel):
    """The `[workspace]` table of a manifest, as far as who it names."""

    name: str = ""


class Claim:
    """Refuses to mirror this workspace over a host root that holds another one.

    Workspaces share a host's default root (`~/.mb-jobs`) unless one gives itself another, and
    the second one's mirror then overwrites the first one's manifest, lock, scripts and
    environment beside its results. The manifest a mirror leaves in the root names its workspace,
    so that is the record: no second file to keep in step with it, and a root mirrored before
    this check existed is already covered. Evidence that is missing counts for nothing, so a root
    with no manifest, or one that is unreadable or names no workspace, is free to take.

    shell: the host's shell, its root placed.
    workspace: this workspace's root.
    """

    @property
    def ours(self) -> str:
        """The name this workspace's manifest declares, its directory's when it declares none."""
        text = Project().manifest(self.workspace).read_text(encoding="utf-8")
        return Claim._named(text) or self.workspace.name

    @staticmethod
    def _named(text: str) -> str:
        """The workspace a manifest's `text` names, empty when it names none or is no manifest."""
        try:
            return _Header.model_validate(tomllib.loads(text).get("workspace", {})).name
        except tomllib.TOMLDecodeError, ValidationError:
            return ""

    def __init__(self, shell: HostShell, workspace: Path) -> None:
        self.shell = shell
        self.workspace = workspace

    def _mirrored(self) -> str:
        """The workspace named by the first manifest the host's root holds, empty when none."""
        reads = (
            f"cat {shlex.quote(f'{self.shell.root}/{name}')} 2>/dev/null"
            for name in Project().manifests
        )
        return Claim._named(self.shell.run(" || ".join([*reads, "true"])))

    def check(self) -> None:
        """Refuse a root whose mirrored manifest names a workspace other than this one."""
        theirs, ours = self._mirrored(), self.ours
        if theirs in {"", ours}:
            return
        host, root = self.shell.plan.host, self.shell.root
        raise MissionError(
            f"{host!r} keeps {root} for workspace {theirs!r}, and this workspace is {ours!r}. "
            f"Mirroring would overwrite that workspace's manifest, lock, scripts and environment. "
            f"Give this one a root of its own with `[hosts.{host}] root = "
            f'"{Project().jobs_roots[0]}-{ours}"` in its manifest, or, when {theirs!r} no longer '
            f"lives on {host!r}, remove the manifest in {root} there and set the host up again."
        )
