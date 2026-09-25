from typing import Self

from ...core.base import Declared
from ...core.project import Project
from .observe import Observe
from .queue import Defaults, QueuePolicy


class Sync(Declared):
    """What ships to a remote host and what the mirror may never delete."""

    include: list[str] = []
    exclude: list[str] = []
    protect: list[str] = []

    def merged(self, over: Self) -> Self:
        """This sync scope layered over `over`: a declared include replaces, the rest add.

        So narrowing a host never drops a workspace-wide protection.
        """
        return type(self)(
            include=self.include if "include" in self.model_fields_set else over.include,
            exclude=_union(over.exclude, extra=self.exclude),
            protect=_union(over.protect, extra=self.protect),
        )


class HostProfile(Declared):
    """One remote (or local) machine's execution profile, inheriting `[hosts.defaults]` per field.

    root: where the workspace lives there, the tool's `~/.<name>-jobs` folder by default; a
        leading `~` is the host's own home as its setup probed it (`%USERPROFILE%` on Windows).
    platform: the pixi platform (`win-64`), probed at setup when empty; it decides whether the
        host is reached through a login `bash` or PowerShell.
    python: the bootstrap interpreter in the remote ssh login shell, quoted as that shell needs;
        standard-library collection needs no environment or Mainboard there.
    vars: read by this machine's backends (an API key, rental parameters), never shipped.
    exports: set for every job after its environment is entered, for facts about the host's
        world (`HF_HUB_OFFLINE = "1"` where compute nodes must never ask the Hub).
    """

    kind: str = "auto"
    root: str = Project().jobs_root
    platform: str = ""
    python: str = "python3"
    account: str = ""
    login_shell: bool = True
    env: str = "default"
    container: str = ""
    modules: dict[str, str] = {}
    scratch: str = ""
    vars: dict[str, str] = {}
    exports: dict[str, str] = {}
    sync: Sync = Sync()
    queues: dict[str, QueuePolicy] = {}
    defaults: Defaults = Defaults()
    observe: Observe = Observe()

    def inheriting(self, base: Self) -> Self:
        """This profile with `base` filling every unset field and merging the tables."""
        fields = self.model_dump(exclude_unset=True)
        fields["sync"] = self.sync.merged(base.sync)
        fields["modules"] = {**base.modules, **self.modules}
        fields["vars"] = {**base.vars, **self.vars}
        fields["exports"] = {**base.exports, **self.exports}
        fields["queues"] = {**base.queues, **self.queues}
        merged = {**base.model_dump(exclude_unset=True), **fields}
        return type(self).model_validate(merged)

    def policy(self, queue: str) -> QueuePolicy:
        """The declared policy for `queue`, permissive when the host names none."""
        return self.queues.get(queue, QueuePolicy())


def _union(base: list[str], *, extra: list[str]) -> list[str]:
    return list(dict.fromkeys([*base, *extra]))
