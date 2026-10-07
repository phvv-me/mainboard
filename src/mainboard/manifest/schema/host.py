from pathlib import Path
from typing import Self

from ...core.base import Declared
from ...core.project import Project
from .queue import Defaults, QueuePolicy


class Sync(Declared):
    """What ships to a remote host and what the mirror may never delete."""

    include: list[str] = []
    exclude: list[str] = []
    protect: list[str] = []

    def shipped(self, root: Path) -> list[str]:
        """What a mirror ships from workspace `root`: the include, plus the committed lock
        wherever the manifest goes, since a host installs nothing without it and a lock reached
        only through a link back to the mirror reads as source escaping the pinned tree."""
        project = Project()
        lock = project.lock(root).name
        if lock in self.include or not any(name in self.include for name in project.manifests):
            return list(self.include)
        return [*self.include, lock] if (root / lock).is_file() else list(self.include)

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

    root: where the workspace lives there. Unset, one of the tool's `~/.<name>-jobs` folders,
        the one the host already uses when it has one (`dispatch.targets.resolve`); a leading
        `~` is the host's own home as its setup probed it.
    platform: the pixi platform (`linux-64`), probed at setup when empty.
    python: the bootstrap interpreter in the remote ssh login shell, quoted as that shell needs;
        standard-library collection needs no environment or Mainboard there.
    vars: read by this machine's backends (an API key, rental parameters), never shipped.
    exports: set for every job after its environment is entered, for facts about the host's
        world (`HF_HUB_OFFLINE = "1"` where compute nodes must never ask the Hub).
    login_memory_gb: what the login node lets one user hold in all, on a site that kills the
        largest tasks past it; setup and environment builds run there stop at a share of it
        (`dispatch.wrapping.guarded`). Zero where the login node sets no such limit.
    """

    kind: str = "auto"
    root: str = Project().jobs_roots[0]
    platform: str = ""
    python: str = "python3"
    account: str = ""
    login_shell: bool = True
    login_memory_gb: float = 0.0
    env: str = "default"
    container: str = ""
    modules: dict[str, str] = {}
    scratch: str = ""
    vars: dict[str, str] = {}
    exports: dict[str, str] = {}
    sync: Sync = Sync()
    queues: dict[str, QueuePolicy] = {}
    defaults: Defaults = Defaults()

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
