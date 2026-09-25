from ...core.base import Declared

_MEGABYTE = 1 << 20


class GitPolicy(Declared):
    """How `git` operates the repository tree (the workspace and its submodules, recursively).

    Only repositories whose remote owner is the root's own or listed are written (pull, commit,
    push); the rest are read only.

    owners: GitHub users or organizations, compared case-insensitively.
    ceiling_mb: the largest file a commit takes; Git LFS pointers are exempt.
    never_commit: git glob pathspecs per repository a commit leaves alone, even when staged by
        hand, except a hand-staged deletion.
    """

    owners: list[str] = []
    ceiling_mb: float = 50.0
    never_commit: list[str] = ["**/evidence/artifacts/**"]

    @property
    def ceiling_bytes(self) -> int:
        return int(self.ceiling_mb * _MEGABYTE)

    @property
    def outside(self) -> list[str]:
        """Pathspecs that leave every `never_commit` path out of what git lists."""
        return [f":(exclude,glob){pattern}" for pattern in self.never_commit]

    @property
    def inside(self) -> list[str]:
        """Pathspecs that list only the `never_commit` paths."""
        return [f":(glob){pattern}" for pattern in self.never_commit]

    def owns(self, owner: str, root_owner: str) -> bool:
        """Whether a repository whose remote names `owner` (empty for none) is this workspace's."""
        mine = {name.casefold() for name in (*self.owners, root_owner) if name}
        return bool(owner) and owner.casefold() in mine
