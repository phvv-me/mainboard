from ...core.base import Declared

_MEGABYTE = 1 << 20


class GitPolicy(Declared):
    """How `git` operates this workspace's repository tree: whose repositories, and what enters.

    The tree is the workspace repository and every submodule under it, recursively, and most of
    that tree is other people's code pinned for reference. A repository is this workspace's own
    when the owner in its remote URL is the workspace root's own owner or one named here, and
    every verb that writes (pull, commit, push) touches only those. The rest are read, never
    committed to, never pushed.

    owners: remote owners (a GitHub user or organization) whose repositories this workspace
        commits to and pushes, compared case-insensitively; the root's own owner always is.
    ceiling_mb: the largest file a commit takes, in megabytes. Git LFS files are exempt, since
        what enters history for them is a pointer of a few hundred bytes.
    never_commit: git glob pathspecs, relative to each repository, whose content a commit leaves
        alone: an untracked file there stays untracked and a change there stays unstaged, even
        one staged by hand. Only a deletion staged by hand goes through, so what history already
        holds can still leave it.
    """

    owners: list[str] = []
    ceiling_mb: float = 50.0
    never_commit: list[str] = ["**/evidence/artifacts/**"]

    @property
    def ceiling_bytes(self) -> int:
        """The file-size ceiling in bytes."""
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
        """Whether a repository whose remote names `owner` is this workspace's own.

        owner: the owner parsed from the repository's remote URL, empty when it has none.
        root_owner: the owner of the workspace root's own remote.
        """
        mine = {name.casefold() for name in (*self.owners, root_owner) if name}
        return bool(owner) and owner.casefold() in mine
