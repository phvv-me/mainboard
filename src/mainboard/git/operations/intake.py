# What `[git]` keeps out of a commit: never-commit content, oversized files, links checked out as
# files, and nested repositories `.gitmodules` does not declare.

from pathlib import PurePosixPath
from stat import S_ISDIR
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ...manifest.schema.project.git import GitPolicy
    from ..repo import Change, Repo


class Intake:
    """Which of a repository's changed paths `[git]` keeps out of a commit."""

    def __init__(self, repo: Repo, policy: GitPolicy, *, named: Collection[str] = ()) -> None:
        self.repo = repo
        self.policy = policy
        self.named = named

    def _embedded(self, changes: Sequence[Change]) -> set[str]:
        """Every untracked nested repository `.gitmodules` does not declare.

        `add` would record such a repository as a pointer with no URL. Git lists one as a path
        ending in `/`.
        """
        declared = set(self.repo.submodule_paths)
        return {
            change.path
            for change in changes
            if change.untracked
            and change.path.endswith("/")
            and change.path.removesuffix("/") not in declared
        }

    def _flattened(self, changes: Sequence[Change]) -> set[str]:
        """Every link the working tree holds as a file, unless the file was named on purpose.

        A file named outright that is more than the link written out replaced it, as `mb agents
        sync` renders `.codex/config.toml` over the link it was, and goes through as the type
        change it is; where git records no file in a link's place (`core.symlinks=false` stages
        the text as the link's target) nothing does.
        """
        linking = self.repo.git.run("config", "--type=bool", "core.symlinks").stdout.strip()
        changed = {change.path for change in changes}
        return {
            path
            for path in self.repo.unlinked()
            if path in changed
            and (path not in self.named or linking == "false" or self.repo.faithful(path))
        }

    def _heavy(self, paths: Sequence[str]) -> set[str]:
        """The paths over the size ceiling that Git LFS does not carry."""
        oversized = [path for path in paths if self._size(path) > self.policy.ceiling_bytes]
        if not oversized:
            return set()
        listing = self.repo.git.out("check-attr", "-z", "filter", "--", *oversized).split("\0")
        # `git check-attr -z` answers in path, attribute, value triples.
        triples = zip(*[iter(listing)] * 3, strict=False)
        lfs = {path for path, _, value in triples if value == "lfs"}
        return set(oversized) - lfs

    def _patterned(self, changes: Sequence[Change]) -> set[str]:
        """The paths under a `never-commit` pattern that carry content.

        They are staged in the index (by hand, since a sweep never lists them) or among `changes`;
        a deletion goes through.
        """
        if not self.policy.inside:
            return set()
        staged = self.repo.git.out(
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=d",
            "--",
            *self.policy.inside,
        )
        named = {
            change.path
            for change in changes
            if not change.deleted
            and any(
                PurePosixPath(change.path).full_match(glob) for glob in self.policy.never_commit
            )
        }
        return set(filter(None, staged.split("\0"))) | named

    def _size(self, path: str) -> int:
        """The bytes `path` puts in history: its own, a symlink's rather than its target's.

        A moved submodule is a directory here and a commit id in history, so it weighs nothing.
        """
        stat = (self.repo.path / path).lstat()
        return 0 if S_ISDIR(stat.st_mode) else stat.st_size

    def withheld(self, changes: Sequence[Change]) -> set[str]:
        """The paths that must stay out of a commit.

        That is `never-commit` content, an oversized file, a link checked out as a file, or a
        nested repository `.gitmodules` does not declare.

        A deletion of a `never-commit` path goes through, which is how something tracked by
        mistake leaves history.
        """
        return (
            self._patterned(changes)
            | self._heavy([c.path for c in changes if not c.deleted])
            | self._flattened(changes)
            | self._embedded(changes)
        )
