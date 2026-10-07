import os
from pathlib import Path, PurePosixPath
from stat import S_ISDIR
from typing import TYPE_CHECKING

from ..core.errors import MissionError
from ..core.project import Project
from .process import said
from .report import Outcome, Step

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ..manifest.schema.git import GitPolicy
    from .repo import Change, Repo
    from .tree import Tree

# How many paths a step names before it only counts the rest.
_NAMED = 3

# What a repository with no commit identity is held for, and the fix.
NO_IDENTITY = "git knows no author here; set `git config --global user.name` and `user.email`"


class Commit:
    """Commit owned repositories bottom-up, so each parent records the commits its submodules made.

    Named paths are committed alone, the way `git commit <paths>` does: each path goes to the
    deepest owned repository holding it, a directory takes every owned repository under it whole,
    and `git commit --only` leaves anything else in that repository's index for whoever staged it.
    Without paths each repository commits what is already staged, a parent its whole index.
    Either way a parent stages the pointer of a submodule that committed in this run, which is the
    bookkeeping this verb exists for, and never another one. `everything` is the deliberate
    sweep: every change in every owned repository, under one message.

    A named path that must not enter a commit (a `never-commit` pattern, a file over the size
    ceiling Git LFS does not carry, a link checked out as a file, a nested repository
    `.gitmodules` does not declare) holds its repository with the reason, rather than committing
    the rest of what was named. A repository git knows no author for, or whose detached HEAD
    cannot be put back on its trunk, is held before anything is staged.
    """

    def __init__(
        self, tree: Tree, message: str, paths: Sequence[Path] = (), *, everything: bool = False
    ) -> None:
        self.tree = tree
        self.message = message
        self.everything = everything
        self.named = _routed(tree, paths)
        self.committed: set[str] = set()

    def run(self) -> list[Step]:
        """Commit bottom-up, refusing up front when there is nothing to commit anywhere or no
        message to commit it with: git refuses an empty message only after the paths are staged,
        and the next commit would carry them under its own."""
        if not self.message.strip():
            raise MissionError("a commit needs a message; nothing was staged")
        if not (self.everything or self.named or self._staged()):
            name = Project().name
            raise MissionError(
                f"nothing is staged in any owned repository; name what to commit "
                f'(`{name} git commit -m "..." PATH...`) or sweep every change with --all'
            )
        return self.tree.upward("commit", self._committed, hold=self.everything)

    def _committed(self, repo: Repo) -> Step:
        step = self._swept(repo) if self.everything else self._chosen(repo)
        if step.outcome is Outcome.DONE:
            self.committed.add(repo.name)
        return step

    def _chosen(self, repo: Repo) -> Step:
        """Commit the paths named under `repo` and the pointers this run moved, or, when the run
        names no path, its whole index with those pointers."""
        pointers = self._pointers(repo)
        paths = [*self.named.get(repo.name, ()), *pointers]
        if not paths and repo.git.ok("diff", "--cached", "--quiet"):
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail="nothing named")
        if held := _unready(repo):
            return held
        if not self.named:
            _stage(repo, pointers)
            return self._finish(repo, [], notes=[])
        changes = repo.changes(within=paths)
        if refused := Intake(repo, self.tree.policy, named=paths).withheld(changes):
            detail = _named("refused", refused) + ", which a commit may not take"
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
        specs = [*_literal(paths), *self._unmoved(repo, paths)]
        _stage(repo, specs)
        return self._finish(repo, specs, notes=[])

    def _swept(self, repo: Repo) -> Step:
        """Commit every change in `repo` but what has to stay out, then merge its upstream."""
        relinked = [path for path in repo.unlinked() if repo.faithful(path)]
        _stage(repo, relinked, "checkout")
        notes = [_named("relinked", set(relinked))]
        changes = repo.changes(self.tree.policy.outside)
        if not changes:
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=notes[0] or "clean")
        if held := _unready(repo):
            return held
        withheld = Intake(repo, self.tree.policy).withheld(changes)
        _stage(repo, [c.path for c in changes if not c.staged and c.path not in withheld])
        _stage(repo, sorted(withheld), "reset", "-q")
        notes.append(_named("withheld", withheld))
        step = self._finish(repo, [], notes=notes)
        if step.outcome is not Outcome.DONE:
            return step
        detail = "; ".join(filter(None, [step.detail, _merged(repo)]))
        return step.model_copy(update={"detail": detail})

    def _finish(self, repo: Repo, specs: Sequence[str], *, notes: list[str]) -> Step:
        """Commit what `specs` match alone (the whole index when empty), or say nothing was
        there to take. `--only` leaves every other staged path in the index for whoever staged it.
        """
        only = ("--only", "--pathspec-from-file=-", "--pathspec-file-nul") if specs else ()
        if repo.git.ok("diff", "--cached", "--quiet", *(("--", *specs) if specs else ())):
            detail = "; ".join(filter(None, notes)) or (
                "nothing changed there" if specs else "nothing staged"
            )
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=detail)
        committed = repo.git.run("commit", "-q", "-m", self.message, *only, stdin="\0".join(specs))
        if not committed.succeeded:
            return Step(repo=repo.name, outcome=Outcome.FAILED, detail=said(committed))
        done = [f"{repo.short('HEAD')} on {repo.branch()}", *notes]
        return Step(repo=repo.name, outcome=Outcome.DONE, detail="; ".join(filter(None, done)))

    def _pointers(self, repo: Repo) -> list[str]:
        """The submodules of `repo` that committed in this run, as the paths of their pointers."""
        return [repo.relative(child) for child in repo.children if child.name in self.committed]

    def _unmoved(self, repo: Repo, paths: Sequence[str]) -> list[str]:
        """Exclusions keeping every pointer under a named directory out unless this run moved it.

        `git add dir/` would otherwise sweep a pointer some other work moved into the commit.
        """
        moved = set(self._pointers(repo))
        return [
            f":(exclude){relative}"
            for child in repo.children
            if (relative := repo.relative(child)) not in moved
            and any(path == "." or relative.startswith(f"{path}/") for path in paths)
        ]

    def _staged(self) -> bool:
        """Whether any owned repository holds staged changes."""
        return any(not repo.git.ok("diff", "--cached", "--quiet") for repo in self.tree.owned())


class Intake:
    """Which of a repository's changed paths `[git]` keeps out of a commit."""

    def __init__(self, repo: Repo, policy: GitPolicy, *, named: Collection[str] = ()) -> None:
        self.repo = repo
        self.policy = policy
        self.named = named

    def withheld(self, changes: Sequence[Change]) -> set[str]:
        """The paths that must stay out: `never-commit` content, an oversized file, a link checked
        out as a file, or a nested repository `.gitmodules` does not declare.

        A deletion of a `never-commit` path goes through, which is how something tracked by
        mistake leaves history.
        """
        return (
            self._patterned(changes)
            | self._heavy([c.path for c in changes if not c.deleted])
            | self._flattened(changes)
            | self._embedded(changes)
        )

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

    def _embedded(self, changes: Sequence[Change]) -> set[str]:
        """Every untracked nested repository `.gitmodules` does not declare, which `add` would
        record as a pointer with no URL. Git lists such a repository as one path ending in `/`."""
        declared = set(self.repo.submodule_paths)
        return {
            change.path
            for change in changes
            if change.untracked
            and change.path.endswith("/")
            and change.path.removesuffix("/") not in declared
        }

    def _patterned(self, changes: Sequence[Change]) -> set[str]:
        """The paths under a `never-commit` pattern with content: staged in the index (by hand,
        since a sweep never lists them) or among `changes`; a deletion goes through."""
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

    def _size(self, path: str) -> int:
        """The bytes `path` puts in history: its own, a symlink's rather than its target's.

        A moved submodule is a directory here and a commit id in history, so it weighs nothing.
        """
        stat = (self.repo.path / path).lstat()
        return 0 if S_ISDIR(stat.st_mode) else stat.st_size


def _routed(tree: Tree, paths: Sequence[Path]) -> dict[str, list[str]]:
    """Each named path as its owning repository spells it, keyed by that repository.

    A path resolves from the current directory, as git resolves it. It belongs to the deepest
    owned repository holding it, and a directory also takes every owned repository under it
    whole. A path inside a foreign submodule, outside the workspace, or naming nothing on disk
    or in the index is refused before anything is staged.
    """
    owned = sorted(tree.owned(), key=lambda repo: len(repo.path.parts), reverse=True)
    routed: dict[str, list[str]] = {}
    for given in paths:
        absolute = Path(os.path.abspath(given))
        holder = next(
            (repo for repo in owned if absolute == repo.path or repo.path in absolute.parents),
            None,
        )
        if holder is None:
            raise MissionError(f"{given} lies outside the workspace's owned repositories")
        relative = absolute.relative_to(holder.path).as_posix()
        foreign = next(
            (
                child
                for child in holder.children
                if not child.owned and child.path in absolute.parents
            ),
            None,
        )
        if foreign is not None:
            raise MissionError(
                f"{given} lies in {foreign.name}, which this workspace does not own; name "
                f"{foreign.name} itself to record its pointer"
            )
        if not (absolute.exists() or holder.git.out("ls-files", "-z", "--", relative)):
            raise MissionError(f"{given} is neither on disk nor tracked; nothing by that name")
        routed.setdefault(holder.name, []).append(relative or ".")
        for repo in owned:
            if repo is not holder and absolute in repo.path.parents:
                routed.setdefault(repo.name, []).append(".")
    return routed


def _unready(repo: Repo) -> Step | None:
    """The hold for a repository that cannot commit yet: no author, or a HEAD off every branch."""
    if not repo.identified():
        return Step(repo=repo.name, outcome=Outcome.HELD, detail=NO_IDENTITY)
    if not repo.branch() and not repo.attach():
        detail = f"detached at {repo.short('HEAD')}, off the line of {repo.trunk()}"
        return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
    return None


def _stage(repo: Repo, paths: Sequence[str], *command: str) -> None:
    """Hand `paths` to `git add -A`, or to `command`, through stdin rather than the command line.

    A tree this size can pass the kernel's limit on a command line. Every path is literal, so a
    file named `*.txt` never globs its neighbours in, unless it carries its own magic
    (`:(exclude)...`).
    """
    if not paths:
        return
    repo.git.out(
        *(command or ("add", "-A")),
        "--pathspec-from-file=-",
        "--pathspec-file-nul",
        stdin="\0".join(_literal(paths)),
    )


def _literal(paths: Sequence[str]) -> list[str]:
    """`paths` as literal pathspecs, a spec already carrying its own magic left as it is."""
    return [path if path.startswith(":(") else f":(literal){path}" for path in paths]


def _merged(repo: Repo) -> str:
    """Merge the upstream a freshly committed repository is behind, saying how that went."""
    upstream = repo.upstream()
    _, behind = repo.counts(upstream)
    if not behind:
        return ""
    if stopped := repo.merge(upstream):
        return f"merging {upstream} stopped on {stopped}; resolve with `git merge {upstream}`"
    return f"merged {behind} from {upstream}"


def _named(verb: str, paths: set[str]) -> str:
    """The step's note naming what `verb` happened to, empty when nothing."""
    if not paths:
        return ""
    named = sorted(paths)[:_NAMED]
    rest = len(paths) - len(named)
    return f"{verb} {', '.join(named)}" + (f" and {rest} more" if rest else "")
