import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ...core.errors import MissionError
from ...core.project import Project
from ..process import said
from ..report import Outcome, Step
from .intake import Intake

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..repo import Repo
    from ..tree import Tree

# How many paths a step names before it only counts the rest.
_NAMED = 3


class Commit:
    """Commit owned repositories bottom-up, so each parent records the commits its submodules made.

    Named paths are committed alone, the way `git commit <paths>` does: each path goes to the
    deepest owned repository holding it, a directory takes every owned repository under it whole,
    and `git commit --only` leaves anything else in that repository's index for whoever staged it.
    Without paths each repository commits what is already staged, a parent its whole index.
    Either way a parent stages the pointer of a submodule that committed in this run, which is the
    bookkeeping this verb exists for, and never another one. `everything` is the deliberate
    sweep: every change in every owned repository.

    Each message is `TEXT`, the default, or `REPO=TEXT` for the owned repository at path `REPO`,
    so one run lands a different commit in each repository. A parent that only records pointers
    and has no message of its own says which submodules moved.

    A named path that must not enter a commit (a `never-commit` pattern, a file over the size
    ceiling Git LFS does not carry, a link checked out as a file, a nested repository
    `.gitmodules` does not declare) holds its repository with the reason, rather than committing
    the rest of what was named. A repository git knows no author for, or whose detached HEAD
    cannot be put back on its trunk, is held before anything is staged.
    """

    # What a repository with no commit identity is held for, and the fix.
    NO_IDENTITY: ClassVar[str] = (
        "git knows no author here; set `git config --global user.name` and `user.email`"
    )

    @staticmethod
    def _addressed(tree: Tree, messages: Sequence[str]) -> tuple[str, dict[str, str]]:
        """The default message and each repository's own, keyed by repository name.

        `REPO=TEXT` addresses the owned repository whose path `REPO` resolves to from the current
        directory, as a named path does; any other message is the default, of which there is one.
        """
        owned = {repo.path: repo.name for repo in tree.owned()}
        defaults: list[str] = []
        addressed: dict[str, str] = {}
        for message in messages:
            where, _, said = message.partition("=")
            name = owned.get(Path(os.path.abspath(where.strip()))) if said.strip() else None
            if name is None:
                defaults.append(message)
            elif addressed.setdefault(name, said.strip()) != said.strip():
                raise MissionError(f"two messages name {name}; give each repository one")
        if len(defaults) > 1:
            raise MissionError(
                "only one message may go without a repository; address the others as REPO=TEXT"
            )
        return (defaults[0].strip() if defaults else ""), addressed

    @staticmethod
    def _holder(owned: Sequence[Repo], *, given: Path, absolute: Path) -> Repo:
        """The deepest of `owned` holding the path `given` (at `absolute`), refusing any other."""
        holder = next(
            (repo for repo in owned if absolute == repo.path or repo.path in absolute.parents),
            None,
        )
        if holder is None:
            raise MissionError(f"{given} lies outside the workspace's owned repositories")
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
        relative = absolute.relative_to(holder.path).as_posix()
        if not (absolute.exists() or Commit._is_tracked(holder, relative)):
            raise MissionError(f"{given} is neither on disk nor tracked; nothing by that name")
        return holder

    @staticmethod
    def _is_tracked(repo: Repo, path: str) -> bool:
        """Whether the index or HEAD names `path`; a `git rm` deletion is in HEAD alone."""
        tree = ("--with-tree=HEAD",) if repo.exists("HEAD") else ()
        return bool(repo.git.out("ls-files", "-z", *tree, "--", path))

    @staticmethod
    def _merged(repo: Repo) -> str:
        """Merge the upstream a freshly committed repository is behind, saying how that went."""
        upstream = repo.upstream()
        _, behind = repo.counts(upstream)
        if not behind:
            return ""
        if stopped := repo.merge(upstream):
            return f"merging {upstream} stopped on {stopped}; resolve with `git merge {upstream}`"
        return f"merged {behind} from {upstream}"

    @staticmethod
    def _merging(repo: Repo, step: Step) -> Step:
        """`step`, with a done commit followed by merging the upstream it is behind."""
        if step.outcome is not Outcome.DONE:
            return step
        detail = "; ".join(filter(None, [step.detail, Commit._merged(repo)]))
        return step.model_copy(update={"detail": detail})

    @staticmethod
    def _routed(tree: Tree, paths: Sequence[Path]) -> dict[str, list[str]]:
        """Each named path as its owning repository spells it, keyed by that repository.

        A path resolves from the current directory, as git resolves it. It belongs to the deepest
        owned repository holding it, and a directory also takes every owned repository under it
        whole. A path inside a foreign submodule, outside the workspace, or naming nothing on disk,
        in the index or at HEAD is refused before anything is staged.
        """
        owned = sorted(tree.owned(), key=lambda repo: len(repo.path.parts), reverse=True)
        routed: dict[str, list[str]] = {}
        for given in paths:
            absolute = Path(os.path.abspath(given))
            holder = Commit._holder(owned, given=given, absolute=absolute)
            relative = absolute.relative_to(holder.path).as_posix()
            routed.setdefault(holder.name, []).append(relative or ".")
            for repo in owned:
                if repo is not holder and absolute in repo.path.parents:
                    routed.setdefault(repo.name, []).append(".")
        return routed

    def __init__(
        self,
        tree: Tree,
        messages: Sequence[str],
        paths: Sequence[Path] = (),
        *,
        everything: bool = False,
    ) -> None:
        self.tree = tree
        self.default, self.messages = Commit._addressed(tree, messages)
        self.everything = everything
        self.named = Commit._routed(tree, paths)
        self.committed: set[str] = set()

    def _chosen(self, repo: Repo) -> Step:
        """Commit the paths named under `repo` and the pointers this run moved.

        When the run names no path, it commits the whole index with those pointers.
        """
        pointers = self._pointers(repo)
        paths = [*self.named.get(repo.name, ()), *pointers]
        if not paths and repo.git.ok("diff", "--cached", "--quiet"):
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail="nothing named")
        if held := _unready(repo):
            return held
        if not self.named:
            _stage(repo, pointers)
            return self._finish(repo, [], notes=[])
        return self._given(repo, paths)

    def _committed(self, repo: Repo) -> Step:
        step = self._swept(repo) if self.everything else self._chosen(repo)
        if step.outcome is Outcome.DONE:
            self.committed.add(repo.name)
        return step

    def _finish(self, repo: Repo, specs: Sequence[str], *, notes: list[str]) -> Step:
        """Commit what `specs` match alone, or say nothing was there to take.

        Empty `specs` take the whole index. `--only` leaves every other staged path in the index
        for whoever staged it.
        """
        only = ("--only", "--pathspec-from-file=-", "--pathspec-file-nul") if specs else ()
        if repo.git.ok("diff", "--cached", "--quiet", *(("--", *specs) if specs else ())):
            detail = "; ".join(filter(None, notes)) or (
                "nothing changed there" if specs else "nothing staged"
            )
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=detail)
        message = self._message(repo)
        committed = repo.git.run("commit", "-q", "-m", message, *only, stdin="\0".join(specs))
        if not committed.succeeded:
            return Step(repo=repo.name, outcome=Outcome.FAILED, detail=said(committed))
        done = [f"{repo.short('HEAD')} on {repo.branch()}", *notes]
        return Step(repo=repo.name, outcome=Outcome.DONE, detail="; ".join(filter(None, done)))

    def _given(self, repo: Repo, paths: Sequence[str]) -> Step:
        """Commit `paths` alone, refusing them all when any must stay out."""
        changes = repo.changes(within=paths)
        if refused := Intake(repo, self.tree.policy, named=paths).withheld(changes):
            detail = _named("refused", refused) + ", which a commit may not take"
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
        specs = [*_literal(paths), *self._unmoved(repo, paths)]
        # Resetting first puts back what `git rm` took out of the index, so `add -A` stages
        # that deletion again instead of matching nothing and failing.
        _stage(repo, specs, "reset", "-q")
        _stage(repo, specs)
        return self._finish(repo, specs, notes=[])

    def _message(self, repo: Repo) -> str:
        """`repo`'s own message, else the default, else what a pointer-only commit records."""
        moved = ", ".join(self._pointers(repo))
        return (
            self.messages.get(repo.name) or self.default or f"Record {moved} at their new commits"
        )

    def _pointers(self, repo: Repo) -> list[str]:
        """The submodules of `repo` that committed in this run, as the paths of their pointers."""
        return [repo.relative(child) for child in repo.children if child.name in self.committed]

    def _staged(self) -> bool:
        """Whether any owned repository holds staged changes."""
        return any(not repo.git.ok("diff", "--cached", "--quiet") for repo in self.tree.owned())

    def _swept(self, repo: Repo) -> Step:
        """Commit every change in `repo` but what has to stay out, then merge its upstream."""
        relinked = [path for path in repo.unlinked() if repo.faithful(path)]
        _stage(repo, relinked, "checkout")
        notes = [_named("relinked", set(relinked))]
        if not (changes := repo.changes(self.tree.policy.outside)):
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=notes[0] or "clean")
        if held := _unready(repo):
            return held
        withheld = Intake(repo, self.tree.policy).withheld(changes)
        _stage(repo, [c.path for c in changes if not c.staged and c.path not in withheld])
        _stage(repo, sorted(withheld), "reset", "-q")
        return Commit._merging(
            repo, self._finish(repo, [], notes=[*notes, _named("withheld", withheld)])
        )

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

    def run(self) -> list[Step]:
        """Commit bottom-up, each repository under its message.

        Nothing is staged when there is nothing to commit anywhere or a repository to commit has
        no message: git refuses an empty message only after the paths are staged, and the next
        commit would carry them under its own.
        """
        unsaid = (
            sorted(set(self.named) - set(self.messages))
            if self.named and not self.everything
            else ["every repository"]
        )
        if not self.default and unsaid:
            raise MissionError(
                f"a commit needs a message, and none was given for {', '.join(unsaid)}; pass "
                "-m TEXT for all of them or -m REPO=TEXT for each. Nothing was staged"
            )
        if not (self.everything or self.named or self._staged()):
            name = Project().name
            raise MissionError(
                f"nothing is staged in any owned repository; name what to commit "
                f'(`{name} git commit -m "..." PATH...`) or sweep every change with --all'
            )
        return self.tree.upward("commit", self._committed, hold=self.everything)


def _unready(repo: Repo) -> Step | None:
    """The hold for a repository that cannot commit yet: no author, or a HEAD off every branch."""
    if not repo.identified():
        return Step(repo=repo.name, outcome=Outcome.HELD, detail=Commit.NO_IDENTITY)
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


def _named(verb: str, paths: set[str]) -> str:
    """The step's note naming what `verb` happened to, empty when nothing."""
    if not paths:
        return ""
    named = sorted(paths)[:_NAMED]
    rest = len(paths) - len(named)
    return f"{verb} {', '.join(named)}" + (f" and {rest} more" if rest else "")
