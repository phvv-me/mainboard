# What a dispatched run is measured from, read from git once and scoped to what the run ships.
#
# A RECEIPT NAMES THE TREE THE JOB RAN ON, NOT THE TREE THE SUBMITTER STOOD IN. The first reading
# of provenance was `git describe --always --dirty` over the repository owning a path in the
# command, falling back to the workspace root, and the root's `git status` counts a submodule's
# content changes as its own. A cutok job dispatched from the monorepo was therefore stamped
# `-dirty` for hours by edits under llm-head and reproducibility that no line of the job could
# reach (2026-09-06). A stamp that cannot tell an edit to the job apart from an edit to something
# beside it is not provenance, it is a timestamp.
#
# So the identity is scoped. A job spelled by file ships a closure, the exact files it imports
# and the directory it lives in, and its identity is the owning repository's HEAD plus a digest
# of those files as they stand on disk, `-dirty` only when one of THEM differs from HEAD or is
# untracked. A command that ships the whole mirror keeps the whole-tree reading, with every nested
# repository's dirt left out of it, since that dirt belongs to another tree's receipts.

import hashlib
import re
import shlex
from enum import StrEnum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from .shared import git

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# What a snapshot key may hold, so a source identity can never name a path outside the sources
# directory: git's text reaches the shell that builds and removes these trees, and a key of `..`
# would aim both at the mirror. Leading dots go with it, so no key can spell a relative step.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

# The suffix `git describe --dirty` appends, spelled once because a receipt strips it by name.
DIRTY = "-dirty"


class Status(StrEnum):
    """How one shipped file stands against the history that owns it.

    CLEAN: tracked and identical to HEAD.
    MODIFIED: tracked, with a working-tree or staged change.
    UNTRACKED: in no commit and not ignored, the file git would list as `??`.
    IGNORED: in no commit because an ignore rule keeps it out, a generated module for instance;
        digested, since the job runs it, but no dirt, since git itself counts none.
    UNVERSIONED: under no repository at all.
    BUILT: a compiled extension the closure ships beside its package's source, found through the
        distribution's own installed record rather than git; digested, never dirt, since a build
        artifact is not something the repository owning its package ever had an opinion on.
    """

    CLEAN = auto()
    MODIFIED = auto()
    UNTRACKED = auto()
    IGNORED = auto()
    UNVERSIONED = auto()
    BUILT = auto()

    @property
    def dirty(self) -> bool:
        """Whether a file in this state makes the tree it ships in dirty."""
        return self in (Status.MODIFIED, Status.UNTRACKED)


class Source(FrozenModel):
    """The dispatching tree read once: what its receipts call it, and where its snapshot goes.

    The two halves are read together on purpose. A dirty tree names no commit, so its key
    carries a digest of the working-tree delta, and asking for that key twice across a slow
    dispatch can answer twice differently. A job rendered against one key while its tree is
    pinned under another is a job pointed at a directory nobody ever created, which is how a
    rented run activated from `.../sources/<the key its render saw>` and found no environment
    there, minutes after the landing had pinned the tree under the key the pin saw (vast
    49867368, 2026-09-04). Reading both at once makes that disagreement unrepresentable.

    identity: `git describe --always` for the tree the job's code lives in, `-dirty` appended
        when what the job ships differs from that commit, the string a job carries into its
        receipts as `MAINBOARD_SOURCE`.
    key: the directory name that tree is pinned under on the host.
    commit: that tree's commit, whole, exported to the job as `MAINBOARD_SOURCE_COMMIT`.
    digest: the content digest of what the job ships, exported as `MAINBOARD_SOURCE_DIGEST`.
        Together those two are what a run seals against on a host that has no history to read:
        the commit says which revision, the digest says these exact bytes.
    """

    identity: str
    key: str
    commit: str = ""
    digest: str = ""

    @property
    def dirty(self) -> bool:
        """Whether the receipts of a run from this source are inadmissible as evidence."""
        return self.identity.endswith(DIRTY)


class Row(FrozenModel):
    """One shipped file in a closure listing: where it is, what its bytes hash to, how it stands.

    path: the workspace-relative path the snapshot holds it under.
    blob: git's own object id for the bytes on disk, so a row can be checked against any index.
    status: how the file stands against the repository that owns it.
    """

    path: str
    blob: str
    status: Status


def blob_of(path: Path) -> str:
    """git's object id for the bytes at `path`, computed here so no repository has to hold them."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def listing(rows: Iterable[Row]) -> str:
    """The listing a job reads through `MAINBOARD_CLOSURE`: one `path blob status` row per file."""
    return "".join(f"{row.path}\t{row.blob}\t{row.status}\n" for row in rows)


def registered(node: Path, rows: Sequence[Row], *, root: Path) -> None:
    """Require the same committed adjacent registration at admission and acquisition."""
    relative = node.relative_to(root).as_posix()
    row = next((item for item in rows if item.path == relative), None)
    if row is None or row.status is not Status.CLEAN:
        raise MissionError(f"{relative} must be committed before this job is acquired")
    if blob_of(node) != row.blob:
        raise MissionError(f"{relative} changed after Mainboard prepared the job")


def named(identity: str) -> str:
    """`identity` as a directory name a snapshot can be pinned under."""
    return _UNSAFE.sub("-", identity)[:96].lstrip(".") or "untracked"


class Repository(FrozenModel):
    """One git working tree, asked the few questions a dispatch has about it.

    Every question is scoped to the paths it is asked about and blind to nested repositories,
    which is what keeps one tree's dirt out of another tree's receipts.

    path: the working tree's top level, absolute.
    """

    path: str

    @classmethod
    def owning(cls, where: Path) -> Repository | None:
        """The repository whose working tree holds `where`, None when no repository does."""
        top = git("-C", str(where), "rev-parse", "--show-toplevel")
        return cls(path=top) if top else None

    @property
    def root(self) -> Path:
        """The working tree as a path."""
        return Path(self.path)

    def describe(self) -> str:
        """`git describe --always` of HEAD, the name a receipt calls this tree by."""
        return self.__read("describe", "--always")

    def head(self) -> str:
        """The whole commit HEAD names, empty for a repository with no commit yet."""
        return self.__read("rev-parse", "HEAD")

    def index(self, paths: Sequence[str] = ()) -> str:
        """`git ls-files -s` over `paths`, the whole tree when none: tracked paths and blobs."""
        return self.__read("ls-files", "-s", "-z", "--", *paths)

    def delta(self, paths: Sequence[str] = ()) -> str:
        """The working tree's departure from HEAD over `paths`, the whole tree when none.

        Both what git would list as changed or untracked and the diff itself, so a delta moves
        exactly when the tree does. Nested repositories are left out on purpose: their content is
        another tree's provenance.
        """
        return self.__read(
            "status", "--porcelain", "--ignore-submodules=all", "--", *paths
        ) + self.__read("diff", "HEAD", "--ignore-submodules=all", "--", *paths)

    def states(self, paths: Sequence[str]) -> dict[str, Status]:
        """How each of `paths` stands against HEAD, keyed by the path as given.

        paths: repository-relative files to ask about.
        """
        tracked = {
            entry.rsplit("\t", maxsplit=1)[1] for entry in self.index(paths).split("\0") if entry
        }
        changed = self.__porcelain(paths)
        return {
            path: changed.get(path, Status.CLEAN if path in tracked else Status.IGNORED)
            for path in paths
        }

    def kept(self, directory: str) -> list[str]:
        """Every file under `directory` git would keep: tracked, or untracked and not ignored.

        The one listing that honours every nested ignore file the way git does, which is what
        a directory shipped in full has to be read through. A tracked file deleted from the
        working tree is listed by the index and skipped here, since nothing can ship it.

        directory: a repository-relative directory.
        """
        listed = self.__read(
            "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", directory
        )
        return sorted(
            {path for path in listed.split("\0") if path and (self.root / path).is_file()}
        )

    def __porcelain(self, paths: Sequence[str]) -> dict[str, Status]:
        """The changed and untracked paths among `paths`, as `git status` reports them."""
        found: dict[str, Status] = {}
        entries = git(
            "-C",
            self.path,
            "status",
            "--porcelain",
            "-z",
            "--ignore-submodules=all",
            "--",
            *paths,
            exact=True,
        ).split("\0")
        pending = iter(entries)
        for entry in pending:
            if not entry:
                continue
            code, path = entry[:2], entry[3:]
            found[path] = Status.UNTRACKED if code == "??" else Status.MODIFIED
            if "R" in code or "C" in code:
                next(pending, "")
        return found

    def __read(self, *args: str) -> str:
        return git("-C", self.path, *args)


def commanded(command: str, root: Path) -> Repository | None:
    """The repository owning the code `command` runs, the workspace's own when no token names one.

    The first command token naming an existing path under `root` picks the repository; a command
    naming no path falls back to the workspace. None when neither is under git at all.

    command: the shell command the job runs.
    root: the local workspace root the dispatch is staged from.
    """
    for token in shlex.split(command):
        candidate = root / token
        if not candidate.exists():
            continue
        found = Repository.owning(candidate if candidate.is_dir() else candidate.parent)
        if found is not None:
            return found
    return Repository.owning(root)


def tree_source(repository: Repository | None) -> Source:
    """The provenance of a command that ships the whole mirror: the tree owning its code.

    The whole-tree reading, with every nested repository's dirt left out: an edit inside a
    submodule is that submodule's provenance and never this tree's.

    repository: the repository owning the command's code, None for a workspace with no git.
    """
    if repository is None:
        return Source(identity="", key=named(""))
    listed = repository.index()
    delta = repository.delta()
    identity = repository.describe() + (DIRTY if delta else "")
    key = named(identity)
    if delta:
        key = f"{key}-{hashlib.blake2s(delta.encode(), digest_size=4).hexdigest()}"
    return Source(
        identity=identity,
        key=key,
        commit=repository.head(),
        digest=hashlib.sha256(f"{listed}\n{delta}".encode()).hexdigest() if listed else "",
    )


class Repositories:
    """The repositories a workspace's files live in, asked once each, and their word on a closure.

    A closure spans repositories: a node under the monorepo imports a house package that is a
    submodule of it, and a vendored dependency resolves to a tree beside the workspace. Every
    file is therefore asked about in the repository that owns it, and the identity names the
    repository that owns the job file while the digest and the dirt cover all of them.

    root: the workspace root the shipped paths are relative to.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repositories: dict[Path, Repository | None] = {}

    def seal(
        self, owner: Repository | None, files: Sequence[str], *, built: Sequence[str] = ()
    ) -> tuple[Source, list[Row]]:
        """The source and the listing of a closure of `files`, owned by `owner`.

        The digest is taken over the listing itself, every shipped file's path and the hash of
        its bytes on disk, so an untracked file, an edit inside a submodule and a vendored tree
        beside the workspace all move it exactly as they move what the host receives.

        owner: the repository holding the job file, whose HEAD names the identity.
        files: the closure's files, workspace-relative, as the snapshot holds them.
        built: the compiled extensions among `files`, recorded `built` outright rather than
            asked of a repository that never had an opinion on a build artifact.
        """
        marked = frozenset(built)
        rows = sorted(
            (
                row
                for repository, shipped in self.__grouped(files).items()
                for row in self.__read(repository, shipped, marked)
            ),
            key=lambda row: row.path,
        )
        digest = hashlib.sha256(listing(rows).encode()).hexdigest()
        dirty = any(row.status.dirty for row in rows)
        identity = (owner.describe() if owner else "") + (DIRTY if dirty else "")
        source = Source(
            identity=identity,
            key=f"{named(identity)}-{digest[:8]}",
            commit=owner.head() if owner else "",
            digest=digest,
        )
        return source, rows

    def owning(self, where: Path) -> Repository | None:
        """The repository holding `where`, asked once per directory."""
        try:
            return self.repositories[where]
        except KeyError:
            found = self.repositories[where] = Repository.owning(where)
            return found

    def kept(self, directory: str) -> list[str]:
        """Every file under the workspace directory `directory` its repository would keep.

        The workspace reaches a vendored dependency through a link, so the directory is asked
        about where it really is and every answer is spelled back the way the workspace reaches
        it, which is the way the snapshot holds it.

        directory: a workspace-relative directory.
        """
        real = (self.root / directory).resolve()
        repository = self.owning(real)
        if repository is None:
            raise MissionError(
                f"{directory} is under no git repository, so nothing can say which of its "
                "files are ignored; a job ships from a repository"
            )
        inside = real.relative_to(repository.root.resolve())
        spelled = PurePosixPath(directory)
        return [
            (spelled / PurePosixPath(path).relative_to(inside.as_posix())).as_posix()
            if inside.parts
            else (spelled / path).as_posix()
            for path in repository.kept(inside.as_posix())
        ]

    def __grouped(self, files: Sequence[str]) -> dict[Repository | None, list[str]]:
        """`files` by the repository owning each one's real location, in path order."""
        grouped: dict[Repository | None, list[str]] = {}
        for path in sorted(files):
            real = (self.root / path).resolve()
            grouped.setdefault(self.owning(real.parent), []).append(path)
        return grouped

    def __relative(self, repository: Repository, path: str) -> str:
        """`path` as `repository` spells it, through whatever link the workspace reaches it by."""
        real = (self.root / path).resolve()
        return real.relative_to(repository.root.resolve()).as_posix()

    def __read(
        self, repository: Repository | None, shipped: Sequence[str], built: frozenset[str]
    ) -> list[Row]:
        """The rows of `shipped` under `repository`: each file's bytes hashed, its status asked.

        The blob is taken over the bytes on disk for every file alike, tracked or not, so the
        digest over the rows moves exactly when the shipped bytes do. A path in `built` is
        stamped outright rather than asked of the repository, which has no opinion on a
        compiled extension the closure found through the distribution's own installed record.
        """
        if repository is None:
            return [
                Row(
                    path=path,
                    blob=blob_of(self.root / path),
                    status=Status.BUILT if path in built else Status.UNVERSIONED,
                )
                for path in shipped
            ]
        relative = {path: self.__relative(repository, path) for path in shipped}
        states = repository.states(list(relative.values()))
        return [
            Row(
                path=path,
                blob=blob_of(self.root / path),
                status=Status.BUILT if path in built else states[spelled],
            )
            for path, spelled in relative.items()
        ]
