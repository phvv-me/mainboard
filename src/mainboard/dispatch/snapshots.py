# The immutable copy of the synced source a dispatch pins its job to, so a later mirror sync
# can never rewrite the code a job that is already queued or running imports.
#
# The mirror stays the rsync target, because an incremental transfer is what makes dispatching
# cheap at all. What a job runs from is a snapshot of that mirror taken at submit time and named
# for the source identity its own receipts carry, so two dispatches of one tree share a snapshot
# and a dispatch of a different tree gets its own. A snapshot holds the shipped source and
# nothing else: the compiled environment, the dispatch state, the workspace's data directories
# and the dispatch's own results path are symlinks back to the mirror, so a job in a snapshot
# activates the mirror's environment, writes its log where every later read already looks, and
# leaves its results where the pull already goes.
#
# The generated tree is the one place those two worlds meet, and it is the one place a symlink
# is not enough. A workspace's generated manifest carries the root it was compiled for, and the
# tool resolves that manifest through the directory it is standing in, so a snapshot whose
# generated tree were a symlink would send every task back to the mirror it points at and pin
# nothing. Its directories are therefore real here, its files hardlinked, and only the installed
# artifacts under them symlinked, which pins the tree while leaving one environment for the host.

import hashlib
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ..core.project import Project
from .shared import git, logger, state_dir
from .sync import Rsync, rsync_argv
from .transport import HostUnreachable, is_transport_failure

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from .transport import Machine

# Where a host keeps its pinned trees, under the dispatch state directory the mirror already
# excludes from every transfer, so a sync can neither ship one nor prune one.
SOURCES = f"{state_dir()}/sources"

# The stamp a finished snapshot carries, naming the tree it froze. Its presence is what makes a
# second dispatch of the same tree free, and what keeps a build interrupted halfway from being
# mistaken for a complete one.
STAMP = ".mainboard-source"

# How many unused snapshots a host keeps after a prune. Enough that a job dispatched moments
# before a sweep still has its tree, small enough to sit inside an inode quota.
KEEP = 3

# What a key may hold, so a source identity can never name a path outside the sources
# directory: git's text reaches the shell that builds and removes these trees, and a key of `..`
# would aim both at the mirror. Leading dots go with it, so no key can spell a relative step.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def source_key(root: Path, *, source: str) -> str:
    """The directory name the tree `source` identifies is pinned under.

    `source` is the identity a job's receipts already carry in `MAINBOARD_SOURCE`, which is what
    lets a row and the tree it was measured on be checked against each other by name alone. A
    committed tree is fully named by it, so every dispatch of that commit shares one snapshot.

    A dirty tree names no commit, so its key carries a digest of the working-tree delta (the
    paths git reports changed or untracked, plus the tracked diff itself). That is what keeps
    every job of one dirty batch on a single snapshot while refusing to hand a later,
    differently dirty dispatch the earlier one's code. The one tree this cannot tell apart is a
    workspace with no git at all, which has no identity to carry into a receipt either.

    root: the local workspace root whose delta disambiguates a dirty identity.
    source: the dispatching tree's identity as `git describe --always --dirty` spelled it.
    """
    named = _UNSAFE.sub("-", source)[:96].lstrip(".") or "untracked"
    if source and not source.endswith("-dirty"):
        return named
    delta = git("-C", str(root), "status", "--porcelain") + git("-C", str(root), "diff", "HEAD")
    return f"{named}-{hashlib.blake2s(delta.encode(), digest_size=4).hexdigest()}"


def containers(sources: Sequence[str]) -> list[str]:
    """Every directory a snapshot has to create to hold `sources`, the tree root included.

    A snapshot fills each of these with symlinks back to the mirror for whatever it did not
    copy, which is how a job reaches the environment, the dispatch state and the data
    directories beside its own source without any of them being copied.
    """
    found = {"."}
    for source in sources:
        parts = PurePosixPath(source).parts[:-1]
        found.update("/".join(parts[: depth + 1]) for depth in range(len(parts)))
    return sorted(found)


def writable(path: str) -> str:
    """`path` as a workspace-relative results path, empty when it is not one.

    A results path is the one caller-typed string a snapshot turns into a symlink pointing back
    at the mirror, so an absolute path or one climbing out of the workspace is refused here
    rather than being spliced into a command that removes it.
    """
    posix = PurePosixPath(path)
    if not path or posix.is_absolute() or ".." in posix.parts:
        return ""
    return posix.as_posix()


class Snapshots:
    """The pinned source trees on one host, under `{root}/{STATE_DIR}/sources/`.

    A snapshot is a hardlink copy of the shipped file set, so it costs no data blocks at all and
    one inode per shipped directory: a file is a second name for the inode the mirror already
    holds, and only a directory has to be new. A workspace shipping a few thousand directories
    therefore costs a few thousand inodes per snapshot, which is why `prune` exists and why it
    keeps so few: Miyabi's personal group allows 102k inodes in total, and a host that runs out
    of them fails every later job with `EDQUOT` rather than with anything about disk space.

    Hardlinking is also what makes the pin correct rather than merely cheap. rsync replaces a
    changed file by writing a new one and renaming it over the old name, so the mirror's
    directory entry moves to a new inode while the snapshot's entry keeps pointing at the one
    the job is already reading.

    root: the workspace root on the host, the mirror every snapshot is taken from and links
        back to.
    keep: how many unused snapshots survive a prune.
    """

    def __init__(self, root: str, *, keep: int = KEEP) -> None:
        self.root = root.rstrip("/")
        self.keep = keep

    @property
    def base(self) -> str:
        """The directory every pinned tree on this host lives in."""
        return f"{self.root}/{SOURCES}"

    def path(self, key: str) -> str:
        """Where the tree `key` names is pinned, whether or not it has been materialised yet.

        Pure path arithmetic on purpose: a dispatch has to render the job script that runs from
        this directory before it opens the connection that creates it.
        """
        return f"{self.base}/{key}"

    def pin(
        self,
        remote: Machine,
        *,
        key: str,
        sources: Sequence[str],
        results: str = "",
        filters: Sequence[str] = (),
        exclude: Sequence[str] = (),
    ) -> str:
        """Materialise the snapshot for `key` on the host and answer the path a job runs from.

        A key already pinned is answered without touching its tree, so a batch of thirty five
        jobs from one commit pays for one snapshot and reuses it thirty four times.

        remote: the open connection to the host.
        key: the tree's identity, from `source_key`.
        sources: the workspace-relative paths the mirror ships, the only thing copied.
        results: the dispatch's declared results path, symlinked back to the mirror so what the
            job writes there is what a later pull brings home; empty for a dispatch that
            declared none.
        filters / exclude: the same rules the mirror transfer used, so the snapshot holds the
            shipped file set and not the artifacts the host wrote beside it. A root-anchored
            merge rule is safe to repeat here because the transfer that just ran ships every
            ancestor ignore file the rules read, so the rule and its file arrive together.
        """
        path = self.path(key)
        program = self.__program(
            path, key=key, sources=sources, results=results, filters=filters, exclude=exclude
        )
        retcode, _, err = remote["bash"][["-lc", program]].run(retcode=None)
        if is_transport_failure(int(retcode), str(err)):
            raise HostUnreachable(str(err).strip()[-200:] or "ssh transport failure")
        if retcode:
            raise SystemExit(f"could not pin the source tree at {path}: {str(err).strip()[-400:]}")
        return path

    def prune(self, remote: Machine, *, live: Collection[str]) -> list[str]:
        """Remove every pinned tree no live job runs from, newest `keep` kept, and name them.

        Reached from the durable sweep rather than from a dispatch, because only the sweep knows
        which jobs are still owed an outcome. The newest few survive whatever the sweep found,
        so a job dispatched between this pass and the last one still has the tree it was pinned
        to even though no record of it had been resolved yet.

        remote: the open connection to the host.
        live: the keys of the trees jobs still in flight run from, never removed.
        """
        listed = remote["bash"][["-lc", f"ls -1t {shlex.quote(self.base)} 2>/dev/null"]](
            retcode=None
        )
        names = [line.strip() for line in str(listed).splitlines() if line.strip()]
        doomed = [name for name in names[self.keep :] if name not in live]
        if not doomed:
            return []
        paths = " ".join(shlex.quote(self.path(name)) for name in doomed)
        remote["bash"][["-lc", f"rm -rf {paths}"]](retcode=None)
        logger.info("pruned %d unused source snapshot(s) under %s", len(doomed), self.base)
        return doomed

    def __program(
        self,
        path: str,
        *,
        key: str,
        sources: Sequence[str],
        results: str,
        filters: Sequence[str],
        exclude: Sequence[str],
    ) -> str:
        """The shell that builds one snapshot, in the order the phases have to happen.

        The shipped paths are hardlinked in with the mirror as `--link-dest`, under the same
        filter rules the transfer used, so nothing the host wrote beside the source is copied.
        The generated tree is rebuilt next, then every directory the copy created is filled with
        symlinks to whatever the mirror holds there and the snapshot does not, which is what puts
        the data directories and the ancestor ignore files back within reach. The declared
        results path is linked back last, so it survives whichever earlier phase also had an
        opinion about it. Only then is the stamp written, so a build cut off halfway is redone
        rather than run from.

        A file that vanished mid-walk (rsync's code 24, which a concurrent mirror sync causes)
        is the one failure absorbed, since the mirror is a moving target. Every other rsync
        failure means the tree is not whole, and a job must never start in one that is not.
        """
        argv = rsync_argv(
            Rsync.ARCHIVE | Rsync.RELATIVE,
            [*sources, f"{path}/"],
            filters=filters,
            exclude=exclude,
            extra=[f"--link-dest={self.root}/"],
        )
        lines = [
            "set -eu",
            f"mb_root={shlex.quote(self.root)}",
            f"mb_snap={shlex.quote(path)}",
            f'if [ -f "$mb_snap/{STAMP}" ]; then exit 0; fi',
            'mkdir -p "$mb_snap"',
            'cd "$mb_root"',
            f'{shlex.join(["rsync", *argv])} || [ "$?" = 24 ]',
            self.__generated(),
            self.__filling(sources),
            *self.__results(results),
            f"printf '%s\\n' {shlex.quote(key)} > \"$mb_snap/{STAMP}\"",
        ]
        return "; ".join(lines)

    def __generated(self) -> str:
        """The lines that rebuild the generated tree as this snapshot's own.

        The environment is the mirror's and the code is the snapshot's, and the generated tree is
        where those two meet, so it is neither copied nor symlinked whole. Its directories down
        to each environment shard are real here, its files are hardlinked in, and everything
        heavy under them, the installed environments, the node modules, the dispatch state and
        the receipts, is a symlink back to the mirror.

        That shape is what a job's own tooling needs. A workspace's generated manifest is
        compiled with the workspace root written into it, so a job standing in a snapshot
        recompiles one for the tree it is standing in, and every file that recompile writes is
        replaced rather than edited in place. Hardlinked here, that lands in this snapshot and
        the mirror's copy is untouched, which is what keeps one job from rewriting the
        environment description every other job on the host activates through. The paths inside
        it still name the mirror's installed environment, so nothing is installed twice.
        """
        out = shlex.quote(Project().out_dir)
        return (
            f'mkdir -p "$mb_snap"/{out}/envs; '
            f'for m in "$mb_root"/{out}/envs/*; do '
            f'if [ -d "$m" ]; then mkdir -p "$mb_snap"/{out}/envs/"${{m##*/}}"; fi; '
            "done; "
            f'for m in "$mb_root"/{out} "$mb_root"/{out}/envs/*; do '
            'if [ ! -d "$m" ]; then continue; fi; '
            'd=${m#"$mb_root"/}; '
            'for e in "$m"/* "$m"/.*; do '
            "n=${e##*/}; "
            'if [ "$n" = "." ] || [ "$n" = ".." ] || [ ! -e "$e" ]; then continue; fi; '
            'if [ -e "$mb_snap/$d/$n" ]; then continue; fi; '
            'if [ -d "$e" ]; then ln -s "$e" "$mb_snap/$d/$n"; else ln "$e" "$mb_snap/$d/$n"; fi; '
            "done; done"
        )

    def __filling(self, sources: Sequence[str]) -> str:
        """The loop symlinking back whatever the mirror holds and the copy did not bring over."""
        return (
            f"for d in {shlex.join(containers(sources))}; do "
            'mkdir -p "$mb_snap/$d"; '
            'for e in "$mb_root/$d"/* "$mb_root/$d"/.*; do '
            "n=${e##*/}; "
            'if [ "$n" = "." ] || [ "$n" = ".." ] || [ ! -e "$e" ]; then continue; fi; '
            'if [ -e "$mb_snap/$d/$n" ]; then continue; fi; '
            'ln -s "$e" "$mb_snap/$d/$n"; '
            "done; done"
        )

    def __results(self, results: str) -> list[str]:
        """The lines that point the dispatch's declared results path back at the mirror."""
        relative = writable(results)
        if not relative:
            return []
        quoted = shlex.quote(relative)
        parent = shlex.quote(str(PurePosixPath(relative).parent))
        return [
            f'mkdir -p "$mb_root"/{quoted} "$mb_snap"/{parent}',
            f'rm -rf "$mb_snap"/{quoted}',
            f'ln -s "$mb_root"/{quoted} "$mb_snap"/{quoted}',
        ]
