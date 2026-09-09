# The immutable copy of the synced source a dispatch pins its job to, so a later mirror sync
# can never rewrite the code a job that is already queued or running imports.
#
# The mirror stays the rsync target, because an incremental transfer is what makes dispatching
# cheap at all. What a job runs from is a snapshot of that mirror taken at submit time and named
# for the source identity its own receipts carry, so two dispatches of one tree share a snapshot
# and a dispatch of a different tree gets its own. A snapshot holds what its image says and
# nothing else: the compiled environment, the dispatch state and the dispatch's own results path
# are symlinks back to the mirror, so a job in a snapshot activates the mirror's environment,
# writes its log where every later read already looks, and leaves its results where the pull
# already goes.
#
# Two images exist. A command that ships the mirror pins the whole synced allowlist, and every
# directory that copy created is filled with links back to whatever else the mirror holds there.
# A job spelled by file pins its closure, the exact files it imports and nothing beside them: the
# mirror is not reachable from that tree except through the environment, the data paths the job
# declared it needs, and its results path.
#
# The generated tree is the one place those two worlds meet, and it is the one place a symlink
# is not enough. A workspace's generated manifest carries the root it was compiled for, and the
# tool resolves that manifest through the directory it is standing in, so a snapshot whose
# generated tree were a symlink would send every task back to the mirror it points at and pin
# nothing. Its directories are therefore real here, its files hardlinked, and only the installed
# artifacts under them symlinked, which pins the tree while leaving one environment for the host.

import shlex
from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.project import Project
from .shared import state_dir
from .sync import Rsync, rsync_argv
from .transport import HostUnreachable, is_transport_failure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .transport import Machine

# Where a host keeps its pinned trees, under the dispatch state directory the mirror already
# excludes from every transfer, so a sync can neither ship one nor prune one.
SOURCES = f"{state_dir()}/sources"

# The completion stamp and frozen closure listing; reuse verifies both before returning a tree.
STAMP = ".mainboard-source"
CLOSURE = ".mainboard-closure"
WRAPPERS = ".mainboard-jobs"


def stamped(key: str, *, commit: str, digest: str) -> str:
    """What a finished snapshot's stamp holds: the tree's key, and the provenance of that tree.

    The key stays the first line and stays alone on it, because it is what the tree is named for
    and what every earlier stamp on every host already holds. The provenance follows as `name
    value` lines, which is what a job reads to say which commit it is running and what the
    shipped bytes hashed to, on a machine that has no history to derive either from.

    key: the tree's identity, the source's key.
    commit / digest: the dispatching tree's commit and content digest, either empty when the
        workspace has no git to answer with.
    """
    lines = [
        key,
        *(f"{name} {value}" for name, value in (("commit", commit), ("digest", digest)) if value),
    ]
    return "\n".join(lines) + "\n"


def containers(sources: Sequence[str]) -> list[str]:
    """Every directory a mirrored snapshot has to create to hold `sources`, the tree root included.

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


class Image(ABC, FrozenModel):
    """What one snapshot copies out of the mirror, and what it reaches back into the mirror for.

    Both halves are shell text the pin runs on the host, in the order the phases have to
    happen: the copy first, the generated tree next, then whatever the image links back.
    """

    @abstractmethod
    def copied(self, root: str) -> str:
        """The line that hardlinks the shipped set out of the mirror at `root` into `$mb_snap`."""

    @abstractmethod
    def filled(self) -> str:
        """The lines that reach back into the mirror from inside the stamp, empty for none."""

    @abstractmethod
    def linked(self) -> list[str]:
        """The lines run on every dispatch, outside the stamp, that link the mirror's data in."""

    def verified(self, digest: str, results: str) -> str:
        """Verify an exact listing when the image carries one; commands carry none."""
        return ""


class Mirrored(Image):
    """The whole synced allowlist, what a command that ships the mirror runs from.

    sources: the workspace-relative paths the mirror ships, the only thing copied.
    filters / exclude: the same rules the mirror transfer used, so the snapshot holds the
        shipped file set and not the artifacts the host wrote beside it. A root-anchored merge
        rule is safe to repeat here because the transfer that just ran ships every ancestor
        ignore file the rules read, so the rule and its file arrive together.
    """

    sources: tuple[str, ...]
    filters: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def copied(self, root: str) -> str:
        """The shipped paths hardlinked in with the mirror as `--link-dest`, under the same rules.

        A vanished file is an incomplete copy, including rsync's exit code 24.
        """
        argv = rsync_argv(
            Rsync.ARCHIVE | Rsync.RELATIVE,
            list(self.sources),
            filters=self.filters,
            exclude=self.exclude,
            extra=[f"--link-dest={root}/"],
        )
        return f'{shlex.join(["rsync", *argv])} "$mb_snap"/'

    def filled(self) -> str:
        """The loop symlinking back whatever the mirror holds and the copy did not bring over."""
        return (
            f"for d in {shlex.join(containers(self.sources))}; do "
            'mkdir -p "$mb_snap/$d"; '
            'for e in "$mb_root/$d"/* "$mb_root/$d"/.*; do '
            "n=${e##*/}; "
            'if [ "$n" = "." ] || [ "$n" = ".." ] || [ ! -e "$e" ]; then continue; fi; '
            'if [ -e "$mb_snap/$d/$n" ]; then continue; fi; '
            'ln -s "$e" "$mb_snap/$d/$n"; '
            "done; done"
        )

    def linked(self) -> list[str]:
        """Nothing: a mirrored tree already reaches everything the mirror holds."""
        return []


class Sealed(Image):
    """A job's closure and nothing beside it, what a job spelled by file runs from.

    listing: the closure listing the mirror carries, workspace-relative, whose first column
        names every shipped file. The snapshot freezes it as `CLOSURE`; the runner reads that
        copy through `MAINBOARD_CLOSURE`, not a later mirror's listing.
    needs: the workspace-relative data paths the job reads, each linked back to the mirror on
        every dispatch. A need the mirror does not hold refuses the dispatch by name, since a
        job that opens a dangling link fails after the queue rather than before it.
    """

    listing: str
    needs: tuple[str, ...] = ()

    def copied(self, root: str) -> str:
        """Hardlink the listed files without applying the mirror's ordinary ignore rules.

        The listing is exact, including vendored files that ordinary mirror rules exclude.
        Only the invariant host-local lease filters remain; dispatch refuses explicit leases,
        and verification below rejects any missing listed file before running the job.
        """
        argv = rsync_argv(
            Rsync.ARCHIVE | Rsync.COPY_LINKS,
            ["./"],
            extra=["--files-from=-", f"--link-dest={root}/"],
        )
        rsync = f'{shlex.join(["rsync", *argv])} "$mb_snap"/'
        return (
            f'cp -- "$mb_root"/{shlex.quote(self.listing)} "$mb_snap/{CLOSURE}"; '
            f'mb_hash=$(sha256sum -- "$mb_snap/{CLOSURE}"); '
            '[ "${mb_hash%% *}" = "$mb_digest" ] || '
            '{ echo "mainboard: closure listing digest mismatch" >&2; false; }; '
            f'cut -f1 "$mb_snap/{CLOSURE}" | {rsync}'
        )

    def verified(self, digest: str, results: str) -> str:
        """Check the frozen listing and every raw Git blob, without status exemptions."""
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("a sealed snapshot requires its complete closure digest")
        live = tuple(writable(path) for path in (*self.needs, *([results] if results else [])))
        if any(
            not path or path in (".", CLOSURE, STAMP, WRAPPERS) or path.startswith(WRAPPERS + "/")
            for path in live
        ):
            raise ValueError("snapshot data and results must be relative paths below the root")
        return "\n".join(
            [
                f'mb_hash=$(sha256sum -- "$mb_snap/{CLOSURE}")',
                f'[ "${{mb_hash%% *}}" = {shlex.quote(digest)} ] || '
                '{ echo "mainboard: closure listing digest mismatch" >&2; false; }',
                "while IFS=$'\\t' read -r mb_file mb_blob mb_status; do",
                'case "/$mb_file/" in //|/*/../*|/*/./*|//*) '
                'echo "mainboard: invalid closure path: $mb_file" >&2; false;; esac',
                f'case "$mb_file" in {CLOSURE}|{STAMP}|{WRAPPERS}|{WRAPPERS}/*) '
                'echo "mainboard: reserved closure path: $mb_file" >&2; false;; esac',
                f"for mb_live in {shlex.join(live)}; do",
                'case "$mb_file" in "$mb_live"|"$mb_live"/*) '
                'echo "mainboard: live path overlaps source: $mb_live" >&2; false;; esac',
                "done",
                '[ -f "$mb_snap/$mb_file" ] && [ ! -L "$mb_snap/$mb_file" ] || '
                '{ echo "mainboard: missing or linked source: $mb_file" >&2; false; }',
                'mb_real=$(realpath --relative-to="$mb_snap" -- "$mb_snap/$mb_file")',
                '[ "$mb_real" = "$mb_file" ] || '
                '{ echo "mainboard: linked source parent: $mb_file" >&2; false; }',
                'mb_hash=$(git hash-object --no-filters -- "$mb_snap/$mb_file")',
                '[ "$mb_hash" = "$mb_blob" ] || '
                '{ echo "mainboard: source blob mismatch: $mb_file" >&2; false; }',
                f'done < "$mb_snap/{CLOSURE}"',
            ]
        )

    def filled(self) -> str:
        """Nothing: a sealed tree reaches the mirror only through what it declared."""
        return ""

    def linked(self) -> list[str]:
        """Each need checked on the mirror and linked into the tree, on every dispatch."""
        lines: list[str] = []
        for need in self.needs:
            quoted = shlex.quote(need)
            parent = shlex.quote(str(PurePosixPath(need).parent))
            absent = shlex.quote(f"mainboard: the need {need} is not on the mirror")
            lines += [
                f'if [ ! -e "$mb_root"/{quoted} ]; then echo {absent} >&2; false; fi',
                f'mkdir -p "$mb_snap"/{parent}',
                f'ln -sfn "$mb_root"/{quoted} "$mb_snap"/{quoted}',
            ]
        return lines


class Snapshots:
    """The pinned source trees on one host, under `{root}/{STATE_DIR}/sources/`.

    Source files share inodes with the mirror; directories and frozen metadata add storage.
    Automatic deletion is unsafe: one workstation's job cache cannot establish ownership
    across other dispatchers or the gap before a submitted job is recorded. Snapshots remain
    until an operator verifies that no queued or running job uses them. Watch inode quotas.

    Hardlinking is also what makes the pin correct rather than merely cheap. rsync replaces a
    changed file by writing a new one and renaming it over the old name, so the mirror's
    directory entry moves to a new inode while the snapshot's entry keeps pointing at the one
    the job is already reading.

    root: the workspace root on the host, the mirror every snapshot is taken from and links
        back to.
    """

    def __init__(self, root: str) -> None:
        self.root = root.rstrip("/")

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

    @staticmethod
    def script(staged: str) -> str:
        """The frozen path of a generated wrapper, whose filename carries its complete digest."""
        name = PurePosixPath(staged).name
        digest = name.removeprefix("job-").removesuffix(".sh")
        if (
            not staged
            or writable(staged) != staged
            or name != f"job-{digest}.sh"
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("a staged wrapper requires a canonical path and full SHA-256 name")
        return f"{WRAPPERS}/{name}"

    def pin(
        self,
        remote: Machine,
        *,
        key: str,
        image: Image,
        results: str = "",
        prefix: str = "",
        environment: str = "default",
        commit: str = "",
        digest: str = "",
        script: str = "",
    ) -> str:
        """Materialise the snapshot for `key` on the host and answer the path a job runs from.

        A key already pinned is verified without rebuilding its tree. Its declared
        results path and its needs are linked back on every dispatch all the same, since those
        belong to the dispatch rather than to the tree and two batches off one commit routinely
        declare different ones.

        remote: the open connection to the host.
        key: the tree's identity, the source's key.
        image: what the snapshot copies out of the mirror and what it links back.
        results: the dispatch's declared results path, symlinked back to the mirror so what the
            job writes there is what a later pull brings home; empty for a dispatch that
            declared none.
        prefix: the immutable environment this tree activates, named by content. Written once,
            when the tree is built, and never repointed: the environment belongs to the tree the
            way its code does, so a wave queued against it keeps it however often the workspace
            re-solves afterwards. Empty leaves the tree reaching the mirror's own environment,
            which is what a workspace with no addressed prefixes still does.
        environment: the environment `prefix` belongs to, which is the directory inside the
            generated tree the link is written into.
        script: staged generated wrapper to freeze and verify before submission; empty for a
            direct remote command whose file is not shipped.
        commit / digest: the dispatching tree's own provenance, written into the stamp beside
            the key so the tree on the host says which commit it is and what its content hashed
            to. A mirror carries no history, so this file is the only place on that machine
            where either can be read.
        """
        if not key or key in (".", "..") or PurePosixPath(key).name != key:
            raise ValueError("a snapshot key must be one directory name")
        path = self.path(key)
        program = self.__program(
            path,
            image=image,
            results=results,
            prefix=prefix,
            environment=environment,
            stamp=stamped(key, commit=commit, digest=digest),
            digest=digest,
            script=script,
        )
        retcode, _, err = remote["bash"][["-lc", program]].run(retcode=None)
        if is_transport_failure(int(retcode), str(err)):
            raise HostUnreachable(str(err).strip()[-200:] or "ssh transport failure")
        if retcode:
            raise SystemExit(f"could not pin the source tree at {path}: {str(err).strip()[-400:]}")
        return path

    def __program(
        self,
        path: str,
        *,
        image: Image,
        results: str,
        prefix: str,
        environment: str,
        stamp: str,
        digest: str = "",
        script: str = "",
    ) -> str:
        """The shell that builds one snapshot, in the order the phases have to happen.

        A host-side lock serializes publication and reuse. Build into a private directory,
        verify its source, then link disjoint live paths and publish atomically. A failed build
        never becomes a completed snapshot; an older incomplete tree refuses for inspection.
        Reuse verifies the stamp and source before adding this dispatch's live paths.
        """
        # The whole build sits inside one `if` rather than behind an early `exit`, because the
        # program runs in a login shell, and a login shell's `exit` runs `.bash_logout`, whose
        # `clear_console` fails without a terminal and under `set -e` becomes the shell's own
        # status: a key already pinned then read as a failed pin. Measured on gold 2026-09-04.
        verify = image.verified(digest, results)
        lines = [
            "set -euo pipefail",
            f"mb_root={shlex.quote(self.root)}",
            f"mb_digest={shlex.quote(digest)}",
            f"mb_final={shlex.quote(path)}",
            f"mkdir -p {shlex.quote(self.base)}",
            f"exec 9>{shlex.quote(self.base + '/.pin.lock')}",
            "flock -x 9",
            'mb_snap="$mb_final"',
            "mb_wrapper=''",
            'trap \'if [ -n "${mb_wrapper:-}" ]; then rm -f -- "$mb_wrapper"; fi; '
            'if [ -n "${mb_snap:-}" ] && [ "$mb_snap" != "$mb_final" ]; '
            'then rm -rf -- "$mb_snap"; fi\' EXIT',
            f'if [ ! -f "$mb_snap/{STAMP}" ]; then',
            'if [ -e "$mb_final" ]; then '
            'echo "mainboard: incomplete snapshot requires inspection: $mb_final" >&2; false; fi',
            f"mb_snap=$(mktemp -d {shlex.quote(self.base + '/.pending.XXXXXXXXXX')})",
            'cd "$mb_root"',
            image.copied(self.root),
            self.__generated(),
            image.filled(),
            *self.__environment(prefix, environment),
            verify,
            *image.linked(),
            *self.__results(results),
            *self.__script(script),
            f"printf '%s' {shlex.quote(stamp)} > \"$mb_snap/{STAMP}\"",
            'mv -T -- "$mb_snap" "$mb_final"',
            'mb_snap="$mb_final"',
            "else",
            f"printf '%s' {shlex.quote(stamp)} | cmp -s - \"$mb_snap/{STAMP}\" || "
            '{ echo "mainboard: snapshot stamp mismatch" >&2; false; }',
            verify,
            # Outside the stamp, because the tree is keyed on the source and a results path is
            # not part of it: two batches off one commit share a snapshot and declare different
            # results paths, and the second one used to get no link at all, so every pull failed
            # on a path that did not exist while its receipts sat inside the snapshot (stream
            # gh200-closure reusing gh200-directed-tree's tree, 2026-09-05). Every dispatch now
            # links its own, which is idempotent for the one that pinned the tree in the first
            # place.
            *image.linked(),
            *self.__results(results),
            *self.__script(script),
            "fi",
        ]
        return "\n".join(line for line in lines if line)

    def __script(self, staged: str) -> list[str]:
        """Freeze and verify the requested wrapper before submission, including source reuse."""
        if not staged:
            return []
        where = self.script(staged)
        digest = PurePosixPath(staged).stem.removeprefix("job-")
        quoted = shlex.quote(where)
        check = (
            '[ "${mb_hash%% *}" = ' + shlex.quote(digest) + " ] || "
            '{ echo "mainboard: wrapper digest mismatch" >&2; false; }'
        )
        return [
            f'[ ! -L "$mb_snap/{WRAPPERS}" ] || '
            '{ echo "mainboard: linked wrapper directory" >&2; false; }',
            f'mkdir -p "$mb_snap/{WRAPPERS}"',
            f'if [ ! -e "$mb_snap"/{quoted} ] && [ ! -L "$mb_snap"/{quoted} ]; then',
            f'mb_wrapper=$(mktemp "$mb_snap/{WRAPPERS}/.pending.XXXXXXXXXX")',
            f'cp --preserve=mode -- "$mb_root"/{shlex.quote(staged)} "$mb_wrapper"',
            'mb_hash=$(sha256sum -- "$mb_wrapper")',
            check,
            f'mv -T -- "$mb_wrapper" "$mb_snap"/{quoted}',
            "mb_wrapper=''",
            "fi",
            f'[ -f "$mb_snap"/{quoted} ] && [ ! -L "$mb_snap"/{quoted} ] || '
            '{ echo "mainboard: missing or linked wrapper" >&2; false; }',
            f'mb_hash=$(sha256sum -- "$mb_snap"/{quoted})',
            check,
        ]

    def __generated(self) -> str:
        """The lines that rebuild the generated tree as this snapshot's own.

        The environment is the mirror's and the code is the snapshot's, and the generated tree is
        where those two meet, so it is neither copied nor symlinked whole. Its directories down
        to each environment shard are real here, its files are hardlinked in, and everything
        heavy under them, the installed environments, the node modules, the dispatch state and
        the receipts, is a symlink back to the mirror. A directory the image already copied into
        (a closure reaching into the vendored tree) is left as the image made it.

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

    def __environment(self, prefix: str, environment: str) -> list[str]:
        """The lines that point this tree's environment at the immutable prefix it names.

        Inside the stamp, and this is the one line where that matters most: a results path
        belongs to the dispatch and is rewritten every time, while the environment belongs to
        the tree. Repointing it on a later dispatch would move a queued wave into an environment
        nobody dispatched it against, which is the whole fault this addressing exists to end.

        prefix: the built environment's directory on this host, empty for a workspace that
            addresses none.
        environment: the environment the link is written for.
        """
        if not prefix:
            return []
        out = shlex.quote(Project().out_dir)
        where = f'"$mb_snap"/{out}/envs/{shlex.quote(environment)}'
        return [
            f"mkdir -p {where}",
            f"rm -rf {where}/.pixi",
            f"ln -s {shlex.quote(prefix)}/.pixi {where}/.pixi",
        ]

    def __results(self, results: str) -> list[str]:
        """The lines that point the dispatch's declared results path back at the mirror.

        Written on every dispatch rather than once per tree, so they only ever remove a link
        they wrote: a real directory the snapshot copied is cleared once and replaced by the
        link, and a link that is already there is replaced by an identical one. `-n` keeps that
        replacement from being followed into the mirror and writing a link inside it.
        """
        relative = writable(results)
        if not relative:
            return []
        if relative in (".", CLOSURE, STAMP, WRAPPERS) or relative.startswith(WRAPPERS + "/"):
            raise ValueError("a results path must not replace snapshot control files")
        quoted = shlex.quote(relative)
        parent = shlex.quote(str(PurePosixPath(relative).parent))
        return [
            f'mkdir -p "$mb_root"/{quoted} "$mb_snap"/{parent}',
            f'if [ ! -L "$mb_snap"/{quoted} ]; then rm -rf "$mb_snap"/{quoted}; fi',
            f'ln -sfn "$mb_root"/{quoted} "$mb_snap"/{quoted}',
        ]
