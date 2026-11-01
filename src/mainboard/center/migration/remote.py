"""The standard-library agent `host setup --center` runs on the destination, sent over SSH stdin.

A machine about to become the center, macOS or Linux, has no Mainboard yet. Everything done
there before the tool is installed is therefore one of these functions, run by a Python that uv
provides and fed through ssh's stdin, so the same code clones, places files and signs in on
either platform. Each answers with plain JSON,
and nothing here ever prints or returns the content it was handed, since some of it is secret.
"""

import getpass
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import IO

from ...probe.census import Census, Json

# The name a replaced file keeps the destination's own copy under, written once.
_BACKUP = ".migrate-backup"

# The anchors a shipped path is relative to: the new workspace root, or the user's home.
_ANCHORS = ("root", "home")

# How long one git or gh command may take: a clone of a large tree is minutes, never hours.
_SECONDS = 3600.0

# The mode a secret is written with: readable and writable by its owner alone, which ssh insists
# on for a private key.
_PRIVATE = stat.S_IRUSR | stat.S_IWUSR


def where(root: str) -> dict[str, Json]:
    """Root, home and user in this machine's spelling; `root` may be relative or use `~`."""
    return {
        "root": str(Path(root).expanduser().absolute()),
        "home": str(Path.home()),
        "user": getpass.getuser(),
    }


def census(root: str) -> dict[str, Json]:
    """This machine's census, the filesystem measured where `root` lives."""
    return Census().survey(root)


def inventory(root: str, parcels: Sequence[Mapping[str, str]]) -> list[Json]:
    """The parcels this machine does not already hold byte for byte, by key.

    A parcel names its anchor, its relative path and a fingerprint: `sha256:<hex>` compares
    content, `mtime:<seconds>:<size>` compares the stamp a shipped tree keeps, which is what lets
    a second migration skip gigabytes of receipts it already moved.

    parcels: each `{"key", "anchor", "path", "fingerprint"}`.
    """
    placement = Placement(root)
    return [
        parcel["key"]
        for parcel in parcels
        if placement.fingerprint(
            placement.target(anchor=parcel["anchor"], relative=parcel["path"]),
            parcel["fingerprint"],
        )
        != parcel["fingerprint"]
    ]


def place(root: str, secret: Sequence[str]) -> dict[str, Json]:
    """Write every file of the tar stream on stdin, keeping a differing original once.

    Each member is named `<anchor>/<relative path>`. A file already holding different bytes is
    moved aside to `<name>.migrate-backup` the first time, so whatever the destination had is
    never lost, and replaced; one holding the same bytes is left alone.

    secret: the member names that are credentials, written readable by this user alone.
    """
    placement = Placement(root)
    private = set(secret)
    written = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as stream:
        for member in stream:
            anchor, _, relative = member.name.partition("/")
            target = placement.target(anchor=anchor, relative=relative)
            if (source := stream.extractfile(member)) is not None:
                written += placement.write(
                    target, source, mtime=member.mtime, private=member.name in private
                )
    return {"written": written}


def ssh(blocks: Sequence[str], *, known: Sequence[str]) -> dict[str, Json]:
    """Add the ssh host blocks and known host lines this machine lacks, keeping all of its own.

    A block is added only when no `Host` line here names its patterns, so a block this machine's
    user adapted stays as they left it, and a known host line only when no line here equals it.

    blocks: whole `Host` blocks, each starting at its `Host` line.
    known: known_hosts lines.
    """
    files = SshFiles(Path.home() / ".ssh")
    config = files.read("config")
    declared = {files.hosts(line) for line in config.splitlines()}
    added = [block for block in blocks if files.hosts(block.splitlines()[0]) not in declared]
    hosts = files.read("known_hosts")
    held = {line.strip() for line in hosts.splitlines()}
    keys = [line for line in dict.fromkeys(known) if line not in held]
    files.append("config", held=config, text="\n\n".join(added), gap="\n")
    files.append("known_hosts", held=hosts, text="\n".join(keys))
    return {"blocks": len(added), "keys": len(keys)}


def clone(
    root: str, *, url: str, branch: str, commit: str, submodules: Sequence[Sequence[str]]
) -> list[dict[str, str]]:
    """Clone the workspace at `commit` and every owned submodule at the pointer it records.

    A root that is already this repository is fetched and moved to `commit` instead, unless it
    holds work `commit` does not contain, which is kept and named rather than reset away. A root
    that holds anything else is refused, since nothing here may overwrite a directory it did not
    make.

    url: the root repository's remote.
    branch: the branch the center is on, empty for a detached HEAD.
    submodules: each owned submodule as `[parent, path, url]`, parents first, paths relative.
    """
    checkout = Checkout(Path(root).expanduser(), url)
    if not _run(("git", "lfs", "version"))[0]:
        # Large files check out as files only once git-lfs's filters are installed.
        _run(("git", "lfs", "install", "--skip-repo"))
    steps = [checkout.at(branch=branch, commit=commit)]
    if steps[0]["outcome"] != "done":
        return steps
    return [*steps, *(checkout.submodule(parent, path=path) for parent, path, _ in submodules)]


def login(token: str) -> dict[str, Json]:
    """Sign gh in with `token` and make it git's credential for GitHub over https.

    token: the GitHub token, read by gh from its stdin and never written anywhere else.
    """
    status, said = _run(("gh", "auth", "login", "--with-token"), stdin=token)
    if status:
        return {"signed": False, "detail": said}
    status, said = _run(("gh", "auth", "setup-git"))
    return {"signed": True, "detail": said if status else ""}


def merge(path: str, *, key: str, entries: Mapping[str, Json]) -> dict[str, Json]:
    """Merge `entries` into the object at `key` of a JSON file under home, created when absent.

    What moves a tool's per-project settings to the new workspace path without replacing the
    file a tool on this machine already keeps its own account in.

    path: the file, relative to home.
    """
    target = Path.home() / path
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        document = {}
    held = document.setdefault(key, {})
    changed = [name for name, value in entries.items() if held.get(name) != value]
    held.update(entries)
    if changed:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
        staged.chmod(_PRIVATE)
        staged.replace(target)
    return {"changed": len(changed)}


class Checkout:
    """The root repository on this machine, cloned when it is not here yet.

    base: where the root repository lives.
    url: the root repository's remote.
    """

    @staticmethod
    def _step(repo: str, found: tuple[int, str], *, done: str) -> dict[str, str]:
        """One repository's row: done with `done`, or failed with git's last words."""
        status, said = found
        if status:
            spoken = [line for line in said.splitlines() if line.strip()]
            return {"repo": repo, "outcome": "failed", "detail": spoken[-1] if spoken else said}
        return {"repo": repo, "outcome": "done", "detail": done}

    def __init__(self, base: Path, url: str) -> None:
        self.base = base
        self.url = url

    def _cloned(self) -> dict[str, str] | None:
        """Why the url could not be cloned at the base, which must be empty; None once it is."""
        if self.base.exists() and any(self.base.iterdir()):
            return {
                "repo": ".",
                "outcome": "held",
                "detail": f"{self.base} is not empty; choose --root",
            }
        found = _run(("git", "clone", "--no-checkout", "--quiet", self.url, str(self.base)))
        return {"repo": ".", "outcome": "failed", "detail": found[1]} if found[0] else None

    def _git(self, *arguments: str) -> tuple[int, str]:
        """`git -C base arguments`, answered as status and output."""
        return _run(("git", "-C", str(self.base), *arguments))

    def _unfit(self, commit: str) -> dict[str, str] | None:
        """Why the repository at the base may not move to `commit`; None when it may.

        It is fetched first. It may not move when it is another repository, or holds commits
        `commit` does not.
        """
        held = {"repo": ".", "outcome": "held"}
        if self._git("remote", "get-url", "origin")[1].strip() != self.url:
            return {**held, "detail": f"{self.base} is another repository"}
        self._git("fetch", "--quiet", "origin")
        head = self._git("rev-parse", "HEAD")[1].strip()
        if head != commit and self._git("merge-base", "--is-ancestor", head, commit)[0]:
            return {
                **held,
                "detail": f"{self.base} holds commits {commit[:12]} does not; left as it is",
            }
        return None

    def at(self, *, branch: str, commit: str) -> dict[str, str]:
        """Put the root repository at `commit`, cloning it first when it is not here yet."""
        refused = self._unfit(commit) if (self.base / ".git").exists() else self._cloned()
        if refused is not None:
            return refused
        moved = (
            self._git("checkout", "-q", "-B", branch, commit)
            if branch
            else self._git("checkout", "-q", "--detach", commit)
        )
        if branch and not moved[0]:
            self._git("branch", "-q", f"--set-upstream-to=origin/{branch}")
        return self._step(".", moved, done=f"{branch or 'detached'} at {commit[:12]}")

    def submodule(self, parent: str, *, path: str) -> dict[str, str]:
        """The submodule at `path` of the repository at `parent`, moved to its recorded pointer."""
        found = _run(
            ("git", "-C", str(self.base / parent), "submodule", "update", "--init", "--", path)
        )
        return self._step(PurePosixPath(parent, path).as_posix(), found, done="at its pointer")


class Placement:
    """Where shipped files land on this machine, and how one lands there.

    root: the new workspace root, relative or using `~`.
    """

    @staticmethod
    def fingerprint(path: Path, kind: str) -> str:
        """`path`'s fingerprint in the same kind `kind` is, empty when there is no file."""
        try:
            status = path.stat()
        except FileNotFoundError:
            return ""
        if kind.startswith("sha256:"):
            with path.open("rb") as held:
                return "sha256:" + hashlib.file_digest(held, "sha256").hexdigest()
        return f"mtime:{int(status.st_mtime)}:{status.st_size}"

    @staticmethod
    def write(target: Path, source: IO[bytes], *, mtime: float, private: bool) -> int:
        """Publish one member at `target`, answering 1 when bytes changed, 0 when they had not."""
        data = source.read()
        with suppress(FileNotFoundError):
            if target.read_bytes() == data:
                return 0
            backup = target.with_name(target.name + _BACKUP)
            if not backup.exists():
                target.replace(backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(data)
        if private:
            staged.chmod(_PRIVATE)
        os.utime(staged, (mtime, mtime))
        staged.replace(target)
        return 1

    def __init__(self, root: str) -> None:
        self.root = root

    def target(self, *, anchor: str, relative: str) -> Path:
        """Where one shipped path lands, refused when it would escape its anchor."""
        if anchor not in _ANCHORS:
            raise ValueError(f"unknown anchor {anchor!r}")
        base = (Path(self.root).expanduser() if anchor == "root" else Path.home()).absolute()
        target = base.joinpath(*PurePosixPath(relative).parts)
        if ".." in PurePosixPath(relative).parts or not target.is_relative_to(base):
            raise ValueError(f"shipped path escapes its anchor: {relative}")
        return target


class SshFiles:
    """The ssh config and known hosts of this machine's user, which only ever gain lines.

    folder: the user's `.ssh` directory.
    """

    @staticmethod
    def hosts(line: str) -> tuple[str, ...]:
        """The patterns a `Host` line names, none for any other line."""
        words = line.split()
        return tuple(words[1:]) if words and words[0].lower() == "host" else ()

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def append(self, name: str, *, held: str, text: str, gap: str = "") -> None:
        """Add `text` as whole lines after `held`, what file `name` holds, `gap` apart from it.

        Appending keeps the file itself, and with it the owner and permissions ssh checks.
        """
        if not text:
            return
        lead = ("\n" if not held.endswith("\n") else "") + gap if held else ""
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / name).open("a", encoding="utf-8", newline="") as sink:
            sink.write(lead + text + "\n")

    def read(self, name: str) -> str:
        """File `name`'s text, empty when there is no such file."""
        try:
            return (self.folder / name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""


def _run(command: Sequence[str], stdin: str = "") -> tuple[int, str]:
    """Run `command`, answering its status and joined output, 127 when it cannot start."""
    try:
        done = subprocess.run(
            list(command),
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SECONDS,
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except OSError, subprocess.SubprocessError:
        return 127, f"{command[0]} could not run"
    return done.returncode, done.stdout + done.stderr
