"""The standard-library agent `center migrate` runs on the destination, sent over SSH stdin.

A machine about to become the center has no Mainboard yet, often no Bash, and may
be Windows, macOS or Linux. Everything done there before the tool is installed is therefore
one of these functions, run by a Python that uv provides and fed through ssh's stdin, so the
same code clones, places files and signs in on every platform. Each answers with plain JSON,
and nothing here ever prints or returns the content it was handed, since some of it is secret.
"""

import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import IO

from ..probe.census import Census, Json

# The name a replaced file keeps the destination's own copy under, written once.
BACKUP = ".migrate-backup"

# The anchors a shipped path is relative to: the new workspace root, or the user's home.
_ANCHORS = ("root", "home")

# How long one git or gh command may take: a clone of a large tree is minutes, never hours.
_SECONDS = 3600.0

# The mode a secret is written with: readable and writable by its owner alone.
_PRIVATE = stat.S_IRUSR | stat.S_IWUSR


def where(root: str) -> dict[str, Json]:
    """The workspace root and the home directory as this machine spells them, both absolute.

    root: the root as given, `~` and relative forms allowed.
    """
    return {
        "root": str(Path(root).expanduser().absolute()),
        "home": str(Path.home()),
        "separator": os.sep,
    }


def census(root: str) -> dict[str, Json]:
    """This machine's census, the filesystem measured where `root` lives."""
    return Census().survey(root)


def inventory(root: str, parcels: Sequence[Mapping[str, str]]) -> list[Json]:
    """The parcels this machine does not already hold byte for byte, by key.

    A parcel names its anchor, its relative path and a fingerprint: `sha256:<hex>` compares
    content, `mtime:<seconds>:<size>` compares the stamp a shipped tree keeps, which is what lets
    a second migration skip gigabytes of receipts it already moved.

    root: the workspace root.
    parcels: each `{"key", "anchor", "path", "fingerprint"}`.
    """
    return [
        parcel["key"]
        for parcel in parcels
        if _fingerprint(_target(root, parcel["anchor"], parcel["path"]), parcel["fingerprint"])
        != parcel["fingerprint"]
    ]


def place(root: str, secret: Sequence[str]) -> dict[str, Json]:
    """Write every file of the tar stream on stdin, keeping a differing original once.

    Each member is named `<anchor>/<relative path>`. A file already holding different bytes is
    moved aside to `<name>.migrate-backup` the first time, so whatever the destination had is
    never lost, and replaced; one holding the same bytes is left alone.

    root: the workspace root.
    secret: the member names that are credentials, written readable by this user alone.
    """
    private = set(secret)
    written = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as stream:
        for member in stream:
            anchor, _, relative = member.name.partition("/")
            target = _target(root, anchor, relative)
            source = stream.extractfile(member)
            if source is None:
                continue
            written += _write(target, source, mtime=member.mtime, private=member.name in private)
    return {"written": written}


def clone(
    root: str, url: str, branch: str, commit: str, submodules: Sequence[Sequence[str]]
) -> list[dict[str, str]]:
    """Clone the workspace at `commit` and every owned submodule at the pointer it records.

    A root that is already this repository is fetched and moved to `commit` instead, unless it
    holds work `commit` does not contain, which is kept and named rather than reset away. A root
    that holds anything else is refused, since nothing here may overwrite a directory it did not
    make.

    root: where the workspace goes.
    url: the root repository's remote.
    branch: the branch the center is on, empty for a detached HEAD.
    commit: the commit the center's HEAD is on.
    submodules: each owned submodule as `[parent, path]`, parents first, both relative.
    """
    base = Path(root).expanduser()
    _prepare()
    steps = [_checkout(base, url, branch, commit)]
    if steps[0]["outcome"] != "done":
        return steps
    for parent, path in submodules:
        found = _git(base / parent, "submodule", "update", "--init", "--", path)
        steps.append(_step(PurePosixPath(parent, path).as_posix(), found, done="at its pointer"))
    return steps


def login(token: str) -> dict[str, Json]:
    """Sign gh in with `token` and make it git's credential for GitHub over https.

    token: the GitHub token, read by gh from its stdin and never written anywhere else.
    """
    signed = _run(("gh", "auth", "login", "--with-token"), stdin=token)
    if signed[0]:
        return {"signed": False, "detail": signed[1]}
    wired = _run(("gh", "auth", "setup-git"))
    return {"signed": True, "detail": "" if not wired[0] else wired[1]}


def merge(path: str, key: str, entries: Mapping[str, Json]) -> dict[str, Json]:
    """Merge `entries` into the object at `key` of a JSON file under home, created when absent.

    What moves a tool's per-project settings to the new workspace path without replacing the
    file a tool on this machine already keeps its own account in.

    path: the file, relative to home.
    key: the top-level key whose object receives the entries.
    entries: the members to set there.
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
        staged.write_text(json.dumps(document, indent=2), encoding="utf-8")
        _restrict(staged)
        staged.replace(target)
    return {"changed": len(changed)}


def _target(root: str, anchor: str, relative: str) -> Path:
    """Where one shipped path lands, refused when it would escape its anchor."""
    if anchor not in _ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}")
    base = (Path(root).expanduser() if anchor == "root" else Path.home()).absolute()
    target = base.joinpath(*PurePosixPath(relative).parts)
    if ".." in PurePosixPath(relative).parts or not target.is_relative_to(base):
        raise ValueError(f"shipped path escapes its anchor: {relative}")
    return target


def _fingerprint(path: Path, kind: str) -> str:
    """`path`'s fingerprint in the same kind `kind` is, empty when there is no file."""
    try:
        status = path.stat()
    except FileNotFoundError:
        return ""
    if kind.startswith("sha256:"):
        with path.open("rb") as held:
            return "sha256:" + hashlib.file_digest(held, "sha256").hexdigest()
    return f"mtime:{int(status.st_mtime)}:{status.st_size}"


def _write(target: Path, source: IO[bytes], *, mtime: float, private: bool) -> int:
    """Publish one member at `target`, answering 1 when bytes changed and 0 when they had not."""
    data = source.read()
    try:
        if target.read_bytes() == data:
            return 0
    except FileNotFoundError:
        pass
    else:
        backup = target.with_name(target.name + BACKUP)
        if not backup.exists():
            target.replace(backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    staged.write_bytes(data)
    if private:
        _restrict(staged)
    os.utime(staged, (mtime, mtime))
    staged.replace(target)
    return 1


def _restrict(path: Path) -> None:
    """Make `path` readable by this user alone, which ssh insists on for a private key."""
    path.chmod(_PRIVATE)
    if os.name == "nt":
        user = os.environ.get("USERNAME", "")
        _run(("icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"))


def _prepare() -> None:
    """The git settings a clone of this tree needs before its first checkout on this machine.

    Large files check out as files only once git-lfs's filters are installed, and on Windows a
    checkout of this tree's deeper paths needs `core.longpaths` before it starts.
    """
    if not _run(("git", "lfs", "version"))[0]:
        _run(("git", "lfs", "install", "--skip-repo"))
    if os.name == "nt":
        _run(("git", "config", "--global", "core.longpaths", "true"))


def _checkout(base: Path, url: str, branch: str, commit: str) -> dict[str, str]:
    """Put the root repository at `commit`, cloning it first when it is not here yet."""
    if (base / ".git").exists():
        origin = _git(base, "remote", "get-url", "origin")
        if origin[1].strip() != url:
            return {"repo": ".", "outcome": "held", "detail": f"{base} is another repository"}
        _git(base, "fetch", "--quiet", "origin")
        head = _git(base, "rev-parse", "HEAD")[1].strip()
        if head != commit and _git(base, "merge-base", "--is-ancestor", head, commit)[0]:
            return {
                "repo": ".",
                "outcome": "held",
                "detail": f"{base} holds commits {commit[:12]} does not; left as it is",
            }
    elif base.exists() and any(base.iterdir()):
        return {"repo": ".", "outcome": "held", "detail": f"{base} is not empty; choose --root"}
    else:
        found = _run(("git", "clone", "--no-checkout", "--quiet", url, str(base)))
        if found[0]:
            return {"repo": ".", "outcome": "failed", "detail": found[1]}
    moved = (
        _git(base, "checkout", "-q", "-B", branch, commit)
        if branch
        else _git(base, "checkout", "-q", "--detach", commit)
    )
    if branch and not moved[0]:
        _git(base, "branch", "-q", f"--set-upstream-to=origin/{branch}")
    return _step(".", moved, done=f"{branch or 'detached'} at {commit[:12]}")


def _step(repo: str, found: tuple[int, str], *, done: str) -> dict[str, str]:
    """One repository's row: done with `done`, or failed with git's last words."""
    status, said = found
    if status:
        spoken = [line for line in said.splitlines() if line.strip()]
        return {"repo": repo, "outcome": "failed", "detail": spoken[-1] if spoken else said}
    return {"repo": repo, "outcome": "done", "detail": done}


def _git(path: Path, *arguments: str) -> tuple[int, str]:
    """`git -C path arguments`, answered as status and output."""
    return _run(("git", "-C", str(path), *arguments))


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
