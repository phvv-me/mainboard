# Mirror a workspace onto a host that has no rsync: the exact file set rsync would send, listed
# by rsync itself without sending any of it, compared against what the host already holds, then
# streamed through `tar` over one ssh channel and unpacked by the host's own tar. Windows ships
# bsdtar and its cmd.exe login shell runs it as it is, which is what makes this route need
# nothing on the host that Windows does not bring.
#
# Nothing here prunes. rsync's `--delete` retires a file the workspace dropped; this route only
# adds or replaces, so a stale file on such a host stays until the tree there is remade.

import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=tar/ssh argv built from typed fields, not untrusted input since=2026-09-11
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING

from ..core.errors import MissionError
from .shared import logger
from .shells import WindowsShell, quoted
from .sync import Rsync, binary, rsync_argv
from .transport import HostUnreachable, is_transport_failure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..context.plan import ExecutionPlan
    from .transport import SshTransport

# One `path\tsize\tmtime` line per file the host holds, unix seconds, forward slashes, relative
# to the workspace root; the same size-and-mtime quick check rsync skips unchanged files by.
_REMOTE_LISTING = """
$root = {root}
$prefix = (Resolve-Path -LiteralPath $root).Path.TrimEnd('\\') + '\\'
function Describe($file) {{
  $relative = $file.FullName.Substring($prefix.Length) -replace '\\\\', '/'
  $written = ([DateTimeOffset]$file.LastWriteTimeUtc).ToUnixTimeSeconds()
  "$relative`t$($file.Length)`t$written"
}}
foreach ($top in @({deep})) {{
  $directory = Join-Path $root $top
  if (Test-Path -LiteralPath $directory -PathType Container) {{
    Get-ChildItem -LiteralPath $directory -Recurse -File -Force | ForEach-Object {{ Describe $_ }}
  }}
}}
foreach ($flat in @({shallow})) {{
  $directory = Join-Path $root $flat
  if (Test-Path -LiteralPath $directory -PathType Container) {{
    Get-ChildItem -LiteralPath $directory -File -Force | ForEach-Object {{ Describe $_ }}
  }}
}}
Get-ChildItem -LiteralPath $root -File -Force | ForEach-Object {{ Describe $_ }}
"""


class Tarball:
    """The rsync-less mirror: list with rsync, diff against the host, stream with tar.

    workspace: the local workspace root every path is relative to.
    ssh: the bounded SSH policy the stream and the listings ride.
    """

    def __init__(self, workspace: Path, ssh: SshTransport) -> None:
        self.workspace = workspace
        self.ssh = ssh

    def listing(
        self,
        paths: Sequence[str],
        *,
        flags: Rsync,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        hide: Sequence[str] = (),
        filters: Sequence[str] = (),
    ) -> list[str]:
        """The workspace-relative files rsync would send under these rules, off its own dry run.

        rsync decides the set so the two mirror routes can never disagree about what ships;
        directories are dropped since tar makes them on the way.
        """
        with TemporaryDirectory(prefix=".mainboard-listing-") as void:
            args = rsync_argv(
                flags,
                paths,
                include=include,
                exclude=exclude,
                hide=hide,
                filters=filters,
                extra=["--dry-run", "--out-format=%n"],
            )
            listed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=rsync argv built from typed fields since=2026-09-11
                [binary(mirror=False), *args, f"{void}/"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                check=True,
            )
        return [line for line in listed.stdout.splitlines() if line and not line.endswith("/")]

    def held(self, shell: WindowsShell, files: Sequence[str]) -> dict[str, tuple[int, int]]:
        """What the host already holds of `files`' directories, as `path -> (size, mtime)`.

        Every top-level directory a file lives under is listed whole, except the generated
        directory, whose environments run to tens of thousands of files nothing here ships:
        under it only the directories actually holding a shipped file are read, and flat.
        """
        deep = sorted({Path(f).parts[0] for f in files if Path(f).parts[0] != ".mainboard"})
        shallow = sorted(
            {Path(f).parent.as_posix() for f in files if Path(f).parts[0] == ".mainboard"}
        )
        script = _REMOTE_LISTING.format(
            root=quoted(shell.root),
            deep=", ".join(quoted(d) for d in deep) or "''",
            shallow=", ".join(quoted(d) for d in shallow) or "''",
        )
        listed = shell.run(script)
        held: dict[str, tuple[int, int]] = {}
        for line in listed.splitlines():
            path, _, rest = line.partition("\t")
            size, _, written = rest.partition("\t")
            if size.isdigit() and written.lstrip("-").isdigit():
                held[path] = (int(size), int(written))
        return held

    def pending(self, files: Sequence[str], held: dict[str, tuple[int, int]]) -> list[str]:
        """The files whose size or mtime differs from what the host holds, or that it lacks."""
        changed: list[str] = []
        for file in files:
            try:
                stat = os.stat(self.workspace / file)
            except FileNotFoundError:
                logger.warning("skipping %s, listed by rsync but gone before the stream", file)
                continue
            if held.get(file) != (stat.st_size, int(stat.st_mtime)):
                changed.append(file)
        return changed

    def stream(self, files: Sequence[str], *, host: str, root: str) -> None:
        """Pack `files` from the workspace and unpack them under `root` on `host`, one channel.

        Links are followed, since the host cannot make them and the vendored tree is links by
        construction; a link to nothing is skipped with a warning rather than ending the mirror.
        """
        destination = self.ssh.destination(host)
        unpack = f'tar -xzf - -C "{root}"'
        with NamedTemporaryFile("w", encoding="utf-8", suffix=".files") as names:
            names.write("\0".join(files))
            names.flush()
            pack = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=tar argv built from typed fields since=2026-09-11
                [
                    "tar",
                    "--create",
                    "--gzip",
                    "--dereference",
                    "--no-recursion",
                    "--null",
                    "--verbatim-files-from",
                    "--ignore-failed-read",
                    f"--files-from={names.name}",
                    "--file=-",
                ],
                cwd=self.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            unpacking = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=ssh argv built from typed fields since=2026-09-11
                ["ssh", *self.ssh.options, destination, unpack],
                stdin=pack.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if pack.stdout is not None:
                pack.stdout.close()
            _, unpack_err = unpacking.communicate()
            _, pack_err = pack.communicate()
        if pack_err:
            logger.warning("tar reported: %s", pack_err.decode("utf-8", "replace").strip()[-400:])
        if is_transport_failure(unpacking.returncode, unpack_err):
            raise HostUnreachable(f"mirror to {host!r} failed: {unpack_err.strip()[-200:]}")
        if unpacking.returncode:
            raise MissionError(
                f"{host!r} could not unpack the mirror: {unpack_err.strip()[-400:] or 'no detail'}"
            )

    def mirror(
        self,
        plan: ExecutionPlan,
        root: str,
        *,
        paths: Sequence[str],
        include: Sequence[str],
        exclude: Sequence[str],
        hide: Sequence[str],
        filters: Sequence[str],
        vendored: str = "",
    ) -> list[str]:
        """Bring `root` on `plan.host` up to date with the workspace and answer what was listed.

        paths, include, exclude, hide, filters: the rsync rules the mirror would run with.
        vendored: the vendored-dependency tree to ship with its links followed, empty for none.
        """
        files = self.listing(
            paths,
            flags=Rsync.RECURSIVE | Rsync.LINKS | Rsync.RELATIVE,
            include=include,
            exclude=exclude,
            hide=hide,
            filters=filters,
        )
        if vendored:
            files += self.listing(
                [vendored], flags=Rsync.RECURSIVE | Rsync.COPY_LINKS | Rsync.RELATIVE
            )
        with WindowsShell(plan, root, ssh=self.ssh) as shell:
            shell.run(f"New-Item -ItemType Directory -Force -Path {quoted(root)} | Out-Null")
            held = self.held(shell, files)
        pending = self.pending(files, held)
        logger.info(
            "mirroring %d of %d file(s) to %s:%s", len(pending), len(files), plan.host, root
        )
        if pending:
            self.stream(pending, host=plan.host, root=root)
        return files
