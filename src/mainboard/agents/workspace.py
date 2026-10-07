# Every coding agent in the workspace configured alike from `.agents`, and judged here.
#
# `sync` renders each harness's files from the shared declarations, removes the ones no longer
# rendered and makes the links a harness reads `.agents` through. `update` installs a missing
# harness and brings every present one to its latest release. `sections` judges what an agent
# started here would meet: a link git could not make (a Windows account without the privilege
# checks one out as a text file, repaired in place since a safe local action can), a file out of
# step with `.agents`, a server whose command or variables this machine lacks, a harness missing
# or logged out, and the Claude Code memory this workspace keeps.

import os
import re
import shutil
import subprocess
import sys
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ..core.errors import MissionError
from ..core.project import Project
from ..core.section import Section, Verdict
from ..git.process import Git
from .harness import Harness, claude_key
from .source import FOLDER, Server, Source

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from .harness import Which

# The mode git records a symbolic link under in its index.
_LINK_MODE = "120000"

# A value that is an absolute path on some machine: `/…` or `C:\…`.
_ABSOLUTE = re.compile(r"^(?:/|[A-Za-z]:[\\/])")

# The launcher that loads the workspace `.env` itself, under any of its names, so a server it
# starts needs nothing else.
_TOOL = Project().names

# Seconds a harness gets to print its version.
_VERSION_SECONDS = 60

type Junction = Callable[[Path, Path], str]


def _junction(link: Path, target: Path) -> str:
    """Make `link` a Windows directory junction to `target`, answering why not, empty on success.

    A junction needs no privilege, so it stands in for a directory link without Developer Mode.
    """
    done = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=argv of fixed words and two paths since=2026-09-25
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return "" if done.returncode == 0 else (done.stdout + done.stderr).strip()


class Agents:
    """Configure and judge every harness from the workspace's `.agents` folder.

    home: the user's home directory, where each harness keeps its per-user state.
    environment: the variables a harness started here sees, this process's own.
    dotenv: the variables the workspace `.env` defines, which `mainboard run` loads.
    which: finds a program on this machine's PATH, None when it is not there.
    """

    def __init__(
        self,
        root: Path,
        *,
        home: Path,
        environment: Mapping[str, str],
        dotenv: Mapping[str, str],
        which: Which,
        junction: Junction = _junction,
    ) -> None:
        self.root = root
        self.home = home
        self.environment = environment
        self.dotenv = dotenv
        self.which = which
        self.junction = junction

    @classmethod
    def at(cls, root: Path, *, home: Path | None = None) -> Agents:
        """The workspace at `root` as this machine sees it."""
        return cls(
            root,
            home=home or Path.home(),
            environment=os.environ,
            dotenv=_dotenv(root / ".env"),
            which=shutil.which,
        )

    @cached_property
    def source(self) -> Source:
        return Source.read(self.root)

    @cached_property
    def harnesses(self) -> list[Harness]:
        return [harness() for harness in Harness.implementations()]

    def rendered(self) -> dict[str, str]:
        """Every file a harness reads, by workspace-relative path, as `.agents` renders it."""
        return {
            path: text
            for harness in self.harnesses
            for path, text in harness.files(self.source).items()
        }

    def drift(self) -> list[str]:
        """Every rendered file that differs from its rendering, and every stale one."""
        rendered = self.rendered()
        differing = [path for path, text in rendered.items() if _current(self.root / path) != text]
        return sorted([*differing, *self._stale(rendered)])

    def sync(self) -> list[str]:
        """Write every drifted file, remove the stale ones and make the links; the paths changed.

        A rendered path that is a link is replaced, never written through, which would rewrite
        the file it points at. A CLAUDE.md is written only where none exists, since its lines
        beside `@AGENTS.md` are a person's.
        """
        rendered = self.rendered()
        changed = []
        for path, text in rendered.items():
            target = self.root / path
            if _current(target) == text:
                continue
            if target.is_symlink():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode())
            changed.append(path)
        for path in self._stale(rendered):
            (self.root / path).unlink()
            changed.append(path)
        claude = self.root / "CLAUDE.md"
        if not claude.exists():
            claude.write_bytes(b"@AGENTS.md\n")
            changed.append("CLAUDE.md")
        return sorted([*changed, *self._linked(), *self._repaired()])

    def update(self) -> list[Section]:
        """Install every missing harness and bring every present one to its latest release.

        A present harness its channel cannot update was installed some other way, and is
        installed through the channel beside it. Each installer's own output reaches the
        terminal, since a vendor script may ask.
        """
        rows = []
        for harness in self.harnesses:
            before = self._version(harness)
            channel = harness.channel
            argv = (channel.update if before else channel.install)(self.home, self.which)
            done = self._run(argv)
            if done.returncode != 0 and before:
                argv = channel.install(self.home, self.which)
                done = self._run(argv)
            after = self._version(harness)
            rows.append(
                Section(
                    section=f"agents: {harness.name}",
                    verdict=Verdict.PASS if done.returncode == 0 and after else Verdict.FAIL,
                    detail=f"{before or 'absent'} -> {after or 'absent'}"
                    if before != after
                    else f"{after} is current",
                    fix=""
                    if done.returncode == 0
                    else f"{' '.join(argv)} exited {done.returncode}",
                )
            )
        return rows

    def sections(self) -> list[Section]:
        """Every agent check, the links first since every other file is read through them."""
        return [
            self.links(),
            self.instructions(),
            self.files(),
            self.servers(),
            self.memory(),
            *self.installs(),
        ]

    def links(self) -> Section:
        """Whether every link the harnesses read and the workspace tracks is a link.

        A tracked link git could not make is a text file holding its target, rebuilt here as a
        junction (a directory) or a hard link (a file); git is told to stop comparing that path,
        since it would otherwise report the stand-in as a change forever.
        """
        try:
            repaired = self._repaired()
        except OSError as refusal:
            return Section(
                section="agents: links",
                verdict=Verdict.FAIL,
                detail=f"cannot stand in for a link: {refusal}",
                fix="enable Developer Mode, then git checkout -- <path>",
            )
        missing = [
            link
            for harness in self.harnesses
            for link in harness.links
            if not _is_link(self.root / link)
        ]
        if missing:
            return Section(
                section="agents: links",
                verdict=Verdict.WARN,
                detail=f"not links: {', '.join(missing)}",
                fix=f"{Project().name} agents sync",
            )
        detail = (
            f"stood in for {len(repaired)} links git could not make: {', '.join(repaired)}"
            if repaired
            else "every link resolves"
        )
        return Section(section="agents: links", verdict=Verdict.PASS, detail=detail)

    def instructions(self) -> Section:
        """Whether AGENTS.md is there and CLAUDE.md brings it in."""
        if not (self.root / "AGENTS.md").is_file():
            return Section(
                section="agents: instructions",
                verdict=Verdict.FAIL,
                detail="no AGENTS.md, the instructions every harness reads",
                fix="git checkout -- AGENTS.md",
            )
        if "@AGENTS.md" not in _text(self.root / "CLAUDE.md"):
            return Section(
                section="agents: instructions",
                verdict=Verdict.WARN,
                detail="CLAUDE.md does not include @AGENTS.md, so Claude Code never reads it",
                fix="add the line @AGENTS.md to CLAUDE.md",
            )
        return Section(
            section="agents: instructions",
            verdict=Verdict.PASS,
            detail="AGENTS.md, and CLAUDE.md bringing it in",
        )

    def files(self) -> Section:
        """Whether every harness's files say what `.agents` declares."""
        try:
            drifted = self.drift()
        except MissionError as fault:
            return Section(section="agents: files", verdict=Verdict.FAIL, detail=str(fault))
        if drifted:
            return Section(
                section="agents: files",
                verdict=Verdict.WARN,
                detail=f"out of step with {FOLDER}: {', '.join(drifted)}",
                fix=f"{Project().name} agents sync",
            )
        return Section(
            section="agents: files",
            verdict=Verdict.PASS,
            detail=f"{len(self.harnesses)} harnesses render from {FOLDER} unchanged",
        )

    def servers(self) -> Section:
        """Whether every MCP server can start here: its command, variables and paths exist."""
        try:
            servers = self.source.servers
        except MissionError as fault:
            return Section(section="agents: servers", verdict=Verdict.FAIL, detail=str(fault))
        problems = [
            problem for name, server in servers.items() for problem in self._server(name, server)
        ]
        if problems:
            return Section(
                section="agents: servers",
                verdict=Verdict.WARN,
                detail="; ".join(problems),
                fix=(
                    "install the missing commands, define the variables in the agents' "
                    "environment, and point paths at this machine"
                ),
            )
        return Section(
            section="agents: servers",
            verdict=Verdict.PASS,
            detail=f"{len(servers)} servers can start here",
        )

    def memory(self) -> Section:
        """Whether Claude Code's memory for this workspace sits under this workspace's own key."""
        folder = self.home / ".claude" / "projects" / claude_key(str(self.root)) / "memory"
        if folder.is_dir():
            count = sum(1 for item in folder.iterdir() if item.is_file())
            return Section(
                section="agents: memory",
                verdict=Verdict.PASS,
                detail=f"{count} memory files under {folder.parent.name}",
            )
        return Section(
            section="agents: memory",
            verdict=Verdict.WARN,
            detail=f"no Claude Code memory under {folder.parent.name}",
            fix=(
                f"run `{Project().name} host setup --center` from the previous center, "
                "which re-keys it"
            ),
        )

    def installs(self) -> list[Section]:
        """Each harness's release here and whether it holds a login."""
        rows = []
        for harness in self.harnesses:
            version = self._version(harness)
            section = f"agents: {harness.name}"
            if not version:
                rows.append(
                    Section(
                        section=section,
                        verdict=Verdict.WARN,
                        detail=f"{harness.binary} is not installed",
                        fix=f"{Project().name} agents update",
                    )
                )
                continue
            unhooked = (
                f", and runs none of the {len(self.source.hooks)} hook events"
                if self.source.hooks and not harness.hooked
                else ""
            )
            logged = harness.logged_in(self.home, self.environment)
            # A key only `.env` holds reaches the harness when `mainboard run` starts it.
            loaded = not logged and harness.logged_in(self.home, self.dotenv)
            login = (
                ""
                if logged
                else f", logged in by .env, so start it as `{Project().name} run {harness.binary}`"
                if loaded
                else ", not logged in"
            )
            rows.append(
                Section(
                    section=section,
                    verdict=Verdict.PASS if logged or loaded else Verdict.WARN,
                    detail=f"{version} at {self.which(harness.binary)}{login}{unhooked}",
                    fix="" if logged or loaded else harness.login,
                )
            )
        return rows

    def _run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        """`argv` run in the terminal, its program found on PATH (an npm `.cmd` on Windows)."""
        return subprocess.run([self.which(argv[0]) or argv[0], *argv[1:]], check=False)  # ruff:ignore[subprocess-without-shell-equals-true]  reason=argv a channel composed since=2026-10-07

    def _version(self, harness: Harness) -> str:
        """The first line `<binary> --version` prints, empty when the binary is not here."""
        path = self.which(harness.binary)
        if path is None:
            return ""
        try:
            done = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=a found binary and a fixed flag since=2026-10-07
                [path, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_VERSION_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"{path} (no version within {_VERSION_SECONDS} s)"
        return next(iter(done.stdout.splitlines()), "").strip() or path

    def _server(self, name: str, server: Server) -> Iterator[str]:
        """What stops one server from starting here, nothing when it can."""
        if not server.remote and not self.which(server.command):
            yield f"{name}: {server.command} is not on PATH"
        loaded = server.command in _TOOL
        for variable in sorted(server.variables):
            if variable in self.environment or (variable in self.dotenv and loaded):
                continue
            where = (
                "only in .env, which this server's launcher does not load"
                if variable in self.dotenv
                else "undefined"
            )
            yield f"{name}: {variable} is {where}"
        for value in server.env.values():
            if _ABSOLUTE.match(value) and not Path(value).exists():
                yield f"{name}: {value} does not exist on this machine"

    def _stale(self, rendered: Mapping[str, str]) -> list[str]:
        """Files a harness renders that `.agents` no longer declares."""
        return [
            path
            for harness in self.harnesses
            for pattern in harness.managed
            for found in self.root.glob(pattern)
            if (path := found.relative_to(self.root).as_posix()) not in rendered
        ]

    def _linked(self) -> list[str]:
        """Make every harness link that is missing; the links made."""
        missing = {
            link: target
            for harness in self.harnesses
            for link, target in harness.links.items()
            if not _is_link(self.root / link) and not (self.root / link).exists()
        }
        for link, target in missing.items():
            self._link(self.root / link, target)
        return list(missing)

    def _link(self, path: Path, target: str) -> None:
        """Link `path` to `target`, a junction for a directory where Windows refuses a link."""
        folder = (path.parent / target).is_dir()
        try:
            path.symlink_to(target, target_is_directory=folder)
        except OSError:
            if sys.platform != "win32" or not folder:
                raise
            if refusal := self.junction(path, (path.parent / target).resolve()):
                raise OSError(refusal) from None

    def _repaired(self) -> list[str]:
        """Stand in for every tracked link checked out as a text file; the paths repaired."""
        repaired = []
        for path, target in self._flattened():
            self._stand_in(path, target)
            relative = path.relative_to(self.root).as_posix()
            Git(self.root).run("update-index", "--skip-worktree", "--", relative)
            repaired.append(relative)
        return repaired

    def _flattened(self) -> list[tuple[Path, Path]]:
        """Every tracked link checked out as a text file, with the target it names."""
        listing = Git(self.root).run("ls-files", "-s", "-z").stdout
        entries = [entry.split("\t", 1) for entry in listing.split("\0") if "\t" in entry]
        found = []
        for stage, relative in entries:
            path = self.root / relative
            if not stage.startswith(_LINK_MODE) or path.is_symlink() or not path.is_file():
                continue
            written = path.read_text(encoding="utf-8").strip()
            target = (path.parent / PurePosixPath(written)).resolve()
            if target.is_relative_to(self.root.resolve()) and target.exists():
                found.append((path, target))
        return found

    def _stand_in(self, path: Path, target: Path) -> None:
        """Replace the text file at `path` with a junction or a hard link to `target`.

        The text file is set aside rather than deleted until its stand-in exists, so a refused
        link leaves the checkout exactly as git wrote it.
        """
        aside = path.with_name(f"{path.name}.{Project().name}-link")
        aside.unlink(missing_ok=True)
        if not target.is_dir():
            os.link(target, aside)
            aside.replace(path)
            return
        path.replace(aside)
        if refusal := self.junction(path, target):
            aside.replace(path)
            raise OSError(refusal)
        aside.unlink()


def _current(path: Path) -> str:
    """A rendered file as written, empty when it is missing or a link stands in its place."""
    return "" if path.is_symlink() else _text(path)


def _text(path: Path) -> str:
    """A file's bytes as text, newlines untouched, empty when it is missing."""
    try:
        return path.read_bytes().decode()
    except FileNotFoundError:
        return ""


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _dotenv(path: Path) -> dict[str, str]:
    """The variables a `.env` file defines, by name; values are never read back out."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    pairs = (line.lstrip().removeprefix("export ").partition("=") for line in text.splitlines())
    return {
        name.strip(): value for name, sign, value in pairs if sign and not name.startswith("#")
    }
