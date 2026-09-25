# The AI agents' configuration in this workspace, read the way each agent will read it here.
#
# Claude Code, Codex and opencode are wired into the workspace through tracked files: AGENTS.md
# and the CLAUDE.md that includes it, `.claude` linked to `.agents`, `.codex/config.toml` linked
# to `.agents/codex.toml`, and the MCP servers of `.mcp.json` and `opencode.json`. Every one of
# those can be present and still broken on a given center: a link checked out as a text file on
# a Windows account that cannot make links, a server whose command is not on this machine's PATH,
# a variable nothing defines, a path that exists only on the machine the file was written on.
# Each is found here; the links, the one thing a safe local action can fix, are repaired in place.

import json
import os
import re
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ..core.section import Section, Verdict
from ..git.process import Git
from .state import claude_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from ..probe.census import Json

# The mode git records a symbolic link under in its index.
_LINK_MODE = "120000"

# How a variable is referenced in each agent's server configuration.
_CLAUDE_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}")
_OPENCODE_VARIABLE = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")

# A value that is an absolute path on some machine: `/…` or `C:\…`.
_ABSOLUTE = re.compile(r"^(?:/|[A-Za-z]:[\\/])")

# The launcher that loads the workspace `.env` itself, so a server it starts needs nothing else.
_TOOL = "mainboard"

type Junction = Callable[[Path, Path], str]


def junction(link: Path, target: Path) -> str:
    """Make `link` a Windows directory junction to `target`, answering why not, empty on success.

    A junction needs no privilege, so it stands in for a directory link without Developer Mode.
    """
    done = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=argv of fixed words and two paths since=2026-09-25
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return "" if done.returncode == 0 else (done.stdout + done.stderr).strip()


class Agents:
    """Validate every agent's workspace configuration here, repairing flattened links.

    home: the user's home directory, where each agent keeps its per-user state.
    environment: the variables an agent started here sees, this process's own.
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
        which: Callable[[str], str | None],
        junction: Junction = junction,
    ) -> None:
        self.root = root
        self.home = home
        self.environment = environment
        self.dotenv = dotenv
        self.which = which
        self.junction = junction

    def sections(self) -> list[Section]:
        """Every agent check, the links first since every other file is read through them."""
        return [
            self.links(),
            self.instructions(),
            *self.servers(),
            self.memory(),
            self.logins(),
        ]

    def links(self) -> Section:
        """Whether every link the workspace tracks is a link, repairing the ones that are not.

        A link git could not make is a text file holding its target. A directory target is
        rebuilt as a junction and a file target as a hard link, and git is told to stop comparing
        that path, since it would otherwise report the stand-in as a change forever.
        """
        repaired: list[str] = []
        broken: list[str] = []
        for path, target in self._flattened():
            relative = path.relative_to(self.root).as_posix()
            try:
                self._stand_in(path, target)
            except OSError as refusal:
                broken.append(f"{relative} ({refusal})")
                continue
            Git(self.root).run("update-index", "--skip-worktree", "--", relative)
            repaired.append(relative)
        if broken:
            return Section(
                section="agents: links",
                verdict=Verdict.FAIL,
                detail=f"cannot stand in for {', '.join(broken)}",
                fix="enable Developer Mode, then git checkout -- <path>",
            )
        detail = (
            f"stood in for {len(repaired)} links git could not make: {', '.join(repaired)}"
            if repaired
            else "every tracked link resolves"
        )
        return Section(section="agents: links", verdict=Verdict.PASS, detail=detail)

    def instructions(self) -> Section:
        """Whether AGENTS.md is there, CLAUDE.md brings it in, and `.claude` reaches `.agents`."""
        missing = [name for name in ("AGENTS.md", "CLAUDE.md") if not (self.root / name).is_file()]
        if missing:
            return Section(
                section="agents: instructions",
                verdict=Verdict.FAIL,
                detail=f"missing {', '.join(missing)}",
                fix="git checkout -- AGENTS.md CLAUDE.md",
            )
        notes = []
        if "@AGENTS.md" not in (self.root / "CLAUDE.md").read_text(encoding="utf-8"):
            notes.append("CLAUDE.md does not include @AGENTS.md, so Claude Code never reads it")
        claude = self.root / ".claude"
        if claude.exists() and not (claude / "settings.json").is_file():
            notes.append(".claude does not reach .agents/settings.json")
        return Section(
            section="agents: instructions",
            verdict=Verdict.WARN if notes else Verdict.PASS,
            detail="; ".join(notes) or "AGENTS.md, CLAUDE.md and .claude -> .agents in place",
        )

    def servers(self) -> list[Section]:
        """Each agent's MCP servers: the command runs here, every variable and path exists."""
        return [
            self._servers(
                ".mcp.json", lambda text: json.loads(text).get("mcpServers", {}), _CLAUDE_VARIABLE
            ),
            self._servers(
                "opencode.json", lambda text: json.loads(text).get("mcp", {}), _OPENCODE_VARIABLE
            ),
            self._servers(
                ".codex/config.toml",
                lambda text: tomllib.loads(text).get("mcp_servers", {}),
                _CLAUDE_VARIABLE,
            ),
        ]

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
            fix="run `mainboard center migrate` from the previous center, which re-keys it",
        )

    def logins(self) -> Section:
        """Which agents hold a login this machine can use, by the files each keeps it in."""
        held = {
            "codex": self.home / ".codex" / "auth.json",
            "opencode": self.home / ".local" / "share" / "opencode" / "auth.json",
        }
        absent = [name for name, path in held.items() if not path.is_file()]
        if absent:
            return Section(
                section="agents: logins",
                verdict=Verdict.WARN,
                detail=f"no login on file for {', '.join(absent)}",
                fix="; ".join(
                    f"{name} login" if name == "codex" else "opencode auth login"
                    for name in absent
                ),
            )
        return Section(
            section="agents: logins",
            verdict=Verdict.PASS,
            detail="codex and opencode logins on file",
        )

    def _servers(
        self, name: str, read: Callable[[str], Mapping[str, Json]], variable: re.Pattern[str]
    ) -> Section:
        """One configuration file's servers, judged."""
        try:
            servers = read((self.root / name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Section(section=f"agents: {name}", verdict=Verdict.PASS, detail="not declared")
        except ValueError as fault:
            return Section(
                section=f"agents: {name}", verdict=Verdict.FAIL, detail=f"does not parse: {fault}"
            )
        problems = [
            problem
            for server, entry in servers.items()
            for problem in self._server(server, entry, variable)
        ]
        if problems:
            return Section(
                section=f"agents: {name}",
                verdict=Verdict.WARN,
                detail="; ".join(problems),
                fix=(
                    "install the missing commands, define the variables in the agent's "
                    "environment, and point paths at this machine"
                ),
            )
        return Section(
            section=f"agents: {name}",
            verdict=Verdict.PASS,
            detail=f"{len(servers)} servers can start here",
        )

    def _server(self, server: str, entry: Json, variable: re.Pattern[str]) -> Iterator[str]:
        """What stops one server from starting here, nothing when it can."""
        if not isinstance(entry, dict):
            return
        command = entry.get("command")
        program = command[0] if isinstance(command, list) and command else command
        if isinstance(program, str) and program and not self.which(program):
            yield f"{server}: {program} is not on PATH"
        for name in sorted(set(variable.findall(json.dumps(entry)))):
            if name in self.environment or (name in self.dotenv and program == _TOOL):
                continue
            where = (
                "only in .env, which this server's launcher does not load"
                if name in self.dotenv
                else "undefined"
            )
            yield f"{server}: {name} is {where}"
        table = entry.get("env", entry.get("environment", {}))
        for value in table.values() if isinstance(table, dict) else ():
            if isinstance(value, str) and _ABSOLUTE.match(value) and not Path(value).exists():
                yield f"{server}: {value} does not exist on this machine"

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
        aside = path.with_name(f"{path.name}.mainboard-link")
        aside.unlink(missing_ok=True)
        if not target.is_dir():
            os.link(target, aside)
            aside.replace(path)
            return
        path.replace(aside)
        refusal = self.junction(path, target)
        if refusal:
            aside.replace(path)
            raise OSError(refusal)
        aside.unlink()


def dotenv(path: Path) -> dict[str, str]:
    """The variables a `.env` file defines, by name; values are never read back out."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    pairs = (line.lstrip().removeprefix("export ").partition("=") for line in text.splitlines())
    return {
        name.strip(): value for name, sign, value in pairs if sign and not name.startswith("#")
    }
