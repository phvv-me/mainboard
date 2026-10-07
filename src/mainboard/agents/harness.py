# The coding agents a workspace configures alike, each a harness with its own formats.
#
# A harness knows where its binary comes from and how it is brought to the latest release, which
# files it reads, and how to write the shared `.agents` declarations into them: the MCP servers,
# the hooks it can run, and the subagents. Everything it renders is a whole file it owns, so a
# rendering is compared and written without merging into anything a person edited; what only one
# harness understands is that harness's own file in `.agents`, laid under the rendering.

import json
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import tomlkit
import yaml
from patos import Registry

from ..core.errors import MissionError
from .source import FOLDER, REFERENCE, Server, Source

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from ..probe.census import Json

type Which = Callable[[str], str | None]


class Channel(ABC):
    """Where a harness's binary comes from, and how it is brought to its latest release."""

    @abstractmethod
    def install(self, home: Path, which: Which) -> list[str]: ...

    def update(self, home: Path, which: Which) -> list[str]:
        return self.install(home, which)


class Script(Channel):
    """The vendor's own installer, and the harness's own command to update itself."""

    def __init__(self, *, posix: str, windows: str, update: Sequence[str]) -> None:
        self.posix = posix
        self.windows = windows
        self.updating = update

    def install(self, home: Path, which: Which) -> list[str]:
        if sys.platform == "win32":
            return ["powershell", "-NoProfile", "-Command", self.windows]
        return ["bash", "-c", self.posix]

    def update(self, home: Path, which: Which) -> list[str]:
        return [*self.updating]


class PixiGlobal(Channel):
    """A conda-forge package `pixi global` installs and exposes on PATH."""

    def __init__(self, package: str) -> None:
        self.package = package

    def install(self, home: Path, which: Which) -> list[str]:
        return [_found("pixi", which), "global", "install", self.package]

    def update(self, home: Path, which: Which) -> list[str]:
        return [_found("pixi", which), "global", "update", self.package]


class Npm(Channel):
    """An npm package installed under `~/.local`, whose `bin` the vendor installers share.

    npm writes a Windows prefix's commands into the prefix itself, so there the prefix is
    `~/.local/bin` for the commands to land in the same directory.

    scripts: the packages whose install scripts must run, which npm otherwise skips (its
        native terminal and keychain bindings, say).
    """

    def __init__(self, package: str, *, scripts: Sequence[str] = ()) -> None:
        self.package = package
        self.scripts = scripts

    def install(self, home: Path, which: Which) -> list[str]:
        prefix = home / ".local" / ("bin" if sys.platform == "win32" else "")
        allowed = [f"--allow-scripts={','.join(self.scripts)}"] if self.scripts else []
        return [
            _found("npm", which),
            "install",
            "--global",
            "--prefix",
            str(prefix),
            *allowed,
            f"{self.package}@latest",
        ]


class Harness(Registry, ABC):
    """One coding agent, configured from the workspace's `.agents` folder.

    binary: the command it runs as.
    channel: where the binary comes from and how it updates.
    credentials: files under the home directory that hold a login.
    keys: variables that log it in without one.
    login: how a person logs it in.
    links: workspace paths that must link to another, the target relative to the link's folder.
    managed: glob patterns of every file it renders, so one no longer rendered is removed.
    hooked: whether it runs the shared hooks.
    """

    binary: ClassVar[str]
    channel: ClassVar[Channel]
    credentials: ClassVar[tuple[str, ...]] = ()
    keys: ClassVar[tuple[str, ...]] = ()
    login: ClassVar[str]
    links: ClassVar[dict[str, str]] = {}
    managed: ClassVar[tuple[str, ...]] = ()
    hooked: ClassVar[bool] = True

    @abstractmethod
    def files(self, source: Source) -> dict[str, str]:
        """Every file this harness reads, by workspace-relative path, rendered from `source`."""

    @abstractmethod
    def server(self, name: str, server: Server) -> dict[str, Json]:
        """One server as this harness's configuration spells it."""

    def servers(self, source: Source) -> dict[str, Json]:
        """Every server, each with its overrides for this harness laid over it."""
        return {
            name: self.server(name, server) | server.overrides.get(self.name, {})
            for name, server in source.servers.items()
        }

    def logged_in(self, home: Path, environment: Mapping[str, str]) -> bool:
        return any((home / path).is_file() for path in self.credentials) or any(
            key in environment for key in self.keys
        )


class Claude(Harness):
    """Claude Code, which reads `.agents` itself through the `.claude` link."""

    binary = "claude"
    channel = Script(
        posix="curl -fsSL https://claude.ai/install.sh | bash",
        windows="irm https://claude.ai/install.ps1 | iex",
        update=("claude", "update"),
    )
    credentials = (".claude/.credentials.json",)
    keys = ("ANTHROPIC_API_KEY",)
    login = "claude, then /login"
    links = {".claude": FOLDER}
    managed = (".mcp.json",)

    def files(self, source: Source) -> dict[str, str]:
        return {".mcp.json": _json({"mcpServers": self.servers(source)})}

    def server(self, name: str, server: Server) -> dict[str, Json]:
        if server.remote:
            return _present(type=server.type, url=server.url, headers=server.headers)
        return _present(command=server.command, args=list(server.args), env=server.env)

    def logged_in(self, home: Path, environment: Mapping[str, str]) -> bool:
        """The macOS login lives in the keychain, every other one in its credentials file."""
        if sys.platform != "darwin":
            return super().logged_in(home, environment)
        found = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=argv of fixed words since=2026-10-07
            ["security", "find-generic-password", "-s", "Claude Code-credentials"],
            capture_output=True,
            check=False,
        )
        return found.returncode == 0 or any(key in environment for key in self.keys)


# The hook events Codex runs, by Claude Code's names, which it shares.
_CODEX_EVENTS = frozenset(
    {
        "SessionStart",
        "SessionEnd",
        "UserPromptSubmit",
        "PreToolUse",
        "PermissionRequest",
        "PostToolUse",
        "PreCompact",
        "PostCompact",
        "SubagentStart",
        "SubagentStop",
        "Stop",
    }
)


class Codex(Harness):
    """OpenAI's Codex CLI, reading `.codex/` (trusted projects only) and `.agents/skills`."""

    binary = "codex"
    channel = PixiGlobal("codex")
    credentials = (".codex/auth.json",)
    login = "codex login"
    managed = (".codex/config.toml", ".codex/hooks.json", ".codex/agents/*.toml")

    def files(self, source: Source) -> dict[str, str]:
        config = tomlkit.parse(source.own("codex.toml"))
        config["mcp_servers"] = self.servers(source)
        files = {".codex/config.toml": tomlkit.dumps(config)}
        if hooks := {
            event: [group.spelled() for group in groups]
            for event, groups in source.hooks.items()
            if event in _CODEX_EVENTS
        }:
            files[".codex/hooks.json"] = _json({"hooks": hooks})
        for agent in source.subagents:
            card = tomlkit.document()
            card["name"] = agent.name
            card["description"] = agent.description
            card["developer_instructions"] = tomlkit.string(agent.body, multiline=True)
            files[f".codex/agents/{agent.name}.toml"] = tomlkit.dumps(card)
        return files

    def server(self, name: str, server: Server) -> dict[str, Json]:
        """Codex reads no reference: a variable is forwarded by name, a header from one."""
        if server.type == "sse":
            raise MissionError(
                f"MCP server {name!r}: Codex reaches stdio and streamable HTTP only"
            )
        return self._remote(name, server) if server.remote else self._local(name, server)

    @staticmethod
    def _local(name: str, server: Server) -> dict[str, Json]:
        literal = {key: value for key, value in server.env.items() if not REFERENCE.search(value)}
        forwarded = [key for key in server.env if key not in literal]
        if renamed := [key for key in forwarded if server.env[key] != f"${{{key}}}"]:
            raise MissionError(
                f"MCP server {name!r}: Codex forwards a variable under its own name only, so "
                f"{', '.join(renamed)} must reference itself"
            )
        return _present(
            command=server.command, args=list(server.args), env=literal, env_vars=forwarded
        )

    @staticmethod
    def _remote(name: str, server: Server) -> dict[str, Json]:
        literal, named, bearer = {}, {}, {}
        for header, value in server.headers.items():
            referenced = REFERENCE.fullmatch(value.removeprefix("Bearer "))
            if not REFERENCE.search(value):
                literal[header] = value
            elif referenced and value.startswith("Bearer ") and header.lower() == "authorization":
                bearer["bearer_token_env_var"] = referenced[1]
            elif referenced and referenced[0] == value:
                named[header] = referenced[1]
            else:
                raise MissionError(
                    f"MCP server {name!r}: Codex sends header {header} only as a literal, a "
                    "whole `${NAME}` or `Bearer ${NAME}`"
                )
        return _present(url=server.url, http_headers=literal, env_http_headers=named) | bearer


class OpenCode(Harness):
    """opencode, reading `opencode.json`, AGENTS.md and `.agents/skills`; it runs no hooks,
    since its lifecycle extensions are JavaScript plugins."""

    name = "opencode"
    binary = "opencode"
    channel = Npm("opencode-ai", scripts=("opencode-ai",))
    credentials = (".local/share/opencode/auth.json",)
    keys = ("OPENROUTER_API_KEY",)
    login = "put OPENROUTER_API_KEY in .env, or opencode auth login"
    managed = ("opencode.json", ".opencode/agents/*.md")
    hooked = False

    def files(self, source: Source) -> dict[str, str]:
        settings = json.loads(source.own("opencode.json") or "{}") | {"mcp": self.servers(source)}
        files = {"opencode.json": _json(settings)}
        for agent in source.subagents:
            files[f".opencode/agents/{agent.name}.md"] = _markdown(
                {"description": agent.description, "mode": "subagent"}, agent.body
            )
        return files

    def server(self, name: str, server: Server) -> dict[str, Json]:
        if server.remote:
            return (
                {"type": "remote"}
                | _present(url=server.url, headers=_spelled(server.headers, "{env:%s}"))
                | {"enabled": True}
            )
        return (
            {"type": "local", "command": [server.command, *server.args]}
            | _present(environment=_spelled(server.env, "{env:%s}"))
            | {"enabled": True}
        )


def claude_key(path: str) -> str:
    """The directory name Claude Code files a workspace's state under: every other char a dash.

    path: the workspace root as that machine spells it, `C:\\Users\\me\\projects` say.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def _found(program: str, which: Which) -> str:
    """`program`'s path, refusing when this machine does not have it."""
    path = which(program)
    if path is None:
        raise MissionError(f"{program} is not on PATH here")
    return path


def _present(**fields: Json) -> dict[str, Json]:
    """The fields that say something, so a rendering never spells an empty default."""
    return {key: value for key, value in fields.items() if value}


def _spelled(values: Mapping[str, str], form: str) -> dict[str, str]:
    """`values` with every `${NAME}` written in a harness's own form, `{env:%s}` say."""
    return {
        key: REFERENCE.sub(lambda found: form % found[1], value) for key, value in values.items()
    }


def _json(document: Mapping[str, Json]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _markdown(front: Mapping[str, Json], body: str) -> str:
    """A Markdown card under YAML front matter, each field kept on one line as Claude writes it."""
    fields = yaml.safe_dump(dict(front), sort_keys=False, allow_unicode=True, width=1 << 20)
    return f"---\n{fields}---\n\n{body}"
