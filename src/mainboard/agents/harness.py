# The coding agents a workspace configures alike, each a harness with its own formats.
#
# A harness knows where its binary comes from and how it is brought to the latest release, which
# files it reads, and how to write the shared `.agents` declarations into them: the MCP servers,
# the hooks it can run, and the subagents. Everything it renders is a whole file it owns, so a
# rendering is compared and written without merging into anything a person edited; what only one
# harness understands is that harness's own file in `.agents`, laid under the rendering.

import json
import re
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import tomlkit
import yaml
from patos import Registry
from tomlkit.exceptions import ParseError

from ..core import MissionError, captured
from .channels import Channel, Npm, PixiGlobal, Script
from .source import Server, Source

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ..probe.census import Json


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

    def logged_in(self, home: Path, environment: Mapping[str, str]) -> bool:
        return any((home / path).is_file() for path in self.credentials) or any(
            key in environment for key in self.keys
        )

    def servers(self, source: Source) -> dict[str, Json]:
        """Every server, each with its overrides for this harness laid over it."""
        return {
            name: self.server(name, server) | server.overrides.get(self.name, {})
            for name, server in source.servers.items()
        }


class Claude(Harness):
    """Claude Code, which reads `.agents` itself through the `.claude` link."""

    binary = "claude"
    channel = Script(
        script="curl -fsSL https://claude.ai/install.sh | bash", update=("claude", "update")
    )
    credentials = (".claude/.credentials.json",)
    keys = ("ANTHROPIC_API_KEY",)
    login = "claude, then /login"
    links = {".claude": Source.FOLDER}
    managed = (".mcp.json",)

    def files(self, source: Source) -> dict[str, str]:
        return {".mcp.json": _json({"mcpServers": self.servers(source)})}

    def logged_in(self, home: Path, environment: Mapping[str, str]) -> bool:
        """The macOS login lives in the keychain, every other one in its credentials file."""
        if sys.platform != "darwin":
            return super().logged_in(home, environment)
        found = captured(["security", "find-generic-password", "-s", "Claude Code-credentials"])
        return found.returncode == 0 or any(key in environment for key in self.keys)

    def server(self, name: str, server: Server) -> dict[str, Json]:
        if server.remote:
            return _present(type=server.type, url=server.url, headers=server.headers)
        return _present(command=server.command, args=list(server.args), env=server.env)


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

    @staticmethod
    def _bearer_token(*, header: str, value: str) -> str | None:
        """The variable a `Bearer ${NAME}` authorization header names, None for other headers."""
        referenced = Server.REFERENCE.fullmatch(value.removeprefix("Bearer "))
        if referenced and value.startswith("Bearer ") and header.lower() == "authorization":
            return referenced[1]
        return None

    @staticmethod
    def _local(name: str, server: Server) -> dict[str, Json]:
        literal = {
            key: value for key, value in server.env.items() if not Server.REFERENCE.search(value)
        }
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
    def _named_variable(name: str, *, header: str, value: str) -> str:
        """The variable a header reads when its whole value is `${NAME}`, refusing any other."""
        whole = Server.REFERENCE.fullmatch(value)
        if whole is None:
            raise MissionError(
                f"MCP server {name!r}: Codex sends header {header} only as a literal, a "
                "whole `${NAME}` or `Bearer ${NAME}`"
            )
        return whole[1]

    @staticmethod
    def _remote(name: str, server: Server) -> dict[str, Json]:
        literal, named, bearer = {}, {}, {}
        for header, value in server.headers.items():
            if not Server.REFERENCE.search(value):
                literal[header] = value
            elif token := Codex._bearer_token(header=header, value=value):
                bearer["bearer_token_env_var"] = token
            else:
                named[header] = Codex._named_variable(name, header=header, value=value)
        return _present(url=server.url, http_headers=literal, env_http_headers=named) | bearer

    @staticmethod
    def parsed(text: str) -> tomlkit.TOMLDocument:
        """A Codex config file read so that its comments and quoting survive a rewrite.

        Raises MissionError when the text is not TOML, as Claude's settings refuse a non-object.
        """
        try:
            return tomlkit.parse(text)
        except ParseError as fault:
            raise MissionError(f"Codex's config is not valid TOML: {fault}") from fault

    def files(self, source: Source) -> dict[str, str]:
        config = Codex.parsed(source.own("codex.toml"))
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

    @staticmethod
    def _markdown(front: Mapping[str, Json], body: str) -> str:
        """A Markdown card under YAML front matter, one line per field as Claude writes it."""
        fields = yaml.safe_dump(dict(front), sort_keys=False, allow_unicode=True, width=1 << 20)
        return f"---\n{fields}---\n\n{body}"

    def files(self, source: Source) -> dict[str, str]:
        settings = json.loads(source.own("opencode.json") or "{}") | {"mcp": self.servers(source)}
        files = {"opencode.json": _json(settings)}
        for agent in source.subagents:
            files[f".opencode/agents/{agent.name}.md"] = OpenCode._markdown(
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


def _present(**fields: Json) -> dict[str, Json]:
    """The fields that say something, so a rendering never spells an empty default."""
    return {key: value for key, value in fields.items() if value}


def _spelled(values: Mapping[str, str], form: str) -> dict[str, str]:
    """`values` with every `${NAME}` written in a harness's own form, `{env:%s}` say."""
    return {
        key: Server.REFERENCE.sub(lambda found: form % found[1], value)
        for key, value in values.items()
    }


def _json(document: Mapping[str, Json]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"
