# The `.agents` folder, the one place a workspace writes what its coding agents share.
#
# Every harness reads its own files in its own format; what they have in common is written here
# once. The MCP servers sit in `mcp.json`, in the `mcpServers` shape most clients read. The
# lifecycle hooks sit in `settings.json` under `hooks`, Claude Code's format, which Codex adopted.
# The subagents are `agents/*.md`, Markdown under YAML front matter. The skills are `skills/`,
# which Codex and opencode read where they are. Settings only one harness
# understands sit beside them as its own file, `codex.toml` say, and are laid under what is
# rendered for it.

import re
from pathlib import Path
from typing import ClassVar, Literal

from patos import FrozenModel, FrozenOpenModel
from pydantic import Field, field_validator

from ..core.errors import MissionError
from ..probe.census import Json

_DEFAULTED_REFERENCE_MESSAGE = (
    "a reference is `${NAME}`; a default would be honored by one harness"
)

_DEFAULTED = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-")


class Server(FrozenModel):
    """One MCP server, spelled as Claude Code's `.mcp.json` spells it.

    type: `stdio` starts `command`; `http` (streamable) and `sse` reach `url`.
    env: variables the started command gets, each value literal or referencing `${NAME}`.
    headers: sent with every request to `url`, each value literal or referencing `${NAME}`.
    overrides: keys laid verbatim over one harness's rendering, by harness name, for what only
        that harness can say (Codex's OAuth scopes, say).
    """

    type: Literal["stdio", "http", "sse"] = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = {}
    url: str = ""
    headers: dict[str, str] = {}
    overrides: dict[str, dict[str, Json]] = {}

    # How a shared file references a variable. Each harness spells it its own way, and no default
    # (`${NAME:-x}`) is accepted, since only Claude Code would honor one.
    REFERENCE: ClassVar[re.Pattern[str]] = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    @field_validator("env", "headers")
    @classmethod
    def undefaulted(cls, values: dict[str, str]) -> dict[str, str]:
        """Refuse a `${NAME:-default}`, which every harness but Claude Code would mangle."""
        if any(_DEFAULTED.search(value) for value in values.values()):
            raise ValueError(_DEFAULTED_REFERENCE_MESSAGE)
        return values

    @property
    def remote(self) -> bool:
        return self.type != "stdio"

    @property
    def variables(self) -> set[str]:
        """Every variable the server's environment or headers reference."""
        return {
            name
            for value in (*self.env.values(), *self.headers.values())
            for name in Server.REFERENCE.findall(value)
        }


class HookGroup(FrozenModel):
    """The hooks one lifecycle event runs where `matcher` (a regex over tool names) matches.

    hooks: each handler as Claude Code spells it (`type`, `command`, `timeout` in seconds, and
        whatever else a harness reads), passed through as written.
    """

    matcher: str = ""
    hooks: tuple[dict[str, Json], ...]

    def spelled(self) -> dict[str, Json]:
        hooks: list[Json] = list(self.hooks)
        return {"matcher": self.matcher, "hooks": hooks} if self.matcher else {"hooks": hooks}


class Subagent(FrozenModel):
    """One subagent: a name, when to call it, and the instructions it works under."""

    name: str
    description: str
    body: str

    @classmethod
    def read(cls, path: Path) -> Subagent:
        """The subagent a `---`-fenced front matter and its Markdown body describe.

        The front matter is read as Claude Code reads it, one `key: value` per line, since a
        description it accepts (a colon inside, `\\n` written out) is often not valid YAML.
        """
        _, front, body = path.read_text(encoding="utf-8").split("---\n", 2)
        fields = {
            key: value for key, _, value in (line.partition(":") for line in front.splitlines())
        }
        return cls(
            name=fields["name"].strip(),
            description=fields["description"].strip(),
            body=body.lstrip(),
        )


class Source(FrozenModel):
    """What the workspace's `.agents` folder declares for every harness alike."""

    root: Path
    servers: dict[str, Server] = {}
    hooks: dict[str, tuple[HookGroup, ...]] = {}
    subagents: tuple[Subagent, ...] = ()

    # The folder every harness's configuration renders from.
    FOLDER: ClassVar[str] = ".agents"

    @classmethod
    def read(cls, root: Path) -> Source:
        """The `.agents` folder under `root`; a file it lacks declares nothing."""
        folder = root / Source.FOLDER
        try:
            return cls(
                root=root,
                servers=_read(_Servers, folder / "mcp.json").servers,
                hooks=_read(_Settings, folder / "settings.json").hooks,
                subagents=tuple(
                    Subagent.read(path) for path in sorted((folder / "agents").glob("*.md"))
                ),
            )
        except (ValueError, KeyError) as fault:
            raise MissionError(f"{Source.FOLDER} does not read: {fault}") from fault

    def own(self, name: str) -> str:
        """One harness's own settings file in `.agents`, empty when it has none."""
        try:
            return (self.root / Source.FOLDER / name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""


class _Servers(FrozenOpenModel):
    """`.agents/mcp.json`, the servers every harness is handed."""

    servers: dict[str, Server] = Field(default={}, alias="mcpServers")


class _Settings(FrozenOpenModel):
    """`.agents/settings.json`, of which the hooks are shared."""

    hooks: dict[str, tuple[HookGroup, ...]] = {}


def _read[File: FrozenOpenModel](model: type[File], path: Path) -> File:
    """`path` read as `model`, which declares nothing when the file is absent."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return model()
    return model.model_validate_json(text)
