"""The user-level telemetry settings of Claude Code and Codex, merged into files the owner edits.

`serve agents sync` points both harnesses at the sink, logs only, with prompts, responses and tool
content off. It writes only the keys it owns and leaves every other byte of each file as it was:
Claude's `env` entries and `otelHeadersHelper` in `settings.json`, Codex's `[otel]` table in
`config.toml`. Both are user-level, since a committed project file is also read by cloud sessions
and hosts, and Codex ignores a project's `otel` altogether.

Claude reads the token through its headers helper, so its settings never hold it. Codex has no
such indirection, so the literal token goes into `[otel]`: it allows appends and nothing else.
Codex still logs the arguments of each tool call (no switch turns them off, 2026-10-29); its tool
output is cut to nothing and its prompts are redacted.

The settings `env` of Claude beats the process environment, which is how the owner's
`CLAUDE_CODE_ENABLE_TELEMETRY = "0"` becomes `"1"` here, and how each `"0"` keeps a variable a
shell exports from turning content back on.
"""

import json
import sys
from abc import ABC, abstractmethod
from inspect import getfile
from shlex import join
from typing import TYPE_CHECKING, ClassVar

import tomlkit

from ...agents import Codex
from ...core import MissionError, write_atomic
from ...core.section import Section
from ...enums import Verdict
from . import headers

if TYPE_CHECKING:
    from pathlib import Path

    from .sink import Sink

# Claude's switches for what its log records carry; each content one off.
_CLAUDE_OFF = (
    "OTEL_LOG_USER_PROMPTS",
    "OTEL_LOG_ASSISTANT_RESPONSES",
    "OTEL_LOG_TOOL_DETAILS",
    "OTEL_LOG_TOOL_CONTENT",
    "OTEL_LOG_RAW_API_BODIES",
)


class Exporter(ABC):
    """One harness's user-level settings file, and the telemetry this tool merges into it.

    name: the harness, the row the sync reports.
    secret: whether the merged file holds the sink's token, and so is kept at 0600.
    """

    name: ClassVar[str]
    secret: ClassVar[bool] = False

    @abstractmethod
    def merged(self, text: str, sink: Sink) -> str:
        """`text`, the file as it stands (empty when absent), with this tool's keys merged in."""

    @abstractmethod
    def path(self, home: Path) -> Path:
        """Where the file is under the home directory `home`."""


class ClaudeExporter(Exporter):
    """Claude Code's `~/.claude/settings.json`: telemetry on, logs alone, to the sink."""

    name = "claude"

    @staticmethod
    def helper(sink: Sink) -> str:
        """The headers helper's command line: this interpreter running `headers.py` by path."""
        return join([sys.executable, "-I", getfile(headers), str(sink.token.path)])

    def merged(self, text: str, sink: Sink) -> str:
        settings = json.loads(text or "{}")
        if not isinstance(settings, dict) or not isinstance(settings.get("env", {}), dict):
            raise MissionError("Claude's settings are not a JSON object with an `env` object")
        settings["env"] = settings.get("env", {}) | {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_METRICS_EXPORTER": "none",
            "OTEL_TRACES_EXPORTER": "none",
            "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": sink.endpoint,
            **dict.fromkeys(_CLAUDE_OFF, "0"),
        }
        settings["otelHeadersHelper"] = ClaudeExporter.helper(sink)
        return json.dumps(settings, indent=2, ensure_ascii=False) + "\n"

    def path(self, home: Path) -> Path:
        return home / ".claude" / "settings.json"


class CodexExporter(Exporter):
    """Codex's `~/.codex/config.toml`: its `[otel]` logs exporter at the sink, content off."""

    name = "codex"
    secret = True

    def merged(self, text: str, sink: Sink) -> str:
        config = Codex.parsed(text)
        otel = config.get("otel") or tomlkit.table()
        exporter = tomlkit.inline_table()
        exporter["otlp-http"] = {
            "endpoint": sink.endpoint,
            "protocol": "json",
            "headers": {"Authorization": f"Bearer {sink.token.read()}"},
        }
        otel["exporter"] = exporter
        otel["log_user_prompt"] = False
        otel["log_agent_responses"] = False
        otel["tool_result"] = tomlkit.inline_table() | {"max_bytes": 0}
        config["otel"] = otel
        return tomlkit.dumps(config)

    def path(self, home: Path) -> Path:
        return home / ".codex" / "config.toml"


_EXPORTERS: tuple[Exporter, ...] = (ClaudeExporter(), CodexExporter())


class AgentTelemetry:
    """The telemetry settings of every harness under one home directory, pointed at one sink.

    sink: the sink every harness exports to.
    home: the home directory the files sit under; the owner's own only through `serve agents
        sync`, any other in a test.
    """

    @staticmethod
    def _text(path: Path) -> str:
        """The file at `path`, empty when there is none."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def __init__(self, sink: Sink, *, home: Path) -> None:
        self.sink = sink
        self.home = home

    def _synced(self, exporter: Exporter) -> Section:
        """Merge into one harness's file, rewritten only when it changes.

        A new file, or one that now holds the token, is kept at 0600; any other keeps its mode.
        """
        path = exporter.path(self.home)
        text = AgentTelemetry._text(path)
        merged = exporter.merged(text, self.sink)
        if merged == text:
            return Section(section=exporter.name, verdict=Verdict.PASS, detail=f"{path} current")
        mode = path.stat().st_mode & 0o777 if text and not exporter.secret else 0o600
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_atomic(path, merged, mode=mode)
        detail = f"{path} now exports logs to {self.sink.endpoint}"
        return Section(section=exporter.name, verdict=Verdict.PASS, detail=detail)

    def sync(self) -> list[Section]:
        """Point every harness at the sink, one row per file, the sink's token made first."""
        self.sink.token.ensure()
        return [self._synced(exporter) for exporter in _EXPORTERS]
