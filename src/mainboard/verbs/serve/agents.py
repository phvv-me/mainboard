from pathlib import Path

from ...render import Output
from ...serve.events.agents import AgentTelemetry
from ...serve.events.sink import Sink
from ..constants import TREE
from ..options import sectioned
from ..session import Verbs

_COMPACT = Output()


class AgentVerbs(Verbs):
    """The `serve agents` verbs: the harnesses' user-level settings, the owner's to write."""

    def sync(self, *, output: Output = _COMPACT) -> int:
        """Point Claude Code and Codex at this workspace's telemetry sink, logs only.

        Merges Claude's telemetry keys into `~/.claude/settings.json` and only `[otel]` into
        `~/.codex/config.toml`, leaving the rest of each file as it was; prompts, responses and
        tool content stay off. Makes the sink's token when there is none: Claude reads it from its
        0600 file, Codex holds it literally. Refuses when an agent runs it.
        """
        telemetry = AgentTelemetry(Sink(self.session.workspace_root()), home=Path.home())
        return sectioned(telemetry.sync(), output, title="serve agents sync")


_AGENTS = AgentVerbs()
agents = TREE.node("serve agents").app(sync=_AGENTS.sync)
