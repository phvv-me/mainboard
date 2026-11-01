from ...agents import Agents
from ...render import Output, progress
from ..constants import TREE
from ..options import sectioned
from ..session import Verbs

# The verbs' default output.
_COMPACT = Output()


class AgentVerbs(Verbs):
    """The `agents` verbs."""

    def agents_check(self, *, output: Output = _COMPACT) -> int:
        """Judge what an agent started here meets, each row with its fix; 1 when one failed.

        The links, AGENTS.md, files out of step with `.agents`, servers whose command or
        variables this machine lacks, the Claude Code memory of this workspace, and each
        harness's release and login.

        Args:
        """
        with progress("checking the agents"):
            sections = Agents.at(self.session.workspace_root()).sections()
        return sectioned(sections, output, title="code agents check")

    def agents_sync(self, *, output: Output = _COMPACT) -> None:
        """Render each harness's files from `.agents`, remove stale ones and make the links.

        `.agents` declares what every harness shares: the MCP servers in `mcp.json`, the hooks
        under `hooks` in `settings.json`, the subagents in `agents/*.md` and the skills in
        `skills/`. Claude Code reads the folder itself through the `.claude` link; Codex,
        opencode and Gemini get their own files (`.codex/`, `opencode.json`, `.gemini/`), each
        with that harness's own settings file in `.agents` (`codex.toml`, `opencode.json`,
        `gemini.json`) laid under it. The rendered files are tracked, so a fresh clone works
        without this tool; edit `.agents`, never them.

        Args:
        """
        changed = Agents.at(self.session.workspace_root()).sync()
        output.print_rows([{"changed": path} for path in changed], title="code agents sync")

    def agents_update(self, *, output: Output = _COMPACT) -> int:
        """Install every missing harness and bring every present one to its latest release.

        Claude Code through its own installer and `claude update`, Codex through `pixi
        global` from conda-forge, opencode and Gemini through npm into `~/.local`.

        Args:
        """
        return sectioned(
            Agents.at(self.session.workspace_root()).update(), output, title="code agents update"
        )


_AGENTS = AgentVerbs()
agents = TREE.node("code agents").app(
    check=_AGENTS.agents_check, sync=_AGENTS.agents_sync, update=_AGENTS.agents_update
)
