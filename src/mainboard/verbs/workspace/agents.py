from typing import TYPE_CHECKING

from ...agents import Agents
from ...engines.compile.ecosystems import Binary
from ...engines.compile.generated import GeneratedFiles
from ...engines.compile.provisioner import Provisioner
from ...manifest.loading import load
from ...render import progress

if TYPE_CHECKING:
    pass


# The values a compact record leaves out: they say nothing an absent field does not.

from ..base import Verbs
from ..options import Output, sectioned

# The verbs' default output and batch declaration.
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
        return sectioned(sections, output, title="agents check")

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
        output.print_rows([{"changed": path} for path in changed], title="agents sync")

    def agents_update(self, *, output: Output = _COMPACT) -> int:
        """Install every missing harness and bring every present one to its latest release.

        Claude Code through its own installer and `claude update`, Codex through `pixi
        global` from conda-forge, opencode and Gemini through npm into `~/.local`.

        Args:
        """
        return sectioned(
            Agents.at(self.session.workspace_root()).update(), output, title="agents update"
        )

    def binary_sync(self, *, env: str = "default", output: Output = _COMPACT) -> None:
        """Install declared binary tools without solving or reinstalling other ecosystems.

        The active platform's [binary.deps] and [binary.dev] ZIP coordinates require an
        HTTPS URL, exact archive SHA-256 and named executable member. No install runs from
        lint; a missing executable is a failed check until this explicit setup succeeds.

        Args:
            env: the environment whose tool directory is installed.
        """
        root = self.session.workspace_root()
        provisioner = Provisioner(root, load(self.session.project.manifest(root)))
        stage = provisioner.stage_for(env)
        with GeneratedFiles(directory=provisioner.out).locked():
            for ecosystem in stage.ecosystems(env):
                if isinstance(ecosystem, Binary):
                    ecosystem.sync()
                    output.print_record(
                        {name: spec.version for name, spec in ecosystem.deps.items()},
                        title="Binary tools",
                    )
