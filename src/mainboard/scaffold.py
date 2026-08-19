# The generator behind `mainboard new`: one research project rendered from the workspace's own
# copier template. copier is the engine and stays the engine, so nothing here renders anything.
# This module decides where a project lives, what its answers are, and what the workspace still
# has to do with the task rows the template writes out.

import shlex
from typing import TYPE_CHECKING

from patos import FrozenModel
from plumbum import local as localhost

from .core.errors import MissionError
from .core.project import Project
from .engines.compile.backend import Process

if TYPE_CHECKING:
    from pathlib import Path

    from .board import Board

# The tool this workspace answers to, so no message below spells the binary's name.
_TOOL = Project().name

# The template this verb renders and where a rendered project lands, both workspace-relative.
# A research project belongs beside its siblings, which is what lets one root manifest own
# every task and one atpx invocation cover every blueprint.
_TEMPLATE = "templates/research"
_HOME = "research"

# The renderer, which ships as a declared Python requirement of this workspace rather than of
# this tool. mainboard installs environments for a living and reaches its own template through
# one, so the engine is asked for by name through the runner instead of imported.
_COPIER = "copier"

# What the template writes for a monorepo project: the task rows the root manifest has to adopt.
_TASKS = f"{_TOOL}.tasks.toml"

# The two homes this verb offers. The template knows a third, a standalone repo managed by uv,
# which is not this tool's business to generate.
_MONOREPO = "monorepo"
_STANDALONE = "standalone-mainboard"


class Scaffolded(FrozenModel):
    """One rendered project and what the workspace still owes it.

    project: the project's slug, the directory name it was rendered under.
    path: where it was rendered.
    tasks: the generated task-row snippet, empty for a standalone project that owns its own.
    paste: the file and table those rows belong in, empty when there are none.
    snippet: the rows themselves, so a caller reads them without opening the file.
    """

    project: str
    path: str
    tasks: str = ""
    paste: str = ""
    snippet: str = ""


class Scaffold:
    """Renders a research project from the workspace's template, through copier.

    The answers come from the name, since every one the template asks for is either derived
    from it or has a default worth taking, which is what turns a questionnaire into one
    argument. What the render leaves behind is reported rather than acted on: the task rows go
    to the caller to paste, because the root manifest is a hand-curated file whose task table
    sits in the middle of it and whose neighbouring `pyproject.toml` needs the same project on
    its type-checker search path, and half of that edit landing automatically is worse than
    none of it.
    """

    def __init__(self, board: Board) -> None:
        """board: the workspace whose template is rendered and whose runner reaches copier."""
        self.board = board

    def render(
        self,
        name: str,
        *,
        standalone: bool = False,
        description: str = "",
        dest: str = "",
    ) -> Scaffolded:
        """Render `name` as a research project and report what it left for the workspace.

        name: the project name, which becomes its slug, its package and its task prefix.
        standalone: render the home that carries its own manifest instead of the monorepo one.
        description: the one sentence the README and the task rows carry.
        dest: where to render it, beside its siblings under the workspace when empty.
        """
        slug = _slug(name)
        home = _STANDALONE if standalone else _MONOREPO
        destination = (self.board.root / dest) if dest else (self.board.root / _HOME / slug)
        if destination.exists():
            raise MissionError(f"{destination} already exists")
        template = self.board.root / _TEMPLATE
        if not (template / "copier.yml").is_file():
            raise MissionError(f"no project template at {template}")
        answers = {
            "project_name": name,
            "description": description or f"{name} research project.",
            "home": home,
        }
        self.copy(template, destination, answers)
        return self.reported(slug, destination)

    def copy(self, template: Path, destination: Path, answers: dict[str, str]) -> None:
        """Run copier over `template` through the workspace runner, refusing on its failure.

        `--defaults` takes the template's own answer for every question these three do not
        settle, so the render is one command rather than a prompt nobody can answer from a
        script.
        """
        data = [
            token
            for question, answer in answers.items()
            for token in ("--data", f"{question}={answer}")
        ]
        command = shlex.join(
            [_COPIER, "copy", "--defaults", *data, str(template), str(destination)]
        )
        staged = self.board.line(command, container="none")
        result = Process.capture(localhost["bash"]["-lc", staged])
        if not result.succeeded:
            result.replay()
            raise MissionError(
                f"`{_COPIER} copy` failed. It is a declared requirement of this workspace, so "
                f"`{_TOOL} install` is what puts it there."
            )

    def reported(self, slug: str, destination: Path) -> Scaffolded:
        """What the render produced, with the task rows read back for the caller to paste."""
        rows = destination / _TASKS
        try:
            snippet = rows.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Scaffolded(project=slug, path=str(destination))
        return Scaffolded(
            project=slug,
            path=str(destination),
            tasks=str(rows),
            paste=f"{self.board.root / self.board.project.manifest} [tasks]",
            snippet=snippet,
        )


def _slug(name: str) -> str:
    """`name` as the template spells a project directory, lowercase and hyphenated."""
    return name.strip().lower().replace(" ", "-").replace("_", "-")
