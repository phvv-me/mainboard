# The generator behind `mainboard new`. copier renders; this module only decides which template a
# name resolves to, where the project lands, its answers, and what the workspace still owes the
# task rows a template writes out.

from typing import TYPE_CHECKING

from patos import FrozenModel

from .core.errors import MissionError
from .core.project import Project
from .engines.compile.provisioner import Provisioner
from .manifest.schema.template import Template

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .board import Board

_TOOL = Project().name

# A declared requirement of this workspace, not of this tool, so it is reached by name through
# the workspace runner instead of imported.
_COPIER = "copier"

# The task rows a monorepo project's template writes for the root manifest to adopt.
_TASKS = f"{_TOOL}.tasks.toml"

# The file that makes a directory a copier template.
_MARKER = "copier.yml"


class Scaffolded(FrozenModel):
    """One rendered project and what the workspace still owes it.

    project: the project's slug, the directory name it was rendered under.
    tasks: the generated task-row file, empty for a project that owns its own.
    paste: the file and table those rows belong in, empty when there are none.
    snippet: the rows themselves, so a caller reads them without opening the file.
    """

    project: str
    path: str
    tasks: str = ""
    paste: str = ""
    snippet: str = ""


class Scaffold:
    """Renders a project from one of the manifest's `[templates]`, through copier.

    The answers come from the project name and what the workspace declared, turning a
    questionnaire into one argument. The task rows are reported for the caller to paste, not
    written: the root manifest is hand-curated with its task table mid-file, its neighbouring
    `pyproject.toml` needs the project on its type-checker path too, and half of that edit
    landing automatically is worse than none of it.
    """

    def __init__(self, board: Board) -> None:
        self.board = board

    def chosen(self, template: str) -> Template:
        """The template `template` names, the workspace's first declared one when empty.

        A declared name matches first, then a declared path, so spelling out a template's own
        directory still gets its declared home and answers. Anything else goes to the engine as
        written, so an undeclared directory or git URL renders too.
        """
        declared = self.board.manifest.templates
        if not template:
            first = next(iter(declared.values()), None)
            if first is None:
                raise MissionError(
                    f"this workspace declares no templates. Add a [templates] table to "
                    f"{self.board.project.manifest}, or name one with --template."
                )
            return first
        known = declared.get(template) or next(
            (entry for entry in declared.values() if entry.path == template), None
        )
        return known or Template(path=template)

    def copy(self, template: str, destination: Path, answers: Mapping[str, str]) -> None:
        """Run copier over `template` through the workspace runner, refusing on its failure.

        `--defaults` takes the template's own answer for every question `answers` leaves open,
        so the render is one command rather than a prompt nobody can answer from a script.
        """
        data = [
            token
            for question, answer in answers.items()
            for token in ("--data", f"{question}={answer}")
        ]
        command = (_COPIER, "copy", "--defaults", *data, template, str(destination))
        plan = self.board.plan(container="none")
        result = Provisioner(self.board.root, self.board.manifest).capture(command, plan.env)
        if not result.succeeded:
            result.replay()
            raise MissionError(
                f"`{_COPIER} copy` failed. It is a declared requirement of this workspace, so "
                f"`{_TOOL} install` is what puts it there."
            )

    def located(self, template: Template) -> str:
        """Where the engine is pointed for `template`, refusing a local directory that is not one.

        A scheme, a `gh:` prefix or a `.git` suffix is fetched by the engine; only a path on this
        disk is checked, before the render is paid for, since a wrong workspace root is the
        usual reason a declared template is not where it says.
        """
        fetched = (
            "://" in template.path
            or template.path.startswith("gh:")
            or template.path.endswith(".git")
        )
        if fetched:
            return template.path
        source = self.board.root / template.path
        if not (source / _MARKER).is_file():
            raise MissionError(f"no project template at {source}")
        return str(source)

    def render(
        self,
        name: str,
        *,
        template: str = "",
        description: str = "",
        dest: str = "",
        answers: Mapping[str, str] = {},
    ) -> Scaffolded:
        """Render `name` from a template and report what the render left for the workspace.

        name: the project name, which becomes its slug, its package and its task prefix.
        template: a declared name or any location copier accepts, else the first declared one.
        description: the one sentence the README and the task rows carry.
        dest: where to render it, under the template's own declared home when empty.
        answers: further template answers, overriding what the manifest declares.
        """
        chosen = self.chosen(template)
        slug = name.strip().lower().replace(" ", "-").replace("_", "-")
        destination = (self.board.root / dest) if dest else (self.board.root / chosen.into / slug)
        if destination.exists():
            raise MissionError(f"{destination} already exists")
        source = self.located(chosen)
        settled = {
            **chosen.answers,
            **answers,
            "project_name": name,
            "description": description or name,
        }
        self.copy(source, destination, settled)
        return self.reported(slug, destination)

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
