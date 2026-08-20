# The generator behind `mainboard new`: one project rendered from a template this workspace
# declares. copier is the engine and stays the engine, so nothing here renders anything. This
# module decides which template a name resolves to, where the project lands, what its answers
# are, and what the workspace still has to do with the task rows a template writes out.

import shlex
from typing import TYPE_CHECKING

from patos import FrozenModel
from plumbum import local as localhost

from .core.errors import MissionError
from .core.project import Project
from .engines.compile.backend import Process
from .manifest.schema.template import Template

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .board import Board

# The tool this workspace answers to, so no message below spells the binary's name.
_TOOL = Project().name

# The renderer, which ships as a declared Python requirement of this workspace rather than of
# this tool. mainboard installs environments for a living and reaches a template through one, so
# the engine is asked for by name through the runner instead of imported.
_COPIER = "copier"

# What a template writes for a monorepo project: the task rows the root manifest has to adopt.
_TASKS = f"{_TOOL}.tasks.toml"

# The file that makes a directory a copier template. A template named as a location to fetch is
# the engine's to resolve, so only one on this disk is checked before the render is paid for.
_MARKER = "copier.yml"


class Scaffolded(FrozenModel):
    """One rendered project and what the workspace still owes it.

    project: the project's slug, the directory name it was rendered under.
    path: where it was rendered.
    tasks: the generated task-row snippet, empty for a project that owns its own.
    paste: the file and table those rows belong in, empty when there are none.
    snippet: the rows themselves, so a caller reads them without opening the file.
    """

    project: str
    path: str
    tasks: str = ""
    paste: str = ""
    snippet: str = ""


class Scaffold:
    """Renders a project from one of the workspace's declared templates, through copier.

    Which templates exist is the manifest's `[templates]` table, so this verb generates whatever
    shapes the workspace keeps rather than the one shape this package would otherwise have to
    know. The answers come from the project name and from what the workspace already declared,
    which is what turns a questionnaire into one argument. What the render leaves behind is
    reported rather than acted on: the task rows go to the caller to paste, because the root
    manifest is a hand-curated file whose task table sits in the middle of it and whose
    neighbouring `pyproject.toml` needs the same project on its type-checker search path, and
    half of that edit landing automatically is worse than none of it.
    """

    def __init__(self, board: Board) -> None:
        """board: the workspace whose templates are rendered and whose runner reaches copier."""
        self.board = board

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
        template: the template to render, a name the manifest declares or any location copier
            accepts; the workspace's first declared template when empty.
        description: the one sentence the README and the task rows carry.
        dest: where to render it, under the template's own declared home when empty.
        answers: further template questions to answer, overriding what the manifest declares.
        """
        chosen = self.chosen(template)
        slug = _slug(name)
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

    def chosen(self, template: str) -> Template:
        """The template `template` names, the workspace's first declared one when empty.

        A declared name is looked up first and a declared path second, so spelling out a
        template's own directory still gets the home and the answers this workspace already
        decided for it rather than a bare render at the root. Anything else is handed to the
        engine as written, which is what lets a directory or a git URL be rendered without
        being declared at all.

        template: the declared name, path or URL the caller asked for.
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

    def located(self, template: Template) -> str:
        """Where the engine is pointed for `template`, refusing a local directory that is not one.

        A template to fetch is the engine's to resolve, so only a path on this disk is checked,
        and checking it is worth the line because a wrong workspace root is the usual reason a
        declared template is not where it says it is.

        template: the resolved template being rendered.
        """
        if _fetched(template.path):
            return template.path
        source = self.board.root / template.path
        if not (source / _MARKER).is_file():
            raise MissionError(f"no project template at {source}")
        return str(source)

    def copy(self, template: str, destination: Path, answers: Mapping[str, str]) -> None:
        """Run copier over `template` through the workspace runner, refusing on its failure.

        `--defaults` takes the template's own answer for every question these do not settle, so
        the render is one command rather than a prompt nobody can answer from a script.

        template: where the engine reads the template from.
        destination: where the project is written.
        answers: the questions this render settles itself.
        """
        data = [
            token
            for question, answer in answers.items()
            for token in ("--data", f"{question}={answer}")
        ]
        command = shlex.join([_COPIER, "copy", "--defaults", *data, template, str(destination)])
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


def _fetched(path: str) -> bool:
    """Whether `path` names a template the engine fetches rather than one on this disk."""
    return "://" in path or path.startswith("gh:") or path.endswith(".git")


def _slug(name: str) -> str:
    """`name` as a template spells a project directory, lowercase and hyphenated."""
    return name.strip().lower().replace(" ", "-").replace("_", "-")
