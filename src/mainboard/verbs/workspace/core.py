import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from cyclopts import App, Parameter
from patos import FrozenModel, Runtime
from pydantic import ConfigDict, Field, model_validator

from ...center.readiness.verify import Verification
from ...center.standalone import Standalone
from ...core.errors import MissionError
from ...lint import Inventory, Linter
from ...lint.files.git import topmost
from ...log import logger
from ...manifest.composing import composition
from ...manifest.loading import load
from ...render import mode_of, progress, record
from ...results import Results
from ..base import Verbs
from ..help import Help
from ..options import Output, answers, query_source, sectioned

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...lint import Report

# What `query` shows unless asked otherwise.
_ROWS = 1000
_BYTES = 10000


@Parameter(name="*")
class Caps(FrozenModel):
    """How much of a query's answer is shown, and how long it may run."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    max_rows: int = Field(default=_ROWS, ge=-1)
    """displayed row limit, 1000 by default; 0 or -1 removes the limit."""
    max_bytes: int = Field(default=_BYTES, ge=-1)
    """encoded data-row byte limit, 10000 by default; 0 or -1 removes the limit. After truncation
    count/hash at most 10000 more rows, then stop the query early. Rows wider than the cap are
    omitted whole; the marker reports displayed rows."""
    timeout: float = 10
    """DuckDB execution limit in seconds, 10 by default; 0 disables it. Timeouts fail the
    command. A client limit cannot interrupt a blocking Quack operator: `mb lake serve
    --timeout` sets the server limit (default 10s)."""

    @property
    def capped(self) -> bool:
        """Whether a display cap was asked for."""
        return (self.max_rows, self.max_bytes) != (_ROWS, _BYTES)


@Parameter(name="*")
class Shown(FrozenModel):
    """Where a query's answer goes: compact rows, JSON, a rich table, or a file."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    json_: Annotated[bool, Parameter(name="--json")] = False
    """print JSON when no output file is requested. Its row-count/hash/snapshot footer goes to
    stderr to keep the array parseable."""
    human: bool = False
    """print a rich table instead of TSV. TSV NULL is bare NULL; text NULL is quoted."""
    out: Path | None = None
    """export a complete new .csv, .parquet, or .json file; display caps and output modes do not
    apply. The execution timeout still applies."""

    @model_validator(mode="after")
    def _one_mode(self) -> Self:
        """Refuse both modes at once before the query runs."""
        mode_of(json_mode=self.json_, human=self.human)
        return self

    @property
    def mode(self) -> str | None:
        """The render key, `None` for the compact default."""
        return mode_of(json_mode=self.json_, human=self.human)


@Parameter(name="*")
class Scaffolding(FrozenModel):
    """Which template `new` renders, and what it answers."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    template: str = ""
    """a declared template name or any location copier accepts."""
    description: str = ""
    """the one sentence the README and the task rows carry."""
    dest: str = ""
    """where to render it, under the template's own declared home when omitted."""
    answer: tuple[str, ...] = ()
    """a further `question=value` for the template, repeatable."""


# The verbs' defaults: compact output, the first template, a query's display caps on screen.
_COMPACT = Output()
_SCAFFOLDING = Scaffolding()
_CAPS = Caps()
_SHOWN = Shown()


class WorkspaceVerbs(Verbs):
    """The `workspace` verbs."""

    app: Runtime[App]

    @staticmethod
    def _answered(
        results: Results, source: str | Path, *, project: str, caps: Caps, shown: Shown
    ) -> None:
        """Show `source`'s answer as `shown` asks, or export it whole to `shown.out`."""
        if shown.out is None:
            with results.stream(source, project=project, timeout=caps.timeout) as query:
                query.render(mode=shown.mode, max_rows=caps.max_rows, max_bytes=caps.max_bytes)
                sys.stdout.flush()
            return
        if shown.json_ or shown.human or caps.capped:
            raise MissionError(
                "--out exports all rows; display caps, --json and --human do not apply"
            )
        exported = results.export(source, shown.out, project=project, timeout=caps.timeout)
        logger.result(exported.as_posix())

    @staticmethod
    def _files(inventory: Inventory, paths: Sequence[Path], *, staged: bool) -> list[Path]:
        """The files a lint pass reads: the index, everything under `paths`, or what changed."""
        if staged:
            return inventory.staged()
        if paths:
            return inventory.under([path.resolve() for path in paths])
        return inventory.changed()

    @staticmethod
    def _reported(report: Report, *, json: bool) -> None:
        """Print a lint report: canonical JSON, or the findings and a summary line."""
        if json:
            record(report.model_dump(mode="json"), mode="json", fields=(), title="lint")
            return
        if report.failures:
            logger.result(report.findings())
        logger.result(report.summary())

    def completion(self, shell: Literal["bash", "zsh", "fish"]) -> None:
        """Print the script that completes this tool's commands and options in a shell.

        Keep it in the shell's startup: `eval "$(mb completion zsh)"` (zsh, bash), or
        `mb completion fish | source` in fish.

        Args:
            shell: the shell the script is for.
        """
        logger.result(
            self.app.generate_completion(prog_name=self.session.project.name, shell=shell)
        )

    def doctor(
        self,
        env: str = "",
        *,
        center: bool = False,
        members: bool = False,
        fix: bool = False,
        output: Output = _COMPACT,
    ) -> int:
        """Say whether this workspace is fit to work in, and exit nonzero when it is not.

        Asked at once and bounded: is the manifest coherent, is what is installed the
        environment it describes, what compute answers, do onboarded hosts match the manifest,
        does a periodic pass settle jobs, does every declared gate hold. Each row names the one
        command that repairs it; a sleeping host or a provider with no key is a word, not a
        failure.

        Args:
            env: the environment to examine, the local profile's own when omitted.
            center: also judge this machine as the workspace's center: git tooling (lfs, symlinks,
                long paths), agent configuration, the default environment on every shell's PATH, a
                torch and CUDA smoke run, scripts that behave differently here.
            members: also check every `[workspace] members` project works cloned alone: no import,
                path or task that only the monorepo satisfies, and a clean `uv` install.
            fix: first repair the CUDA library clashes the `libraries` rows name, as a manifest
                edit (the owning package raised to its newest release, then re-locked and
                installed) or a link inside the environment, never a system package, and print
                one row for each change. The report that follows shows what is left.
        """
        local = self.session.board("local")
        if fix:
            with progress("repairing library clashes"):
                repaired = local.doctor(env).repair()
            if repaired:
                sectioned(repaired, output, title="fix", heading=True)
        with progress("examining the workspace"):
            sections = Verification(local).sections() if center else local.doctor(env).sections()
        code = sectioned(sections, output, title="doctor")
        if members:
            standalone = Standalone(
                composition(self.session.project.manifest(self.session.workspace_root()))
            )
            with progress("checking the members alone"):
                alone = standalone.sections(())
            code |= sectioned(alone, output, title="members", heading=True)
        return code

    def help_(self, *query: str) -> None:
        """Show command help or search shipped docs and Python docstrings without a workspace.

        Args:
            query: an exact command path such as `batch run`, or words such as `Log.read_table`.
        """
        Help(self.app).show(" ".join(query))

    def lint(
        self,
        *paths: Path,
        check: bool = False,
        only: str = "",
        staged: bool = False,
        json: bool = False,
    ) -> int:
        """Fix what can be fixed, then check, over the changed files or everything under PATHS.

        With no path the pass reads every file that differs from HEAD or is new, owned
        submodules entered and deletions included, so the everyday call costs what the edit did.
        A path widens it to every file git tracks or would track at or beneath it, so `lint .` at
        the root reads the whole workspace. The exit is nonzero when a file was rewritten or a
        step failed, the one answer a person, an agent, a hook and a CI job all act on.

        Args:
            paths: files or directories, relative to the working directory.
            check: write nothing: run each tool's read-only `check` command and report what the
                text hygiene would repair, the mode for CI and for verifying a tree.
            only: the steps to run, comma-separated tool names with `text` for the built-in
                hygiene; every step when empty, so `--only ruff-format` is a formatter alone.
            staged: read the files in the index of the root and of each owned submodule, what a
                commit would hold, the file set of the pre-commit hook. It answers to the
                topmost superproject's workspace, wherever it runs. A fixer rewrites the
                working file, so the commit stops once and the files are staged again.
            json: print the report as canonical JSON instead of the findings and a summary line.
        """
        if staged and paths:
            raise MissionError("--staged reads the index; name no paths with it")
        root = (topmost(Path.cwd()) if staged else None) or self.session.workspace_root()
        manifest = load(self.session.project.manifest(root))
        files = WorkspaceVerbs._files(Inventory(root, manifest.git), paths, staged=staged)
        steps = [step.strip() for step in only.split(",") if step.strip()]
        report = Linter(root, manifest, check=check, only=steps).lint(files)
        WorkspaceVerbs._reported(report, json=json)
        return 0 if report.clean else 1

    def new(
        self,
        name: str,
        *,
        scaffolding: Scaffolding = _SCAFFOLDING,
        output: Output = _COMPACT,
    ) -> None:
        """Scaffold a project from one of this workspace's declared templates.

        Which templates exist is the manifest's `[templates]` table, and the first declared one
        is what this renders when none is named. Every answer a template asks for comes from
        the name, from what the workspace already declared, or from the template's own default,
        so a project stays one argument while `--answer` covers the rest. The task rows a
        template generates are printed rather than pasted, since the root manifest's task table
        is hand-curated and the same project has to reach the type checker's search path beside
        it, and half of that edit landing on its own is worse than none of it.

        Args:
            name: the project name, which becomes its slug, its package and its task prefix.
        """
        with progress(f"rendering {name}"):
            made = (
                self.session.board("local")
                .scaffold()
                .render(
                    name,
                    template=scaffolding.template,
                    description=scaffolding.description,
                    dest=scaffolding.dest,
                    answers=answers(scaffolding.answer),
                )
            )
        payload = made.model_dump()
        # The rows print whole and pasteable at a terminal, since wrapped inside a table cell
        # they are harder to copy back out. A compact mode keeps the field, the only place a
        # caller reading the record gets them.
        if output.mode is None and made.snippet:
            logger.result(made.snippet)
            payload.pop("snippet")
        output.print_record(payload, title="new")

    def query(
        self,
        sql: str | None = None,
        *,
        file: Path | None = None,
        project: str = "",
        caps: Caps = _CAPS,
        shown: Shown = _SHOWN,
    ) -> None:
        """Explore collected results across servers; each query sees newly arrived files.

        The lake is attached read-only (writes go to session scratch tables), and DuckLake
        time travel (`AT (VERSION => n)`) recovers snapshots `lake compact` has not yet expired
        (older than 30 days).

        Views: runs, trials, events, metrics, artifacts, jobs, and every table of the
        workspace's state lake as `lake.<table>` (lake.events, lake.log_lines, lake.costs...).
        Project scopes science views;
        jobs always shows the fleet. Monitor refreshes tracked jobs; collect also imports
        results from native remote runs. Neither requires a shared database service.

        The hash identifies rendered values, not raw SQL types or an integrity seal. The
        snapshot_id covers lake.* only; file-backed views and jobs are not snapshot-pinned.

        Args:
            sql: one DuckDB statement; defaults to SELECT * FROM runs.
                Use unique column aliases for results that return rows.
            file: read SQL from a UTF-8 file instead of the positional statement.
                Both the file and paths inside SQL resolve from the caller's current directory.
            project: restrict scientific rows to this research project.
        """
        source = query_source(sql, file) or "SELECT * FROM runs"
        results = Results(self.session.workspace_root())
        try:
            WorkspaceVerbs._answered(results, source, project=project, caps=caps, shown=shown)
        except ValueError as fault:
            raise MissionError(str(fault)) from None
